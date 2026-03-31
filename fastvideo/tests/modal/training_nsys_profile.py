# SPDX-License-Identifier: Apache-2.0
"""
Modal app: FastVideo training MFU profiling with NVTX + Nsight Systems.

Runs 2-GPU torchrun training wrapped by nsys on Modal, producing a .nsys-rep
and .sqlite file with full NVTX labels (from PR #1087) plus MFU printed to
stdout.

Outputs written to the 'fastvideo-nsys-training' volume:
  mfu_training.nsys-rep  — Nsight Systems report (CUDA kernels + NVTX)
  mfu_training.sqlite    — SQLite export for programmatic queries

Usage:
  modal run training_nsys_profile.py
  TRAIN_NUM_STEPS=5 modal run training_nsys_profile.py

Environment variables (set before ``modal run``):
  TRAIN_MODEL_ID      : HuggingFace model ID
                        (default: Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
  TRAIN_NUM_GPUS      : number of GPUs / torchrun nproc_per_node (default: 2)
  TRAIN_NUM_STEPS     : max_train_steps (default: 5)
  TRAIN_GRAD_ACCUM    : gradient accumulation steps (default: 1)
  HF_API_KEY          : HuggingFace token for gated models
  WANDB_API_KEY       : WandB API key (optional; uses offline mode if absent)
  IMAGE_VERSION       : fastvideo-dev image tag suffix (default: py3.12-latest)

By default clones hao-ai-lab/FastVideo and fetches PR #1087 (Ohm's NVTX
tracer). Override with BUILDKITE_REPO / BUILDKITE_COMMIT / BUILDKITE_PULL_REQUEST.
"""

import os
import subprocess
import sys
import textwrap

import modal

app = modal.App()

model_vol = modal.Volume.from_name("hf-model-weights")
results_vol = modal.Volume.from_name(
    "fastvideo-nsys-training",
    create_if_missing=True,
)

local_fastvideo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


image_version = os.getenv("IMAGE_VERSION", "py3.12-latest")
image_tag = f"ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:{image_version}"
print(f"Using image for training profiling: {image_tag}")

image = (
    modal.Image.from_registry(image_tag, add_python="3.12")
    .add_local_dir(
        local_fastvideo_dir,
        remote_path="/FastVideo",
        ignore=[".git", ".venv", "nsys_results", "__pycache__", "comfyui"],
        copy=True,
    )
    .run_commands("rm -rf /FastVideo/.git")
    .apt_install(
        "cmake",
        "pkg-config",
        "build-essential",
        "curl",
        "libssl-dev",
        "ffmpeg",
        "gnupg",
    )
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf "
        "https://sh.rustup.rs | sh -s -- -y --default-toolchain stable"
    )
    .run_commands("echo 'source ~/.cargo/env' >> ~/.bashrc")
    # nsys CLI
    .run_commands(
        "set -e && "
        ". /etc/lsb-release 2>/dev/null || true; "
        "UBUNTU_VER=${DISTRIB_RELEASE:-22.04}; "
        "UBUNTU_VER=$(echo $UBUNTU_VER | tr -d .); "
        "apt-get update && "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y "
        "  --no-install-recommends gnupg && "
        "echo \"deb http://developer.download.nvidia.com/devtools/"
        "repos/ubuntu${UBUNTU_VER}/$(dpkg --print-architecture) /\" "
        "> /etc/apt/sources.list.d/nvidia-devtools.list && "
        "apt-key adv --fetch-keys "
        "http://developer.download.nvidia.com/compute/cuda/repos/"
        "ubuntu1804/x86_64/7fa2af80.pub || true && "
        "apt-get update && "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y "
        "  --no-install-recommends nsight-systems-cli"
    )
    .env({
        "PATH": "/root/.cargo/bin:$PATH",
        "BUILDKITE_REPO": os.environ.get("BUILDKITE_REPO", ""),
        "BUILDKITE_COMMIT": os.environ.get("BUILDKITE_COMMIT", ""),
        "BUILDKITE_PULL_REQUEST": os.environ.get(
            "BUILDKITE_PULL_REQUEST", ""
        ),
        "IMAGE_VERSION": os.environ.get("IMAGE_VERSION", ""),
    })
)

# ---------------------------------------------------------------------------
# MFU training worker — written to /tmp at runtime and run via torchrun.
# NOTE: uses only single-quoted string literals to avoid nesting issues.
# ---------------------------------------------------------------------------

# Build the training script as a list of lines to avoid triple-quote nesting.
_MFU_LINES = [
    "import os",
    "import sys",
    "from pathlib import Path",
    "",
    "sys.path.insert(0, '/FastVideo')",
    "os.environ.setdefault('PYTHONPATH', '/FastVideo')",
    "",
    "from fastvideo.utils import FlexibleArgumentParser",
    "from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs",
    "from fastvideo.training.wan_training_pipeline import WanTrainingPipeline",
    "from fastvideo.utils import logger",
    "from torch.profiler import ProfilerActivity, profile, record_function",
    "",
    "MODEL_PATH = os.environ.get('TRAIN_MODEL_ID',"
    " 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers')",
    "DATA_PATH  = '/data/crush-smol_processed_t2v"
    "/combined_parquet_dataset/worker_0/'",
    "OUTPUT_DIR = Path('/tmp/mfu_output')",
    "NUM_GPUS   = os.environ.get('TRAIN_NUM_GPUS', '2')",
    "NUM_STEPS  = os.environ.get('TRAIN_NUM_STEPS', '5')",
    "GRAD_ACCUM = os.environ.get('TRAIN_GRAD_ACCUM', '1')",
    "",
    "os.environ['FASTVIDEO_ATTENTION_BACKEND'] = 'FLASH_ATTN'",
    "os.environ.setdefault('WANDB_MODE', 'offline')",
    "os.environ['HF_HOME'] = '/root/data/.cache'",
    "",
    "parser = FlexibleArgumentParser()",
    "parser = TrainingArgs.add_cli_args(parser)",
    "parser = FastVideoArgs.add_cli_args(parser)",
    "",
    "args = parser.parse_args([",
    "    '--model_path',                    MODEL_PATH,",
    "    '--inference_mode',                'False',",
    "    '--pretrained_model_name_or_path', MODEL_PATH,",
    "    '--data_path',                     DATA_PATH,",
    "    '--dataloader_num_workers',        '1',",
    "    '--train_batch_size',              '1',",
    "    '--train_sp_batch_size',           '1',",
    "    '--gradient_accumulation_steps',   GRAD_ACCUM,",
    "    '--num_latent_t',                  '20',",
    "    '--num_height',                    '720',",
    "    '--num_width',                     '1280',",
    "    '--num_frames',                    '77',",
    "    '--enable_gradient_checkpointing_type', 'full',",
    "    '--max_train_steps',               NUM_STEPS,",
    "    '--learning_rate',                 '5e-5',",
    "    '--mixed_precision',               'bf16',",
    "    '--weight_only_checkpointing_steps',    '999999',",
    "    '--training_state_checkpointing_steps', '999999',",
    "    '--weight_decay',                  '1e-4',",
    "    '--max_grad_norm',                 '1.0',",
    "    '--num_euler_timesteps',           '50',",
    "    '--multi_phased_distill_schedule', '4000-1',",
    "    '--not_apply_cfg_solver',",
    "    '--training_cfg_rate',             '0.1',",
    "    '--ema_start_step',                '0',",
    "    '--dit_precision',                 'fp32',",
    "    '--output_dir',                    str(OUTPUT_DIR),",
    "    '--tracker_project_name',          'mfu_nsys_profile',",
    "    '--checkpoints_total_limit',       '1',",
    "    '--validation_steps',              '999999',",
    "    '--num_gpus',                      NUM_GPUS,",
    "    '--sp_size',                       NUM_GPUS,",
    "    '--tp_size',                       '1',",
    "    '--hsdp_replicate_dim',            NUM_GPUS,",
    "    '--hsdp_shard_dim',                '1',",
    "])",
    "",
    "pipeline = WanTrainingPipeline.from_pretrained(",
    "    args.pretrained_model_name_or_path, args=args)",
    "",
    "pipeline.train()",
    "",
    "logger.info('MFU training run complete.')",
]

_MFU_TRAINING_SCRIPT = "\n".join(_MFU_LINES) + "\n"

# ---------------------------------------------------------------------------
# Profiler sanity-check script — runs training with torch.profiler ONLY,
# NO Nsight Systems, so CUPTI is available to torch.profiler.
# After training, sums FLOPs across all SP ranks and compares to Ohm formula.
# ---------------------------------------------------------------------------

_PROFILER_SANITY_SCRIPT = textwrap.dedent("""\
import os
import sys
import json
import torch
import torch.distributed as dist
from pathlib import Path

sys.path.insert(0, '/FastVideo')
os.environ.setdefault('PYTHONPATH', '/FastVideo')

from fastvideo.utils import FlexibleArgumentParser
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.training.wan_training_pipeline import WanTrainingPipeline
from fastvideo.utils import logger
from torch.profiler import ProfilerActivity, profile, record_function

MODEL_PATH = os.environ.get('TRAIN_MODEL_ID', 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers')
DATA_PATH  = '/data/crush-smol_processed_t2v/combined_parquet_dataset/worker_0/'
OUTPUT_DIR = Path('/tmp/mfu_output')
NUM_GPUS   = os.environ.get('TRAIN_NUM_GPUS', '2')
NUM_STEPS  = os.environ.get('TRAIN_NUM_STEPS', '5')
GRAD_ACCUM = os.environ.get('TRAIN_GRAD_ACCUM', '1')
PEAK_FLOPS_PER_GPU = float(os.environ.get('PEAK_FLOPS_PER_GPU', '3.62e14'))

os.environ['FASTVIDEO_ATTENTION_BACKEND'] = 'FLASH_ATTN'
os.environ.setdefault('WANDB_MODE', 'offline')
os.environ['HF_HOME'] = '/root/data/.cache'

parser = FlexibleArgumentParser()
parser = TrainingArgs.add_cli_args(parser)
parser = FastVideoArgs.add_cli_args(parser)

args = parser.parse_args([
    '--model_path',                    MODEL_PATH,
    '--inference_mode',                'False',
    '--pretrained_model_name_or_path', MODEL_PATH,
    '--data_path',                     DATA_PATH,
    '--dataloader_num_workers',        '1',
    '--train_batch_size',              '1',
    '--train_sp_batch_size',           '1',
    '--gradient_accumulation_steps',   GRAD_ACCUM,
    '--num_latent_t',                  '20',
    '--num_height',                    '720',
    '--num_width',                     '1280',
    '--num_frames',                    '77',
    '--enable_gradient_checkpointing_type', 'full',
    '--max_train_steps',               NUM_STEPS,
    '--learning_rate',                 '5e-5',
    '--mixed_precision',               'bf16',
    '--weight_only_checkpointing_steps',    '999999',
    '--training_state_checkpointing_steps', '999999',
    '--weight_decay',                  '1e-4',
    '--max_grad_norm',                 '1.0',
    '--num_euler_timesteps',           '50',
    '--multi_phased_distill_schedule', '4000-1',
    '--not_apply_cfg_solver',
    '--training_cfg_rate',             '0.1',
    '--ema_start_step',                '0',
    '--dit_precision',                 'fp32',
    '--output_dir',                    str(OUTPUT_DIR),
    '--tracker_project_name',          'mfu_profiler_sanity',
    '--checkpoints_total_limit',       '1',
    '--validation_steps',              '999999',
    '--num_gpus',                      NUM_GPUS,
    '--sp_size',                       NUM_GPUS,
    '--tp_size',                       '1',
    '--hsdp_replicate_dim',            NUM_GPUS,
    '--hsdp_shard_dim',                '1',
])

pipeline = WanTrainingPipeline.from_pretrained(
    args.pretrained_model_name_or_path, args=args)

pipeline.train()

logger.info('MFU training run complete.')

local_rank = int(os.environ.get('LOCAL_RANK', '0'))
max_train_steps = int(NUM_STEPS)
world_size = int(NUM_GPUS)

# Print profiler table and collect per-rank FLOPs
try:
    if local_rank == 0:
        table = prof.key_averages().table(sort_by='flops', row_limit=20)
        logger.info('Torch profiler FLOPs table (top 20):\\n%s', table)
    rank_flops = sum(getattr(e, 'flops', 0) or 0 for e in prof.key_averages())
except Exception as exc:
    logger.warning('Profiler summary failed: %s', exc)
    rank_flops = 0.0

# Sum FLOPs across all SP ranks so we get the full model's FLOPs
flops_t = torch.tensor(rank_flops, dtype=torch.float64)
try:
    if dist.is_initialized():
        dist.all_reduce(flops_t, op=dist.ReduceOp.SUM)
except Exception as exc:
    logger.warning('dist.all_reduce failed, using rank-0 FLOPs only: %s', exc)
total_profiler_flops = flops_t.item()
profiler_flops_per_step = total_profiler_flops / max_train_steps

if local_rank == 0:
    # Load wandb summary written by rank 0 during training
    summary_path = (OUTPUT_DIR / 'tracker' / 'wandb' /
                    'latest-run' / 'files' / 'wandb-summary.json')
    wandb_data = {}
    if summary_path.exists():
        with summary_path.open() as f:
            wandb_data = json.load(f)
        logger.info('Loaded wandb summary from %s', summary_path)
    else:
        logger.warning('wandb summary not found at %s; using defaults', summary_path)

    batch_size    = wandb_data.get('batch_size',    1)
    seq_len       = wandb_data.get('dit_seq_len',   31200)
    context_len   = wandb_data.get('context_len',   512)
    hidden_dim    = wandb_data.get('hidden_dim',    1536)
    num_layers    = wandb_data.get('num_layers',    30)
    ffn_dim       = wandb_data.get('ffn_dim',       8960)
    avg_step_time = wandb_data.get('avg_step_time', 0.0)

    # Formula FLOPs (Ohm ground truth — same as mfu_calculation.py)
    qkv_out    = 8 * hidden_dim * hidden_dim * seq_len
    cross_proj = (4 * hidden_dim * hidden_dim * seq_len +
                  4 * hidden_dim * hidden_dim * context_len)
    mlp        = 4 * hidden_dim * ffn_dim * seq_len
    self_attn  = 4 * seq_len * seq_len * hidden_dim
    cross_attn = 4 * seq_len * context_len * hidden_dim
    flops_per_layer = qkv_out + cross_proj + mlp + self_attn + cross_attn
    formula_flops_per_step = (batch_size * flops_per_layer * num_layers *
                              4 * int(GRAD_ACCUM))

    total_peak   = PEAK_FLOPS_PER_GPU * world_size
    ratio        = (profiler_flops_per_step / formula_flops_per_step
                   if formula_flops_per_step else 0.0)
    mfu_formula  = (formula_flops_per_step / avg_step_time / total_peak * 100
                   if avg_step_time > 0 else 0.0)
    mfu_profiler = (profiler_flops_per_step / avg_step_time / total_peak * 100
                   if avg_step_time > 0 else 0.0)

    lines = [
        '=== Torch Profiler vs Formula MFU Sanity Check ===',
        f'world_size={world_size}  max_train_steps={max_train_steps}',
        f'seq_len={seq_len}  context_len={context_len}',
        f'hidden_dim={hidden_dim}  ffn_dim={ffn_dim}  num_layers={num_layers}',
        f'batch_size={batch_size}  avg_step_time={avg_step_time:.4f}s',
        f'peak_flops_per_gpu={PEAK_FLOPS_PER_GPU:.3e}',
        '',
        'FLOPs per step (summed across all ranks via dist.all_reduce):',
        f'  formula  (Ohm ground truth): {formula_flops_per_step:.4e}',
        f'  profiler (torch.profiler):   {profiler_flops_per_step:.4e}',
        f'  ratio    (profiler/formula): {ratio:.4f}',
        '',
        'MFU (using avg_step_time from wandb):',
        f'  formula  MFU: {mfu_formula:.4f}%',
        f'  profiler MFU: {mfu_profiler:.4f}%',
        '',
        'Note: torch.profiler under-counts FLOPs for custom CUDA kernels',
        '(FlashAttention etc.) not registered with the FLOPs counter.',
        'Expect ratio < 1 — the gap shows FlashAttn FLOPs missing from profiler.',
    ]
    report = '\\n'.join(lines) + '\\n'
    logger.info('\\n%s', report)

    out_path = Path('/results/profiler_sanity_check.txt')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    logger.info('Sanity check report saved to %s', out_path)
""")


# ---------------------------------------------------------------------------
# Helpers (reused from nsys_profile.py)
# ---------------------------------------------------------------------------

def _build_checkout_command(
    git_commit: str | None,
    pr_number: str | None,
) -> str:
    if pr_number and pr_number != "false":
        return (
            "git fetch --prune origin "
            f"refs/pull/{pr_number}/head && "
            "git checkout FETCH_HEAD"
        )
    if git_commit:
        return f"git checkout {git_commit}"
    return "git checkout HEAD"


def _build_install_script(git_repo: str, checkout_command: str) -> str:
    return f"""
    set -euo pipefail
    ulimit -s unlimited || ulimit -s 65536 || true
    mkdir -p /results/tmp_build
    export TMPDIR=/results/tmp_build
    export TEMP=/results/tmp_build
    export TMP=/results/tmp_build
    source $HOME/.local/bin/env
    source /opt/venv/bin/activate
    cd /FastVideo &&
    git submodule update --init --recursive -q || true &&
    export MAX_JOBS=1 &&
    export CMAKE_BUILD_PARALLEL_LEVEL=1 &&
    export SKBUILD_BUILD_ARGS="-j 1" &&
    cd fastvideo-kernel &&
    echo "[install] Building fastvideo-kernel..." &&
    if ! ./build.sh > /results/tmp_build/build.log 2>&1; then
        echo "[ERROR] fastvideo-kernel build failed! Last 50 lines:" >&2
        tail -n 50 /results/tmp_build/build.log >&2
        exit 1
    fi &&
    cd .. &&
    echo "[install] Installing FastVideo package..." &&
    if ! uv pip install -e .[test] > /results/tmp_build/install.log 2>&1; then
        echo "[ERROR] FastVideo install failed! Last 50 lines:" >&2
        tail -n 50 /results/tmp_build/install.log >&2
        exit 1
    fi &&
    export HF_HOME="/root/data/.cache" &&
    ( [ -z "${{HF_API_KEY:-}}" ] || \\
      hf auth login --token "$HF_API_KEY" --quiet || true )
    """


# ---------------------------------------------------------------------------
# Modal function — 2x L40S for 2-GPU torchrun training
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S:2",
    image=image,
    timeout=7200,
    memory=65536,
    secrets=[
        modal.Secret.from_dict({
            "HF_API_KEY": os.environ.get("HF_API_KEY", ""),
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        })
    ],
    volumes={"/root/data": model_vol, "/results": results_vol},
)
def run_fastvideo_training_nsys_profile(
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    num_gpus: int = 2,
    num_steps: int = 5,
    pr_number: str = "1087",
) -> int:
    """Profile FastVideo training with Nsight Systems + NVTX.

    Checks out PR #1087 (Ohm's NVTX tracer) from hao-ai-lab/FastVideo, runs
    2-GPU torchrun training wrapped by nsys, exports SQLite, and writes results
    to the 'fastvideo-nsys-training' volume.

    MFU value is printed to stdout from the training pipeline log.
    """
    git_repo = (
        os.environ.get("BUILDKITE_REPO")
        or "https://github.com/hao-ai-lab/FastVideo.git"
    )
    git_commit = os.environ.get("BUILDKITE_COMMIT")
    env_pr = os.environ.get("BUILDKITE_PULL_REQUEST") or pr_number
    checkout_command = _build_checkout_command(git_commit, env_pr)

    print(
        f"[training-nsys] model={model_id}  gpus={num_gpus}  steps={num_steps}"
    )
    print(f"[training-nsys] repo={git_repo}  checkout={checkout_command}")

    # ----- Phase 1: clone, build, install -----
    install_script = _build_install_script(git_repo, checkout_command)
    result = subprocess.run(
        ["/bin/bash", "-c", install_script],
        stdout=sys.stdout,
        stderr=subprocess.PIPE,
        check=False,
    )
    stderr_text = (result.stderr or b"").decode(errors="replace")
    if stderr_text:
        sys.stderr.write(stderr_text)
    if result.returncode != 0:
        print(
            f"[training-nsys] Build/install failed (exit {result.returncode}).",
            file=sys.stderr,
        )
        return result.returncode

    # ----- Phase 2: download crush-smol dataset -----
    dataset_dir = "/data/crush-smol_processed_t2v"
    if not os.path.isdir(os.path.join(dataset_dir, "combined_parquet_dataset")):
        print("[training-nsys] Downloading crush-smol dataset...")
        dl_script = (
            "from huggingface_hub import snapshot_download; "
            f"snapshot_download("
            f"repo_id='wlsaidhi/crush-smol_processed_t2v',"
            f"local_dir='{dataset_dir}',"
            f"repo_type='dataset',"
            f"local_dir_use_symlinks=False,"
            f"); print('Dataset downloaded.')"
        )
        dl_result = subprocess.run(
            ["/opt/venv/bin/python", "-c", dl_script],
            env={**os.environ, "HF_HOME": "/root/data/.cache"},
            stdout=sys.stdout,
            stderr=subprocess.PIPE,
            check=False,
        )
        if dl_result.returncode != 0:
            sys.stderr.write(
                (dl_result.stderr or b"").decode(errors="replace"))
            return dl_result.returncode
    else:
        print(f"[training-nsys] Dataset already at {dataset_dir}")

    # ----- Phase 3: write training worker script -----
    training_script = "/tmp/fv_mfu_training.py"
    with open(training_script, "w") as f:
        f.write(_MFU_TRAINING_SCRIPT)

    train_env = {
        **os.environ,
        "TRAIN_MODEL_ID": model_id,
        "TRAIN_NUM_GPUS": str(num_gpus),
        "TRAIN_NUM_STEPS": str(num_steps),
        "HF_HOME": "/root/data/.cache",
        "WANDB_MODE": "offline",
        "TMPDIR": "/tmp",
        "TEMP": "/tmp",
        "TMP": "/tmp",
        "FASTVIDEO_ATTENTION_BACKEND": "FLASH_ATTN",
    }

    # ----- Phase 4: nsys profile wrapping torchrun -----
    # --pytorch=autograd-nvtx: operator-level NVTX in all workers
    # --cpuctxsw=none --sample=none: reduce QuadD errors on Modal cloud
    command = (
        "set -e && "
        "source /opt/venv/bin/activate && "
        "cd /FastVideo && "
        # QuadD UUID workaround
        "NSYS_CFG=$(nsys -z 2>/dev/null || true) && "
        "if [ -n \"$NSYS_CFG\" ]; then "
        "  mkdir -p \"$(dirname \"$NSYS_CFG\")\" && "
        "  echo 'CuptiUseRawGpuTimestamps=false' >> \"$NSYS_CFG\"; "
        "fi && "
        # Write to /tmp (local), copy to /results after.
        # Use ; (not &&) after torchrun so nsys-rep is preserved even on OOM.
        "nsys profile "
        "  --force-overwrite=true "
        "  -o /tmp/mfu_training "
        "  --trace=cuda,nvtx "
        "  --pytorch=autograd-nvtx "
        "  --cpuctxsw=none "
        "  --sample=none "
        "  --stats=false "
        f" -- torchrun "
        f"     --nnodes=1 "
        f"     --nproc_per_node={num_gpus} "
        f"     --master_port=29504 "
        f"     {training_script} ; "
        "NSYS_EXIT=$? && "
        "echo \"[training-nsys] nsys+torchrun exit=$NSYS_EXIT\" && "
        # Export SQLite in-container (runs regardless of training exit code)
        "if [ -f /tmp/mfu_training.nsys-rep ]; then "
        "  echo '[training-nsys] Exporting SQLite...' && "
        "  nsys export "
        "    --type sqlite "
        "    --force-overwrite=true "
        "    -o /tmp/mfu_training.sqlite "
        "    /tmp/mfu_training.nsys-rep && "
        "  echo '[training-nsys] SQLite export done.' ; "
        "fi && "
        # Copy results to persistent volume
        "[ -f /tmp/mfu_training.nsys-rep ] && "
        "  cp /tmp/mfu_training.nsys-rep /results/mfu_training.nsys-rep || true && "
        "[ -f /tmp/mfu_training.sqlite ] && "
        "  cp /tmp/mfu_training.sqlite /results/mfu_training.sqlite || true && "
        "[ -f /tmp/mfu_training.qdstrm ] && "
        "  cp /tmp/mfu_training.qdstrm /results/mfu_training.qdstrm || true && "
        "exit $NSYS_EXIT"
    )

    result = subprocess.run(
        ["/bin/bash", "-c", command],
        env=train_env,
        stdout=sys.stdout,
        stderr=subprocess.PIPE,
        check=False,
    )
    stderr_text = (result.stderr or b"").decode(errors="replace")
    if stderr_text:
        sys.stderr.write(stderr_text)

    has_cuda_errors = (
        "Errors occurred while processing the raw events" in stderr_text
        or "TargetProfilingFailed" in stderr_text
    )
    if has_cuda_errors:
        print(
            "[training-nsys] QuadD UUID errors detected (known Modal "
            "limitation). The .nsys-rep may still contain valid kernel data.",
            file=sys.stderr,
        )

    # ----- Commit volume -----
    try:
        results_vol.commit()
    except Exception as exc:  # noqa: BLE001
        print(f"[training-nsys] Volume commit failed: {exc}", file=sys.stderr)

    # ----- Report -----
    report_exists = os.path.isfile("/results/mfu_training.nsys-rep")
    sqlite_exists = os.path.isfile("/results/mfu_training.sqlite")

    if report_exists:
        extra = (
            "\n  # mfu_training.sqlite already exported in-container"
            if sqlite_exists else
            "\n\nConvert locally (if .sqlite missing):"
            "\n  nsys export --type sqlite --force-overwrite=true"
            "\n    -o mfu_training.sqlite mfu_training.nsys-rep"
        )
        print(
            "\n[training-nsys] Done. Results in volume 'fastvideo-nsys-training'."
            "\n\nDownload:"
            "\n  modal volume get fastvideo-nsys-training mfu_training.nsys-rep ."
            "\n  modal volume get fastvideo-nsys-training mfu_training.sqlite ."
            "\n  modal volume get fastvideo-nsys-training mfu_training.qdstrm ."
            + extra
        )
        return 0

    print(
        f"[training-nsys] nsys failed (exit {result.returncode}). "
        "No report written.",
        file=sys.stderr,
    )
    return int(result.returncode)


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S:2",
    image=image,
    timeout=3600,
    memory=65536,
    secrets=[
        modal.Secret.from_dict({
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        })
    ],
    volumes={"/root/data": model_vol, "/results": results_vol},
)
def run_profiler_sanity_check(
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    num_gpus: int = 2,
    num_steps: int = 5,
    pr_number: str = "1087",
) -> int:
    """Torch profiler vs formula MFU sanity check (no Nsight Systems).

    Runs torchrun WITHOUT nsys so torch.profiler can use CUPTI without
    conflict. Sums FLOPs across all SP ranks via dist.all_reduce, then
    compares against the Ohm formula FLOPs. Saves a text report to the
    'fastvideo-nsys-training' volume as profiler_sanity_check.txt.
    """
    git_repo = (
        os.environ.get("BUILDKITE_REPO")
        or "https://github.com/hao-ai-lab/FastVideo.git"
    )
    git_commit = os.environ.get("BUILDKITE_COMMIT")
    env_pr = os.environ.get("BUILDKITE_PULL_REQUEST") or pr_number
    checkout_command = _build_checkout_command(git_commit, env_pr)

    print(
        f"[profiler-sanity] model={model_id}  gpus={num_gpus}  steps={num_steps}"
    )
    print(f"[profiler-sanity] repo={git_repo}  checkout={checkout_command}")

    # Phase 1: clone, build, install
    install_script = _build_install_script(git_repo, checkout_command)
    result = subprocess.run(
        ["/bin/bash", "-c", install_script],
        stdout=sys.stdout,
        stderr=subprocess.PIPE,
        check=False,
    )
    stderr_text = (result.stderr or b"").decode(errors="replace")
    if stderr_text:
        sys.stderr.write(stderr_text)
    if result.returncode != 0:
        print(
            f"[profiler-sanity] Build/install failed (exit {result.returncode}).",
            file=sys.stderr,
        )
        return result.returncode

    # Phase 2: download crush-smol dataset
    dataset_dir = "/data/crush-smol_processed_t2v"
    if not os.path.isdir(os.path.join(dataset_dir, "combined_parquet_dataset")):
        print("[profiler-sanity] Downloading crush-smol dataset...")
        dl_script = (
            "from huggingface_hub import snapshot_download; "
            f"snapshot_download("
            f"repo_id='wlsaidhi/crush-smol_processed_t2v',"
            f"local_dir='{dataset_dir}',"
            f"repo_type='dataset',"
            f"local_dir_use_symlinks=False,"
            f"); print('Dataset downloaded.')"
        )
        dl_result = subprocess.run(
            ["/opt/venv/bin/python", "-c", dl_script],
            env={**os.environ, "HF_HOME": "/root/data/.cache"},
            stdout=sys.stdout,
            stderr=subprocess.PIPE,
            check=False,
        )
        if dl_result.returncode != 0:
            sys.stderr.write(
                (dl_result.stderr or b"").decode(errors="replace"))
            return dl_result.returncode
    else:
        print(f"[profiler-sanity] Dataset already at {dataset_dir}")

    # Phase 3: write profiler sanity script
    sanity_script_path = "/tmp/fv_profiler_sanity.py"
    with open(sanity_script_path, "w") as f:
        f.write(_PROFILER_SANITY_SCRIPT)

    train_env = {
        **os.environ,
        "TRAIN_MODEL_ID": model_id,
        "TRAIN_NUM_GPUS": str(num_gpus),
        "TRAIN_NUM_STEPS": str(num_steps),
        "PEAK_FLOPS_PER_GPU": "3.62e14",  # L40S bf16 peak
        "HF_HOME": "/root/data/.cache",
        "WANDB_MODE": "offline",
        "TMPDIR": "/tmp",
        "TEMP": "/tmp",
        "TMP": "/tmp",
        "FASTVIDEO_ATTENTION_BACKEND": "FLASH_ATTN",
    }

    # Phase 4: torchrun WITHOUT nsys (torch.profiler owns CUPTI cleanly)
    command = (
        "set -e && "
        "source /opt/venv/bin/activate && "
        "cd /FastVideo && "
        f"torchrun "
        f"  --nnodes=1 "
        f"  --nproc_per_node={num_gpus} "
        f"  --master_port=29505 "
        f"  {sanity_script_path}"
    )
    result = subprocess.run(
        ["/bin/bash", "-c", command],
        env=train_env,
        stdout=sys.stdout,
        stderr=subprocess.PIPE,
        check=False,
    )
    stderr_text = (result.stderr or b"").decode(errors="replace")
    if stderr_text:
        sys.stderr.write(stderr_text)

    # Commit volume
    try:
        results_vol.commit()
    except Exception as exc:  # noqa: BLE001
        print(f"[profiler-sanity] Volume commit failed: {exc}", file=sys.stderr)

    sanity_exists = os.path.isfile("/results/profiler_sanity_check.txt")
    if sanity_exists:
        print(
            "\n[profiler-sanity] Done. Report in volume 'fastvideo-nsys-training'."
            "\n\nDownload:"
            "\n  modal volume get fastvideo-nsys-training profiler_sanity_check.txt ."
        )
        return 0

    print(
        f"[profiler-sanity] Run failed (exit {result.returncode}). No report written.",
        file=sys.stderr,
    )
    return int(result.returncode)


@app.local_entrypoint()
def main() -> None:
    """Trigger FastVideo training MFU+NVTX profiling on Modal.

    Checks out PR #1087 (Ohm's NVTX tracer) from hao-ai-lab/FastVideo,
    runs 2-GPU torchrun training under nsys, exports SQLite, and saves
    everything to the 'fastvideo-nsys-training' Modal volume.

    Results are written to the Modal volume 'fastvideo-nsys-training'.
    Download after the run with:
      modal volume get fastvideo-nsys-training mfu_training.nsys-rep .
      modal volume get fastvideo-nsys-training mfu_training.sqlite .

    MFU value is printed to stdout during the run.
    """
    model_id = os.environ.get(
        "TRAIN_MODEL_ID", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    )
    num_gpus = int(os.environ.get("TRAIN_NUM_GPUS", "2"))
    num_steps = int(os.environ.get("TRAIN_NUM_STEPS", "5"))
    pr_number = os.environ.get("TRAIN_PR_NUMBER", "1087")

    print(
        f"[local] model={model_id}  gpus={num_gpus}  steps={num_steps}"
        f"  pr={pr_number}"
    )

    exit_code = run_fastvideo_training_nsys_profile.remote(
        model_id=model_id,
        num_gpus=num_gpus,
        num_steps=num_steps,
        pr_number=pr_number,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


@app.local_entrypoint()
def sanity_check() -> None:
    """Torch profiler vs Ohm formula MFU sanity check (no Nsight Systems).

    Runs torchrun + torch.profiler WITHOUT nsys so CUPTI is available to the
    profiler. Sums FLOPs across SP ranks, compares to formula FLOPs, and
    saves profiler_sanity_check.txt to the 'fastvideo-nsys-training' volume.

    Usage:
      uv run modal run fastvideo/tests/modal/training_nsys_profile.py::sanity_check

    Download after the run:
      modal volume get fastvideo-nsys-training profiler_sanity_check.txt .
    """
    model_id = os.environ.get(
        "TRAIN_MODEL_ID", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    )
    num_gpus = int(os.environ.get("TRAIN_NUM_GPUS", "2"))
    num_steps = int(os.environ.get("TRAIN_NUM_STEPS", "5"))
    pr_number = os.environ.get("TRAIN_PR_NUMBER", "1087")

    print(
        f"[local] profiler sanity check  model={model_id}  gpus={num_gpus}"
        f"  steps={num_steps}  pr={pr_number}"
    )

    exit_code = run_profiler_sanity_check.remote(
        model_id=model_id,
        num_gpus=num_gpus,
        num_steps=num_steps,
        pr_number=pr_number,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)
