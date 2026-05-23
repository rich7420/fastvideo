# SPDX-License-Identifier: Apache-2.0
"""Validate NCHANNELS=1 winner from L40S 4xL40S 5-cell NCCL tuning A/B
on H100x2 NVLink.

L40S result: NCCL_MIN/MAX_NCHANNELS=1 = -6.80% wall (vs baseline 215.10s
-> 200.47s). The win is from cutting per-a2a kernel count in half (2
channels x 3 peer hops -> 1 x 3 = 3 kernels).

Open question: does the same kernel-launch-overhead phenomenon hold on
H100 with NVLink, where per-call bandwidth is ~10x higher? Two cells,
same container, same seed:

  A_default        - NCCL defaults
  D_single_channel - NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1

Usage:
  .venv/bin/modal run fastvideo/tests/modal/nccl_nchannels_h100.py
"""

import json
import os
import pathlib
import subprocess
import sys

import modal

app = modal.App()

try:
    _DEFAULT_LOCAL_ROOT = pathlib.Path(__file__).resolve().parents[3]
except IndexError:
    _DEFAULT_LOCAL_ROOT = pathlib.Path("/FastVideo")
LOCAL_ROOT = pathlib.Path(os.environ.get("FASTVIDEO_LOCAL_ROOT", str(_DEFAULT_LOCAL_ROOT)))

image_version = os.getenv("IMAGE_VERSION", "py3.12-latest")
image_tag = f"ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:{image_version}"

model_vol = modal.Volume.from_name("hf-model-weights")
results_vol = modal.Volume.from_name("fastvideo-nsys-rep", create_if_missing=True)

_IGNORE = [
    ".git/**", ".venv/**", "**/__pycache__/**", "**/*.pyc",
    "*.nsys-rep", "*.sqlite", "*.nsys-cache/**", "*.qdstrm",
    "nsys_results/**", "fastvideo/tests/performance/results/**",
    "fastvideo/tests/performance/generated_videos/**",
    ".parquet_build_*/**",
    "fastvideo-kernel/build/**", "fastvideo-kernel/_skbuild/**",
    "build/**", "dist/**", "*.egg-info/**", "profile_results/**",
]

image = (modal.Image.from_registry(image_tag, add_python="3.12").apt_install("libgl1", "libglib2.0-0",
                                                                             "git").run_commands("rm -rf /FastVideo").
         add_local_dir(str(LOCAL_ROOT), remote_path="/FastVideo", ignore=_IGNORE))

_PROMPT = ("Will Smith casually eats noodles, his relaxed demeanor contrasting "
           "with the energetic background of a bustling street food market. The "
           "scene captures a mix of humor and authenticity. Mid-shot framing, "
           "vibrant lighting.")

_CELLS: dict[str, dict[str, str]] = {
    "A_default": {},
    "D_single_channel": {
        "NCCL_MIN_NCHANNELS": "1",
        "NCCL_MAX_NCHANNELS": "1",
    },
}

# H100x2 -> sp_size=2, num_gpus=2.
_GEN_SCRIPT = """
import json, os, sys, time
import torch

if __name__ == '__main__':
    out_dir = sys.argv[1]
    cell_label = sys.argv[2]

    print(f'[gen] cell={cell_label}', flush=True)
    print(f'[gen] torch={torch.__version__} nccl={torch.cuda.nccl.version()}', flush=True)
    for k in ('NCCL_MIN_NCHANNELS', 'NCCL_MAX_NCHANNELS'):
        print(f'[gen]   {k}={os.environ.get(k, "<unset>")}', flush=True)

    from fastvideo import VideoGenerator
    gen = VideoGenerator.from_pretrained(
        model_path='Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
        num_gpus=2,
        flow_shift=7.0,
        sp_size=2,
        tp_size=1,
        vae_sp=True,
        vae_tiling=True,
        text_encoder_precisions=('fp32',),
    )

    prompt = ${PROMPT_JSON}
    os.makedirs(out_dir, exist_ok=True)

    walls = []
    peaks = []
    for run_idx in range(2):
        out_path = f'{out_dir}/{cell_label}_run{run_idx}.mp4'
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        gen.generate_video(
            prompt,
            height=720, width=1280, num_frames=77,
            num_inference_steps=30,
            guidance_scale=3, embedded_cfg_scale=6,
            seed=1024, fps=24,
            output_path=out_path,
            save_video=True,
        )
        wall = time.perf_counter() - t0
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        walls.append(wall); peaks.append(peak_mb)
        print(f'[gen] cell={cell_label} run{run_idx} wall={wall:.2f}s '
              f'peak={peak_mb:.1f}MB', flush=True)

    sidecar = f'{out_dir}/{cell_label}.json'
    with open(sidecar, 'w') as f:
        json.dump({
            'cell': cell_label,
            'env': {k: os.environ.get(k) for k in ('NCCL_MIN_NCHANNELS', 'NCCL_MAX_NCHANNELS')},
            'walls_s': walls,
            'peaks_mb': peaks,
            'warmup_wall_s': walls[0],
            'measure_wall_s': walls[1] if len(walls) > 1 else None,
        }, f, indent=2)
"""


def _run_cell(out_dir: str, cell_label: str, cell_env: dict[str, str]) -> int:
    print(f"[nccl-h100] === {cell_label} === env={cell_env}", flush=True)
    script_path = "/tmp/gen.py"
    body = _GEN_SCRIPT.replace("${PROMPT_JSON}", json.dumps(_PROMPT))
    with open(script_path, "w") as f:
        f.write(body)

    full_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/root/data/.cache",
        "FASTVIDEO_STAGE_LOGGING": "1",
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,TUNING",
        **cell_env,
    }
    cmd = ("set -e && "
           "source /opt/venv/bin/activate && "
           "cd /FastVideo && "
           '( [ -z "${HF_API_KEY:-}" ] || hf auth login --token "$HF_API_KEY" --quiet || true ) && '
           f"python {script_path} {out_dir} {cell_label}")
    return subprocess.run(["/bin/bash", "-c", cmd], env=full_env).returncode


_SUMMARY_SCRIPT = """
import json, pathlib
out_dir = '/results/nccl_nchannels_h100'
rows = []
for p in sorted(pathlib.Path(out_dir).glob('*.json')):
    if p.name == 'summary.json':
        continue
    rows.append(json.loads(p.read_text()))
print()
print('=== NCHANNELS=1 H100x2 A/B ===')
print(f'{"cell":18s}  {"warmup":>8s}  {"measure":>8s}  {"peak_mb":>9s}  env')
for d in rows:
    w = d.get('warmup_wall_s'); m = d.get('measure_wall_s'); pk = d.get('measure_peak_mb')
    env_short = ' '.join(f'{k}={v}' for k, v in (d.get('env') or {}).items() if v)
    print(f"{d['cell']:18s}  {w if w is None else f'{w:.2f}':>8s}  "
          f"{m if m is None else f'{m:.2f}':>8s}  "
          f"{pk if pk is None else f'{pk:.0f}':>9s}  {env_short}")
baseline = next((r for r in rows if r['cell'] == 'A_default'), None)
if baseline and baseline.get('measure_wall_s'):
    for r in rows:
        if r['cell'] == 'A_default': continue
        m = r.get('measure_wall_s')
        if m is None: continue
        d = m - baseline['measure_wall_s']
        print(f"{r['cell']:18s}  delta={d:+.2f}s ({d/baseline['measure_wall_s']*100:+.2f}%)")
"""


@app.function(
    gpu="H100:2",
    image=image,
    timeout=7200,
    memory=65536,
    secrets=[modal.Secret.from_dict({"HF_API_KEY": os.environ.get("HF_API_KEY", "")})],
    volumes={"/root/data": model_vol, "/results": results_vol},
)
def nccl_nchannels_h100() -> int:
    out_dir = "/results/nccl_nchannels_h100"
    os.makedirs(out_dir, exist_ok=True)

    for cell_label, cell_env in _CELLS.items():
        sidecar = pathlib.Path(out_dir) / f"{cell_label}.json"
        if sidecar.exists():
            print(f"[nccl-h100] {sidecar} exists -- skip", flush=True)
            continue
        rc = _run_cell(out_dir, cell_label, cell_env)
        if rc != 0:
            print(f"[nccl-h100] {cell_label} rc={rc}", file=sys.stderr)

    summary_path = "/tmp/summary.py"
    with open(summary_path, "w") as f:
        f.write(_SUMMARY_SCRIPT)
    subprocess.run(
        ["/bin/bash", "-c", "set -e && source /opt/venv/bin/activate && python " + summary_path],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    try:
        results_vol.commit()
    except Exception as exc:
        print(f"commit failed: {exc}", file=sys.stderr)
    return 0


@app.local_entrypoint()
def main() -> None:
    print(f"[local] uploading source from: {LOCAL_ROOT}")
    raise SystemExit(nccl_nchannels_h100.remote() or 0)
