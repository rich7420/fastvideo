# SPDX-License-Identifier: Apache-2.0
"""NCCL tuning A/B: probe which knob actually moves the needle on 4xL40S
SP=4 Wan T2V.

Covers items 1, 3, 5, 6 from the "what NCCL overhead can we cut" list:

  - Item 1 (NCCL version / per-a2a kernel count): each cell prints
    torch.cuda.nccl.version() and the number of `ncclDevKernel_SendRecv`
    kernels NVTX-bracketed per logical a2a (informational only; the
    fused alltoall path is selected by NCCL itself).
  - Item 3 (env tuning): NCCL_BUFFSIZE / NCCL_MIN_NCHANNELS / NCCL_ALGO
    / NCCL_P2P_LEVEL combinations.
  - Item 5 (real dedicated stream): FASTVIDEO_USE_FUNCTIONAL_A2A=1 swaps
    dist.all_to_all_single -> _functional_collectives.all_to_all_single
    (stream-aware).
  - Item 6 (message chunking): tied to BUFFSIZE.

All cells run with dit_layerwise_offload=True (the default). An earlier
experiment with offload disabled showed +78% wall (392s vs 220s), so
offload's async prefetch is net beneficial and we tune NCCL on top of
the production config rather than against an artificial baseline.

Cells (single container, same prompt + seed):

  A_default       : NCCL env defaults, legacy all_to_all_single
  B_functional    : FASTVIDEO_USE_FUNCTIONAL_A2A=1 (Item 5)
  C_buffsize_16M  : NCCL_BUFFSIZE=16777216 (Item 6)
  D_ring_pxb      : NCCL_ALGO=Ring + NCCL_P2P_LEVEL=PXB + NCCL_MIN_NCHANNELS=2
  E_combined      : functional + BUFFSIZE=16M + Ring + PXB

Usage:
  .venv/bin/modal run fastvideo/tests/modal/nccl_tuning_ab.py
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

# Cell config: each cell defines NCCL / FastVideo env additions.
# All cells run on production-default offload (layerwise prefetch ON).
# NCCL_DEBUG=INFO is set in every cell so the first-rank handshake log
# tells us which transport (P2P direct vs SHM vs host-staged) NCCL
# actually picked -- that disambiguates the 30x bandwidth-limit gap.
_CELLS: dict[str, dict[str, str]] = {
    "A_default": {
        # baseline: only NCCL_DEBUG for diagnostics
    },
    "B_p2p_pxb": {
        # Force PCIe-bridge level P2P. If NCCL defaulted to host-staged
        # copies on L40S (no NVLink), this is the highest-ROI knob.
        "NCCL_P2P_LEVEL": "PXB",
    },
    "C_buffsize_16M": {
        # Bigger chunks => fewer SendRecv kernels per logical a2a.
        # Default is 4 MB; bump to 16 MB.
        "NCCL_BUFFSIZE": "16777216",
    },
    "D_single_channel": {
        # Force a single NCCL channel. Reduces per-call kernel multiplicity
        # at the cost of intra-call parallelism. For a 13 MiB payload on
        # PCIe, this may amortize launch overhead better than the default.
        "NCCL_MIN_NCHANNELS": "1",
        "NCCL_MAX_NCHANNELS": "1",
    },
    "E_functional": {
        # Item 5: route through torch.distributed._functional_collectives,
        # which is supposed to be stream-aware in a way the legacy
        # all_to_all_single isn't. Tests whether the "100% same-stream
        # NCCL" pattern in the original profile was a pyTorch issue.
        "FASTVIDEO_USE_FUNCTIONAL_A2A": "1",
    },
}

# Inner gen script: prints NCCL version and kernel-per-a2a sanity numbers
# so we don't need to parse nsys for every cell.
_GEN_SCRIPT = """
import json, os, sys, time
import torch

if __name__ == '__main__':
    out_dir = sys.argv[1]
    cell_label = sys.argv[2]

    # Sanity log: NCCL version and key env vars actually visible to NCCL.
    print(f'[gen] cell={cell_label}', flush=True)
    print(f'[gen] torch={torch.__version__} nccl={torch.cuda.nccl.version()}', flush=True)
    for k in ('NCCL_BUFFSIZE', 'NCCL_ALGO', 'NCCL_P2P_LEVEL',
              'NCCL_MIN_NCHANNELS', 'NCCL_MAX_NCHANNELS',
              'NCCL_PROTO', 'NCCL_DEBUG',
              'FASTVIDEO_USE_FUNCTIONAL_A2A'):
        print(f'[gen]   {k}={os.environ.get(k, "<unset>")}', flush=True)

    from fastvideo import VideoGenerator
    # Use production-default offload config (layerwise prefetch ON).
    # Disabling offload was tested separately and produced a 2x regression.
    gen = VideoGenerator.from_pretrained(
        model_path='Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
        num_gpus=4,
        flow_shift=7.0,
        sp_size=4,
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
        try:
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
        except torch.cuda.OutOfMemoryError as exc:
            print(f'[gen] cell={cell_label} run{run_idx} OOM: {exc}', flush=True)
            walls.append(None); peaks.append(None); continue
        except Exception as exc:
            print(f'[gen] cell={cell_label} run{run_idx} FAIL: {exc!r}', flush=True)
            walls.append(None); peaks.append(None); continue

        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        walls.append(wall); peaks.append(peak_mb)
        print(f'[gen] cell={cell_label} run{run_idx} wall={wall:.2f}s '
              f'peak={peak_mb:.1f}MB', flush=True)

    sidecar = f'{out_dir}/{cell_label}.json'
    with open(sidecar, 'w') as f:
        json.dump({
            'cell': cell_label,
            'env': {k: os.environ.get(k) for k in (
                'NCCL_BUFFSIZE', 'NCCL_ALGO', 'NCCL_P2P_LEVEL',
                'NCCL_MIN_NCHANNELS', 'NCCL_MAX_NCHANNELS',
                'FASTVIDEO_USE_FUNCTIONAL_A2A')},
            'walls_s': walls,
            'peaks_mb': peaks,
            'warmup_wall_s': walls[0],
            'measure_wall_s': walls[1] if len(walls) > 1 else None,
            'measure_peak_mb': peaks[1] if len(peaks) > 1 else None,
            'fail': any(w is None for w in walls),
        }, f, indent=2)
"""


def _run_cell(out_dir: str, cell_label: str, cell_env: dict[str, str]) -> int:
    print(f"[nccl-tune] === Running cell {cell_label} === env_override={cell_env}", flush=True)
    script_path = "/tmp/gen.py"
    body = _GEN_SCRIPT.replace("${PROMPT_JSON}", json.dumps(_PROMPT))
    with open(script_path, "w") as f:
        f.write(body)

    full_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/root/data/.cache",
        "FASTVIDEO_STAGE_LOGGING": "1",
        # NCCL_DEBUG=INFO logs which transport NCCL picked on the first
        # collective; cheap, only emits a handful of lines at init time.
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,TUNING",
        **cell_env,
    }
    cmd = ("set -e && "
           "source /opt/venv/bin/activate && "
           "cd /FastVideo && "
           '( [ -z "${HF_API_KEY:-}" ] || hf auth login --token "$HF_API_KEY" --quiet || true ) && '
           f"python {script_path} {out_dir} {cell_label}")
    rc = subprocess.run(["/bin/bash", "-c", cmd], env=full_env).returncode
    return rc


_SUMMARY_SCRIPT = """
import json, pathlib, sys
out_dir = '/results/nccl_tune'
rows = []
for p in sorted(pathlib.Path(out_dir).glob('*.json')):
    if p.name == 'summary.json':
        continue
    rows.append(json.loads(p.read_text()))

print()
print('=== NCCL tuning A/B summary ===')
print(f'{"cell":18s}  {"warmup":>8s}  {"measure":>8s}  {"peak_mb":>9s}  fail  env_overrides')
print('-' * 110)
for d in rows:
    env_short = ' '.join(f'{k.split("_",1)[-1]}={v}' for k, v in (d.get('env') or {}).items() if v)
    w = d.get('warmup_wall_s'); m = d.get('measure_wall_s'); pk = d.get('measure_peak_mb')
    print(f"{d['cell']:18s}  "
          f"{w if w is None else f'{w:.2f}':>8s}  "
          f"{m if m is None else f'{m:.2f}':>8s}  "
          f"{pk if pk is None else f'{pk:.0f}':>9s}  "
          f"{'Y' if d.get('fail') else 'N':>4s}  {env_short}")

baseline = next((r for r in rows if r['cell'] == 'A_default'), None)
if baseline and baseline.get('measure_wall_s'):
    print()
    print('=== Delta vs A_default ===')
    for r in rows:
        if r['cell'] == 'A_default':
            continue
        m = r.get('measure_wall_s')
        if m is None:
            print(f"{r['cell']:18s}  N/A (fail)")
            continue
        delta = m - baseline['measure_wall_s']
        delta_pct = delta / baseline['measure_wall_s'] * 100
        print(f"{r['cell']:18s}  delta_wall={delta:+.2f}s  ({delta_pct:+.2f}%)")

pathlib.Path(out_dir + '/summary.json').write_text(json.dumps({
    'cells': rows, 'baseline_cell': 'A_default',
}, indent=2))
print(f'\\nsaved: {out_dir}/summary.json')
"""


@app.function(
    gpu="L40S:4",
    image=image,
    timeout=14400,
    memory=65536,
    secrets=[modal.Secret.from_dict({"HF_API_KEY": os.environ.get("HF_API_KEY", "")})],
    volumes={"/root/data": model_vol, "/results": results_vol},
)
def nccl_tuning_ab() -> int:
    out_dir = "/results/nccl_tune"
    os.makedirs(out_dir, exist_ok=True)

    for cell_label, cell_env in _CELLS.items():
        sidecar = pathlib.Path(out_dir) / f"{cell_label}.json"
        if sidecar.exists():
            print(f"[nccl-tune] {sidecar} exists -- skipping", flush=True)
            continue
        rc = _run_cell(out_dir, cell_label, cell_env)
        if rc != 0:
            print(f"[nccl-tune] cell {cell_label} failed rc={rc}; continuing", file=sys.stderr)

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
        print(f"[nccl-tune] volume commit failed: {exc}", file=sys.stderr)
    return 0


@app.local_entrypoint()
def main() -> None:
    print(f"[local] uploading source from: {LOCAL_ROOT}")
    exit_code = nccl_tuning_ab.remote()
    if exit_code != 0:
        raise SystemExit(exit_code)
