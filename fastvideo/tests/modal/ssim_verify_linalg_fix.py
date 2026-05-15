# SPDX-License-Identifier: Apache-2.0
"""SSIM verification for the UniPC linalg.solve → CPU fix.

Single Modal job that:
  1. Generates a fixed-seed video with the current (patched) code → fix.mp4
  2. Sed-reverts the .cpu() patches in-container
  3. Generates the same prompt + seed again → baseline.mp4
  4. Computes per-frame SSIM between the two
  5. Returns mean/min SSIM

Both runs use the same model load, same prompt, same seed, same generation
kwargs. The only difference is whether the scheduler's linalg.solve runs
on CPU or GPU.

Usage:
  .venv/bin/modal run fastvideo/tests/modal/ssim_verify_linalg_fix.py
"""

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
LOCAL_ROOT = pathlib.Path(
    os.environ.get("FASTVIDEO_LOCAL_ROOT", str(_DEFAULT_LOCAL_ROOT)))

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

image = (modal.Image.from_registry(image_tag, add_python="3.12")
         .pip_install("scikit-image>=0.21", "imageio[ffmpeg]>=2.30")
         .run_commands("rm -rf /FastVideo")
         .add_local_dir(str(LOCAL_ROOT), remote_path="/FastVideo", ignore=_IGNORE))


# Generation driver script — runs one inference with fixed seed and saves mp4.
# Worker spawn re-imports this as __mp_main__, so guard with __main__.
_GEN_SCRIPT = """
import os, sys, json, time
import torch
import numpy as np

if __name__ == '__main__':
    from fastvideo import VideoGenerator

    out_path = sys.argv[1]
    print(f'[gen] output={out_path}', flush=True)

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

    prompt = ('Will Smith casually eats noodles, his relaxed demeanor contrasting '
              'with the energetic background of a bustling street food market. The '
              'scene captures a mix of humor and authenticity. Mid-shot framing, '
              'vibrant lighting.')

    t0 = time.perf_counter()
    result = gen.generate_video(
        prompt,
        height=720, width=1280, num_frames=77,
        num_inference_steps=30,
        guidance_scale=3, embedded_cfg_scale=6,
        seed=1024, fps=24,
        output_path=out_path,
        save_video=True,
        return_frames=True,
    )
    elapsed = time.perf_counter() - t0
    print(f'[gen] done in {elapsed:.1f}s', flush=True)

    # Save frames as .npy for SSIM (avoid mp4 codec drift)
    npy_path = out_path.replace('.mp4', '.npy')
    if isinstance(result, dict):
        frames = result.get('frames')
        if frames is not None:
            arr = np.stack(frames, axis=0)
            np.save(npy_path, arr)
            print(f'[gen] frames -> {npy_path} shape={arr.shape}', flush=True)
"""


@app.function(
    gpu="L40S:4",
    image=image,
    timeout=3600,
    memory=65536,
    secrets=[modal.Secret.from_dict({"HF_API_KEY": os.environ.get("HF_API_KEY", "")})],
    volumes={"/root/data": model_vol, "/results": results_vol},
)
def ssim_verify() -> int:
    # Files we toggle (relative to /FastVideo)
    sched_a = "fastvideo/models/schedulers/scheduling_flow_unipc_multistep.py"
    sched_b = "fastvideo/models/schedulers/scheduling_unipc_multistep.py"

    script_path = "/tmp/gen.py"
    with open(script_path, "w") as f:
        f.write(_GEN_SCRIPT)

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/root/data/.cache",
    }

    def run_inference(out_path: str, tag: str) -> int:
        cmd = (
            "set -e && "
            "source /opt/venv/bin/activate && "
            "cd /FastVideo && "
            '( [ -z "${HF_API_KEY:-}" ] || hf auth login --token "$HF_API_KEY" --quiet || true ) && '
            f"python {script_path} {out_path}"
        )
        print(f"[verify] === run inference ({tag}) ===", flush=True)
        result = subprocess.run(["/bin/bash", "-c", cmd], env=env)
        return result.returncode

    # ---- Pass 1: with fix (current code) ----
    # Skip if a previous run already produced the npy (cheap restart after
    # mid-script failure).
    if os.path.isfile("/results/ssim_fix.npy"):
        print("[verify] /results/ssim_fix.npy exists — skipping Pass 1", flush=True)
    else:
        rc = run_inference("/results/ssim_fix.mp4", "FIX (cpu solve)")
        if rc != 0:
            print(f"[verify] fix run failed rc={rc}", file=sys.stderr)
            return rc

    # ---- Revert the fix in-container ----
    # Patch the solve_ex(check_errors=False)[0] form back to plain
    # linalg.solve(...).to(device) to reproduce baseline numerics.
    print("[verify] reverting solve_ex patch in-container", flush=True)
    revert_script_path = "/tmp/revert.py"
    with open(revert_script_path, "w") as f:
        f.write(
            "import re, pathlib\n"
            "files = [\n"
            "    'fastvideo/models/schedulers/scheduling_flow_unipc_multistep.py',\n"
            "    'fastvideo/models/schedulers/scheduling_unipc_multistep.py',\n"
            "]\n"
            "for rel in files:\n"
            "    p = pathlib.Path('/FastVideo') / rel\n"
            "    t = p.read_text()\n"
            "    # Predictor (multi-line):\n"
            "    #   torch.linalg.solve_ex(R[:-1, :-1], b[:-1],\\n  check_errors=False)[0].to(x.dtype)\n"
            "    # -> torch.linalg.solve(R[:-1, :-1], b[:-1]).to(device).to(x.dtype)\n"
            "    t2 = re.sub(\n"
            "        r'torch\\.linalg\\.solve_ex\\(R\\[:-1, :-1\\], b\\[:-1\\],\\s+check_errors=False\\)\\[0\\]\\.to\\(x\\.dtype\\)',\n"
            "        'torch.linalg.solve(R[:-1, :-1], b[:-1]).to(device).to(x.dtype)', t)\n"
            "    # Corrector (single line):\n"
            "    t2 = re.sub(\n"
            "        r'torch\\.linalg\\.solve_ex\\(R, b, check_errors=False\\)\\[0\\]\\.to\\(x\\.dtype\\)',\n"
            "        'torch.linalg.solve(R, b).to(device).to(x.dtype)', t2)\n"
            "    p.write_text(t2)\n"
            "    n = t.count('solve_ex')\n"
            "    n2 = t2.count('solve_ex')\n"
            "    print(f'{rel}: solve_ex count {n} -> {n2}', flush=True)\n"
        )
    subprocess.run(["python", revert_script_path], check=True)

    # Verify revert was complete
    grep = subprocess.run(
        ["bash", "-c", "grep -n 'cpu()' /FastVideo/" + sched_a + " /FastVideo/" + sched_b + " || true"],
        capture_output=True, text=True,
    )
    print(f"[verify] remaining .cpu() in scheduler files:\n{grep.stdout}", flush=True)

    # ---- Pass 2: baseline (no fix) ----
    rc = run_inference("/results/ssim_baseline.mp4", "BASELINE (gpu solve)")
    if rc != 0:
        print(f"[verify] baseline run failed rc={rc}", file=sys.stderr)
        return rc

    # ---- SSIM comparison ----
    ssim_script = """
import sys, json, pathlib
import numpy as np
from skimage.metrics import structural_similarity as ssim

fix = np.load('/results/ssim_fix.npy')
base = np.load('/results/ssim_baseline.npy')

print(f'fix shape: {fix.shape} dtype={fix.dtype}', flush=True)
print(f'base shape: {base.shape} dtype={base.dtype}', flush=True)

assert fix.shape == base.shape, 'frame shape mismatch'

per_frame = []
for i in range(fix.shape[0]):
    s = ssim(fix[i], base[i], channel_axis=-1, data_range=255)
    per_frame.append(float(s))

result = {
    'num_frames': len(per_frame),
    'ssim_mean': float(np.mean(per_frame)),
    'ssim_min':  float(np.min(per_frame)),
    'ssim_max':  float(np.max(per_frame)),
    'ssim_std':  float(np.std(per_frame)),
    'per_frame': per_frame,
}
pathlib.Path('/results/ssim_result.json').write_text(json.dumps(result, indent=2))
print('=== SSIM result ===', flush=True)
print(f"  frames: {result['num_frames']}", flush=True)
print(f"  mean:   {result['ssim_mean']:.5f}", flush=True)
print(f"  min:    {result['ssim_min']:.5f}", flush=True)
print(f"  max:    {result['ssim_max']:.5f}", flush=True)
print(f"  std:    {result['ssim_std']:.5f}", flush=True)

# Gate: mean SSIM must be > 0.99 for fix to be merge-ready
gate = result['ssim_mean'] > 0.99
print(f"  gate (mean > 0.99): {'PASS' if gate else 'FAIL'}", flush=True)
sys.exit(0 if gate else 2)
"""
    subprocess.run(["python", "-c", ssim_script], check=False)

    try:
        results_vol.commit()
    except Exception as exc:
        print(f"[verify] volume commit failed: {exc}", file=sys.stderr)

    return 0


@app.local_entrypoint()
def main() -> None:
    print(f"[local] uploading source from: {LOCAL_ROOT}")
    exit_code = ssim_verify.remote()
    if exit_code != 0:
        raise SystemExit(exit_code)
