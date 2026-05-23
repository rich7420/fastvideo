# SPDX-License-Identifier: Apache-2.0
"""CFG batching A/B: same-seed generation under CFG_BATCH=0 vs CFG_BATCH=1.

Single Modal job that:
  1. Generates N fixed-seed videos with FASTVIDEO_CFG_BATCH=0
     (separate cond+uncond, baseline)             -> /results/cfgb_baseline/
  2. Generates same N prompts with FASTVIDEO_CFG_BATCH=1
     (fused [neg, pos] B=2 forward)               -> /results/cfgb_batched/
  3. Computes per-prompt frame-wise SSIM + LPIPS between the matched pair
     and emits a JSON report.

We do the comparison in one container so cross-machine variance does
not contaminate the timing delta.

Usage:
  .venv/bin/modal run fastvideo/tests/modal/cfg_batch_ab.py
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

# lpips for perceptual diff; scikit-image for SSIM (frame-wise);
# decord for fast video read.
image = (modal.Image.from_registry(image_tag, add_python="3.12").apt_install("libgl1", "libglib2.0-0", "git").
         run_commands("/opt/venv/bin/python -m pip install --no-cache-dir "
                      "  lpips scikit-image decord").run_commands("rm -rf /FastVideo").add_local_dir(
                          str(LOCAL_ROOT), remote_path="/FastVideo", ignore=_IGNORE))

# Use 3 prompts (subset of TGATE's 5 to keep wall-time reasonable). These
# are deliberately varied so a single bad case is easier to spot.
_PROMPTS = [
    ("noodle",
     "Will Smith casually eats noodles, his relaxed demeanor contrasting "
     "with the energetic background of a bustling street food market. The "
     "scene captures a mix of humor and authenticity. Mid-shot framing, "
     "vibrant lighting."),
    ("mountain",
     "A peaceful mountain lake at sunset, mist rising from the water, "
     "golden hour light reflecting on the still surface, pine trees on "
     "the shore swaying gently in the breeze."),
    ("car",
     "A red sports car driving along a winding mountain road at high speed, "
     "sunlight flashing through the dense forest canopy, camera tracking "
     "alongside the vehicle, dynamic motion blur."),
]

# Generation matches wan-t2v-1.3b-l40s-hires (720x1280 / 77f / 30 steps).
# We log per-video timing + peak memory inside the subprocess and dump it
# to a sidecar JSON so the parent can aggregate.
# num_gpus / sp_size are passed via env so the same script powers both
# the 4xL40S and the 2xH100 variants.
_GEN_SCRIPT = """
import json, os, sys, time
import torch

if __name__ == '__main__':
    from fastvideo import VideoGenerator

    out_dir = sys.argv[1]
    prompt_idx = int(sys.argv[2])
    prompt = sys.argv[3]

    num_gpus = int(os.environ.get('CFGB_NUM_GPUS', '4'))
    sp_size = int(os.environ.get('CFGB_SP_SIZE', '4'))

    print(f'[gen] FASTVIDEO_CFG_BATCH={os.getenv("FASTVIDEO_CFG_BATCH", "<unset>")}', flush=True)
    print(f'[gen] FASTVIDEO_TGATE_STEP={os.getenv("FASTVIDEO_TGATE_STEP", "<unset>")}', flush=True)
    print(f'[gen] num_gpus={num_gpus} sp_size={sp_size}', flush=True)

    gen = VideoGenerator.from_pretrained(
        model_path='Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
        num_gpus=num_gpus,
        flow_shift=7.0,
        sp_size=sp_size,
        tp_size=1,
        vae_sp=True,
        vae_tiling=True,
        text_encoder_precisions=('fp32',),
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = f'{out_dir}/p{prompt_idx}.mp4'
    side_path = f'{out_dir}/p{prompt_idx}.json'

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    gen.generate_video(
        prompt,
        height=720, width=1280, num_frames=77,
        num_inference_steps=30,
        guidance_scale=3, embedded_cfg_scale=6,
        seed=1024 + prompt_idx, fps=24,
        output_path=out_path,
        save_video=True,
    )
    wall = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    with open(side_path, 'w') as f:
        json.dump({
            'prompt_idx': prompt_idx,
            'wall_s': wall,
            'peak_memory_mb': peak_mb,
            'env_cfg_batch': os.getenv('FASTVIDEO_CFG_BATCH', '<unset>'),
            'env_tgate_step': os.getenv('FASTVIDEO_TGATE_STEP', '<unset>'),
        }, f)
    print(f'[gen] p{prompt_idx} wall={wall:.2f}s peak={peak_mb:.1f}MB -> {out_path}', flush=True)
"""


def _run_generation(out_dir: str, env: dict, tag: str) -> int:
    """Generate ALL prompts under the given env."""
    print(f"[cfgb-ab] === Generating {tag} ===", flush=True)
    script_path = "/tmp/gen.py"
    with open(script_path, "w") as f:
        f.write(_GEN_SCRIPT)

    for idx, (_slug, prompt) in enumerate(_PROMPTS):
        out_path = f"{out_dir}/p{idx}.mp4"
        if os.path.isfile(out_path):
            print(f"[cfgb-ab] {out_path} exists -- skipping prompt {idx}", flush=True)
            continue
        cmd = ("set -e && "
               "source /opt/venv/bin/activate && "
               "cd /FastVideo && "
               '( [ -z "${HF_API_KEY:-}" ] || hf auth login --token "$HF_API_KEY" --quiet || true ) && '
               f"python {script_path} {out_dir} {idx} {repr(prompt)}")
        rc = subprocess.run(["/bin/bash", "-c", cmd], env=env).returncode
        if rc != 0:
            print(f"[cfgb-ab] prompt {idx} failed rc={rc}", file=sys.stderr)
            return rc
    return 0


# Per-frame SSIM + LPIPS comparison. Decord reads both videos; LPIPS
# (AlexNet backbone) gives a perceptual delta; SSIM gives structural
# similarity. Aggregates to per-video mean / min so a single bad frame
# doesn't get averaged away.
_SCORE_SCRIPT = """
import json, os, pathlib, sys
import numpy as np
import torch
from decord import VideoReader, cpu
from skimage.metrics import structural_similarity as ssim
import lpips

baseline_dir = os.environ['CFGB_BASELINE_DIR']
variant_dir  = os.environ['CFGB_VARIANT_DIR']

# AlexNet LPIPS (default; lighter than VGG) on CUDA.
loss_fn = lpips.LPIPS(net='alex').cuda()

def to_tensor_minmax(arr):
    # arr: (H, W, 3) uint8 -> (1, 3, H, W) float32 in [-1, 1] on CUDA
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return (t / 127.5 - 1.0).cuda()

def compare_one(path_a, path_b):
    va = VideoReader(path_a, ctx=cpu(0))
    vb = VideoReader(path_b, ctx=cpu(0))
    n = min(len(va), len(vb))
    if n == 0:
        return None

    ssim_vals, lpips_vals = [], []
    for i in range(n):
        a = va[i].asnumpy()  # (H, W, 3) uint8
        b = vb[i].asnumpy()
        # SSIM is per-channel; use channel_axis on the last axis. data_range
        # is 255 since uint8.
        s = ssim(a, b, channel_axis=-1, data_range=255)
        ssim_vals.append(float(s))
        with torch.no_grad():
            d = loss_fn(to_tensor_minmax(a), to_tensor_minmax(b)).item()
        lpips_vals.append(float(d))

    return {
        'frames': n,
        'ssim_mean': float(np.mean(ssim_vals)),
        'ssim_min':  float(np.min(ssim_vals)),
        'ssim_max':  float(np.max(ssim_vals)),
        'lpips_mean': float(np.mean(lpips_vals)),
        'lpips_max':  float(np.max(lpips_vals)),
    }

def read_sidecar(folder, idx):
    p = pathlib.Path(folder) / f'p{idx}.json'
    if p.exists():
        return json.loads(p.read_text())
    return {}

report = {'pairs': []}
for p_path in sorted(pathlib.Path(baseline_dir).glob('p*.mp4')):
    idx = int(p_path.stem.lstrip('p'))
    v_path = pathlib.Path(variant_dir) / f'p{idx}.mp4'
    if not v_path.exists():
        print(f'[skip] missing variant pair for p{idx}', flush=True)
        continue

    sc = compare_one(str(p_path), str(v_path))
    base_meta = read_sidecar(baseline_dir, idx)
    var_meta  = read_sidecar(variant_dir, idx)
    report['pairs'].append({
        'prompt_idx': idx,
        'baseline_path': str(p_path),
        'variant_path':  str(v_path),
        'baseline_wall_s': base_meta.get('wall_s'),
        'variant_wall_s':  var_meta.get('wall_s'),
        'baseline_peak_mb': base_meta.get('peak_memory_mb'),
        'variant_peak_mb':  var_meta.get('peak_memory_mb'),
        **(sc or {}),
    })

# Aggregate
if report['pairs']:
    ssim_means = [p['ssim_mean'] for p in report['pairs'] if 'ssim_mean' in p]
    ssim_mins  = [p['ssim_min']  for p in report['pairs'] if 'ssim_min'  in p]
    lpips_means = [p['lpips_mean'] for p in report['pairs'] if 'lpips_mean' in p]
    lpips_maxes = [p['lpips_max']  for p in report['pairs'] if 'lpips_max'  in p]
    report['aggregate'] = {
        'ssim_mean_mean': float(np.mean(ssim_means)) if ssim_means else None,
        'ssim_min_worst': float(np.min(ssim_mins)) if ssim_mins else None,
        'lpips_mean_mean': float(np.mean(lpips_means)) if lpips_means else None,
        'lpips_max_worst': float(np.max(lpips_maxes)) if lpips_maxes else None,
    }
    # Timing aggregate (only when both sides have sidecar)
    base_walls = [p['baseline_wall_s'] for p in report['pairs'] if p.get('baseline_wall_s') is not None]
    var_walls  = [p['variant_wall_s']  for p in report['pairs'] if p.get('variant_wall_s')  is not None]
    if base_walls and var_walls:
        report['aggregate']['avg_baseline_wall_s'] = float(np.mean(base_walls))
        report['aggregate']['avg_variant_wall_s']  = float(np.mean(var_walls))
        report['aggregate']['avg_delta_pct'] = (
            (float(np.mean(var_walls)) - float(np.mean(base_walls))) / float(np.mean(base_walls)) * 100.0
        )

out_path = os.environ.get('CFGB_REPORT_PATH', '/results/cfgb_ab_report.json')
pathlib.Path(out_path).write_text(json.dumps(report, indent=2))

print()
print('=== CFG batching A/B summary ===')
for p in report['pairs']:
    print(f"p{p['prompt_idx']:>2}  SSIM mean={p.get('ssim_mean'):.4f} "
          f"min={p.get('ssim_min'):.4f}  LPIPS mean={p.get('lpips_mean'):.4f} "
          f"max={p.get('lpips_max'):.4f}  "
          f"base={p.get('baseline_wall_s')}s var={p.get('variant_wall_s')}s")
agg = report.get('aggregate', {})
if agg:
    print()
    print(f"AGGREGATE: SSIM mean={agg.get('ssim_mean_mean'):.4f} "
          f"worst-min={agg.get('ssim_min_worst'):.4f}  "
          f"LPIPS mean={agg.get('lpips_mean_mean'):.4f} "
          f"worst-max={agg.get('lpips_max_worst'):.4f}")
    if 'avg_delta_pct' in agg:
        print(f"           wall: baseline={agg['avg_baseline_wall_s']:.2f}s  "
              f"variant={agg['avg_variant_wall_s']:.2f}s  "
              f"delta={agg['avg_delta_pct']:+.2f}%")
print(f'\\n[done] saved {out_path}', flush=True)
"""


def _run_ab(
    tgate_step: float,
    baseline_dir: str,
    variant_dir: str,
    num_gpus: int,
    sp_size: int,
    report_path: str,
) -> int:
    """Shared L40S / H100 A/B body."""
    base_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/root/data/.cache",
        "FASTVIDEO_TGATE_STEP": f"{tgate_step}",
        "CFGB_NUM_GPUS": str(num_gpus),
        "CFGB_SP_SIZE": str(sp_size),
    }

    # Pass 1: baseline (FASTVIDEO_CFG_BATCH=0).
    rc = _run_generation(baseline_dir, {
        **base_env,
        "FASTVIDEO_CFG_BATCH": "0",
    }, "BASELINE (FASTVIDEO_CFG_BATCH=0)")
    if rc != 0:
        return rc

    # Pass 2: variant (FASTVIDEO_CFG_BATCH=1).
    rc = _run_generation(variant_dir, {
        **base_env,
        "FASTVIDEO_CFG_BATCH": "1",
    }, "BATCHED (FASTVIDEO_CFG_BATCH=1)")
    if rc != 0:
        return rc

    score_path = "/tmp/score.py"
    with open(score_path, "w") as f:
        f.write(_SCORE_SCRIPT)
    cmd = ("set -e && "
           "source /opt/venv/bin/activate && "
           "cd /FastVideo && "
           f"python {score_path}")
    score_env = {
        **base_env,
        "CFGB_BASELINE_DIR": baseline_dir,
        "CFGB_VARIANT_DIR": variant_dir,
        "CFGB_REPORT_PATH": report_path,
    }
    rc = subprocess.run(["/bin/bash", "-c", cmd], env=score_env).returncode

    try:
        results_vol.commit()
    except Exception as exc:
        print(f"[cfgb-ab] volume commit failed: {exc}", file=sys.stderr)

    return rc


@app.function(
    gpu="L40S:4",
    image=image,
    timeout=7200,
    memory=65536,
    secrets=[modal.Secret.from_dict({"HF_API_KEY": os.environ.get("HF_API_KEY", "")})],
    volumes={"/root/data": model_vol, "/results": results_vol},
)
def cfg_batch_ab_l40s_4(tgate_step: float = 1.0) -> int:
    """A/B on 4xL40S (PCIe; the original profile target)."""
    return _run_ab(
        tgate_step=tgate_step,
        baseline_dir="/results/cfgb_baseline",
        variant_dir="/results/cfgb_batched",
        num_gpus=4,
        sp_size=4,
        report_path="/results/cfgb_ab_report.json",
    )


@app.function(
    gpu="H100:2",
    image=image,
    timeout=7200,
    memory=65536,
    secrets=[modal.Secret.from_dict({"HF_API_KEY": os.environ.get("HF_API_KEY", "")})],
    volumes={"/root/data": model_vol, "/results": results_vol},
)
def cfg_batch_ab_h100_2(tgate_step: float = 1.0) -> int:
    """A/B on 2xH100 (NVLink). Tests the hypothesis that CFG batching is
    bandwidth-bound on L40S PCIe but gives real wall savings when the
    interconnect can amortise the larger per-call payload."""
    return _run_ab(
        tgate_step=tgate_step,
        baseline_dir="/results/cfgb_h100_baseline",
        variant_dir="/results/cfgb_h100_batched",
        num_gpus=2,
        sp_size=2,
        report_path="/results/cfgb_h100_ab_report.json",
    )


@app.local_entrypoint()
def main(tgate_step: float = 1.0, hardware: str = "l40s_4") -> None:
    """hardware: 'l40s_4' (default) or 'h100_2'."""
    print(f"[local] hardware={hardware} tgate_step={tgate_step} "
          f"uploading source from: {LOCAL_ROOT}")
    if hardware == "l40s_4":
        exit_code = cfg_batch_ab_l40s_4.remote(tgate_step=tgate_step)
    elif hardware == "h100_2":
        exit_code = cfg_batch_ab_h100_2.remote(tgate_step=tgate_step)
    else:
        raise SystemExit(f"unknown hardware={hardware!r}; expected 'l40s_4' or 'h100_2'")
    if exit_code != 0:
        raise SystemExit(exit_code)
