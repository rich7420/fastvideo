# SPDX-License-Identifier: Apache-2.0
"""VBench quality evaluation: TGATE variant vs baseline (no gating).

Single Modal job that:
  1. Generates N fixed-seed videos with FASTVIDEO_TGATE_STEP=1.0
     (baseline two-pass CFG) → /results/q_tg_baseline/
  2. Generates same N prompts with FASTVIDEO_TGATE_STEP=<tgate_step>
     (TGATE active) → /results/q_tg_<tgate_step>/
  3. Runs VBench scoring on both sets and reports per-dimension delta.

Mirrors quality_eval_linalg_fix.py — but switching variants is simply
re-invoking the gen subprocess with a different env var, so the
sed-patching machinery from the linalg_fix version is not needed.

Usage:
  .venv/bin/modal run fastvideo/tests/modal/quality_eval_tgate.py
  .venv/bin/modal run fastvideo/tests/modal/quality_eval_tgate.py --tgate-step 0.3
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

# VBench scoring needs openai-clip / pyiqa / easydict / decord / lpips
# (matches quality_eval_score.py).  No sed-patching tools needed since
# TGATE switches via env var.
image = (modal.Image.from_registry(image_tag, add_python="3.12").apt_install("libgl1", "libglib2.0-0", "git").
         run_commands("/opt/venv/bin/python -m pip install --no-cache-dir "
                      "  openai-clip pyiqa easydict decord lpips").run_commands("rm -rf /FastVideo").add_local_dir(
                          str(LOCAL_ROOT), remote_path="/FastVideo", ignore=_IGNORE))

# Same 5 prompts as quality_eval_linalg_fix.py so VBench numbers are
# cross-comparable with prior experiments.
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
    ("portrait",
     "Close-up portrait of an elderly Japanese woman with deep wrinkles "
     "smiling warmly, soft natural lighting illuminating her face, slight "
     "head movement, traditional kimono visible at the bottom of the frame."),
    ("metal",
     "Liquid mercury flowing through an intricate brass clockwork mechanism, "
     "macro lens, cinematic dramatic lighting, gears turning slowly, "
     "reflective metallic surfaces, dark background with golden highlights."),
]

# Matches wan-t2v-1.3b-l40s-hires benchmark config (720x1280 / 77f / 30
# steps).  The TGATE summary log inside DenoisingStage proves the env var
# wired through; assert it stays on after generation by echoing
# `FASTVIDEO_TGATE_STEP` from the gen process.
_GEN_SCRIPT = """
import os, sys, time
import torch

if __name__ == '__main__':
    from fastvideo import VideoGenerator

    out_dir = sys.argv[1]
    prompt_idx = int(sys.argv[2])
    prompt = sys.argv[3]

    print(f'[gen] FASTVIDEO_TGATE_STEP={os.getenv("FASTVIDEO_TGATE_STEP", "<unset>")}', flush=True)

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

    os.makedirs(out_dir, exist_ok=True)
    out_path = f'{out_dir}/p{prompt_idx}.mp4'

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
    print(f'[gen] p{prompt_idx} done in {time.perf_counter()-t0:.1f}s -> {out_path}', flush=True)
"""


def _run_generation(out_dir: str, env: dict, tag: str) -> int:
    """Generate ALL prompts under the given env (env already has FASTVIDEO_TGATE_STEP set)."""
    print(f"[quality-tgate] === Generating {tag} ===", flush=True)
    script_path = "/tmp/gen.py"
    with open(script_path, "w") as f:
        f.write(_GEN_SCRIPT)

    for idx, (slug, prompt) in enumerate(_PROMPTS):
        out_path = f"{out_dir}/p{idx}.mp4"
        if os.path.isfile(out_path):
            print(f"[quality-tgate] {out_path} exists — skipping prompt {idx}", flush=True)
            continue
        cmd = ("set -e && "
               "source /opt/venv/bin/activate && "
               "cd /FastVideo && "
               '( [ -z "${HF_API_KEY:-}" ] || hf auth login --token "$HF_API_KEY" --quiet || true ) && '
               f"python {script_path} {out_dir} {idx} {repr(prompt)}")
        rc = subprocess.run(["/bin/bash", "-c", cmd], env=env).returncode
        if rc != 0:
            print(f"[quality-tgate] prompt {idx} failed rc={rc}", file=sys.stderr)
            return rc
    return 0


# VBench dimensions that don't require the (uninitialized) `vbench` git
# submodule — matches the 5-metric subset used by quality_eval_score.py
# so numbers are comparable across experiments.
_VBENCH_SCRIPT = """
import os, sys, json, pathlib
from fastvideo.eval import create_evaluator

baseline_dir = os.environ['TGATE_BASELINE_DIR']
variant_dir = os.environ['TGATE_VARIANT_DIR']
variant_label = os.environ['TGATE_VARIANT_LABEL']

metrics = [
    'vbench.subject_consistency',
    'vbench.background_consistency',
    'vbench.aesthetic_quality',
    'vbench.imaging_quality',
    'vbench.temporal_flickering',
]

evaluator = create_evaluator(metrics=metrics, num_gpus=1)

def score(folder, label):
    samples = []
    for p in sorted(pathlib.Path(folder).glob('*.mp4')):
        samples.append({'video': str(p), 'fps': 24.0, 'prompt': p.stem})
    if not samples:
        print(f'[{label}] no videos in {folder}', flush=True)
        return {}, []
    print(f'[{label}] scoring {len(samples)} videos', flush=True)
    all_results = evaluator.evaluate(samples=samples)
    by_metric = {}
    per_video = []
    for sample, results in zip(samples, all_results):
        sd = {}
        for name, r in results.items():
            by_metric.setdefault(name, []).append(r.score)
            sd[name] = r.score
        per_video.append({'video': sample['video'], 'scores': sd})
    avg = {n: (sum(v)/len(v) if v else None) for n, v in by_metric.items()}
    return avg, per_video

scores_base, per_base = score(baseline_dir, 'BASELINE')
scores_var, per_var = score(variant_dir, variant_label.upper())
evaluator.shutdown()

print()
print('=== VBench per-dimension comparison ===')
print(f'{"metric":42s}  {"baseline":>10s}  {variant_label:>10s}  {"delta":>11s}')
for m in metrics:
    b = scores_base.get(m)
    v = scores_var.get(m)
    if b is None or v is None:
        print(f'{m:42s}  {"-":>10s}  {"-":>10s}  {"-":>11s}')
        continue
    delta = v - b
    arrow = '↑' if delta > 0 else ('↓' if delta < 0 else '=')
    print(f'{m:42s}  {b:10.4f}  {v:10.4f}  {delta:+11.4f} {arrow}')

result = {
    'variant_label': variant_label,
    'baseline_avg': scores_base,
    'variant_avg': scores_var,
    'baseline_per_video': per_base,
    'variant_per_video': per_var,
}
out_path = f'/results/tgate_quality_eval_{variant_label}.json'
pathlib.Path(out_path).write_text(json.dumps(result, indent=2))
print(f'\\n[done] saved {out_path}', flush=True)
"""


def _variant_label(tgate_step: float) -> str:
    """Filesystem-safe label, e.g. 0.5 -> 'tg_0.5'."""
    return f"tg_{tgate_step:.2f}".rstrip("0").rstrip(".") if "." in f"{tgate_step:.2f}" else f"tg_{tgate_step}"


@app.function(
    gpu="L40S:4",
    image=image,
    timeout=7200,
    memory=65536,
    secrets=[modal.Secret.from_dict({"HF_API_KEY": os.environ.get("HF_API_KEY", "")})],
    volumes={"/root/data": model_vol, "/results": results_vol},
)
def quality_eval_tgate(tgate_step: float = 0.5) -> int:
    if not 0.0 <= tgate_step < 1.0:
        print(f"[quality-tgate] tgate_step={tgate_step} not in [0, 1); abort", file=sys.stderr)
        return 1

    variant_label = _variant_label(tgate_step)
    baseline_dir = "/results/q_tg_baseline"
    variant_dir = f"/results/q_{variant_label}"

    base_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/root/data/.cache",
    }

    # ---- Pass 1: baseline (FASTVIDEO_TGATE_STEP=1.0, identical to no-gating CFG) ----
    rc = _run_generation(baseline_dir, {
        **base_env,
        "FASTVIDEO_TGATE_STEP": "1.0",
    }, f"BASELINE (FASTVIDEO_TGATE_STEP=1.0)")
    if rc != 0:
        return rc

    # ---- Pass 2: TGATE variant ----
    rc = _run_generation(variant_dir, {
        **base_env,
        "FASTVIDEO_TGATE_STEP": f"{tgate_step}",
    }, f"TGATE (FASTVIDEO_TGATE_STEP={tgate_step})")
    if rc != 0:
        return rc

    # ---- VBench scoring ----
    print("[quality-tgate] running VBench scoring on both sets", flush=True)
    vbench_path = "/tmp/vbench_eval.py"
    with open(vbench_path, "w") as f:
        f.write(_VBENCH_SCRIPT)
    cmd = ("set -e && "
           "source /opt/venv/bin/activate && "
           "cd /FastVideo && "
           f"python {vbench_path}")
    score_env = {
        **base_env,
        "TGATE_BASELINE_DIR": baseline_dir,
        "TGATE_VARIANT_DIR": variant_dir,
        "TGATE_VARIANT_LABEL": variant_label,
    }
    rc = subprocess.run(["/bin/bash", "-c", cmd], env=score_env).returncode

    try:
        results_vol.commit()
    except Exception as exc:
        print(f"[quality-tgate] volume commit failed: {exc}", file=sys.stderr)

    return rc


@app.local_entrypoint()
def main(tgate_step: float = 0.5) -> None:
    print(f"[local] tgate_step={tgate_step} uploading source from: {LOCAL_ROOT}")
    exit_code = quality_eval_tgate.remote(tgate_step=tgate_step)
    if exit_code != 0:
        raise SystemExit(exit_code)
