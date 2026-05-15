# SPDX-License-Identifier: Apache-2.0
"""VBench quality evaluation: .cpu() variant vs baseline (solve) variant.

Single Modal job that:
  1. Generates N fixed-seed videos with the current (.cpu()) code → /results/q_cpu/
  2. Sed-reverts .cpu() patches in-container
  3. Generates same N prompts with baseline (solve) code → /results/q_baseline/
  4. Runs VBench scoring on both sets
  5. Reports per-dimension score comparison

Goal: prove (or disprove) that the .cpu() linalg fix preserves video quality
despite SSIM 0.98 vs the baseline.

Usage:
  .venv/bin/modal run fastvideo/tests/modal/quality_eval_linalg_fix.py
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


# 5 diverse prompts spanning motion / scene / portrait / abstract dimensions.
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


_GEN_SCRIPT = """
import os, sys, time
import torch

if __name__ == '__main__':
    from fastvideo import VideoGenerator

    out_dir = sys.argv[1]
    prompt_idx = int(sys.argv[2])
    prompt = sys.argv[3]

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
    """Generate ALL prompts in one Modal container (one VideoGenerator init)."""
    # Each subprocess invocation re-loads VideoGenerator (workers respawn),
    # which would amortize poorly. Instead, run inline as separate processes
    # so generator state is reset and the on-disk scheduler file is re-read
    # for each prompt.
    print(f"[quality] === Generating {tag} ===", flush=True)
    script_path = "/tmp/gen.py"
    with open(script_path, "w") as f:
        f.write(_GEN_SCRIPT)

    for idx, (slug, prompt) in enumerate(_PROMPTS):
        out_path = f"{out_dir}/p{idx}.mp4"
        if os.path.isfile(out_path):
            print(f"[quality] {out_path} exists — skipping prompt {idx}", flush=True)
            continue
        cmd = (
            "set -e && "
            "source /opt/venv/bin/activate && "
            "cd /FastVideo && "
            '( [ -z "${HF_API_KEY:-}" ] || hf auth login --token "$HF_API_KEY" --quiet || true ) && '
            f"python {script_path} {out_dir} {idx} {repr(prompt)}"
        )
        rc = subprocess.run(["/bin/bash", "-c", cmd], env=env).returncode
        if rc != 0:
            print(f"[quality] prompt {idx} failed rc={rc}", file=sys.stderr)
            return rc
    return 0


def _revert_cpu_to_solve() -> None:
    """Patch in-container scheduler files: .cpu() form back to plain solve."""
    print("[quality] reverting .cpu() patch in-container", flush=True)
    revert_script_path = "/tmp/revert_cpu.py"
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
            "    # Predictor multi-line\n"
            "    t2 = re.sub(\n"
            "        r'torch\\.linalg\\.solve\\(R\\[:-1, :-1\\]\\.cpu\\(\\),\\s+b\\[:-1\\]\\.cpu\\(\\)\\)\\.to\\(device\\)\\.to\\(x\\.dtype\\)',\n"
            "        'torch.linalg.solve(R[:-1, :-1], b[:-1]).to(device).to(x.dtype)', t)\n"
            "    # Corrector single-line\n"
            "    t2 = re.sub(\n"
            "        r'torch\\.linalg\\.solve\\(R\\.cpu\\(\\), b\\.cpu\\(\\)\\)\\.to\\(device\\)\\.to\\(x\\.dtype\\)',\n"
            "        'torch.linalg.solve(R, b).to(device).to(x.dtype)', t2)\n"
            "    p.write_text(t2)\n"
            "    n = t.count('.cpu()')\n"
            "    n2 = t2.count('.cpu()')\n"
            "    print(f'{rel}: .cpu() count {n} -> {n2}', flush=True)\n"
        )
    subprocess.run(["python", revert_script_path], check=True)


_VBENCH_SCRIPT = """
import os, sys, json, pathlib
from fastvideo.eval import create_evaluator

cpu_dir = '/results/q_cpu'
base_dir = '/results/q_baseline'

# Per-video VBench dimensions that don't require an external prompt corpus
# match — they're purely visual quality / consistency metrics on the mp4.
metrics = [
    'vbench.subject_consistency',
    'vbench.background_consistency',
    'vbench.motion_smoothness',
    'vbench.aesthetic_quality',
    'vbench.imaging_quality',
    'vbench.overall_consistency',
]

evaluator = create_evaluator(metrics=metrics, num_gpus=1)

def score(folder, label):
    samples = []
    for p in sorted(pathlib.Path(folder).glob('*.mp4')):
        samples.append({'video': str(p), 'fps': 24.0, 'prompt': p.stem})
    if not samples:
        print(f'[{label}] no videos in {folder}', flush=True)
        return {}
    print(f'[{label}] scoring {len(samples)} videos', flush=True)
    all_results = evaluator.evaluate(samples=samples)
    by_metric = {}
    for results in all_results:
        for name, r in results.items():
            by_metric.setdefault(name, []).append(r.score)
    return {n: (sum(v)/len(v) if v else None) for n, v in by_metric.items()}

scores_cpu = score(cpu_dir, 'CPU')
scores_base = score(base_dir, 'BASELINE')
evaluator.shutdown()

print('\\n=== VBench per-dimension comparison ===')
print(f'{"metric":40s}  {"baseline":>10s}  {"cpu":>10s}  {"delta":>10s}')
for m in metrics:
    b = scores_base.get(m)
    c = scores_cpu.get(m)
    if b is None or c is None:
        print(f'{m:40s}  {"-":>10s}  {"-":>10s}  {"-":>10s}')
        continue
    delta = c - b
    arrow = '↑' if delta > 0 else ('↓' if delta < 0 else '=')
    print(f'{m:40s}  {b:10.4f}  {c:10.4f}  {delta:+10.4f} {arrow}')

result = {
    'baseline_scores': scores_base,
    'cpu_scores': scores_cpu,
}
pathlib.Path('/results/quality_eval_result.json').write_text(json.dumps(result, indent=2))
print('\\n[done] saved /results/quality_eval_result.json', flush=True)
"""


@app.function(
    gpu="L40S:4",
    image=image,
    timeout=7200,
    memory=65536,
    secrets=[modal.Secret.from_dict({"HF_API_KEY": os.environ.get("HF_API_KEY", "")})],
    volumes={"/root/data": model_vol, "/results": results_vol},
)
def quality_eval() -> int:
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/root/data/.cache",
    }

    # ---- Pass 1: generate with .cpu() variant (current on-disk code) ----
    rc = _run_generation("/results/q_cpu", env, "CPU variant (.cpu())")
    if rc != 0:
        return rc

    # ---- Revert .cpu() -> solve in-container ----
    _revert_cpu_to_solve()

    # ---- Pass 2: generate with baseline (solve) ----
    rc = _run_generation("/results/q_baseline", env, "BASELINE (solve)")
    if rc != 0:
        return rc

    # ---- VBench scoring ----
    print("[quality] running VBench scoring on both sets", flush=True)
    vbench_path = "/tmp/vbench_eval.py"
    with open(vbench_path, "w") as f:
        f.write(_VBENCH_SCRIPT)
    cmd = (
        "set -e && "
        "source /opt/venv/bin/activate && "
        "cd /FastVideo && "
        f"python {vbench_path}"
    )
    rc = subprocess.run(["/bin/bash", "-c", cmd], env=env).returncode

    try:
        results_vol.commit()
    except Exception as exc:
        print(f"[quality] volume commit failed: {exc}", file=sys.stderr)

    return rc


@app.local_entrypoint()
def main() -> None:
    print(f"[local] uploading source from: {LOCAL_ROOT}")
    exit_code = quality_eval.remote()
    if exit_code != 0:
        raise SystemExit(exit_code)
