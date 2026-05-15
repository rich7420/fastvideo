# SPDX-License-Identifier: Apache-2.0
"""Quality eval — scoring half. Runs VBench on /results/q_cpu and /results/q_baseline.

Run AFTER both quality_eval_gen.py variants have produced 5 videos each.

Usage:
  .venv/bin/modal run fastvideo/tests/modal/quality_eval_score.py
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
         .apt_install("libgl1", "libglib2.0-0", "git")
         # Modal's pip_install targets the image-builder Python, NOT
         # /opt/venv where fastvideo runs. Install into the venv directly.
         .run_commands(
             "/opt/venv/bin/python -m pip install --no-cache-dir "
             "  openai-clip pyiqa easydict decord lpips"
         )
         .run_commands("rm -rf /FastVideo")
         .add_local_dir(str(LOCAL_ROOT), remote_path="/FastVideo", ignore=_IGNORE))


_VBENCH_SCRIPT = """
import os, sys, json, pathlib, importlib.util

# Sanity check: which clip module is installed where
for mod_name in ('clip', 'openai_clip', 'pyiqa', 'easydict', 'decord', 'lpips'):
    spec = importlib.util.find_spec(mod_name)
    if spec is None:
        print(f'[mod] {mod_name}: NOT FOUND', flush=True)
    else:
        print(f'[mod] {mod_name}: {spec.origin}', flush=True)

from fastvideo.eval import create_evaluator

cpu_dir = '/results/q_cpu'
base_dir = '/results/q_baseline'

# Per-video VBench dimensions that don't require an external prompt corpus
# match — purely visual quality / consistency metrics on the mp4.
# Metrics that work without the (uninitialized) `vbench` git submodule
# (the metric files at `motion_smoothness`/`overall_consistency` import
# `vbench.third_party.amt` which lives in the submodule).
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

scores_cpu, per_cpu = score(cpu_dir, 'CPU')
scores_base, per_base = score(base_dir, 'BASELINE')
evaluator.shutdown()

print('\\n=== VBench per-dimension comparison ===')
print(f'{"metric":42s}  {"baseline":>10s}  {"cpu":>10s}  {"delta":>11s}')
for m in metrics:
    b = scores_base.get(m)
    c = scores_cpu.get(m)
    if b is None or c is None:
        print(f'{m:42s}  {"-":>10s}  {"-":>10s}  {"-":>11s}')
        continue
    delta = c - b
    arrow = '↑' if delta > 0 else ('↓' if delta < 0 else '=')
    print(f'{m:42s}  {b:10.4f}  {c:10.4f}  {delta:+11.4f} {arrow}')

result = {
    'baseline_avg': scores_base,
    'cpu_avg': scores_cpu,
    'baseline_per_video': per_base,
    'cpu_per_video': per_cpu,
}
pathlib.Path('/results/quality_eval_result.json').write_text(json.dumps(result, indent=2))
print('\\n[done] saved /results/quality_eval_result.json', flush=True)
"""


@app.function(
    gpu="L40S:1",
    image=image,
    timeout=3600,
    memory=32768,
    secrets=[modal.Secret.from_dict({"HF_API_KEY": os.environ.get("HF_API_KEY", "")})],
    volumes={"/root/data": model_vol, "/results": results_vol},
)
def quality_score() -> int:
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/root/data/.cache",
    }

    # Sanity: are 10 videos there?
    n_cpu = len(list(pathlib.Path("/results/q_cpu").glob("*.mp4")))
    n_base = len(list(pathlib.Path("/results/q_baseline").glob("*.mp4")))
    print(f"[score] q_cpu has {n_cpu} mp4s, q_baseline has {n_base} mp4s")
    if n_cpu == 0 or n_base == 0:
        print("[score] missing videos — abort", file=sys.stderr)
        return 1

    script_path = "/tmp/vbench_eval.py"
    with open(script_path, "w") as f:
        f.write(_VBENCH_SCRIPT)
    cmd = (
        "set -e && "
        "source /opt/venv/bin/activate && "
        "cd /FastVideo && "
        f"python {script_path}"
    )
    rc = subprocess.run(["/bin/bash", "-c", cmd], env=env).returncode

    try:
        results_vol.commit()
    except Exception as exc:
        print(f"[score] volume commit failed: {exc}", file=sys.stderr)
    return rc


@app.local_entrypoint()
def main() -> None:
    print(f"[local] uploading source from: {LOCAL_ROOT}")
    rc = quality_score.remote()
    if rc != 0:
        raise SystemExit(rc)
