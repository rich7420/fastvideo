# SPDX-License-Identifier: Apache-2.0
"""VBench quality evaluation on H100 with FA3: TGATE 1.0 vs 0.7 vs 0.5.

Same 5 prompts as quality_eval_tgate.py so numbers are cross-comparable
with the L40S/FA2 results in perf_compile_root_cause_report §21.4. The
critical difference: FA3 (WGMMA on Hopper) uses a different accumulation
order than FA2 (HMMA), so bf16 numerics drift between backends — even
when prompt + seed + steps are bit-exact. §23.12 follow-up #1.

One Modal job (~50 min wall on cache hit):
  1. FA3 cache restore (~30 s, requires /fa3_cache populated by a prior
     perf_nsys_profile.py FA3 run, see §23.11.3) — or full source build
     fallback (~90 min).
  2. Generate 5 prompts × 3 TGATE variants on H100:1 = 15 videos to
     /results/q_h100_fa3_tg_1.0|0.7|0.5/
  3. VBench score all 3 sets (5 metrics × 3 variants) → write per-variant
     JSON to /results/tgate_h100_fa3_quality_*.json

Usage:
  .venv/bin/modal run fastvideo/tests/modal/quality_eval_tgate_h100_fa3.py

Download results after:
  .venv/bin/modal volume get fastvideo-nsys-rep tgate_h100_fa3_quality_summary.json .
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
# FA3 build cache — see perf_nsys_profile.py §23.11.3
fa3_cache_vol = modal.Volume.from_name("fa3-build-cache", create_if_missing=True)

_IGNORE = [
    ".git/**", ".venv/**", "**/__pycache__/**", "**/*.pyc",
    "*.nsys-rep", "*.sqlite", "*.nsys-cache/**", "*.qdstrm",
    "nsys_results/**", "fastvideo/tests/performance/results/**",
    "fastvideo/tests/performance/generated_videos/**",
    ".parquet_build_*/**",
    "fastvideo-kernel/build/**", "fastvideo-kernel/_skbuild/**",
    "build/**", "dist/**", "*.egg-info/**", "profile_results/**",
]

image = (
    modal.Image.from_registry(image_tag, add_python="3.12")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .run_commands(
        "/opt/venv/bin/python -m pip install --no-cache-dir "
        "  openai-clip pyiqa easydict decord lpips"
    )
    .run_commands("rm -rf /FastVideo")
    .add_local_dir(str(LOCAL_ROOT), remote_path="/FastVideo", ignore=_IGNORE)
)

# Same 5 prompts as quality_eval_tgate.py / §21.4 / §23.5 — cross-comparable.
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

# Single-H100, sp=1, vae_sp=false (matches benchmark wan-t2v-1.3b-h100-sp1
# and the §23.10/§23.11 measurement configs).
_GEN_SCRIPT = """
import os, sys, time
import torch

if __name__ == '__main__':
    from fastvideo import VideoGenerator

    out_dir = sys.argv[1]
    prompt_idx = int(sys.argv[2])
    prompt = sys.argv[3]

    print(f'[gen] FASTVIDEO_TGATE_STEP={os.getenv("FASTVIDEO_TGATE_STEP", "<unset>")}', flush=True)
    # Probe FA version so the log records which backend each variant ran on
    try:
        from fastvideo.attention.backends.flash_attn import fa_version
        print(f'[gen] fa_version={fa_version}', flush=True)
    except Exception as e:
        print(f'[gen] could not import fa_version: {e}', flush=True)

    gen = VideoGenerator.from_pretrained(
        model_path='Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
        num_gpus=1,
        flow_shift=7.0,
        sp_size=1,
        tp_size=1,
        vae_sp=False,
        vae_tiling=True,
        text_encoder_precisions=('fp32',),
        enable_torch_compile=True,
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


def _fa3_install_cmd() -> str:
    """Bash snippet that ensures flash_attn_interface is importable. Mirrors
    perf_nsys_profile.py:run_perf_nsys_h100_sp1_fa3 — see §23.8 for the
    three-path strategy (already-present → cache restore → source build)."""
    return (
        "echo '=== FA3 install ===' && "
        "SP=/opt/venv/lib/python3.12/site-packages && "
        "CACHE_DIR=/fa3_cache/site-packages-${IMAGE_VERSION:-unknown} && "
        "CACHE_MARKER=$CACHE_DIR/.INSTALLED && "
        "if python -c 'import flash_attn_interface' 2>/dev/null; then "
        "  echo '[fa3] already present in site-packages, skipping install'; "
        "elif [ -f \"$CACHE_MARKER\" ]; then "
        "  echo '[fa3] cache hit at' $CACHE_DIR && "
        "  cp -r $CACHE_DIR/. $SP/ && "
        "  python -c 'import flash_attn_interface' 2>/dev/null && "
        "  echo '[fa3] cache restore OK'; "
        "else "
        "  echo '[fa3] cache miss; building flash-attention/hopper from source' && "
        "  pip install -q ninja && "
        "  rm -rf /tmp/flash-attention && "
        "  git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git "
        "    /tmp/flash-attention && "
        "  cd /tmp/flash-attention/hopper && "
        "  echo '[fa3] build start:' $(date +%H:%M:%S) && "
        "  MAX_JOBS=8 python setup.py install && "
        "  echo '[fa3] build done:' $(date +%H:%M:%S) && "
        "  cd /FastVideo && "
        "  if python -c 'import flash_attn_interface' 2>/dev/null; then "
        "    echo '[fa3] saving build artifacts to cache' && "
        "    STAGE=/fa3_cache/staging-$$ && "
        "    mkdir -p $STAGE && "
        "    for pat in 'flash_attn_interface*' 'flash_attn_3' 'flash_attn_3-*' "
        "               'flash_attn_3*.egg-info' 'flash_attn_3*.dist-info'; do "
        "      find $SP -maxdepth 1 -name \"$pat\" "
        "        -exec cp -r {} $STAGE/ \\; 2>/dev/null; "
        "    done && "
        "    rm -rf $CACHE_DIR && mv $STAGE $CACHE_DIR && touch $CACHE_MARKER && "
        "    echo '[fa3] cache saved to' $CACHE_DIR; "
        "  fi; "
        "fi ; "
        "python -c 'import flash_attn_interface; from fastvideo.attention.backends.flash_attn import fa_version; "
        "print(f\"[fa3] OK module loaded, fa_version={fa_version}\")' 2>&1 || "
        "echo '[fa3] WARN: flash_attn_interface not importable — will fall back to FA2'"
    )


def _run_generation(out_dir: str, env: dict, tag: str) -> int:
    """Generate ALL 5 prompts under the given env."""
    print(f"[quality-tgate-h100-fa3] === Generating {tag} ===", flush=True)
    script_path = "/tmp/gen.py"
    with open(script_path, "w") as f:
        f.write(_GEN_SCRIPT)
    for idx, (slug, prompt) in enumerate(_PROMPTS):
        out_path = f"{out_dir}/p{idx}.mp4"
        if os.path.isfile(out_path):
            print(f"[quality-tgate-h100-fa3] {out_path} exists — skipping prompt {idx}", flush=True)
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
            print(f"[quality-tgate-h100-fa3] prompt {idx} failed rc={rc}", file=sys.stderr)
            return rc
    return 0


# VBench scoring (5-metric subset — same as quality_eval_tgate.py / §21.4).
# Now scores N variants and emits an N-way per-dimension comparison JSON.
_VBENCH_SCRIPT = """
import os, sys, json, pathlib
from fastvideo.eval import create_evaluator

# Variants passed as comma-separated label=dir pairs in TGATE_VARIANT_PAIRS
pairs = []
for chunk in os.environ['TGATE_VARIANT_PAIRS'].split(','):
    label, d = chunk.split('=', 1)
    pairs.append((label.strip(), d.strip()))

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

all_results = {}
for label, d in pairs:
    avg, per_video = score(d, label.upper())
    all_results[label] = {'avg': avg, 'per_video': per_video}
evaluator.shutdown()

baseline_label = pairs[0][0]  # first variant is baseline (TGATE 1.0)
print()
print(f'=== VBench per-dimension comparison (baseline = {baseline_label}) ===')
header = f'{"metric":42s}'
for label, _ in pairs:
    header += f'  {label:>11s}'
for label, _ in pairs[1:]:
    header += f'  {"Δ "+label:>13s}'
print(header)

for m in metrics:
    row = f'{m:42s}'
    base = all_results[baseline_label]['avg'].get(m)
    for label, _ in pairs:
        v = all_results[label]['avg'].get(m)
        row += f'  {v:>11.4f}' if v is not None else f'  {"-":>11s}'
    for label, _ in pairs[1:]:
        v = all_results[label]['avg'].get(m)
        if base is None or v is None:
            row += f'  {"-":>13s}'
        else:
            d = v - base
            arrow = '↑' if d > 0 else ('↓' if d < 0 else '=')
            row += f'  {d:+11.4f} {arrow}'
    print(row)

result = {
    'baseline_label': baseline_label,
    'variants': {label: all_results[label] for label, _ in pairs},
}
out_path = '/results/tgate_h100_fa3_quality_summary.json'
pathlib.Path(out_path).write_text(json.dumps(result, indent=2))
print(f'\\n[done] saved {out_path}', flush=True)
"""


@app.function(
    gpu="H100:1",
    image=image,
    timeout=10800,  # 180 min — accommodates cache-miss source build (~90 min)
    memory=65536,
    secrets=[modal.Secret.from_dict({
        "HF_API_KEY": os.environ.get("HF_API_KEY", ""),
        "IMAGE_VERSION": image_version,
    })],
    volumes={
        "/root/data": model_vol,
        "/results": results_vol,
        "/fa3_cache": fa3_cache_vol,
    },
)
def quality_eval_tgate_h100_fa3() -> int:
    """Run TGATE 1.0 (baseline) / 0.7 / 0.5 quality A/B on H100 with FA3."""
    # Three variants: filesystem-safe label → tgate_step float
    variants = [
        ("tg_1.0", "1.0"),
        ("tg_0.7", "0.7"),
        ("tg_0.5", "0.5"),
    ]

    print("[quality-tgate-h100-fa3] === FA3 install ===", flush=True)
    rc = subprocess.run(
        ["/bin/bash", "-c",
         "set +e && source /opt/venv/bin/activate && " + _fa3_install_cmd()],
        env={**os.environ, "PYTHONUNBUFFERED": "1", "HF_HOME": "/root/data/.cache"},
    ).returncode
    if rc != 0:
        print(f"[quality-tgate-h100-fa3] FA3 install failed rc={rc} — proceeding anyway", file=sys.stderr)
    # Commit the cache volume early in case we just populated it (save the
    # ~90 min for a subsequent run even if pytest fails later).
    try:
        fa3_cache_vol.commit()
    except Exception as exc:
        print(f"[quality-tgate-h100-fa3] cache commit failed: {exc}", file=sys.stderr)

    base_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/root/data/.cache",
    }

    pairs_for_vbench = []
    for label, step_str in variants:
        out_dir = f"/results/q_h100_fa3_{label}"
        env = {**base_env, "FASTVIDEO_TGATE_STEP": step_str}
        rc = _run_generation(out_dir, env, f"{label} (FASTVIDEO_TGATE_STEP={step_str})")
        if rc != 0:
            return rc
        # Commit per variant so partial progress survives a later crash.
        try:
            results_vol.commit()
        except Exception as exc:
            print(f"[quality-tgate-h100-fa3] results commit failed: {exc}", file=sys.stderr)
        pairs_for_vbench.append(f"{label}={out_dir}")

    print("[quality-tgate-h100-fa3] === Running VBench scoring on 3 variants ===", flush=True)
    vbench_path = "/tmp/vbench_eval.py"
    with open(vbench_path, "w") as f:
        f.write(_VBENCH_SCRIPT)
    cmd = (
        "set -e && "
        "source /opt/venv/bin/activate && "
        "cd /FastVideo && "
        f"python {vbench_path}"
    )
    score_env = {
        **base_env,
        "TGATE_VARIANT_PAIRS": ",".join(pairs_for_vbench),
    }
    rc = subprocess.run(["/bin/bash", "-c", cmd], env=score_env).returncode

    try:
        results_vol.commit()
    except Exception as exc:
        print(f"[quality-tgate-h100-fa3] final results commit failed: {exc}", file=sys.stderr)

    print(f"\n[quality-tgate-h100-fa3] === Outcome === scoring exit={rc}")
    print("\nDownload:")
    print("  modal volume get fastvideo-nsys-rep tgate_h100_fa3_quality_summary.json .")
    return rc


@app.local_entrypoint()
def main() -> None:
    print(f"[local] uploading source from: {LOCAL_ROOT}")
    exit_code = quality_eval_tgate_h100_fa3.remote()
    if exit_code != 0:
        raise SystemExit(exit_code)
