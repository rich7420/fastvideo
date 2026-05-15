# SPDX-License-Identifier: Apache-2.0
"""Quality eval — generation half. Run with --variant cpu OR --variant baseline.

Each invocation patches the scheduler IN-CONTAINER to match the requested
variant, then generates 5 fixed-seed videos.

Pair with quality_eval_score.py once both variants finish.

Usage:
  .venv/bin/modal run fastvideo/tests/modal/quality_eval_gen.py --variant cpu
  .venv/bin/modal run fastvideo/tests/modal/quality_eval_gen.py --variant baseline
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
         .run_commands("rm -rf /FastVideo")
         .add_local_dir(str(LOCAL_ROOT), remote_path="/FastVideo", ignore=_IGNORE))


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


def _patch_to_variant(variant: str) -> None:
    """Patch scheduler files in-container to match requested variant.

    Disk on the mount comes from local source (whatever code state at
    image build time). This patcher rewrites to the canonical form for
    `variant` so the two parallel jobs converge to known states.
    """
    print(f"[patch] applying variant={variant}", flush=True)
    patch_script = "/tmp/patch_variant.py"

    cpu_pred_to_baseline = (
        r"torch\.linalg\.solve\(R\[:-1, :-1\]\.cpu\(\),\s+b\[:-1\]\.cpu\(\)\)\.to\(device\)\.to\(x\.dtype\)"
    )
    cpu_corr_to_baseline = (
        r"torch\.linalg\.solve\(R\.cpu\(\), b\.cpu\(\)\)\.to\(device\)\.to\(x\.dtype\)"
    )
    se_pred_to_baseline = (
        r"torch\.linalg\.solve_ex\(R\[:-1, :-1\], b\[:-1\],\s+check_errors=False\)\[0\]\.to\(x\.dtype\)"
    )
    se_corr_to_baseline = (
        r"torch\.linalg\.solve_ex\(R, b, check_errors=False\)\[0\]\.to\(x\.dtype\)"
    )

    # The patcher always normalizes whatever variant is on disk to "baseline"
    # (plain solve) first, then optionally re-applies the .cpu() form.
    with open(patch_script, "w") as f:
        f.write(
            "import re, pathlib, sys\n"
            f"variant = {variant!r}\n"
            "files = [\n"
            "    'fastvideo/models/schedulers/scheduling_flow_unipc_multistep.py',\n"
            "    'fastvideo/models/schedulers/scheduling_unipc_multistep.py',\n"
            "]\n"
            "for rel in files:\n"
            "    p = pathlib.Path('/FastVideo') / rel\n"
            "    t = p.read_text()\n"
            "    # Step 1: collapse any variant to baseline (plain solve)\n"
            f"    t = re.sub(r'{cpu_pred_to_baseline}',\n"
            "        'torch.linalg.solve(R[:-1, :-1], b[:-1]).to(device).to(x.dtype)', t)\n"
            f"    t = re.sub(r'{cpu_corr_to_baseline}',\n"
            "        'torch.linalg.solve(R, b).to(device).to(x.dtype)', t)\n"
            f"    t = re.sub(r'{se_pred_to_baseline}',\n"
            "        'torch.linalg.solve(R[:-1, :-1], b[:-1]).to(device).to(x.dtype)', t)\n"
            f"    t = re.sub(r'{se_corr_to_baseline}',\n"
            "        'torch.linalg.solve(R, b).to(device).to(x.dtype)', t)\n"
            "    # Step 2: if variant=cpu, apply .cpu() form\n"
            "    if variant == 'cpu':\n"
            "        t = re.sub(\n"
            "            r'torch\\.linalg\\.solve\\(R\\[:-1, :-1\\], b\\[:-1\\]\\)\\.to\\(device\\)\\.to\\(x\\.dtype\\)',\n"
            "            'torch.linalg.solve(R[:-1, :-1].cpu(), b[:-1].cpu()).to(device).to(x.dtype)', t)\n"
            "        t = re.sub(\n"
            "            r'torch\\.linalg\\.solve\\(R, b\\)\\.to\\(device\\)\\.to\\(x\\.dtype\\)',\n"
            "            'torch.linalg.solve(R.cpu(), b.cpu()).to(device).to(x.dtype)', t)\n"
            "    p.write_text(t)\n"
            "    has_cpu = '.cpu()' in t\n"
            "    has_solve_ex = 'solve_ex' in t\n"
            "    print(f'{rel}: cpu={has_cpu} solve_ex={has_solve_ex}', flush=True)\n"
        )
    subprocess.run(["python", patch_script], check=True)


@app.function(
    gpu="L40S:4",
    image=image,
    timeout=3600,
    memory=65536,
    secrets=[modal.Secret.from_dict({"HF_API_KEY": os.environ.get("HF_API_KEY", "")})],
    volumes={"/root/data": model_vol, "/results": results_vol},
)
def quality_gen(variant: str = "cpu") -> int:
    assert variant in ("cpu", "baseline"), f"unknown variant: {variant}"
    out_dir = f"/results/q_{variant}"

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/root/data/.cache",
    }

    _patch_to_variant(variant)

    script_path = "/tmp/gen.py"
    with open(script_path, "w") as f:
        f.write(_GEN_SCRIPT)

    for idx, (slug, prompt) in enumerate(_PROMPTS):
        out_path = f"{out_dir}/p{idx}.mp4"
        if os.path.isfile(out_path):
            print(f"[quality-{variant}] {out_path} exists — skip", flush=True)
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
            print(f"[quality-{variant}] p{idx} failed rc={rc}", file=sys.stderr)
            return rc

    try:
        results_vol.commit()
    except Exception as exc:
        print(f"[quality-{variant}] commit failed: {exc}", file=sys.stderr)
    return 0


@app.local_entrypoint()
def main(variant: str = "cpu") -> None:
    print(f"[local] variant={variant} uploading from: {LOCAL_ROOT}")
    rc = quality_gen.remote(variant=variant)
    if rc != 0:
        raise SystemExit(rc)
