# SPDX-License-Identifier: Apache-2.0
"""Modal app: paired inference perf benchmark + nsys profile (4x L40S).

Mounts local FastVideo code (no git clone), so it picks up uncommitted
changes on `inference-profile`. Relies on fastvideo-dev image already
having fastvideo + kernel installed editably at /FastVideo — we just
overlay our Python source on top.

Single Modal job produces:

  /results/perf_<benchmark_id>_<ts>.json  — metrics (E2E latency, throughput,
                                            peak memory, per-stage times)
  /results/perf.nsys-rep                  — Nsight Systems trace with NVTX
                                            ranges for every pipeline stage

Usage from repo root:
  .venv/bin/modal run fastvideo/tests/modal/perf_nsys_profile.py
  PERF_BENCHMARK_ID=wan-t2v-1.3b-l40s-hires .venv/bin/modal run fastvideo/tests/modal/perf_nsys_profile.py

Env vars:
  PERF_BENCHMARK_ID : pytest -k filter; matches benchmark_id in
                      .buildkite/performance-benchmarks/tests/*.json
                      (default: wan-t2v-1.3b-l40s-hires)
  HF_API_KEY        : HuggingFace token
  IMAGE_VERSION     : fastvideo-dev image tag (default: py3.12-latest)
  FASTVIDEO_LOCAL_ROOT : local repo root to mount (default: derived from
                         this file's path)

Download results after the run:
  .venv/bin/modal volume get fastvideo-nsys-rep perf.nsys-rep .
  .venv/bin/modal volume get fastvideo-nsys-rep perf_json/<filename> .
"""

import os
import pathlib
import subprocess
import sys

import modal

app = modal.App()

model_vol = modal.Volume.from_name("hf-model-weights")
results_vol = modal.Volume.from_name(
    "fastvideo-nsys-rep",
    create_if_missing=True,
)
# Cache for flash-attention/hopper (FA3) build artifacts. Avoids the
# ~90 min nvcc compile (293 templated kernels) on every fresh Modal
# container. Cache key includes image_version so an image upgrade
# (different torch/cuda ABI) invalidates and rebuilds automatically.
fa3_cache_vol = modal.Volume.from_name(
    "fa3-build-cache",
    create_if_missing=True,
)

# Repo root: <repo>/fastvideo/tests/modal/perf_nsys_profile.py -> <repo>
# Modal re-imports this module inside the container where the path layout
# is different (script lives at /root/perf_nsys_profile.py). The container
# never actually mounts code from a local path, so we just need a value
# that won't IndexError at import.
try:
    _DEFAULT_LOCAL_ROOT = pathlib.Path(__file__).resolve().parents[3]
except IndexError:
    _DEFAULT_LOCAL_ROOT = pathlib.Path("/FastVideo")
LOCAL_ROOT = pathlib.Path(
    os.environ.get("FASTVIDEO_LOCAL_ROOT", str(_DEFAULT_LOCAL_ROOT)))

image_version = os.getenv("IMAGE_VERSION", "py3.12-latest")
image_tag = f"ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:{image_version}"

# Patterns excluded from the local-source upload. Keep this list narrow —
# anything fastvideo imports at runtime must be present.
_IGNORE = [
    ".git/**",
    ".venv/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.pytest_cache/**",
    "**/.mypy_cache/**",
    "**/.ruff_cache/**",
    "**/node_modules/**",
    # Profile / benchmark artifacts that pile up in repo root
    "*.nsys-rep",
    "**/*.nsys-rep",
    "*.sqlite",
    "**/*.sqlite",
    "*.nsys-cache/**",
    "**/*.nsys-cache/**",
    "*.qdstrm",
    "**/*.qdstrm",
    "nsys_results/**",
    "profile_results/**",
    "fastvideo/tests/performance/results/**",
    "fastvideo/tests/performance/generated_videos/**",
    ".parquet_build_*/**",
    "**/.parquet_build_*/**",
    # Pre-existing builds we don't want to overwrite the image's kernel with
    "fastvideo-kernel/build/**",
    "fastvideo-kernel/_skbuild/**",
    "build/**",
    "dist/**",
    "*.egg-info/**",
]

image = (
    modal.Image.from_registry(image_tag, add_python="3.12").apt_install(
        "wget",
        "gnupg",
        "ca-certificates",
    ).run_commands(
        # nsys CLI: install from NVIDIA devtools repo into the image.
        "wget -qO - https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nvidia.pub | "
        "  gpg --dearmor -o /usr/share/keyrings/nvidia-devtools-keyring.gpg && "
        "echo 'deb [signed-by=/usr/share/keyrings/nvidia-devtools-keyring.gpg] "
        "  https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/ /' "
        "  > /etc/apt/sources.list.d/nvidia-devtools.list && "
        "apt-get update && "
        "apt-get install -y --no-install-recommends nsight-systems-cli"
    ).run_commands(
        # Wipe the image's pre-baked /FastVideo before overlaying local
        # source — otherwise stale subdirs from old builds (e.g. removed
        # stepvideo pipeline) get walked by pkgutil and break imports.
        # fastvideo_kernel lives in site-packages, so this is safe.
        "rm -rf /FastVideo"
    ).add_local_dir(
        str(LOCAL_ROOT),
        remote_path="/FastVideo",
        ignore=_IGNORE,
    ))


@app.function(
    gpu="L40S:4",
    image=image,
    timeout=3600,
    memory=65536,
    secrets=[modal.Secret.from_dict({
        "HF_API_KEY": os.environ.get("HF_API_KEY", ""),
        "FASTVIDEO_TGATE_STEP": os.environ.get("FASTVIDEO_TGATE_STEP", "1.0"),
    })],
    volumes={"/root/data": model_vol, "/results": results_vol},
)
def run_perf_nsys(benchmark_id: str = "wan-t2v-1.3b-l40s-hires") -> int:
    print(f"[perf-nsys] benchmark_id={benchmark_id}")
    print(f"[perf-nsys] mode=local-mount (no git clone)")

    profile_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "FASTVIDEO_STAGE_LOGGING": "1",
        "HF_HOME": "/root/data/.cache",
    }

    # nsys flags: --trace=cuda,nvtx (kernels + NVTX ranges from
    # stages/base.py::__call__), --pytorch=autograd-nvtx (free torch op
    # markers), --cpuctxsw/--sample=none (suppress QuadD errors on Modal).
    # `;` instead of `&&` after pytest so the trace is copied even when
    # pytest exits non-zero (threshold violation is expected on first run).
    # --assert=plain: disable assertion rewriting to dodge a Python 3.12 +
    # pytest 8 + anyio bug where rewrite.find_spec gets a non-string path
    # ("'AssertionRewritingHook' object has no attribute 'rpartition'").
    # Our threshold asserts work fine without rewriting.
    # Exact-match by surrounding with square brackets so substring-overlap
    # configs (e.g. wan-t2v-1.3b-l40s-compile-sp1) don't get pulled in.
    pytest_cmd = (
        f"pytest -vs --assert=plain "
        f"./fastvideo/tests/performance/test_inference_performance.py "
        f"-k \"[{benchmark_id}]\"")
    # NCCL diagnostic env vars (override-able via PERF_NCCL_* local env vars
    # for sweeping algo/proto without redeploying).
    nccl_debug = os.environ.get("PERF_NCCL_DEBUG", "INFO")  # default: log algo selection
    nccl_algo = os.environ.get("PERF_NCCL_ALGO", "")  # "" = NCCL auto
    nccl_proto = os.environ.get("PERF_NCCL_PROTO", "")
    nccl_buffsize = os.environ.get("PERF_NCCL_BUFFSIZE", "")
    nccl_exports = (
        f"export NCCL_DEBUG={nccl_debug} ; "
        + (f"export NCCL_ALGO={nccl_algo} ; " if nccl_algo else "")
        + (f"export NCCL_PROTO={nccl_proto} ; " if nccl_proto else "")
        + (f"export NCCL_BUFFSIZE={nccl_buffsize} ; " if nccl_buffsize else "")
    )

    command = (
        "set +e && "
        "source /opt/venv/bin/activate && "
        "cd /FastVideo && "
        # HF login if a token was provided (gated models).
        '( [ -z "${HF_API_KEY:-}" ] || hf auth login --token "$HF_API_KEY" --quiet || true ) && '
        # Dump GPU topology + driver info so NCCL behaviour can be reproduced.
        "echo '=== nvidia-smi topo -m ===' ; nvidia-smi topo -m 2>&1 || true ; "
        "echo '=== nvidia-smi --query-gpu=name,driver_version,pcie.link.gen.current,pcie.link.width.current --format=csv ===' ; "
        "nvidia-smi --query-gpu=name,driver_version,pcie.link.gen.current,pcie.link.width.current --format=csv 2>&1 || true ; "
        # NCCL env (logged + overridable).
        + nccl_exports +
        "echo '=== NCCL env ===' ; env | grep NCCL_ 2>&1 || true ; "
        # QuadD UUID workaround.
        "NSYS_CFG=$(nsys -z 2>/dev/null || true) && "
        "if [ -n \"$NSYS_CFG\" ]; then "
        "  mkdir -p \"$(dirname \"$NSYS_CFG\")\" && "
        "  echo 'CuptiUseRawGpuTimestamps=false' >> \"$NSYS_CFG\"; "
        "fi && "
        "nsys profile "
        "  --force-overwrite=true "
        "  -o /tmp/perf "
        "  --trace=cuda,nvtx "
        "  --pytorch=autograd-nvtx "
        "  --cpuctxsw=none "
        "  --sample=none "
        "  --cuda-memory-usage=false "
        "  --stats=false "
        f" -- bash -c '{pytest_cmd}' ; "
        "PYTEST_RC=$? ; "
        # Copy nsys trace from /tmp (local disk) to /results (FUSE).
        "cp -v /tmp/perf.nsys-rep /results/perf.nsys-rep 2>&1 || true ; "
        # Copy perf JSON output(s) — written by the pytest test.
        "mkdir -p /results/perf_json && "
        "find ./fastvideo/tests/performance/results -name 'perf_*.json' "
        "  -exec cp -v {} /results/perf_json/ \\; 2>&1 || true ; "
        # Copy generated videos for downstream SSIM regression.
        "mkdir -p /results/perf_videos && "
        "find ./fastvideo/tests/performance/generated_videos -name '*.mp4' "
        "  -exec cp -v {} /results/perf_videos/ \\; 2>&1 || true ; "
        "exit $PYTEST_RC")

    result = subprocess.run(
        ["/bin/bash", "-c", command],
        env=profile_env,
        stdout=sys.stdout,
        stderr=subprocess.PIPE,
        check=False,
    )
    stderr_text = (result.stderr or b"").decode(errors="replace")
    if stderr_text:
        sys.stderr.write(stderr_text)

    try:
        results_vol.commit()
    except Exception as exc:
        print(f"[perf-nsys] volume commit failed: {exc}", file=sys.stderr)

    has_nsys = os.path.isfile("/results/perf.nsys-rep")
    json_files = []
    if os.path.isdir("/results/perf_json"):
        json_files = sorted(os.listdir("/results/perf_json"))

    print("\n[perf-nsys] === Outcome ===")
    print(f"  pytest exit code : {result.returncode}")
    print(f"  nsys trace       : {'OK' if has_nsys else 'MISSING'}")
    print(f"  perf json files  : {json_files or 'none'}")
    print("\nDownload:")
    print("  modal volume get fastvideo-nsys-rep perf.nsys-rep .")
    for fn in json_files:
        print(f"  modal volume get fastvideo-nsys-rep perf_json/{fn} .")

    return int(result.returncode)


@app.function(
    gpu="L40S:2",
    image=image,
    timeout=3600,
    memory=65536,
    secrets=[modal.Secret.from_dict({
        "HF_API_KEY": os.environ.get("HF_API_KEY", ""),
        "FASTVIDEO_TGATE_STEP": os.environ.get("FASTVIDEO_TGATE_STEP", "1.0"),
    })],
    volumes={"/root/data": model_vol, "/results": results_vol},
)
def run_perf_nsys_sp2(benchmark_id: str = "wan-t2v-1.3b-l40s-hires-sp2") -> int:
    # Mirrors run_perf_nsys but targets 2x L40S. Output filenames use the
    # benchmark_id, so the sp=2 trace lands at /results/perf_sp2.nsys-rep to
    # avoid clobbering the baseline.
    print(f"[perf-nsys-sp2] benchmark_id={benchmark_id}")

    profile_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "FASTVIDEO_STAGE_LOGGING": "1",
        "HF_HOME": "/root/data/.cache",
    }
    # Exact-match via bracket so we don't accidentally pull in compile-sp1.
    pytest_cmd = (
        f"pytest -vs --assert=plain "
        f"./fastvideo/tests/performance/test_inference_performance.py "
        f"-k \"[{benchmark_id}]\"")
    command = (
        "set +e && "
        "source /opt/venv/bin/activate && "
        "cd /FastVideo && "
        '( [ -z "${HF_API_KEY:-}" ] || hf auth login --token "$HF_API_KEY" --quiet || true ) && '
        "NSYS_CFG=$(nsys -z 2>/dev/null || true) && "
        "if [ -n \"$NSYS_CFG\" ]; then "
        "  mkdir -p \"$(dirname \"$NSYS_CFG\")\" && "
        "  echo 'CuptiUseRawGpuTimestamps=false' >> \"$NSYS_CFG\"; "
        "fi && "
        "nsys profile "
        "  --force-overwrite=true "
        "  -o /tmp/perf_sp2 "
        "  --trace=cuda,nvtx "
        "  --pytorch=autograd-nvtx "
        "  --cpuctxsw=none "
        "  --sample=none "
        "  --cuda-memory-usage=false "
        "  --stats=false "
        f" -- bash -c '{pytest_cmd}' ; "
        "PYTEST_RC=$? ; "
        "cp -v /tmp/perf_sp2.nsys-rep /results/perf_sp2.nsys-rep 2>&1 || true ; "
        "mkdir -p /results/perf_json && "
        "find ./fastvideo/tests/performance/results -name 'perf_*.json' "
        "  -exec cp -v {} /results/perf_json/ \\; 2>&1 || true ; "
        "exit $PYTEST_RC")

    result = subprocess.run(
        ["/bin/bash", "-c", command],
        env=profile_env,
        stdout=sys.stdout,
        stderr=subprocess.PIPE,
        check=False,
    )
    stderr_text = (result.stderr or b"").decode(errors="replace")
    if stderr_text:
        sys.stderr.write(stderr_text)

    try:
        results_vol.commit()
    except Exception as exc:
        print(f"[perf-nsys-sp2] volume commit failed: {exc}", file=sys.stderr)

    has_nsys = os.path.isfile("/results/perf_sp2.nsys-rep")
    print(f"\n[perf-nsys-sp2] === Outcome === pytest={result.returncode}, nsys={'OK' if has_nsys else 'MISSING'}")
    return int(result.returncode)


@app.function(
    gpu="L40S:1",
    image=image,
    timeout=3600,
    memory=65536,
    secrets=[modal.Secret.from_dict({
        "HF_API_KEY": os.environ.get("HF_API_KEY", ""),
        "FASTVIDEO_TGATE_STEP": os.environ.get("FASTVIDEO_TGATE_STEP", "1.0"),
    })],
    volumes={"/root/data": model_vol, "/results": results_vol},
)
def run_perf_nsys_sp1(benchmark_id: str = "wan-t2v-1.3b-l40s-compile-sp1") -> int:
    # Single-GPU oracle: no NCCL at all. Used to calibrate the true cost of SP
    # all_to_all in the sp=4 baseline.
    print(f"[perf-nsys-sp1] benchmark_id={benchmark_id}")

    profile_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "FASTVIDEO_STAGE_LOGGING": "1",
        "HF_HOME": "/root/data/.cache",
    }
    pytest_cmd = (
        f"pytest -vs --assert=plain "
        f"./fastvideo/tests/performance/test_inference_performance.py "
        f"-k \"[{benchmark_id}]\"")
    command = (
        "set +e && "
        "source /opt/venv/bin/activate && "
        "cd /FastVideo && "
        '( [ -z "${HF_API_KEY:-}" ] || hf auth login --token "$HF_API_KEY" --quiet || true ) && '
        "NSYS_CFG=$(nsys -z 2>/dev/null || true) && "
        "if [ -n \"$NSYS_CFG\" ]; then "
        "  mkdir -p \"$(dirname \"$NSYS_CFG\")\" && "
        "  echo 'CuptiUseRawGpuTimestamps=false' >> \"$NSYS_CFG\"; "
        "fi && "
        "nsys profile "
        "  --force-overwrite=true "
        "  -o /tmp/perf_sp1 "
        "  --trace=cuda,nvtx "
        "  --pytorch=autograd-nvtx "
        "  --cpuctxsw=none "
        "  --sample=none "
        "  --cuda-memory-usage=false "
        "  --stats=false "
        f" -- bash -c '{pytest_cmd}' ; "
        "PYTEST_RC=$? ; "
        "cp -v /tmp/perf_sp1.nsys-rep /results/perf_sp1.nsys-rep 2>&1 || true ; "
        "mkdir -p /results/perf_json && "
        "find ./fastvideo/tests/performance/results -name 'perf_*.json' "
        "  -exec cp -v {} /results/perf_json/ \\; 2>&1 || true ; "
        "exit $PYTEST_RC")

    result = subprocess.run(
        ["/bin/bash", "-c", command],
        env=profile_env,
        stdout=sys.stdout,
        stderr=subprocess.PIPE,
        check=False,
    )
    stderr_text = (result.stderr or b"").decode(errors="replace")
    if stderr_text:
        sys.stderr.write(stderr_text)

    try:
        results_vol.commit()
    except Exception as exc:
        print(f"[perf-nsys-sp1] volume commit failed: {exc}", file=sys.stderr)

    has_nsys = os.path.isfile("/results/perf_sp1.nsys-rep")
    print(f"\n[perf-nsys-sp1] === Outcome === pytest={result.returncode}, nsys={'OK' if has_nsys else 'MISSING'}")
    return int(result.returncode)


@app.function(
    gpu="H100:1",
    image=image,
    timeout=3600,
    memory=65536,
    secrets=[modal.Secret.from_dict({
        "HF_API_KEY": os.environ.get("HF_API_KEY", ""),
        "FASTVIDEO_TGATE_STEP": os.environ.get("FASTVIDEO_TGATE_STEP", "1.0"),
        "FASTVIDEO_ATTENTION_BACKEND": os.environ.get("FASTVIDEO_ATTENTION_BACKEND", ""),
    })],
    volumes={"/root/data": model_vol, "/results": results_vol},
)
def run_perf_nsys_h100_sp1(benchmark_id: str = "wan-t2v-1.3b-h100-sp1") -> int:
    # Single H100 baseline (sp=1). Hopper sm90 path: FA3 fires automatically
    # when flash_attn_interface is importable. Trace lands at
    # /results/perf_h100_sp1.nsys-rep so it doesn't clobber the L40S baselines.
    print(f"[perf-nsys-h100-sp1] benchmark_id={benchmark_id}")

    profile_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "FASTVIDEO_STAGE_LOGGING": "1",
        "HF_HOME": "/root/data/.cache",
    }
    pytest_cmd = (
        f"pytest -vs --assert=plain "
        f"./fastvideo/tests/performance/test_inference_performance.py "
        f"-k \"[{benchmark_id}]\"")
    command = (
        "set +e && "
        "source /opt/venv/bin/activate && "
        "cd /FastVideo && "
        '( [ -z "${HF_API_KEY:-}" ] || hf auth login --token "$HF_API_KEY" --quiet || true ) && '
        # Dump GPU info so we can confirm we got an H100 (sm90).
        "echo '=== nvidia-smi ===' ; nvidia-smi 2>&1 || true ; "
        "echo '=== nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv ===' ; "
        "nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv 2>&1 || true ; "
        # Probe FA version: the flash_attn backend module logs 'Using FlashAttention-X'
        # on import. Print it explicitly so it's easy to spot in the run output.
        "python -c \"from fastvideo.attention.backends.flash_attn import fa_version; "
        "print(f'[probe] flash_attn fa_version={fa_version}')\" 2>&1 || true ; "
        "NSYS_CFG=$(nsys -z 2>/dev/null || true) && "
        "if [ -n \"$NSYS_CFG\" ]; then "
        "  mkdir -p \"$(dirname \"$NSYS_CFG\")\" && "
        "  echo 'CuptiUseRawGpuTimestamps=false' >> \"$NSYS_CFG\"; "
        "fi && "
        "nsys profile "
        "  --force-overwrite=true "
        "  -o /tmp/perf_h100_sp1 "
        "  --trace=cuda,nvtx "
        "  --pytorch=autograd-nvtx "
        "  --cpuctxsw=none "
        "  --sample=none "
        "  --cuda-memory-usage=false "
        "  --stats=false "
        f" -- bash -c '{pytest_cmd}' ; "
        "PYTEST_RC=$? ; "
        "cp -v /tmp/perf_h100_sp1.nsys-rep /results/perf_h100_sp1.nsys-rep 2>&1 || true ; "
        "mkdir -p /results/perf_json && "
        "find ./fastvideo/tests/performance/results -name 'perf_*.json' "
        "  -exec cp -v {} /results/perf_json/ \\; 2>&1 || true ; "
        "mkdir -p /results/perf_videos && "
        "find ./fastvideo/tests/performance/generated_videos -name '*.mp4' "
        "  -exec cp -v {} /results/perf_videos/ \\; 2>&1 || true ; "
        "exit $PYTEST_RC")

    result = subprocess.run(
        ["/bin/bash", "-c", command],
        env=profile_env,
        stdout=sys.stdout,
        stderr=subprocess.PIPE,
        check=False,
    )
    stderr_text = (result.stderr or b"").decode(errors="replace")
    if stderr_text:
        sys.stderr.write(stderr_text)

    try:
        results_vol.commit()
    except Exception as exc:
        print(f"[perf-nsys-h100-sp1] volume commit failed: {exc}", file=sys.stderr)

    has_nsys = os.path.isfile("/results/perf_h100_sp1.nsys-rep")
    print(f"\n[perf-nsys-h100-sp1] === Outcome === pytest={result.returncode}, nsys={'OK' if has_nsys else 'MISSING'}")
    print("\nDownload:")
    print("  modal volume get fastvideo-nsys-rep perf_h100_sp1.nsys-rep .")
    return int(result.returncode)


@app.function(
    gpu="H100:1",
    image=image,
    timeout=10800,  # 180 min — FA3 source build takes ~75-90 min on Modal H100 container CPU (293 nvcc kernels, MAX_JOBS=8); cache hit drops this to ~30 s
    memory=65536,
    secrets=[modal.Secret.from_dict({
        "HF_API_KEY": os.environ.get("HF_API_KEY", ""),
        "FASTVIDEO_TGATE_STEP": os.environ.get("FASTVIDEO_TGATE_STEP", "1.0"),
        "FASTVIDEO_ATTENTION_BACKEND": os.environ.get("FASTVIDEO_ATTENTION_BACKEND", ""),
        "IMAGE_VERSION": image_version,  # FA3 cache key — see fa3_cache_vol
    })],
    volumes={
        "/root/data": model_vol,
        "/results": results_vol,
        "/fa3_cache": fa3_cache_vol,
    },
)
def run_perf_nsys_h100_sp1_fa3(benchmark_id: str = "wan-t2v-1.3b-h100-sp1") -> int:
    # H100 sp=1 with Flash Attention 3 installed. fastvideo's flash_attn.py
    # auto-picks FA3 when `flash_attn_interface` is importable, so the install
    # is the only delta vs run_perf_nsys_h100_sp1.
    # Trace lands at /results/perf_h100_sp1_fa3.nsys-rep.
    print(f"[perf-nsys-h100-sp1-fa3] benchmark_id={benchmark_id}")

    profile_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "FASTVIDEO_STAGE_LOGGING": "1",
        "HF_HOME": "/root/data/.cache",
    }
    pytest_cmd = (
        f"pytest -vs --assert=plain "
        f"./fastvideo/tests/performance/test_inference_performance.py "
        f"-k \"[{benchmark_id}]\"")
    # FA3 install: three paths checked in order, fall through on failure.
    #   1. already-present     — image baked it in (future-proof; not today)
    #   2. restore from cache  — Modal volume hit; ~30 s copy
    #   3. build from source   — follows docs/inference/optimizations.md:60-68;
    #                             ~90 min nvcc compile (293 templated kernels)
    # On a successful build (path 3), we also save the freshly-installed files
    # to the cache volume so the *next* fresh container hits path 2.
    #
    # Cache invalidation: the cache marker filename embeds $IMAGE_VERSION
    # (forwarded via Secret). Bumping IMAGE_VERSION ⇒ different marker ⇒
    # cache miss ⇒ rebuild against the new torch/cuda ABI.
    #
    # We use `python setup.py install` (not `pip install .`) so nvcc compile
    # output is unbuffered to stdout — pip wraps the build and swallows
    # progress. `flash-attn-3` is NOT on PyPI as of 2026-05.
    fa3_install = (
        "echo '=== FA3 install ===' && "
        "SP=/opt/venv/lib/python3.12/site-packages && "
        "CACHE_DIR=/fa3_cache/site-packages-${IMAGE_VERSION:-unknown} && "
        "CACHE_MARKER=$CACHE_DIR/.INSTALLED && "
        # Path 1: already present (no-op fast path)
        "if python -c 'import flash_attn_interface' 2>/dev/null; then "
        "  echo '[fa3] already present in site-packages, skipping install'; "
        # Path 2: restore from Modal volume cache
        "elif [ -f \"$CACHE_MARKER\" ]; then "
        "  echo '[fa3] cache hit at' $CACHE_DIR && "
        "  cp -r $CACHE_DIR/. $SP/ && "
        "  python -c 'import flash_attn_interface' 2>/dev/null && "
        "  echo '[fa3] cache restore OK'; "
        # Path 3: source build (and save to cache on success)
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
        # Save to cache: list every freshly-installed FA3 artifact, copy
        # transactionally (staging → rename) so a partial commit doesn't
        # poison the cache.
        "  if python -c 'import flash_attn_interface' 2>/dev/null; then "
        "    echo '[fa3] saving build artifacts to cache' && "
        "    STAGE=/fa3_cache/staging-$$ && "
        "    mkdir -p $STAGE && "
        "    for pat in 'flash_attn_interface*' 'flash_attn_3' 'flash_attn_3-*' "
        "               'flash_attn_3*.egg-info' 'flash_attn_3*.dist-info'; do "
        "      find $SP -maxdepth 1 -name \"$pat\" "
        "        -exec cp -r {} $STAGE/ \\; 2>/dev/null; "
        "    done && "
        "    rm -rf $CACHE_DIR && "
        "    mv $STAGE $CACHE_DIR && "
        "    touch $CACHE_MARKER && "
        "    echo '[fa3] cache saved to' $CACHE_DIR && "
        "    ls -la $CACHE_DIR; "
        "  else "
        "    echo '[fa3] WARN: build claimed success but import fails — not caching'; "
        "  fi; "
        "fi ; "
        # Final verification independent of path taken; proceed either way so
        # we always get a profile (the probe right after will show fa_version).
        "python -c 'import flash_attn_interface; print(\"[fa3] OK module loaded\")' 2>&1 || "
        "echo '[fa3] WARN: flash_attn_interface still not importable — will run as FA2' "
    )
    command = (
        "set +e && "
        "source /opt/venv/bin/activate && "
        "cd /FastVideo && "
        '( [ -z "${HF_API_KEY:-}" ] || hf auth login --token "$HF_API_KEY" --quiet || true ) && '
        "echo '=== nvidia-smi ===' ; nvidia-smi 2>&1 || true ; "
        "echo '=== nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv ===' ; "
        "nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv 2>&1 || true ; "
        + fa3_install +
        " ; "
        # Probe FA version after install attempt.
        "python -c \"from fastvideo.attention.backends.flash_attn import fa_version; "
        "print(f'[probe] flash_attn fa_version={fa_version}')\" 2>&1 || true ; "
        "NSYS_CFG=$(nsys -z 2>/dev/null || true) && "
        "if [ -n \"$NSYS_CFG\" ]; then "
        "  mkdir -p \"$(dirname \"$NSYS_CFG\")\" && "
        "  echo 'CuptiUseRawGpuTimestamps=false' >> \"$NSYS_CFG\"; "
        "fi && "
        "nsys profile "
        "  --force-overwrite=true "
        "  -o /tmp/perf_h100_sp1_fa3 "
        "  --trace=cuda,nvtx "
        "  --pytorch=autograd-nvtx "
        "  --cpuctxsw=none "
        "  --sample=none "
        "  --cuda-memory-usage=false "
        "  --stats=false "
        f" -- bash -c '{pytest_cmd}' ; "
        "PYTEST_RC=$? ; "
        "cp -v /tmp/perf_h100_sp1_fa3.nsys-rep /results/perf_h100_sp1_fa3.nsys-rep 2>&1 || true ; "
        "mkdir -p /results/perf_json && "
        "find ./fastvideo/tests/performance/results -name 'perf_*.json' "
        "  -exec cp -v {} /results/perf_json/ \\; 2>&1 || true ; "
        "exit $PYTEST_RC")

    result = subprocess.run(
        ["/bin/bash", "-c", command],
        env=profile_env,
        stdout=sys.stdout,
        stderr=subprocess.PIPE,
        check=False,
    )
    stderr_text = (result.stderr or b"").decode(errors="replace")
    if stderr_text:
        sys.stderr.write(stderr_text)

    try:
        results_vol.commit()
    except Exception as exc:
        print(f"[perf-nsys-h100-sp1-fa3] results volume commit failed: {exc}", file=sys.stderr)
    # Persist FA3 build cache so the next fresh container hits the restore path.
    # Safe to call even on a cache-hit run (no-op if nothing changed).
    try:
        fa3_cache_vol.commit()
    except Exception as exc:
        print(f"[perf-nsys-h100-sp1-fa3] fa3 cache volume commit failed: {exc}", file=sys.stderr)

    has_nsys = os.path.isfile("/results/perf_h100_sp1_fa3.nsys-rep")
    print(f"\n[perf-nsys-h100-sp1-fa3] === Outcome === pytest={result.returncode}, nsys={'OK' if has_nsys else 'MISSING'}")
    print("\nDownload:")
    print("  modal volume get fastvideo-nsys-rep perf_h100_sp1_fa3.nsys-rep .")
    return int(result.returncode)


@app.local_entrypoint()
def main() -> None:
    # PERF_GPU selects which entrypoint to call:
    #   "h100-sp1"     -> 1x H100 baseline (FA2 default)
    #   "h100-sp1-fa3" -> 1x H100 with FA3 source-installed
    # Otherwise falls back to PERF_GPU_COUNT (legacy):
    #   1 -> sp=1 L40S oracle, 2 -> sp=2 L40S, else 4x L40S sp=4.
    gpu_sel = os.environ.get("PERF_GPU", "").lower()
    if gpu_sel == "h100-sp1":
        benchmark_id = os.environ.get("PERF_BENCHMARK_ID", "wan-t2v-1.3b-h100-sp1")
        print(f"[local] gpu=h100-sp1, benchmark_id={benchmark_id}")
        print(f"[local] uploading source from: {LOCAL_ROOT}")
        exit_code = run_perf_nsys_h100_sp1.remote(benchmark_id=benchmark_id)
        if exit_code != 0:
            print(f"[local] pytest exit code: {exit_code} "
                  "(non-zero may just mean threshold violation; "
                  "check Modal volume for perf*.nsys-rep)")
        return
    if gpu_sel == "h100-sp1-fa3":
        benchmark_id = os.environ.get("PERF_BENCHMARK_ID", "wan-t2v-1.3b-h100-sp1")
        print(f"[local] gpu=h100-sp1-fa3, benchmark_id={benchmark_id}")
        print(f"[local] uploading source from: {LOCAL_ROOT}")
        exit_code = run_perf_nsys_h100_sp1_fa3.remote(benchmark_id=benchmark_id)
        if exit_code != 0:
            print(f"[local] pytest exit code: {exit_code} "
                  "(non-zero may just mean threshold violation; "
                  "check Modal volume for perf_h100_sp1_fa3.nsys-rep)")
        return

    gpu_count = int(os.environ.get("PERF_GPU_COUNT", "4"))
    if gpu_count == 1:
        benchmark_id = os.environ.get("PERF_BENCHMARK_ID", "wan-t2v-1.3b-l40s-compile-sp1")
        print(f"[local] gpu_count=1, benchmark_id={benchmark_id}")
        print(f"[local] uploading source from: {LOCAL_ROOT}")
        exit_code = run_perf_nsys_sp1.remote(benchmark_id=benchmark_id)
    elif gpu_count == 2:
        benchmark_id = os.environ.get("PERF_BENCHMARK_ID", "wan-t2v-1.3b-l40s-hires-sp2")
        print(f"[local] gpu_count=2, benchmark_id={benchmark_id}")
        print(f"[local] uploading source from: {LOCAL_ROOT}")
        exit_code = run_perf_nsys_sp2.remote(benchmark_id=benchmark_id)
    else:
        benchmark_id = os.environ.get("PERF_BENCHMARK_ID", "wan-t2v-1.3b-l40s-hires")
        print(f"[local] gpu_count=4, benchmark_id={benchmark_id}")
        print(f"[local] uploading source from: {LOCAL_ROOT}")
        exit_code = run_perf_nsys.remote(benchmark_id=benchmark_id)
    if exit_code != 0:
        print(f"[local] pytest exit code: {exit_code} "
              "(non-zero may just mean threshold violation; "
              "check Modal volume for perf*.nsys-rep)")
