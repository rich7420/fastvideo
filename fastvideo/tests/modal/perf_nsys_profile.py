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


@app.function(
    gpu="H100:2",
    image=image,
    timeout=10800,
    memory=65536,
    secrets=[modal.Secret.from_dict({
        "HF_API_KEY": os.environ.get("HF_API_KEY", ""),
        "FASTVIDEO_TGATE_STEP": os.environ.get("FASTVIDEO_TGATE_STEP", "1.0"),
        "FASTVIDEO_ATTENTION_BACKEND": os.environ.get("FASTVIDEO_ATTENTION_BACKEND", ""),
        "IMAGE_VERSION": image_version,
    })],
    volumes={
        "/root/data": model_vol,
        "/results": results_vol,
        "/fa3_cache": fa3_cache_vol,
    },
)
def run_perf_h100_sp2_fa3(benchmark_id: str = "wan-t2v-1.3b-h100-sp2") -> int:
    # H100:2 sp=2 with FA3. Verifies the disable-removed attention path under
    # sequence parallelism (all-to-all collective now inside the compiled
    # graph). No nsys wrapper — we only need does-it-run + perf JSON. The FA3
    # install is the same 3-tier strategy as run_perf_nsys_h100_sp1_fa3.
    print(f"[perf-h100-sp2-fa3] benchmark_id={benchmark_id}")
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
    fa3_install = (
        "echo '=== FA3 install ===' && "
        "SP=/opt/venv/lib/python3.12/site-packages && "
        "CACHE_DIR=/fa3_cache/site-packages-${IMAGE_VERSION:-unknown} && "
        "CACHE_MARKER=$CACHE_DIR/.INSTALLED && "
        "if python -c 'import flash_attn_interface' 2>/dev/null; then "
        "  echo '[fa3] already present'; "
        "elif [ -f \"$CACHE_MARKER\" ]; then "
        "  echo '[fa3] cache hit at' $CACHE_DIR && cp -r $CACHE_DIR/. $SP/ && "
        "  python -c 'import flash_attn_interface' 2>/dev/null && echo '[fa3] cache restore OK'; "
        "else "
        "  echo '[fa3] cache miss; building from source' && pip install -q ninja && "
        "  rm -rf /tmp/flash-attention && "
        "  git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && "
        "  cd /tmp/flash-attention/hopper && MAX_JOBS=8 python setup.py install && cd /FastVideo && "
        "  if python -c 'import flash_attn_interface' 2>/dev/null; then "
        "    STAGE=/fa3_cache/staging-$$ && mkdir -p $STAGE && "
        "    for pat in 'flash_attn_interface*' 'flash_attn_3' 'flash_attn_3-*' "
        "               'flash_attn_3*.egg-info' 'flash_attn_3*.dist-info'; do "
        "      find $SP -maxdepth 1 -name \"$pat\" -exec cp -r {} $STAGE/ \\; 2>/dev/null; "
        "    done && rm -rf $CACHE_DIR && mv $STAGE $CACHE_DIR && touch $CACHE_MARKER; "
        "  fi; "
        "fi ; "
        "python -c 'from fastvideo.attention.backends.flash_attn import fa_version; "
        "print(f\"[probe] fa_version={fa_version}\")' 2>&1 || true"
    )
    command = (
        "set +e && source /opt/venv/bin/activate && cd /FastVideo && "
        '( [ -z "${HF_API_KEY:-}" ] || hf auth login --token "$HF_API_KEY" --quiet || true ) && '
        "echo '=== nvidia-smi ===' ; nvidia-smi --query-gpu=name,compute_cap --format=csv 2>&1 || true ; "
        + fa3_install + " ; "
        + pytest_cmd + " ; PYTEST_RC=$? ; "
        "mkdir -p /results/perf_json && "
        "find ./fastvideo/tests/performance/results -name 'perf_*.json' "
        "  -exec cp -v {} /results/perf_json/ \\; 2>&1 || true ; "
        "exit $PYTEST_RC")
    result = subprocess.run(["/bin/bash", "-c", command], env=profile_env,
                            stdout=sys.stdout, stderr=subprocess.PIPE, check=False)
    stderr_text = (result.stderr or b"").decode(errors="replace")
    if stderr_text:
        sys.stderr.write(stderr_text)
    try:
        results_vol.commit()
        fa3_cache_vol.commit()
    except Exception as exc:
        print(f"[perf-h100-sp2-fa3] volume commit failed: {exc}", file=sys.stderr)
    print(f"\n[perf-h100-sp2-fa3] === Outcome === pytest={result.returncode}")
    print("  Download: modal volume get fastvideo-nsys-rep perf_json/ . --recursive")
    return int(result.returncode)


# Shared FA3 source-install block (3-tier: present / cache / build-from-source).
# Identical to the inline block in run_perf_h100_sp2_fa3; factored out so the
# nsys-wrapping multi-GPU variants below stay in sync.
_FA3_INSTALL_SH = (
    "echo '=== FA3 install ===' && "
    "SP=/opt/venv/lib/python3.12/site-packages && "
    "CACHE_DIR=/fa3_cache/site-packages-${IMAGE_VERSION:-unknown} && "
    "CACHE_MARKER=$CACHE_DIR/.INSTALLED && "
    "if python -c 'import flash_attn_interface' 2>/dev/null; then "
    "  echo '[fa3] already present'; "
    "elif [ -f \"$CACHE_MARKER\" ]; then "
    "  echo '[fa3] cache hit at' $CACHE_DIR && cp -r $CACHE_DIR/. $SP/ && "
    "  python -c 'import flash_attn_interface' 2>/dev/null && echo '[fa3] cache restore OK'; "
    "else "
    "  echo '[fa3] cache miss; building from source' && pip install -q ninja && "
    "  rm -rf /tmp/flash-attention && "
    "  git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && "
    "  cd /tmp/flash-attention/hopper && MAX_JOBS=8 python setup.py install && cd /FastVideo && "
    "  if python -c 'import flash_attn_interface' 2>/dev/null; then "
    "    STAGE=/fa3_cache/staging-$$ && mkdir -p $STAGE && "
    "    for pat in 'flash_attn_interface*' 'flash_attn_3' 'flash_attn_3-*' "
    "               'flash_attn_3*.egg-info' 'flash_attn_3*.dist-info'; do "
    "      find $SP -maxdepth 1 -name \"$pat\" -exec cp -r {} $STAGE/ \\; 2>/dev/null; "
    "    done && rm -rf $CACHE_DIR && mv $STAGE $CACHE_DIR && touch $CACHE_MARKER; "
    "  fi; "
    "fi ; "
    "python -c 'from fastvideo.attention.backends.flash_attn import fa_version; "
    "print(f\"[probe] fa_version={fa_version}\")' 2>&1 || true"
)


def _run_h100_nsys_fa3(benchmark_id: str, sp: int, log_tag: str) -> int:
    """Shared body for the nsys-wrapped multi-GPU H100 FA3 runs (sp2 / sp4).

    Unlike run_perf_h100_sp2_fa3 (JSON-only), this wraps the run in `nsys
    profile` so we get the per-collective NCCL breakdown + compute<->comm
    overlap on H100's NVSwitch fabric. Output: /results/perf_h100_sp{sp}_fa3.nsys-rep.
    """
    out_stem = f"perf_h100_sp{sp}_fa3"
    print(f"[{log_tag}] benchmark_id={benchmark_id} sp={sp}")
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
    nccl_debug = os.environ.get("PERF_NCCL_DEBUG", "INFO")
    command = (
        "set +e && source /opt/venv/bin/activate && cd /FastVideo && "
        '( [ -z "${HF_API_KEY:-}" ] || hf auth login --token "$HF_API_KEY" --quiet || true ) && '
        # Topology so NVLink/NVSwitch vs PCIe is recorded alongside the trace.
        "echo '=== nvidia-smi topo -m ===' ; nvidia-smi topo -m 2>&1 || true ; "
        "nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv 2>&1 || true ; "
        f"export NCCL_DEBUG={nccl_debug} ; "
        + _FA3_INSTALL_SH + " ; "
        # QuadD UUID workaround (non-fatal cloud nsys error).
        "NSYS_CFG=$(nsys -z 2>/dev/null || true) && "
        "if [ -n \"$NSYS_CFG\" ]; then "
        "  mkdir -p \"$(dirname \"$NSYS_CFG\")\" && "
        "  echo 'CuptiUseRawGpuTimestamps=false' >> \"$NSYS_CFG\"; "
        "fi ; "
        "nsys profile "
        "  --force-overwrite=true "
        f"  -o /tmp/{out_stem} "
        "  --trace=cuda,nvtx "
        "  --pytorch=autograd-nvtx "
        "  --cpuctxsw=none "
        "  --sample=none "
        "  --cuda-memory-usage=false "
        "  --stats=false "
        f" -- bash -c '{pytest_cmd}' ; "
        "PYTEST_RC=$? ; "
        f"cp -v /tmp/{out_stem}.nsys-rep /results/{out_stem}.nsys-rep 2>&1 || true ; "
        "mkdir -p /results/perf_json && "
        "find ./fastvideo/tests/performance/results -name 'perf_*.json' "
        "  -exec cp -v {} /results/perf_json/ \\; 2>&1 || true ; "
        "exit $PYTEST_RC")
    result = subprocess.run(["/bin/bash", "-c", command], env=profile_env,
                            stdout=sys.stdout, stderr=subprocess.PIPE, check=False)
    stderr_text = (result.stderr or b"").decode(errors="replace")
    if stderr_text:
        sys.stderr.write(stderr_text)
    try:
        results_vol.commit()
        fa3_cache_vol.commit()
    except Exception as exc:
        print(f"[{log_tag}] volume commit failed: {exc}", file=sys.stderr)
    has_nsys = os.path.isfile(f"/results/{out_stem}.nsys-rep")
    print(f"\n[{log_tag}] === Outcome === pytest={result.returncode}, nsys={'OK' if has_nsys else 'MISSING'}")
    print(f"  Download: modal volume get fastvideo-nsys-rep {out_stem}.nsys-rep .")
    return int(result.returncode)


@app.function(
    gpu="H100:2",
    image=image,
    timeout=10800,
    memory=65536,
    secrets=[modal.Secret.from_dict({
        "HF_API_KEY": os.environ.get("HF_API_KEY", ""),
        "FASTVIDEO_TGATE_STEP": os.environ.get("FASTVIDEO_TGATE_STEP", "1.0"),
        "FASTVIDEO_ATTENTION_BACKEND": os.environ.get("FASTVIDEO_ATTENTION_BACKEND", ""),
        "IMAGE_VERSION": image_version,
    })],
    volumes={
        "/root/data": model_vol,
        "/results": results_vol,
        "/fa3_cache": fa3_cache_vol,
    },
)
def run_perf_nsys_h100_sp2_fa3(benchmark_id: str = "wan-t2v-1.3b-h100-sp2") -> int:
    return _run_h100_nsys_fa3(benchmark_id, sp=2, log_tag="perf-nsys-h100-sp2-fa3")


@app.function(
    gpu="H100:4",
    image=image,
    timeout=10800,
    memory=65536,
    secrets=[modal.Secret.from_dict({
        "HF_API_KEY": os.environ.get("HF_API_KEY", ""),
        "FASTVIDEO_TGATE_STEP": os.environ.get("FASTVIDEO_TGATE_STEP", "1.0"),
        "FASTVIDEO_ATTENTION_BACKEND": os.environ.get("FASTVIDEO_ATTENTION_BACKEND", ""),
        "IMAGE_VERSION": image_version,
    })],
    volumes={
        "/root/data": model_vol,
        "/results": results_vol,
        "/fa3_cache": fa3_cache_vol,
    },
)
def run_perf_nsys_h100_sp4_fa3(benchmark_id: str = "wan-t2v-1.3b-h100-sp4") -> int:
    return _run_h100_nsys_fa3(benchmark_id, sp=4, log_tag="perf-nsys-h100-sp4-fa3")


def _deep_profile_secrets():
    return [modal.Secret.from_dict({
        "HF_API_KEY": os.environ.get("HF_API_KEY", ""),
        "FASTVIDEO_TGATE_STEP": os.environ.get("FASTVIDEO_TGATE_STEP", "1.0"),
        "FASTVIDEO_ATTENTION_BACKEND": os.environ.get("FASTVIDEO_ATTENTION_BACKEND", ""),
        "IMAGE_VERSION": image_version,
    })]


_DEEP_VOLS = {"/root/data": model_vol, "/results": results_vol, "/fa3_cache": fa3_cache_vol}


@app.function(gpu="H100:1", image=image, timeout=10800, memory=65536,
              secrets=_deep_profile_secrets(), volumes=_DEEP_VOLS)
def run_ncu_roofline(benchmark_id: str = "wan-t2v-1.3b-h100-sp1") -> int:
    # EXPERIMENT 1: Nsight Compute roofline on the hot kernels (FA3 attn / nvjet GEMM /
    # VAE conv) — answers "how far from peak" per kernel. NOTE: ncu HW counters may be
    # blocked on cloud (ERR_NVGPUCTRPERM); we probe and report either way.
    print(f"[ncu-roofline] benchmark_id={benchmark_id}")
    profile_env = {**os.environ, "PYTHONUNBUFFERED": "1",
                   "FASTVIDEO_STAGE_LOGGING": "1", "HF_HOME": "/root/data/.cache"}
    pytest_cmd = ("pytest -q --assert=plain "
                  "./fastvideo/tests/performance/test_inference_performance.py "
                  f"-k [{benchmark_id}]")
    command = (
        "set +e && source /opt/venv/bin/activate && cd /FastVideo && "
        '( [ -z "${HF_API_KEY:-}" ] || hf auth login --token "$HF_API_KEY" --quiet || true ) && '
        "nvidia-smi --query-gpu=name,compute_cap --format=csv 2>&1 | head -2 ; "
        + _FA3_INSTALL_SH + " ; "
        # The toolkit-bundled ncu (CUDA 12.8) is too old for Modal's driver 580 →
        # 'LibraryNotLoaded'. Install driver-matched nsight-compute-2026.1.0 from the
        # NVIDIA repo (recipe: gist Ubospica/622acde9...).
        "echo '[ncu] installing nsight-compute-2026.1.0 from NVIDIA repo' ; "
        "apt-get install -y -qq wget gnupg 2>&1 | tail -1 ; "
        ". /etc/os-release 2>/dev/null ; "
        "case \"${ID}${VERSION_ID}\" in debian12) REPO=debian12 ;; ubuntu24.04) REPO=ubuntu2404 ;; ubuntu20.04) REPO=ubuntu2004 ;; *) REPO=ubuntu2204 ;; esac ; "
        "echo \"[ncu] os=${ID}${VERSION_ID} repo=$REPO\" ; "
        "wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/$REPO/x86_64/3bf863cc.pub | gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg 2>/dev/null ; "
        "echo \"deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/$REPO/x86_64/ /\" > /etc/apt/sources.list.d/cuda-ncu.list ; "
        "apt-get update -qq 2>&1 | tail -2 ; apt-get install -y -qq nsight-compute-2026.1.0 2>&1 | tail -3 ; "
        "NCU=$(ls -d /opt/nvidia/nsight-compute/2026*/ncu 2>/dev/null | sort -V | tail -1) ; "
        "[ -z \"$NCU\" ] && NCU=$(ls -d /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | sort -V | tail -1) ; "
        "[ -z \"$NCU\" ] && NCU=$(command -v ncu) ; "
        "echo \"[ncu] using: ${NCU:-MISSING}\" ; [ -n \"$NCU\" ] && \"$NCU\" --version 2>&1 | head -3 ; "
        # roofline on key kernels; skip warmup, bounded count; no clock-control (avoids priv)
        "\"$NCU\" --set roofline --target-processes all --clock-control none "
        "  --kernel-name regex:'FlashAttnFwdSm90|nvjet|xmma_fprop' "
        "  --launch-skip 600 --launch-count 30 "
        "  -f -o /results/ncu_h100_sp1_fa3 "
        "  -- bash -c \"" + pytest_cmd + "\" 2>&1 | tail -60 ; NCU_RC=${PIPESTATUS[0]} ; "
        "echo \"[ncu] exit=$NCU_RC\" ; ls -la /results/ncu_h100_sp1_fa3.ncu-rep 2>&1 || true ; "
        "exit $NCU_RC")
    result = subprocess.run(["/bin/bash", "-c", command], env=profile_env,
                            stdout=sys.stdout, stderr=subprocess.PIPE, check=False)
    if result.stderr:
        sys.stderr.write(result.stderr.decode(errors="replace"))
    try:
        results_vol.commit(); fa3_cache_vol.commit()
    except Exception as exc:
        print(f"[ncu-roofline] commit failed: {exc}", file=sys.stderr)
    print(f"[ncu-roofline] === Outcome === rc={result.returncode}")
    print("  Download: modal volume get fastvideo-nsys-rep ncu_h100_sp1_fa3.ncu-rep .")
    return int(result.returncode)


@app.function(gpu="H100:1", image=image, timeout=10800, memory=65536,
              secrets=_deep_profile_secrets(), volumes=_DEEP_VOLS)
def run_cutracer_sass(benchmark_id: str = "wan-t2v-1.3b-h100-sp1") -> int:
    # EXPERIMENT 2: CUTracer (NVBit SASS instruction histogram) on the hot kernels.
    # NVBit runs in user space → works even where ncu HW counters are blocked.
    # Heavy: kernel-filtered + CUTRACER_MAX_ITERS=1 to bound instrumentation cost.
    print(f"[cutracer] benchmark_id={benchmark_id}")
    # v2: target the FA3 attention (cutlass Sm90 — SASS resolves, unlike cuBLAS nvjet) to answer
    # "is 72%-MFU attention tensor-core-saturated?" (ncu can't — blocked on Modal). MMA-only
    # instrumentation = far fewer instrumented points → dodges the earlier no-data deadlock on the
    # warp-specialized Sm90 kernel; longer no-data timeout as belt-and-suspenders.
    profile_env = {**os.environ, "PYTHONUNBUFFERED": "1",
                   "FASTVIDEO_STAGE_LOGGING": "1", "HF_HOME": "/root/data/.cache",
                   "CUTRACER_MAX_ITERS": "1",
                   "CUTRACER_INSTR_CATEGORIES": os.environ.get("CUTRACER_INSTR_CATEGORIES", "mma"),
                   "CUTRACER_NO_DATA_TIMEOUT_S": "300"}
    pytest_cmd = ("pytest -q --assert=plain "
                  "./fastvideo/tests/performance/test_inference_performance.py "
                  f"-k [{benchmark_id}]")
    # 'device_kernel' matches the cutlass FA3 wrapper symbol; 'gelu' = a triton FFN kernel that
    # is guaranteed to resolve (fallback signal that CUTracer is working).
    kfilter = os.environ.get("CUTRACER_KFILTER", "device_kernel,gelu")
    command = (
        "set +e && source /opt/venv/bin/activate && cd /FastVideo && "
        '( [ -z "${HF_API_KEY:-}" ] || hf auth login --token "$HF_API_KEY" --quiet || true ) && '
        + _FA3_INSTALL_SH + " ; "
        # build CUTracer NVBit .so (nvcc present in dev image; libzstd via apt)
        "apt-get update -qq 2>&1 | tail -1 ; apt-get install -y -qq libzstd-dev 2>&1 | tail -1 ; "
        "pip install -q cutracer 2>&1 | tail -2 || true ; "
        "SO=/root/.nsys-ai/cutracer/lib/cutracer.so ; "
        "if [ ! -f \"$SO\" ]; then echo '[cutracer] building .so' ; "
        "  rm -rf /opt/CUTracer && git clone --depth=1 https://github.com/facebookresearch/CUTracer /opt/CUTracer 2>&1 | tail -2 && "
        "  (cd /opt/CUTracer && ./install_third_party.sh 2>&1 | tail -3 && make 2>&1 | tail -5) && "
        "  mkdir -p /root/.nsys-ai/cutracer/lib && cp /opt/CUTracer/lib/cutracer.so $SO 2>&1 ; fi ; "
        "echo \"[cutracer] so: $([ -f $SO ] && echo OK || echo MISSING)\" ; "
        "mkdir -p /results/cutracer_out ; "
        "cutracer trace --cutracer-so $SO --analysis proton_instr_histogram "
        f"  --kernel-filters {kfilter} --output-dir /results/cutracer_out "
        "  -- bash -c \"" + pytest_cmd + "\" 2>&1 | tail -50 ; CT_RC=${PIPESTATUS[0]} ; "
        "echo \"[cutracer] exit=$CT_RC\" ; ls -la /results/cutracer_out 2>&1 || true ; "
        "exit $CT_RC")
    result = subprocess.run(["/bin/bash", "-c", command], env=profile_env,
                            stdout=sys.stdout, stderr=subprocess.PIPE, check=False)
    if result.stderr:
        sys.stderr.write(result.stderr.decode(errors="replace"))
    try:
        results_vol.commit(); fa3_cache_vol.commit()
    except Exception as exc:
        print(f"[cutracer] commit failed: {exc}", file=sys.stderr)
    print(f"[cutracer] === Outcome === rc={result.returncode}")
    print("  Download: modal volume get fastvideo-nsys-rep cutracer_out/ . --recursive")
    return int(result.returncode)


@app.function(gpu="H100:4", image=image, timeout=10800, memory=65536,
              secrets=_deep_profile_secrets(), volumes=_DEEP_VOLS)
def run_nsys_cpu_trace(benchmark_id: str = "wan-t2v-1.3b-h100-sp4") -> int:
    # EXPERIMENT 3: nsys with CPU-side tracing (osrt + python sampling + cpu ctxsw) to see
    # what the CPU does during the GPU bubbles / 45% sync time. The earlier traces were
    # --trace=cuda,nvtx only (CPU invisible). Output perf_h100_sp4_cpu.nsys-rep.
    print(f"[nsys-cpu] benchmark_id={benchmark_id}")
    profile_env = {**os.environ, "PYTHONUNBUFFERED": "1",
                   "FASTVIDEO_STAGE_LOGGING": "1", "HF_HOME": "/root/data/.cache"}
    pytest_cmd = ("pytest -vs --assert=plain "
                  "./fastvideo/tests/performance/test_inference_performance.py "
                  f"-k \"[{benchmark_id}]\"")
    command = (
        "set +e && source /opt/venv/bin/activate && cd /FastVideo && "
        '( [ -z "${HF_API_KEY:-}" ] || hf auth login --token "$HF_API_KEY" --quiet || true ) && '
        "nvidia-smi --query-gpu=name,compute_cap --format=csv 2>&1 | head -4 ; "
        f"export NCCL_DEBUG=INFO ; "
        + _FA3_INSTALL_SH + " ; "
        "NSYS_CFG=$(nsys -z 2>/dev/null || true) && "
        "if [ -n \"$NSYS_CFG\" ]; then mkdir -p \"$(dirname \"$NSYS_CFG\")\" && "
        "  echo 'CuptiUseRawGpuTimestamps=false' >> \"$NSYS_CFG\"; fi ; "
        # NOTE: --python-sampling=true segfaults this workload at startup (nsys 2026.2.1);
        # native CPU sampling + osrt still classifies sync-bound vs dispatch-bound.
        "nsys profile --force-overwrite=true -o /tmp/perf_h100_sp4_cpu "
        "  --trace=cuda,nvtx,osrt "
        "  --pytorch=autograd-nvtx "
        "  --sample=process-tree --cpuctxsw=process-tree "
        "  --cuda-memory-usage=false --stats=false "
        f" -- bash -c '{pytest_cmd}' ; PYTEST_RC=$? ; "
        "cp -v /tmp/perf_h100_sp4_cpu.nsys-rep /results/perf_h100_sp4_cpu.nsys-rep 2>&1 || true ; "
        "exit $PYTEST_RC")
    result = subprocess.run(["/bin/bash", "-c", command], env=profile_env,
                            stdout=sys.stdout, stderr=subprocess.PIPE, check=False)
    if result.stderr:
        sys.stderr.write(result.stderr.decode(errors="replace"))
    try:
        results_vol.commit(); fa3_cache_vol.commit()
    except Exception as exc:
        print(f"[nsys-cpu] commit failed: {exc}", file=sys.stderr)
    has = os.path.isfile("/results/perf_h100_sp4_cpu.nsys-rep")
    print(f"[nsys-cpu] === Outcome === rc={result.returncode} nsys={'OK' if has else 'MISSING'}")
    print("  Download: modal volume get fastvideo-nsys-rep perf_h100_sp4_cpu.nsys-rep .")
    return int(result.returncode)


@app.local_entrypoint()
def main() -> None:
    # PERF_GPU selects which entrypoint to call:
    #   "h100-sp1"     -> 1x H100 baseline (FA2 default)
    #   "h100-sp1-fa3" -> 1x H100 with FA3 source-installed
    #   "h100-sp2-fa3" -> 2x H100 sp=2 with FA3 (verify distributed compile path)
    #   "h100-sp2-nsys" -> 2x H100 sp=2 FA3, nsys-wrapped (comm-overlap trace)
    #   "h100-sp4-nsys" -> 4x H100 sp=4 FA3, nsys-wrapped (comm-overlap trace)
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
    if gpu_sel == "h100-sp2-fa3":
        benchmark_id = os.environ.get("PERF_BENCHMARK_ID", "wan-t2v-1.3b-h100-sp2")
        print(f"[local] gpu=h100-sp2-fa3, benchmark_id={benchmark_id}")
        print(f"[local] uploading source from: {LOCAL_ROOT}")
        exit_code = run_perf_h100_sp2_fa3.remote(benchmark_id=benchmark_id)
        if exit_code != 0:
            print(f"[local] pytest exit code: {exit_code} "
                  "(non-zero may just mean threshold violation; "
                  "check Modal volume for perf_json/)")
        return
    if gpu_sel == "h100-sp2-nsys":
        benchmark_id = os.environ.get("PERF_BENCHMARK_ID", "wan-t2v-1.3b-h100-sp2")
        print(f"[local] gpu=h100-sp2-nsys, benchmark_id={benchmark_id}")
        print(f"[local] uploading source from: {LOCAL_ROOT}")
        exit_code = run_perf_nsys_h100_sp2_fa3.remote(benchmark_id=benchmark_id)
        if exit_code != 0:
            print(f"[local] pytest exit code: {exit_code} "
                  "(non-zero may just mean threshold violation; "
                  "check Modal volume for perf_h100_sp2_fa3.nsys-rep)")
        return
    if gpu_sel == "h100-sp4-nsys":
        benchmark_id = os.environ.get("PERF_BENCHMARK_ID", "wan-t2v-1.3b-h100-sp4")
        print(f"[local] gpu=h100-sp4-nsys, benchmark_id={benchmark_id}")
        print(f"[local] uploading source from: {LOCAL_ROOT}")
        exit_code = run_perf_nsys_h100_sp4_fa3.remote(benchmark_id=benchmark_id)
        if exit_code != 0:
            print(f"[local] pytest exit code: {exit_code} "
                  "(non-zero may just mean threshold violation; "
                  "check Modal volume for perf_h100_sp4_fa3.nsys-rep)")
        return
    if gpu_sel == "ncu-roofline":
        print("[local] gpu=ncu-roofline (H100:1 FA3)")
        run_ncu_roofline.remote()
        return
    if gpu_sel == "cutracer":
        print("[local] gpu=cutracer (H100:1 FA3)")
        run_cutracer_sass.remote()
        return
    if gpu_sel == "cpu-trace":
        print("[local] gpu=cpu-trace (H100:4 FA3)")
        run_nsys_cpu_trace.remote()
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
