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
    secrets=[modal.Secret.from_dict({"HF_API_KEY": os.environ.get("HF_API_KEY", "")})],
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
    secrets=[modal.Secret.from_dict({"HF_API_KEY": os.environ.get("HF_API_KEY", "")})],
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
    secrets=[modal.Secret.from_dict({"HF_API_KEY": os.environ.get("HF_API_KEY", "")})],
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


@app.local_entrypoint()
def main() -> None:
    # PERF_GPU_COUNT=1 selects sp=1 (single-GPU oracle), =2 selects sp=2 (2 ranks),
    # otherwise defaults to the 4x L40S sp=4 run.
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
