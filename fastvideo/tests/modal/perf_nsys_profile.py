# SPDX-License-Identifier: Apache-2.0
"""Modal app: paired inference perf benchmark + nsys profile (4x L40S).

Runs the pytest perf suite (FASTVIDEO_STAGE_LOGGING=1, per-stage NVTX) under
nsys, so a single Modal job produces:

  /results/perf_<benchmark_id>_<ts>.json  — metrics (E2E latency, throughput,
                                            peak memory, per-stage times)
  /results/perf.nsys-rep                  — Nsight Systems trace with NVTX
                                            ranges for every pipeline stage

Usage:
  modal run fastvideo/tests/modal/perf_nsys_profile.py
  PERF_BENCHMARK_ID=wan-t2v-1.3b-l40s-hires modal run fastvideo/tests/modal/perf_nsys_profile.py

Env vars:
  PERF_BENCHMARK_ID : pytest -k filter; matches benchmark_id in
                      .buildkite/performance-benchmarks/tests/*.json
                      (default: wan-t2v-1.3b-l40s-hires)
  HF_API_KEY        : HuggingFace token
  IMAGE_VERSION     : fastvideo-dev image tag (default: py3.12-latest)
  BUILDKITE_REPO    : git repo to clone (default: hao-ai-lab/FastVideo)
  BUILDKITE_COMMIT  : commit to check out (overrides PR)
  BUILDKITE_PULL_REQUEST : PR number to check out

Download results after the run:
  modal volume get fastvideo-nsys-rep perf.nsys-rep .
  modal volume get fastvideo-nsys-rep <perf_json_filename> .
"""

import os
import subprocess
import sys

import modal

app = modal.App()

model_vol = modal.Volume.from_name("hf-model-weights")
results_vol = modal.Volume.from_name(
    "fastvideo-nsys-rep",
    create_if_missing=True,
)

image_version = os.getenv("IMAGE_VERSION", "py3.12-latest")
image_tag = f"ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:{image_version}"

image = (
    modal.Image.from_registry(image_tag, add_python="3.12")
    .run_commands("rm -rf /FastVideo")
    .apt_install(
        "cmake",
        "pkg-config",
        "ninja-build",
        "git",
        "wget",
        "gnupg",
        "ca-certificates",
    )
    .run_commands(
        "wget -qO - https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nvidia.pub | "
        "  gpg --dearmor -o /usr/share/keyrings/nvidia-devtools-keyring.gpg && "
        "echo 'deb [signed-by=/usr/share/keyrings/nvidia-devtools-keyring.gpg] "
        "  https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/ /' "
        "  > /etc/apt/sources.list.d/nvidia-devtools.list && "
        "apt-get update && "
        "apt-get install -y --no-install-recommends nsight-systems-cli"
    )
    .env({
        "PATH": "/root/.cargo/bin:$PATH",
        "BUILDKITE_REPO": os.environ.get("BUILDKITE_REPO", ""),
        "BUILDKITE_COMMIT": os.environ.get("BUILDKITE_COMMIT", ""),
        "BUILDKITE_PULL_REQUEST": os.environ.get("BUILDKITE_PULL_REQUEST", ""),
        "IMAGE_VERSION": os.environ.get("IMAGE_VERSION", ""),
    })
)


def _build_checkout_command(git_commit: str | None, pr_number: str | None) -> str:
    if pr_number and pr_number != "false":
        return (f"git fetch --prune origin refs/pull/{pr_number}/head && "
                "git checkout FETCH_HEAD")
    if git_commit:
        return f"git checkout {git_commit}"
    return "git checkout HEAD"


def _build_install_script(git_repo: str, checkout_command: str) -> str:
    return f"""
    set -euo pipefail
    ulimit -s unlimited || ulimit -s 65536 || true
    mkdir -p /results/tmp_build
    export TMPDIR=/results/tmp_build
    export TEMP=/results/tmp_build
    export TMP=/results/tmp_build
    source $HOME/.local/bin/env
    source /opt/venv/bin/activate
    git clone -q {git_repo} /FastVideo &&
    cd /FastVideo &&
    ( {checkout_command} ) > /dev/null 2>&1 &&
    git submodule update --init --recursive -q &&
    export MAX_JOBS=1 &&
    export CMAKE_BUILD_PARALLEL_LEVEL=1 &&
    export SKBUILD_BUILD_ARGS="-j 1" &&
    cd fastvideo-kernel &&
    if ! ./build.sh > /results/tmp_build/build.log 2>&1; then
        echo "[ERROR] kernel build failed. Last 50 lines:" >&2
        tail -n 50 /results/tmp_build/build.log >&2
        exit 1
    fi &&
    cd .. &&
    if ! uv pip install -e .[test] > /results/tmp_build/install.log 2>&1; then
        echo "[ERROR] fastvideo install failed. Last 50 lines:" >&2
        tail -n 50 /results/tmp_build/install.log >&2
        exit 1
    fi &&
    export HF_HOME="/root/data/.cache" &&
    ( [ -z "${{HF_API_KEY:-}}" ] || \\
      hf auth login --token "$HF_API_KEY" --quiet || true )
    """


@app.function(
    gpu="L40S:4",
    image=image,
    timeout=3600,
    memory=65536,
    secrets=[modal.Secret.from_dict({"HF_API_KEY": os.environ.get("HF_API_KEY", "")})],
    volumes={"/root/data": model_vol, "/results": results_vol},
)
def run_perf_nsys(benchmark_id: str = "wan-t2v-1.3b-l40s-hires") -> int:
    git_repo = (os.environ.get("BUILDKITE_REPO")
                or "https://github.com/hao-ai-lab/FastVideo.git")
    git_commit = os.environ.get("BUILDKITE_COMMIT")
    pr_number = os.environ.get("BUILDKITE_PULL_REQUEST")
    checkout_command = _build_checkout_command(git_commit, pr_number)

    print(f"[perf-nsys] benchmark_id={benchmark_id}")
    print(f"[perf-nsys] repo={git_repo}")

    install_script = _build_install_script(git_repo, checkout_command)
    install_proc = subprocess.run(
        ["/bin/bash", "-c", install_script],
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=False,
    )
    if install_proc.returncode != 0:
        return install_proc.returncode

    profile_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "FASTVIDEO_STAGE_LOGGING": "1",
    }

    # nsys flags: --trace=cuda,nvtx (kernels + NVTX ranges from
    # stages/base.py::__call__), --pytorch=autograd-nvtx (free torch op
    # markers), --cpuctxsw/--sample=none (suppress QuadD errors on Modal).
    # `;` instead of `&&` after pytest so the trace is copied even when
    # pytest exits non-zero (threshold violation is expected on first run).
    pytest_cmd = (
        f"pytest -vs ./fastvideo/tests/performance/test_inference_performance.py "
        f"-k {benchmark_id}")
    command = (
        "set +e && "
        "source /opt/venv/bin/activate && "
        "cd /FastVideo && "
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
        "  --stats=false "
        f" -- bash -c '{pytest_cmd}' ; "
        "PYTEST_RC=$? ; "
        # Copy nsys trace from /tmp (local disk) to /results (FUSE).
        "cp -v /tmp/perf.nsys-rep /results/perf.nsys-rep 2>&1 || true ; "
        # Copy perf JSON output(s) — written by the pytest test.
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


@app.local_entrypoint()
def main() -> None:
    benchmark_id = os.environ.get("PERF_BENCHMARK_ID", "wan-t2v-1.3b-l40s-hires")
    print(f"[local] benchmark_id={benchmark_id}")
    exit_code = run_perf_nsys.remote(benchmark_id=benchmark_id)
    if exit_code != 0:
        # Non-zero is expected on first run (threshold seeding); only raise
        # if the trace itself is missing (treat as infra failure).
        print(f"[local] pytest exit code: {exit_code} "
              "(non-zero may just mean threshold violation; "
              "check Modal volume for perf.nsys-rep)")
