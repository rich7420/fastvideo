# SPDX-License-Identifier: Apache-2.0
"""One-off Modal job: nsys export .nsys-rep -> .sqlite on the cloud side.

Local nsys (2026.1.1) refuses to read traces produced by Modal's container
nsys (2026.2.1). This script runs the export inside a Modal container so
the resulting .sqlite can be downloaded and analyzed locally with
nsys-ai (which reads SQLite directly, no nsys version check).

Usage:
  .venv/bin/modal run fastvideo/tests/modal/export_nsys_sqlite.py
"""

import os
import subprocess
import sys

import modal

app = modal.App()
results_vol = modal.Volume.from_name("fastvideo-nsys-rep")

image = (
    modal.Image.debian_slim()
    .apt_install("wget", "gnupg", "ca-certificates")
    .run_commands(
        "wget -qO - https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nvidia.pub | "
        "  gpg --dearmor -o /usr/share/keyrings/nvidia-devtools-keyring.gpg && "
        "echo 'deb [signed-by=/usr/share/keyrings/nvidia-devtools-keyring.gpg] "
        "  https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/ /' "
        "  > /etc/apt/sources.list.d/nvidia-devtools.list && "
        "apt-get update && "
        "apt-get install -y --no-install-recommends nsight-systems-cli"))


@app.function(image=image, timeout=1800, volumes={"/results": results_vol})
def export_to_sqlite(input_name: str = "perf.nsys-rep",
                     output_name: str = "perf.sqlite") -> int:
    src = f"/results/{input_name}"
    dst = f"/results/{output_name}"
    if not os.path.isfile(src):
        print(f"[export] not found: {src}", file=sys.stderr)
        return 1
    print(f"[export] {src} -> {dst}")
    cmd = [
        "nsys", "export",
        "--type", "sqlite",
        "--force-overwrite=true",
        "--include-blobs=true",
        "-o", dst,
        src,
    ]
    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if result.returncode == 0:
        try:
            results_vol.commit()
        except Exception as exc:
            print(f"[export] commit failed: {exc}", file=sys.stderr)
        size_mb = os.path.getsize(dst) / (1024 * 1024)
        print(f"[export] OK ({size_mb:.1f} MB). Download:")
        print(f"  modal volume get fastvideo-nsys-rep {output_name} .")
    return result.returncode


@app.local_entrypoint()
def main() -> None:
    exit_code = export_to_sqlite.remote()
    if exit_code != 0:
        raise SystemExit(exit_code)
