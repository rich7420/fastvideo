# SPDX-License-Identifier: Apache-2.0
"""Modal job: 2-rank correctness test for the SP all-to-all stream split.

Verifies that the comm-stream-split implementation in
`fastvideo.distributed.device_communicators.base_device_communicator` returns
bitwise-identical results to the original single-stream pattern.

Usage:
  .venv/bin/modal run fastvideo/tests/modal/test_all_to_all_correctness.py
"""

import os
import pathlib
import subprocess
import sys
import textwrap

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

_IGNORE = [
    ".git/**", ".venv/**", "**/__pycache__/**", "**/*.pyc",
    "*.nsys-rep", "*.sqlite", "*.nsys-cache/**", "*.qdstrm",
    "nsys_results/**", "fastvideo/tests/performance/results/**",
    "fastvideo/tests/performance/generated_videos/**",
    ".parquet_build_*/**",
    "fastvideo-kernel/build/**", "fastvideo-kernel/_skbuild/**",
    "build/**", "dist/**", "*.egg-info/**",
]

image = (modal.Image.from_registry(image_tag, add_python="3.12")
         .run_commands("rm -rf /FastVideo")
         .add_local_dir(str(LOCAL_ROOT), remote_path="/FastVideo", ignore=_IGNORE))


# Worker script written to /tmp inside the container. Runs torchrun with
# 2 ranks. Worker re-imports this file as __mp_main__ in spawn-mode, so
# guard with __main__.
_WORKER_SCRIPT = textwrap.dedent("""\
    import os
    import sys
    import torch
    import torch.distributed as dist

    if __name__ == "__main__":
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        group = dist.group.WORLD

        # Import after init so the module sees a valid group.
        from fastvideo.distributed.device_communicators.base_device_communicator import (
            DistributedAutograd,
        )
        import torch.distributed as _dist

        # Deterministic input shared across ranks via broadcast.
        torch.manual_seed(2026 + rank)
        bs, shard_seqlen, hn, hd = 2, 8, world_size * 4, 32
        x = torch.randn(bs, shard_seqlen, hn, hd, device=device, dtype=torch.bfloat16)

        # Path A — new code: comm-stream split path
        out_new = DistributedAutograd.AllToAll4D.apply(group, x.clone(), world_size, 2, 1)

        # Path B — reference: inline the original single-stream impl
        def reference_a2a_4d(input_, group, world_size, scatter_dim, gather_dim):
            assert scatter_dim == 2 and gather_dim == 1
            bs, shard_seqlen, hn, hd = input_.shape
            shard_hn = hn // world_size
            input_ = input_.transpose(0, 2).contiguous()
            output = torch.empty_like(input_)
            _dist.all_to_all_single(output, input_, group=group)
            output = torch.cat(output.split(shard_hn), dim=1)
            output = output.transpose(0, 2).contiguous()
            return output

        out_ref = reference_a2a_4d(x.clone(), group, world_size, 2, 1)

        # Bitwise equality across both paths.
        torch.cuda.synchronize()
        same = torch.equal(out_new, out_ref)
        # Reduce across ranks so any failure surfaces on rank 0.
        ok = torch.tensor([1 if same else 0], device=device)
        dist.all_reduce(ok)

        # Round-trip: do scatter-then-gather and recover x (up to padding).
        round_trip = DistributedAutograd.AllToAll4D.apply(group, out_new, world_size, 1, 2)
        # Shape comparison: round_trip should be (bs, shard_seqlen, hn, hd)
        shape_ok = round_trip.shape == x.shape

        if rank == 0:
            print(f"[rank {rank}] new vs ref bitwise equal on all {world_size} ranks: "
                  f"{ok.item() == world_size}")
            print(f"[rank {rank}] round-trip shape correct: {shape_ok}")
            print(f"[rank {rank}] input  shape: {x.shape}")
            print(f"[rank {rank}] out_new shape: {out_new.shape}")
            print(f"[rank {rank}] round_trip shape: {round_trip.shape}")
            assert ok.item() == world_size, "BITWISE MISMATCH between new and reference"
            assert shape_ok, "round-trip shape mismatch"
            print("[rank 0] PASS: all checks green")

        dist.destroy_process_group()
""")


@app.function(gpu="L40S:2", image=image, timeout=600)
def run() -> int:
    script_path = "/tmp/worker.py"
    with open(script_path, "w") as f:
        f.write(_WORKER_SCRIPT)

    cmd = [
        "/opt/venv/bin/torchrun",
        "--standalone",
        "--nproc_per_node=2",
        script_path,
    ]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    print("[corr] launching torchrun with 2 ranks")
    result = subprocess.run(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
    return int(result.returncode)


@app.local_entrypoint()
def main() -> None:
    exit_code = run.remote()
    if exit_code != 0:
        raise SystemExit(exit_code)
