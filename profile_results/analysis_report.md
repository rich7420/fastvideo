# GPU Profile Analysis Report — perf.nsys-rep

**Workload:** FastVideo distributed video diffusion inference
**Profile:** 777 s wall, 4 GPUs (0-3), 2.2 M kernels, 685 K memcpys, 16 M NVTX events
**Analysis window:** 400–420 s (steady-state middle of generation-2 denoising, post-warmup)

---

## 1. Workload Shape

| Stage                    | Count | Total ms (sum across 4 ranks) | Per-stage wall |
| ------------------------ | ----- | ----------------------------- | -------------- |
| stage::TextEncodingStage | 12    | 45,364                        | ~4 s           |
| stage::DenoisingStage    | 12    | 2,439,349                     | ~206 s         |
| stage::DecodingStage     | 12    | 231,076                       | ~58 s          |

12 instances = 4 ranks × 3 video generations. Each pipeline: TextEnc → Denoise (dominant) → Decode → ~20 s gap. No `_bwd` / `wgrad` / `dgrad` kernels → confirmed inference workload.

## 2. Per-GPU Activity (20 s window, GPU 0)

```
Total wall                       : 20,000 ms
Kernel busy time                 : 19,072 ms  (95.4%)
Pure idle                        :    928 ms   (4.6%)
Sync-stall density (STREAM_SYNCHRONIZE)
                                 : 91.56 %  ← see §3
NCCL kernels                     :    363  ( 8,766 ms = 43.0%)
flash_fwd kernels                :    357  ( 5,418 ms = 27.1%)
GEMM/elementwise/etc.            : 16,087  ( 4,888 ms = 24.3%)
```

Per-GPU symmetry across ranks 0–3:

| GPU | kernels | busy      | bubble |
| --- | ------- | --------- | ------ |
| 0   | 16,807  | 19,072 ms | 3.77 % |
| 1   | 16,807  | 19,078 ms | 3.71 % |
| 2   | 16,807  | 19,168 ms | 3.31 % |
| 3   | 16,807  | 19,145 ms | 3.41 % |

All four ranks balanced — no straggler, no rank-asymmetric idle.

## 3. Root Cause — Single-stream NCCL/Compute Serialization

| Evidence                           | Value                                  | Source                     |
| ---------------------------------- | -------------------------------------- | -------------------------- |
| overlap_pct (compute vs NCCL)      | 0.0 %                                  | overlap_breakdown          |
| same_stream_diagnosis              | ["7"]                                  | overlap_breakdown          |
| compute ⊥ nccl_sendrecv overlap    | 0 ms                                   | kernel_overlap_matrix      |
| All 16,807 GPU-0 kernels on stream | 7 (single stream)                      | sqlite per-stream group-by |
| sync_density_pct                   | 91.56 % (18,249 ms STREAM_SYNCHRONIZE) | sync_cost_analysis         |
| SendRecv : flash_fwd call ratio    | 357 : 357 (1-to-1 per attention layer) | top_kernels + sqlite       |
| SendRecv kernel avg duration       | 23.9 ms (min 11.1, max 58.4)           | nccl_breakdown             |

**Mechanism:** Ulysses sequence-parallel attention requires an all-to-all (manifested here as NCCL SendRecv) before and after every attention layer to redistribute Q/K/V heads across the SP group. FastVideo issues these collectives on the same CUDA stream as `flash_fwd` (stream 7). Because a CUDA stream is in-order, compute kernels physically cannot start until the preceding NCCL kernel drains, and vice versa. The `STREAM_SYNCHRONIZE` density of 91.56 % is the host-side launch queue stalling on stream-7 drain points — not user-issued `torch.cuda.synchronize()`. With NCCL = 43 % and flash = 27 % of wall, maximum theoretical overlap = min(8.6 s, 5.4 s) = **5.4 s per 20 s window**, fully lost today.

**Secondary findings** (not the bottleneck — log only):

- `gpu_idle_gaps` shows 3-4 gaps of 110–244 ms in the window, all attributed to `cudaStreamSynchronize_v3020` after `reduce_kernel`. NVTX shows 4 instances of `aten::is_nonzero` + `aten::item` totaling ~3.6 s — likely a periodic NaN check or stopping-criterion `.item()` inside the denoising loop. Total cost = ~550 ms / 20 s ≈ 2.7 %. Marginal vs the structural comm issue.
- `aten::pin_memory` appears for 29 s total (NVTX). Spread across all 3 video generations during text-encoding ingest. Worth fixing later but not the dominant cost.

## 4. Compute Health (positive findings)

| Check                      | Result                                                                                          |
| -------------------------- | ----------------------------------------------------------------------------------------------- |
| flash_fwd dtype            | bf16 (Flash-Attention-2 with cutlass::bfloat16_t)                                               |
| GEMM                       | cutlass_80_tensorop_bf16_s16816gemm — Ampere tensor cores                                       |
| tc_active for flash_fwd    | true (from top_kernels)                                                                         |
| tc_active for top GEMM     | true                                                                                            |
| Memcpy on dedicated stream | Stream 21 carries 15.8 GB of D2D (NCCL inner buffers) — properly isolated from compute stream 7 |
| H2D bandwidth (window)     | ~9 GB/s per rank — not the bottleneck                                                           |

Compute kernels themselves are well-tuned (correct dtype, tensor cores active). The bottleneck is **placement, not per-kernel efficiency**.

## 5. Implementation Diagnosis — Why Everything Lands on Stream 7

The serialization is rooted in three pieces of code. All four ranks hit it identically, which is why the profile is symmetric.

### 5.1 Dataflow per attention layer (`fastvideo/attention/layer.py:87-142`)

`DistributedAttention.forward` issues every operation onto whatever CUDA stream the caller is on (the default compute stream — `7` in the trace):

```python
qkv = torch.cat([q, k, v], dim=0)                     # alloc + cat — stream 7
qkv = sequence_model_parallel_all_to_all_4D(qkv,      # NCCL #1 — stream 7
                                            scatter_dim=2, gather_dim=1)
qkv = qkv[:, :original_seq_len, :, :]                 # slice — stream 7
qkv = _apply_rotary_emb(qkv, cos, sin, ...)           # rope — stream 7
qkv = self.attn_impl.preprocess_qkv(qkv, ...)         # preprocess — stream 7
q, k, v = qkv.chunk(3, dim=0)
output = self.attn_impl.forward(q, k, v, ...)         # flash_fwd — stream 7
output = self.attn_impl.postprocess_output(output, ...)
output = F.pad(output, (0, 0, 0, 0, 0, pad_seq_len))
output = sequence_model_parallel_all_to_all_4D(output, # NCCL #2 — stream 7
                                               scatter_dim=1, gather_dim=2)
return output
```

Every layer is a strict three-segment sequence: **NCCL → flash_fwd → NCCL**, all in-order on a single stream. With 30 transformer blocks × 30 inference steps, this serialization is repeated 900 times per generation, exactly matching the 357 SendRecv : 357 flash_fwd 1:1 ratio observed in the trace.

### 5.2 Root cause — `dist.all_to_all_single` inherits the caller's stream

The Ulysses SP all-to-all is implemented in `fastvideo/distributed/device_communicators/base_device_communicator.py:155, 175`:

```python
# forward case scatter_dim=2, gather_dim=1
input_ = input_.transpose(0, 2).contiguous()
output = torch.empty_like(input_)
dist.all_to_all_single(output, input_, group=group)   # ← inherits current_stream
output = torch.cat(output.split(shard_hn), dim=1)
output = output.transpose(0, 2).contiguous()
return output
```

`torch.distributed.all_to_all_single` with the NCCL backend launches its kernels on `torch.cuda.current_stream()`. The caller has not switched streams, so the NCCL kernel is enqueued onto stream 7 directly behind the preceding `transpose+contiguous`, and the following `cat+transpose+contiguous` enqueue immediately after — every op gets serialized on the same stream by construction.

Note: nsys reports `ncclSendRecv` rather than `ncclAllToAll` because `all_to_all_single` with non-uniform peer chunks lowers to a fan of `ncclSend`/`ncclRecv` pairs. This is the expected NCCL kernel name for this collective, not a separate problem.

### 5.3 Call-site pattern provides no opportunity for overlap

Even with NCCL on its own stream, the in-layer flow `pre-all-to-all → flash → post-all-to-all` has no compute work in parallel with either collective — `flash` depends on the pre-all-to-all output, and the next operation (residual add / norm of the next block) depends on the post-all-to-all output. Useful overlap only appears at the **block boundary**: post-all-to-all of block N can overlap with the residual / norm / QKV projection of block N+1 on the compute stream.

### 5.4 Why this is a program issue, not a benchmark issue

`sp_size=4` is the realistic configuration for multi-GPU inference; running with `sp_size=1` would mask the issue but defeats the purpose of multi-GPU deployment. The benchmark numbers reflect what real users would experience. The fix belongs in the framework, not the benchmark config.

## 6. Fix Plan

| Tier | Change | File(s) | Expected NCCL-time recovered |
|------|--------|---------|------------------------------|
| **A** | Dedicate a per-group NCCL stream; issue `all_to_all_single` on it; use `record_stream` + cross-stream events for correctness | `base_device_communicator.py`, `parallel_state.py` | ~30–50 % (recovers cross-block overlap) |
| **B** | Refactor `DistributedAttention.forward` to expose async pre/post handles so the next block can issue its `pre-all-to-all` before consuming the current block's `post-all-to-all` | `attention/layer.py` | additional ~20–30 % on top of A |
| **C** | Reuse intermediate transpose / output buffers across layers | `base_device_communicator.py` | marginal |

This report's Fix section implements **Tier A** (the foundation) plus the safe portion of Tier B (no semantic change to attention call sites). Tier B's full async restructure is deferred — it requires changes inside every model's transformer block and is best done in a follow-up after Tier A is validated.

## 7. Update — Tier A Attempt and What We Learned

### 7.1 What happened

We implemented Tier A as described: a dedicated comm CUDA stream per ProcessGroup, `record_stream` bookkeeping, and `wait_stream` handshakes around `dist.all_to_all_single`. We verified bitwise equivalence vs the original single-stream impl with a 2-rank Modal test. **The change was numerically correct.**

We then re-ran the 4×L40S benchmark. The result was a **regression**, not a speedup:

| Metric | Baseline | After Tier A |
|---|---|---|
| Per-stage DenoisingStage (warmup) | 201.8 s | 299.6 s (+48 %) |
| Container stability | clean | crashed on Measurement 2/2 |

We reverted the change. The reverted state is what is currently on the `inference-profile` branch.

### 7.2 Why Tier A was wrong (the part the original report got wrong)

After reverting, we read `torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp` on the PyTorch main branch. The relevant facts:

- `AllToAllOptions::asyncOp` defaults to `true` (Types.hpp). With `asyncOp=true`, PyTorch already pulls a stream from `getStreamFromPool(high_priority)` (line ~3258) and caches it per-PG in `ncclStreams_`.
- The internal handshake is already implemented (lines ~200-206, ~3812-3863): record event on caller's `current_stream` → NCCL's per-PG stream `block`s on that event → kernel → record end event → `work.wait()` blocks caller on the end event.
- `recordStream` is already called internally via `stashed_for_allocator_safety_` (`TensorShelf`) at line ~3853, released on `wait()`.

In other words: **PyTorch already does the stream split + event sync + allocator bookkeeping for us.** Our wrapper layered a second handshake on top of PyTorch's first one, with:
- redundant `cudaEventRecord` + `cudaStreamWaitEvent` pairs (each ~1–3 µs of host overhead);
- duplicate `record_stream` calls (extra allocator hash-map updates);
- and most importantly, no new overlap opportunity, because of §7.3.

vLLM's `vllm/distributed/device_communicators/pynccl.py` (lines ~268-290) confirms the consensus pattern: drive NCCL on the caller's current stream and rely on PyTorch's internal sync. **No manual stream wrapping.**

### 7.3 The actual hard constraint — data dependency

Tracing every dependency in `WanTransformerBlock.forward` (fastvideo/models/dits/wanvideo.py:314-394) and `DistributedAttention.forward` (fastvideo/attention/layer.py:87-142):

```
hidden_states_in
  → norm1 → q/k/v_proj
    → cat(q,k,v)
    → pre-a2a (NCCL #1)           ← consumer: flash needs full-seq + sharded-heads QKV
    → slice + rope + preprocess
    → flash                        ← consumer: post-a2a needs flash output
    → postprocess + pad
    → post-a2a (NCCL #2)           ← consumer: to_out needs full-heads output
  → to_out
  → self_attn_residual_norm        ← consumer: cross-attn norm needs this
  → cross_attn (LOCAL, no a2a — text-encoder KV is replicated; flash on sharded Q)
  → cross_attn_residual_norm
  → ffn (MLP, no a2a)
  → mlp_residual
hidden_states_out
```

Every arrow above is a true data dependency: the consumer reads the producer's output buffer. **Within a block there is no compute that is independent of the all-to-all output and that could be scheduled in parallel with it.** Across blocks, the residual stream `hidden_states_in → hidden_states_out` enforces sequential ordering between blocks.

This is the real root cause. The original report's framing — "single stream serialization, fixable by stream split" — was wrong. The correct framing is:

> **The Ulysses SP all-to-all sits on the critical path of every self-attention, and within a 1-batch inference step there is no independent compute window to overlap it with.** Switching streams cannot create work that does not exist.

### 7.4 Theoretical MFU ceiling under current parallelism

From the trace's 20-second steady-state window:

- compute (flash + GEMM + element-wise + VAE) = 27.1 % + 24.3 % ≈ 51.4 % of wall
- NCCL = 43.0 % of wall
- idle = ~5.6 %

If NCCL is strictly serial with compute (which it is, per §7.3), then MFU is bounded by:

```
MFU_ceiling = compute_time / (compute_time + comm_time)
            ≈ 51.4 / (51.4 + 43.0)
            ≈ 54.4 %
```

Cross-attention is local (no a2a), so this number is generous — only self-attention is comm-bound. A tighter bound, considering only the attention-only fraction, suggests the practical inference MFU under Ulysses SP=4 with no overlap is in the **35–45 %** range, regardless of how we tune the comm primitive.

## 8. Forward Plan (Structural Options)

Given §7.3, real speedups come from one of:
- **(P1) Change the parallelism scheme** so the collective is not strictly on the critical path, or so the collective is structurally smaller.
- **(P2) Change the attention algorithm** to one whose collective overlaps with attention compute by construction.
- **(P3) Lower-tier wins** that do not touch SP at all (VAE, host syncs, pin_memory). Ceiling around 15 % E2E, cumulative.

Below: concrete options, mapped to the FastVideo code.

### Option R — Replace Ulysses SP with Ring Attention

The canonical answer to "SP attention with overlap" is Ring Attention (Liu et al. 2023): each rank holds a chunk of K/V; for each Q chunk, K/V is rotated around the ring (point-to-point) while attention is computed locally on whichever K/V chunk currently sits on the rank. Comm and compute overlap by construction inside the attention kernel.

- **Where:** introduce a new attention backend under `fastvideo/attention/backends/ring_attn.py`; modify `DistributedAttention.forward` (`fastvideo/attention/layer.py`) to dispatch to it when configured.
- **External dep:** an existing ring-attention kernel — candidates: `flash-attn`'s ring-attention variant, the `ring-flash-attention` package, or `xformers`'s implementation.
- **Correctness verification:** (a) bitwise on a tiny synthetic 2-rank test (likely will NOT be bitwise — different reduction order); (b) numerical tolerance test (`atol=1e-3, rtol=1e-3` for bf16); (c) SSIM regression on a fixed seed against the baseline mp4 output, accept if SSIM > 0.99.
- **Risk:** **High.** Different reduction order means non-bitwise outputs; masking edge cases (cross-attn-free, causal-free in our case but still); needs the ring kernel to support our exact (head_dim, num_heads, dtype) combo on L40S.
- **Estimated gain:** literature reports 50-80 % of comm time hidden. For our 43 % NCCL share, that's ~20-35 % E2E speedup. Likely **150-180 s per generation** vs current 226 s.
- **Estimated effort:** ~3-5 days including bring-up + SSIM gates.

### Option C — Chunked Ulysses (pipeline within attention)

Keep Ulysses but chunk along the SP-shard dim: split the seq into N=2 or 4 chunks, pipeline so chunk_{k+1}'s pre-a2a runs while chunk_k's flash runs. This is a smaller change than Option R but still touches the attention kernel boundary.

- **Where:** `DistributedAttention.forward` only. Outer API unchanged. Internal: chunk q/k/v along seq, issue async pre-a2a per chunk, flash per chunk, async post-a2a per chunk.
- **Correctness:** bitwise should hold (we're not changing reductions, just enqueue order). Verify with our existing 2-rank test extended.
- **Risk:** **Medium.** Async correctness across chunks (waits, allocator pressure). Needs to maintain the same final tensor shapes.
- **Estimated gain:** at N=2, hide ~30-40 % of one collective per chunk. Roughly **5-15 % E2E**.
- **Estimated effort:** ~2-3 days.

### Option T — Switch SP → TP (tensor parallel)

Tensor parallelism shards along the head/hidden dim (not seq), so the collective is per-MLP-allreduce instead of per-attention-a2a. Megatron-LM's "async-TP" pattern overlaps the allreduce with the matmul that produces it.

- **Where:** large change. Every Linear becomes column- or row-parallel; QKV proj is column-parallel; output proj is row-parallel; MLP up is column-parallel, MLP down is row-parallel. Requires a TP-aware module port.
- **Correctness:** numerical tolerance (different reduction order). SSIM regression.
- **Risk:** **Very high.** Touches every Linear in the model. Easy to get wrong; hard to roll back.
- **Estimated gain:** comm volume similar magnitude but with overlap-friendly placement. **10-20 % E2E** if Megatron async-TP applies cleanly.
- **Estimated effort:** ~1-2 weeks.

### Option L — Lower-tier wins (do NOT touch SP)

Independent of the SP question:

| Item | Current cost | Mechanism | Estimated gain |
|---|---|---|---|
| `.item()` syncs in scheduler step | ~2.7 % wall | rewrite scheduler index lookup to tensor-only or gate on log cadence | 1-2 % E2E |
| `pin_memory` during text-encode ingest | ~3.7 % wall (one-time per gen) | pre-pinned buffer pool, `non_blocking=True` H2D | 1-2 % E2E |
| VAE decode | 8.5 % E2E | re-profile DecodingStage; `vae_sp` / `vae_tiling` tile size sweep | 2-4 % E2E |
| **Total** | — | — | **5-10 % E2E, no structural risk** |

### Recommended sequence

1. **Option L first** (low risk, no SP touch). Take the ~5-10 % E2E. **No structural change yet.**
2. **Option C** (Chunked Ulysses). Most conservative structural change. ~5-15 % additional E2E. Bitwise verifiable.
3. **Option R** (Ring Attention). Largest single win if it lands. Numerical-tolerance verifiable (SSIM gate).
4. **Option T** (TP) only if R is blocked or insufficient.

The original report's call to do "Tier B (async pre-issue across blocks)" should be retired: §7.3 shows there is no independent compute window between blocks to issue into. Any structural change must come from the Option R / C / T family.

### Correctness verification framework (applies to all structural options)

We will not merge any structural change without all four of:
1. **Unit test:** an `all_to_all_*` round-trip on 2 ranks. Bitwise where possible, atol/rtol otherwise.
2. **Numerical block test:** a single `WanTransformerBlock.forward` invocation, fixed seed, 2-rank vs 4-rank, atol=1e-3 / rtol=1e-3 in bf16.
3. **SSIM regression:** fixed-seed video generation, compare mp4 frame-by-frame against the baseline mp4 (the one currently in `profile_results/`). SSIM > 0.99 per frame.
4. **Perf delta:** the new `perf_*.json` must show `avg_generation_time_s` strictly lower than 226.1 s, AND a fresh nsys trace must show the collective wall-time share dropping correspondingly.

A change that passes 1-3 but fails 4 is reverted (we got the Tier A revert from this exact failure mode). A change that passes 4 but fails 1-3 is reverted as a correctness regression.

## 9. Implementation Plan — Ring Attention

Decision: implement Option R (Ring Attention) per §8. This section is the working plan; it gates implementation.

### 9.1 Why Ring Attention is the right pick (recap)

Per-block timing in steady state (corrected from §3 with the asymmetric AtoA finding):

```
AtoA_pre (3× qkv payload) : 34 ms
flash_fwd                 : 31 ms
AtoA_post (1× output)     : 12 ms
to_out + cross + MLP + next-block QKV proj : 28 ms
─────────────────────────────────────────────
per-block wall             : 105 ms
```

`compute` (excluding NCCL) ≈ 59 ms / block; `comm` ≈ 46 ms / block.

Ring attention **changes the algorithm**, not the comm primitive: instead of doing two large AtoAs around a monolithic flash, it rotates K/V chunks around the SP ring while computing partial attention. Comm of round `r+1` overlaps with compute of round `r` *inside the attention kernel itself*. Per-attention timing target:

| Stage | Ulysses (measured) | Ring (idealized, perfect overlap) |
|---|---|---|
| pre-AtoA | 34 ms | 0 (removed) |
| flash (monolithic) | 31 ms | – |
| flash (4 chunks, sp=4) | – | 4 × 8 = 32 ms total compute |
| K/V rotations × 3, overlapped with compute | – | `max(8, 12) × 3 ≈ 36 ms` wall |
| post-AtoA | 12 ms | 0 (removed) |
| online-softmax merge epilog | 0 | ~2 ms |
| **attention total wall** | **77 ms** | **~50 ms theoretical** |

Per-attention savings (theoretical): 27 ms. Practical ring impls land 60–80 % of theory → ~16–22 ms saved per attention. Over 30 blocks × 30 steps × 3 generations:

- **Optimistic (80 % of theory):** 22 ms × 30 × 30 × 3 / 4 ranks = ~15 s saved per gen → E2E **~210 s** (≈ 7 % faster than 226 s).
- **Stretch (theory ceiling):** ~27 s saved per gen → E2E **~199 s** (≈ 12 % faster).

These numbers are lower than the original §8 estimate of 20–35 %. After the self-review, **the honest target is 8–15 % E2E**, not 20 %. The 20 % case only triggers if rotation transfer is genuinely overlapping with sizeable compute *and* the per-rotation epilog is cheap on ring-flash-attn's L40S codepath — neither of which is certain a priori.

MFU effect: same FLOPs in 5–12 % less wall → MFU bumps from 38.7 % to **~42–45 %**. Not the ~53 % I originally claimed.

### 9.2 Sharding scheme change (the part that makes Ring NOT a drop-in)

| Stage | Ulysses (current) | Ring |
|---|---|---|
| QKV proj input | seq sharded per rank | seq sharded per rank |
| **Pre-attention reshape** | `all_to_all(scatter=heads, gather=seq)` → full-seq, sharded-heads | **none** — keep seq sharded |
| Attention compute | local flash on full-seq, sharded-heads | ring rotation of K/V chunks; per rank computes Q × K_chunk for each rotation, accumulates via online softmax |
| **Post-attention reshape** | `all_to_all(scatter=seq, gather=heads)` → seq sharded, full heads | **none** — output already seq sharded |
| `to_out` input | seq sharded, full heads | seq sharded, full heads ✓ same |

The Ring path **deletes** both AtoAs and replaces them with `(sp_size − 1) × 2` small K/V SendRecv pairs interleaved inside the attention compute. Total comm volume is similar; the win is from comm/compute interleaving.

### 9.3 Library choice

| Lib | Stars | License | Last update | Notes |
|---|---|---|---|---|
| `ring-flash-attention` (zhuzilin) | 1018 | MIT | 2025-09 | Pure Python wrapper around `flash_attn`. Three variants: `ring`, `stripe`, `zigzag`. Simplest. |
| `yunchang` (USP) | 667 | Apache-2.0 | 2026-01 | Drop-in for Ulysses+Ring+Hybrid. Higher ceiling but heavier. |

**Pick `ring-flash-attention` for first attempt.** Reasons:
- Same `flash_attn` backend already in the fastvideo-dev image — no kernel compile changes.
- Smaller surface area to debug.
- Falls back trivially to Ulysses if we hit a blocker.

If first attempt yields < 10 % E2E gain, escalate to `yunchang` for the hybrid 2D ring×Ulysses pattern.

### 9.4 Integration design

Two options:

**Design A — new backend that owns the comm.** Add `fastvideo/attention/backends/ring_attn.py`. Modify `DistributedAttention.forward` (`fastvideo/attention/layer.py:87-142`) to detect when the backend is ring and **skip** the two `sequence_model_parallel_all_to_all_4D` calls. Pass the SP group into the backend.

**Design B — parallel `RingDistributedAttention` class.** Add a sibling class in `fastvideo/attention/layer.py`. Switched at `WanTransformerBlock.__init__` based on a config flag.

**Pick Design A.** Reasons:
- Existing backend registry (`AttentionBackendEnum`) already supports per-model backend choice — natural extension point.
- Keeps the `DistributedAttention` outer class as the single attention API for the rest of the codebase.
- The `preprocess_qkv` / `postprocess_output` hooks in the backend interface (abstract.py:124-163) are exactly the right contract — we just add a "no-comm" indicator and have `DistributedAttention.forward` consult it.

**Backend marker naming.** Instead of `requires_no_outer_a2a` (leaky abstraction; assumes outer parallelism is always AtoA), use **`handles_sequence_parallel_internally: bool = False`** on the `AttentionImpl` class. The `DistributedAttention` dispatch reads this flag.

**Pre-AtoA work that the ring path must take over.** When ring backend is active, `DistributedAttention.forward` skips both `sequence_model_parallel_all_to_all_4D` calls and these subsidiary operations must move into or wrap around the ring call:

1. **RoPE application.** Current code applies `_apply_rotary_emb` with full-sequence `cos/sin` tables AFTER the pre-AtoA materializes the full sequence. In ring mode each rank only sees its own seq slice — RoPE must be applied with `cos[local_start:local_end], sin[same]`. Either compute the slice in the dispatch wrapper before calling ring, or push the full table + a seq offset into the backend and let it slice. *Concrete*: in step R-4 add `cos_local, sin_local = cos[local_start:local_end], sin[...]` and pass.
2. **SP padding (`compute_padding_for_sp`).** Today, padding is applied at `sequence_model_parallel_shard` (upstream of the block) and the AtoA path slices `qkv[:, :original_seq_len]` AFTER the AtoA. In ring mode, the per-rank slice is already padded to `(seq_padded/sp)` and the slicing must happen AFTER ring attention output (still on the per-rank slice) — or we keep the padding through ring and only strip it in `sequence_model_parallel_all_gather_with_unpad` at the model output. *Concrete*: in step R-4 do **not** strip padding before ring; ring sees the padded chunks; verify the padded tokens contribute zero to attention output (they should, given how padding is filled).
3. **`replicated_qkv` concat path** (`layer.py:114-121`). This branch slices replicated tokens by heads (assumes sharded heads) and concats along seq. In ring mode each rank has full heads + sharded seq, so the slicing direction reverses. **Audit first**: grep for which callers pass `replicated_q != None`. If Wan T2V never uses this path (likely — it's mainly for I2V image tokens), the simplest fix is to **assert it's None** in the ring dispatch branch and defer ring support for replicated-token attention to a follow-up.

### 9.5 Config surface

Two ways to enable:

1. **Env var** (least intrusive for benchmarking):
   `FASTVIDEO_ATTENTION_BACKEND=RING_FLASH_ATTN` — backend resolver in `fastvideo/attention/__init__.py` already supports backend selection via env. This is what the perf script can flip.
2. **Pipeline config field** (proper API): add `attention_parallel: Literal["ulysses", "ring"] = "ulysses"` to `pipeline_config` later, after env-var validates the change.

Start with env var; promote to config field once benchmarked.

### 9.6 Correctness gates (must all pass before merging)

| Gate | Check | Tool |
|------|-------|------|
| G1 | `ring-flash-attn` standalone gives correct attention output vs `torch.nn.functional.scaled_dot_product_attention` on a single GPU, with `atol=1e-3, rtol=1e-3` in bf16 | new pytest under `fastvideo/tests/distributed/` |
| G2 | Ring path output matches Ulysses path output on 2 ranks, same seed, same Q/K/V, atol/rtol 1e-3 | extend `fastvideo/tests/modal/test_all_to_all_correctness.py` |
| G3 | `WanTransformerBlock.forward` single-block fixed-seed, Ulysses vs Ring, atol=5e-3 / rtol=5e-3 (bf16 with many ops looses precision a bit) | new Modal test |
| G4 | Full generation SSIM ≥ 0.99 per frame vs baseline mp4 (`profile_results/generated_videos/.../*.mp4`) | new Modal SSIM script |
| G5 | `perf_*.json::avg_generation_time_s` strictly lower than 226.1 s | reuse `perf_nsys_profile.py` |
| G6 | New `perf.nsys-rep` shows `overlap_pct > 0` and total NCCL wall share lower than baseline 43 % | nsys-ai `overlap_breakdown` + `kernel_overlap_matrix` |

Gates G1, G2, G3 must pass before running G5 (don't waste Modal credits on broken outputs). G4 and G6 are run together after G5 succeeds.

### 9.7 Implementation steps (ordered, each ends in a verifiable state)

1. **Install probe** — confirm `ring-flash-attn` installs cleanly into the fastvideo-dev Modal image (no compile, no torch version mismatch). Add to `pyproject.toml`'s optional `[ring]` extra OR pip-install inside the Modal entry script. *No fastvideo code change yet.* Verification: a Modal job that imports it and prints version.
2. **Standalone correctness (G1)** — Modal 1-GPU test: feed the same Q/K/V to ring_flash_attn and SDPA, assert close. *No fastvideo code change yet.*
3. **Backend skeleton** — add `fastvideo/attention/backends/ring_attn.py` that implements `AttentionImpl` and returns `AttentionBackend` with a `requires_no_outer_a2a: bool = True` marker. Register in `AttentionBackendEnum` and the resolver in `fastvideo/attention/__init__.py`. *No call-site change yet, won't be picked up by default.*
4. **DistributedAttention dispatch** — in `fastvideo/attention/layer.py::DistributedAttention.forward`, branch on `self.attn_impl.requires_no_outer_a2a` (default False). When True: skip the two AtoAs, call `attn_impl.forward(q, k, v, ...)` directly with sharded inputs, return its output as-is. Default path unchanged.
5. **2-rank distributed correctness (G2)** — extend `test_all_to_all_correctness.py` to compare Ulysses path vs ring path on 2 ranks. Bitwise will fail (different reduction order); use `torch.testing.assert_close(rtol=1e-3, atol=1e-3)`.
6. **Block-level correctness (G3)** — new Modal test: instantiate `WanTransformerBlock`, run forward twice (ulysses backend, ring backend), check outputs close.
7. **Run perf+nsys with ring backend (G5+G6)** — set `FASTVIDEO_ATTENTION_BACKEND=RING_FLASH_ATTN` in the existing `perf_nsys_profile.py`. Compare new perf JSON and nsys trace vs baseline.
8. **SSIM regression (G4)** — new Modal job: generate video with same seed using ring backend, compare to baseline mp4 frame-by-frame. SSIM > 0.99.
9. **Decide** — if G5 passes (avg_time < 226.1 s) AND G4 passes, commit. Otherwise revert and document.

### 9.8 Risks and rollback plan

| Risk | Mitigation |
|------|------------|
| `ring-flash-attn` doesn't support head_dim=128 / num_heads=12 / bf16 / non-causal on L40S Ampere | G1 catches this in <5 min on Modal. If it fails, switch to `yunchang` or abort. |
| Output numerically diverges enough to break SSIM (different reduction order in online softmax) | G3 catches at block level before full E2E. Tolerance is bf16-realistic. |
| Ring overhead per rotation is high → no perf win on small SP world (sp=4) | G5 catches. If avg_time not better, revert and try `yunchang` 2D variant. |
| Container OOMs because ring intermediate K/V chunks aren't freed | nsys peak memory + `record_stream` audit. Headroom is 27 GB so unlikely. |
| RoPE wrong on sharded seq (correctness, 🔴) | R-4 explicitly slices cos/sin per rank; G3 block-level test would fail loudly |
| `replicated_q` path silently mis-shaped (correctness, 🔴) | R-4 dispatch branch asserts `replicated_q is None` for ring; audit Wan callers in step R-3 |
| Other models (LTX-2 inline AtoA, Hunyuan) silently regress | These don't route through DistributedAttention.forward, so they keep Ulysses regardless. **Only Wan benefits from this change.** Document this. |
| PCIe-only L40S (no NVLink) makes K/V rotation as slow as the AtoA it replaced | G5 reveals; rollback. Consider larger sp_chunk_size on yunchang in that case. |

**Rollback** is a single env-var change (drop `FASTVIDEO_ATTENTION_BACKEND`) or a one-line revert of step 4's dispatch branch.

## 10. Second Self-Review — Ring Attention Cancelled

After writing the §9 plan, we did a deeper literature and hardware reality check. The findings overturned the plan.

### 10.1 Why Ring is wrong for *this* hardware

| Evidence | Source | Implication |
|---|---|---|
| `ring-flash-attention` README: *"NVLink between GPUs are required for high performance."* | [zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention) | L40S has **no NVLink**, only PCIe Gen4. Library's own author flags our config as out-of-spec. |
| Measured L40S P2P bandwidth: ~10–13 GB/s post-IOMMU-passthrough fix; some configs hang NCCL ring tests outright | [NVIDIA dev forum](https://forums.developer.nvidia.com/t/nccl-hangs-on-l40s-gpus-pcie-resolved-via-iommu-passthrough/368169), [joshlk gist](https://gist.github.com/joshlk/bbb1aca6e70b11d251886baee6423dcb) | Each ring rotation needs ~192 MB transferred → ~15–19 ms one-way per rotation. Per-block flash is 31 ms split into 4 chunks of ~8 ms. **Comm-critical-path per block ≈ 45–57 ms vs current Ulysses AtoA total 46 ms.** Same order, possibly worse. |
| USP paper Table 3 on 8×L20 PCIe (closest published analog): pure Ulysses dominates short seq; Hybrid 2D only beats pure Ring by **7.5 %** at 32K seq; "*Ulysses dominates high-bandwidth networks*"; "*Ring-Attention can lead to a decrease in computation efficiency... Even if communication and computation fully overlap, the total execution time lags behind.*" | [arXiv:2405.07719v5](https://arxiv.org/html/2405.07719v5) | At our SP=4 with 31200 seq, expected ring win over Ulysses is < 10 %, possibly negative. |
| xDiT (the most polished DiT-parallel framework) Flux 1024px on **8×L40 PCIe** actually **slows down** vs 4 GPUs due to cross-socket QPI traffic; xDiT publishes no Wan2.1 numbers | [xDiT flux.md](https://github.com/xdit-project/xDiT/blob/main/docs/performance/flux.md), [xDiT README](https://github.com/xdit-project/xDiT) | High-resolution DiT on PCIe-only multi-GPU is *known* to scale poorly; this is exactly our 720×1280 regime. |
| L40S compute-to-comm ratio: 362 BF16 TFLOPS ÷ ~13 GB/s ≈ 28000:1 | derived | Original Ring paper assumption (comm BW ≥ compute BW) is off by 4 orders of magnitude. Ring's overlap premise doesn't apply. |

**Conclusion:** Ring Attention's expected win on our hardware is in the noise (< 5 %), and there's a real chance of regression because of online-softmax overhead per rotation. The library's own author tells us this is out-of-spec for PCIe-only GPUs.

### 10.2 What we should have done — DMD is the real lever

The single largest available speedup is **already published** as part of FastVideo upstream:

- **FastWan2.1-T2V-1.3B-Diffusers** (DMD-distilled): 3-step inference vs our 30 steps
- **60× denoising speedup at 480p, 90× at 720p** (the denoising stage is 89 % of our E2E, so this is roughly an order-of-magnitude E2E gain)
- Sources: [FastVideo DMD docs](https://haoailab.com/FastVideo/distillation/dmd/), [FastVideo post-training blog](https://haoailab.com/blogs/fastvideo_post_training/)
- Caveat: published without SSIM/FVD/VBench numbers; quality regression needs to be measured by us

DMD changes the model (different weights, different scheduler), so it is *categorically different* from the infra optimizations we've been planning (Ring, comm-stream, etc.) — those preserve the exact model output, DMD does not. The question is whether the project's goal is "make the same model faster" (infra optimizations) or "make video generation faster" (algorithmic optimizations).

### 10.3 Revised priority order (ROI-ranked)

| # | Option | Mechanism | Expected E2E gain | Risk | Touches model output? |
|---|--------|-----------|---|------|----|
| 1 | **DMD-distilled checkpoint (FastWan2.1)** | 3 steps vs 30 steps | **~10–30× on DenoisingStage** (~75–80 % E2E) | Medium (quality regression unknown without our own SSIM/FVD eval) | **Yes** — different weights |
| 2 | **sp_size = 2 instead of 4** | Halves AtoA fan (single SendRecv vs 4-way) | ~5–15 % E2E | Low (Ulysses-2 is well-tested) | No |
| 3 | **FP8 (E4M3) for linears** via Transformer Engine on L40S Ada FP8 cores | ~1.45× on linear-bound stages | ~5–10 % E2E | Medium (TE integration) | Minor (quantization noise) |
| 4 | **xDiT Ulysses-2 × Ring-2 hybrid** | 7-8 % over pure Ring per USP paper at SP=4 PCIe | ~5–8 % E2E | High (new framework) | No |
| 5 | Pure Ring Attention SP=4 (the original §9 plan) | Comm/compute overlap inside attention | 0–8 % E2E, possible regression | High | No |
| 6 | Async-TP (Megatron-style) | Replace SP with TP, allreduce in MLP | unknown for video DiT | Very high | No |

DMD wins by an order of magnitude. Among the "preserve model output" options, **sp=2 is the lowest-risk concrete win**. Ring drops to #5.

### 10.4 Plan adjustment

§9 (Ring Attention implementation plan) is **cancelled**. The eight R-* tasks are deleted. The Ring section stays in this report as a documented dead-end so we don't re-investigate it.

Recommended next move:

**(a) Validate DMD output quality with SSIM/FVD against the baseline mp4.** If quality is acceptable, swap the perf benchmark to use the DMD checkpoint and re-run §3-§4 — this becomes the new baseline.

**(b) In parallel, run a 5-minute sp=2 Modal job** (drop `num_gpus=4, sp_size=4` to `num_gpus=2, sp_size=2` in `wan-t2v-1.3b-l40s-hires.json`, no code change) to measure the AtoA reduction empirically. This is essentially free to test.

**(c) If both (a) and (b) land, consider FP8 for the MLP linears.** Otherwise stop here.

The infrastructure work we did (per-stage timing, NVTX, paired perf+nsys Modal job, nsys-ai analysis pipeline) all stays — it's what enabled this evaluation in the first place, and is the foundation for CI perf tracking.

### 9.9 Estimated effort

| Step | Time |
|------|------|
| 1 (install probe) | 30 min |
| 2 (G1) | 1 h |
| 3 (backend skeleton) | 2 h |
| 4 (dispatch) | 1 h |
| 5 (G2) | 1 h |
| 6 (G3) | 2 h |
| 7 (G5+G6 Modal run + analysis) | 30 min Modal + 1 h analysis |
| 8 (G4) | 1 h |
| 9 (decide / commit / cleanup) | 1 h |
| **Total** | **~10 h** focused work |

Modal cost estimate: ~6 short runs (1-GPU correctness + 2-rank tests + 4-rank perf + SSIM gen) ≈ $15-25 in credits.

