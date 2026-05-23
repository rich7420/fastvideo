# NCCL Optimization — Direction Options (forward-looking)

_Updated: 2026-05-16. Companion to `nccl-optimization-status.md`._

**Context**: Wan T2V 1.3B / 720p / 77f / SP=4 / 4×L40S PCIe / quality-preserving.

## TL;DR — sorted by realism × impact

| Direction | Wall Δ (validated or expected) | Quality | Effort | Verdict |
|---|---|---|---|---|
| **TGATE 0.7 / 0.5** | **−14.6% / −22%** (measured) | approximate; SSIM/VBench passed | low | **ship now** |
| **H100 / NVLink deploy** | **−45%** (measured H100×2 vs L40S×4) | none | none (commercial) | **business win** |
| NCCL env tuning | 0 ~ few % | none | low | document per-hw only — not general |
| AllToAll4D layout cleanup | 0–2% (noise) | none | low-medium | minor cleanup PR, don't expect a lot |
| FP8 / int8 collective compression | 5–15% | medium-high (off main line) | medium | not on "no quality loss" path |
| Sparse (VSA / BSA) | 10%+ | medium-high | medium | off quality line |
| **Pure Ring Attention** | 10–20%+ | low (exact); impl risk high | high | overkill / fragile |
| **Hybrid Ulysses-Ring (USP-style)** | likely > pure Ring in practice | low-medium | high | **best architectural direction** |

## Current state — concrete details

### Per attention layer (Wan T2V 1.3B, SP=4, sequence 31200 tokens)

Each `DistributedAttention.forward` call (`fastvideo/attention/layer.py:60`) does:

```
Input per rank:  q, k, v each [B=1, shard_seq=7800, num_heads=12, head_dim=128]
   └─ torch.cat([q,k,v], dim=0)              ── [3B=3, 7800, 12, 128]
   └─ a2a_in (scatter=2 gather=1)            ── [3B, full_seq=31200, head_shard=3, 128]
   └─ RoPE on Q,K  (post-a2a, full seq)
   └─ flash_attn(q, k, v)                    ── [B, 31200, 3, 128]
   └─ a2a_out (scatter=1 gather=2)           ── [B, shard_seq=7800, 12, 128]
Output per rank: [B, 7800, 12, 128]
```

Two AllToAll4D per layer (`base_device_communicator.py:142-197`). Wan T2V has 30 such layers; with CFG that's 60 a2a per forward, ×30 timesteps ×2 forwards = 3,600 a2a per video.

Each a2a payload p50 = 13.18 MiB (full QKV bundle). After fused NCCL with default 2 channels × 3 ring hops, every logical a2a maps to 4–7 `ncclDevKernel_SendRecv` (profile §11).

### Per-iteration time decomposition (profile §7b, iter=311)

```
Total step:       3,087 ms       (this is iter 311 of denoising; non-warmup typical ~6.4 s/step covers 2 forwards)
  NCCL ring/launch/barrier   1,617 ms  (54.7%)   ← the cost we are trying to attack
  flash attention              920 ms  (31.1%)
  GEMM + elementwise + other   550 ms  (14.2%)
```

Stream-level (profile §3, per rank over 677 s full profile):
```
compute_only:  311 s  (46%)
nccl_only  :  271 s  (40%)
idle       :   94 s  (14%)
overlap_ms :  0.02 ms                       ← compute and NCCL strictly serialized on stream 7
```

### Why "make NCCL faster" plateaued

Already ruled out:
- offload contention · flash metadata · framework per-step · CFG batching
- NCCL P2P not enabled (NCCL_DEBUG confirmed P2P + PHB/PIX active by default)
- Layout copies (Item A −0.6% — copies hide behind NCCL latency)

The remaining cost inside the AllToAll4D wrapper:
```
NCCL ring + launch + barrier overhead :  197–247 s / rank  (≈ 64–80% of wrapper total)
```

Pre-attention QKV exchange dominates because it moves Q/K/V together (3× post-attention payload). No `.contiguous()` or env-var tweak touches the dependency chain `qkv_proj → a2a_in → flash → a2a_out → out_proj` — this is what Hybrid/Ring breaks.

### Files / call sites referenced

```
fastvideo/attention/layer.py:60                DistributedAttention.forward (Ulysses)
fastvideo/attention/layer.py:100               sequence_model_parallel_all_to_all_4D scatter=2/gather=1
fastvideo/attention/layer.py:140               sequence_model_parallel_all_to_all_4D scatter=1/gather=2
fastvideo/distributed/device_communicators/base_device_communicator.py:142-197
                                               _all_to_all_4D_forward (the actual NCCL call site)
fastvideo/distributed/parallel_state.py:259    graph_capture context (defines comm stream)
fastvideo/models/dits/wanvideo.py:651          freqs_cis (RoPE) cached per spatial shape
```

## Faster-than-Ring options (already covered)

### A. TGATE — short-term winner
```
0.7:  −14.6%  conservative, quality indistinguishable
0.5:  −22.0%  sweet spot
0.3:  −31.9%  imaging_quality regresses (−5%) — not recommended
```
Already implemented (`17047d5a`), VBench / SSIM validated, env-gated default OFF. **PR-ready**.

### B. H100 / NVLink deployment — business path
4×L40S 216s → 2×H100 121s. Hardware speedup with zero code risk; only blocker is cost. Recommended if latency is the product goal.

### C. NCCL tuning — _hardware-conditional_ knob, not a general win
`NCCL_MIN/MAX_NCHANNELS=1`: −6.8% on L40S, **+19.4% on H100**. PCIe = serial trunk (1 channel optimal); NVLink = parallel lanes (multi-channel optimal). Ship only as documented per-hw deployment knob, never as default.

## Why _pure_ Ring isn't necessarily best

Ring Attention's core (Ring Attention paper, Liu et al. 2023) is blockwise exact attention with overlapped K/V transfer — not a sparse approximation. This **does** address Ulysses bulk a2a, but pure Ring has costs:
1. Blockwise attention kernel must be rebuilt or repurposed
2. Loses FastVideo's mature FlashAttention integration
3. RoPE / causal / TGATE / CFG paths all need re-handling
4. At small SP counts (4 ranks), the design overhead may dominate the gain

## What's likely better: Hybrid Ulysses + Ring (USP-style)

USP / Unified Sequence Parallelism (Fang & Zhao, 2024) treats Ulysses and Ring as composable. For N=4 ranks, factor `4 = Nu × Nr`:
- **Outer Ulysses (Nu)**: head split via a2a — within a smaller group, so smaller payload + fewer hops
- **Inner Ring (Nr)**: K/V rotation with compute-comm overlap via online softmax

### Concrete 2×2 layout for 4×L40S

```
                Ulysses axis (a2a, head split)
                 ──────────────────────────────►
       Ring  ┌──────────────┬──────────────┐
       axis  │   rank 0     │   rank 1     │      Each rank holds:
       (K/V  │  (seq grp A) │  (seq grp A) │        seq_shard/Nr   = 7800/2 = 3900 tokens
       ring) │  (head 0..5) │  (head 6..11)│        head_shard/Nu  = 12/2  = 6 heads
             ├──────────────┼──────────────┤        ─ each rank tensor:
             │   rank 2     │   rank 3     │           [B, 3900, 6, 128]  vs current [B, 7800, 12, 128]
             │  (seq grp B) │  (seq grp B) │           = 1/4 the activation footprint
             │  (head 0..5) │  (head 6..11)│
             └──────────────┴──────────────┘

  Ulysses a2a (rank 0 ⇄ rank 1):  payload ~6.5 MiB, 1 hop  (vs current 13 MiB / 3 hops)
  Ring K/V (rank 0 ⇄ rank 2):     payload ~3.25 MiB per step, 1 step total,
                                  OVERLAPPED with attention compute via online softmax
```

### Per-layer comparison

|  | Current Ulysses (N=4) | Hybrid 2×2 |
|---|---|---|
| a2a payload | 13.18 MiB | **6.5 MiB** (½) |
| a2a hops | 3 | **1** |
| a2a kernels (default channels) | 4–7 per logical | **2–3** per logical |
| Ring K/V transfers | 0 | 1 step × 3.25 MiB |
| Compute-comm overlap | ❌ | **✓** (Ring side hidden behind flash on Ulysses side) |
| FlashAttention used | yes (single call) | yes (1–2 calls per layer with LSE merge) |
| Ulysses fallback | n/a | automatic when Nr=1 |

### Reasons hybrid wins over pure Ring

- Reuses existing FlashAttention via `return_softmax_lse` for blockwise merge — no kernel rewrite
- Falls back cleanly to Ulysses when Nr=1 — same code path covers all configs (sets `FASTVIDEO_HYBRID_UR_NR=1`)
- Smaller a2a payloads also relieve PCIe pressure (separate axis of improvement)
- Cleaner SSIM equivalence proof (online softmax is well-understood; FA already returns LSE)
- 2-rank a2a sub-group inherits the L40S PCIe NCHANNELS=1 win (smaller communicators → less channel overhead matters more)

## FP8 collective compression — explicit non-recommendation

Halving payload would cut NCCL wall ~half. But quantizing Q/K/V before exchange:
- changes softmax logits' numerical envelope
- needs per-token scale/dequant
- requires full SSIM/VBench sweep
- moves into the same "trade quality for speed" zone as TGATE

If product later accepts a second approximate knob, FP8 comm is worth a try. Until then, **not on the no-quality-loss main line**.

## Recommended next step

Do **not** start pure Ring coding. Write a hybrid design doc first:

```
Hybrid Ulysses-Ring Attention for FastVideo SP Inference
```

Design doc must answer:
1. What does the current Ulysses pre-attention AllToAll4D actually move (Q/K/V layout, shapes, padding)?
2. Can Q stay local while K/V rotates? Where does Q's RoPE land?
3. Block size for the Ring dimension; how it interacts with FlashAttention's tile size
4. Online softmax merge — `return_softmax_lse` interface across fa2/fa3/fa4
5. RoPE — applied before or after ring; positional indices in sharded view
6. Output layout return path back to sequence-sharded hidden state
7. Scope: self-attn only first; cross-attn unchanged
8. Stacking with TGATE — does the TGATE cache-hit path bypass Ring?
9. Expected wall ceiling separately for PCIe (L40S) and NVLink (H100)
10. SSIM ≥ 0.999 + per-step latent drift bounded — explicit gates and harness reuse

After the design doc, prototype on Wan T2V flash self-attn only, env-gated, with the Ulysses path as fallback.

## Bottom line

- **Fastest to ship**: TGATE PR (already done, just needs branching to `tgate-cfg-reuse`) → H100 deployment (commercial) → NCCL per-hw knob docs.
- **Architecturally right**: Hybrid Ulysses-Ring (USP-style), not pure Ring. Design doc before implementation.
- **Out of scope for "no quality loss"**: FP8 comm, sparse attention. Park as future experiments.
