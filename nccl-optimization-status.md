# FastVideo NCCL Optimization — Status & Learnings

_Updated: 2026-05-16. Branch: `inference-profile` @ `cacc3d17`. Upstream `main` @ `8acd8e21`._

## What landed (kept)

| Where | Change | Wall | Risk |
|---|---|---|---|
| `cacc3d17` (inference-profile) | Validator NaN check → `FASTVIDEO_DEBUG_NAN_CHECK` gate | −0.5~1% | bit-exact |
| `17047d5a` (inference-profile) | TGATE — env-gated stale-uncond CFG reuse | −14.6% (0.7) / −22% (0.5) | SSIM/VBench passed, quality near-baseline |

**Total kept improvement: TGATE alone gives the biggest validated wall savings (−14~22%)**.

## What we tried and rolled back

| Attempt | Wall Δ | Quality | Rejected because |
|---|---|---|---|
| CFG batching (B=1 → B=2 fused forward) | −0.5% (L40S) / −0.9% (H100) | SSIM 0.91 vs 0.999 gate | drift > gate, no measurable wall gain |
| NCCL `MIN/MAX_NCHANNELS=1` | −6.8% (L40S) / **+19.4% (H100)** | bit-exact | hardware-specific, not general |
| NCCL `P2P_LEVEL=PXB` / `BUFFSIZE=16M` | ±1% | bit-exact | NCCL was already optimal here |
| `_functional_collectives` a2a path | +1% (worse) | bit-exact | AsyncCollectiveTensor wrapper overhead; legacy already stream-correct |
| a2a layout-copy cleanup (3 copies → 2) | −0.6% | bit-exact | within Modal cross-run noise; copies were hiding behind NCCL latency |
| Disable layerwise offload | +0~+78% (machine variance) | bit-exact | net neutral; first run hit slow Modal machine and misled |

## Key empirical findings

1. **NCCL is bandwidth-bound on PCIe**, not stream-bound or sync-bound. The 24 ms / 13 MiB kernel time (~50× naïve PCIe ceiling) comes from ring algorithm hops + SM contention, not host-staging or kernel decomposition. `NCCL_DEBUG=INFO` confirmed NCCL was already using P2P direct (PHB/PIX), 2 channels, ring algo SIMPLE proto.
2. **NVLink reverses PCIe-optimal knobs.** NCHANNELS=1 helps L40S because PCIe is a serial trunk (multi-channel = launch overhead, no bandwidth gain). NVLink has parallel lanes — same knob loses 19%.
3. **Layerwise offload is net neutral on Wan T2V 1.3B / 4×L40S.** The 2.7 TB H2D from the original profile is real but doesn't contend with NCCL meaningfully — Modal machine variance dominates the apparent regression.
4. **Per-step framework overhead is already squeezed.** Flash attention metadata builder is no-op, scheduler `scale_model_input` is a no-op, denoising loop invariants are hoisted (commit `8f4587fd`). No remaining slack at this layer.
5. **Software stream rearrangement cannot move the NCCL bottleneck on PCIe** — confirmed by both the H100×2 -0.85% CFG batching result (NVLink should have helped if bandwidth were the issue) and the existing repo note in `8f4587fd`'s commit message.

## Architectural picture (from original profile, unchanged by experiments)

```
compute_only_ms:   311s  (46.0%)    | same_stream_compute_pct: 100%
nccl_only_ms:      271s  (40.1%)    | same_stream_nccl_pct:    100%
idle_ms:            94s  (13.9%)    | overlap_ms:                0
                                    | iteration spend: NCCL 54.7% + flash 31.1%
```

Compute and NCCL are strictly serialized on one stream. Software can rearrange the order; it cannot create real overlap unless the data dependency chain breaks.

## What can still move the needle

Only architectural change can break the strict serialization:

- **Hybrid Ulysses-Ring Attention (next direction)**: factor SP=N into Nu × Nr (e.g. 4 = 2×2). Outer Ulysses does head a2a inside small group (smaller payload, fewer hops). Inner Ring rotates K/V inside the other group with compute-comm overlap via online softmax. Estimated 3-4 weeks; expected wall improvement 20-30% on L40S with PCIe + good fall-back behavior on NVLink.
- **Hardware swap to NVLink** (out of scope for code, but acknowledged: 4×L40S → 2×H100 alone is −45% wall in our measurements).

## Branch / PR plan

- `inference-profile` — primary dev branch; keep stacking exploratory work
- `tgate-cfg-reuse` — to be opened from `upstream/main`, cherry-pick TGATE only (no auto-attribution; see prior session for plan)
- `hybrid-ulysses-ring` — future PR branch after impl + SSIM validation

## Reference files (Modal harnesses, untracked, reusable)

- `fastvideo/tests/modal/cfg_batch_ab.py` — per-frame SSIM/LPIPS + wall A/B (reusable for Hybrid validation)
- `fastvideo/tests/modal/nccl_tuning_ab.py` — NCCL env A/B framework
- `fastvideo/tests/modal/nccl_nchannels_h100.py` — hardware-aware NCCL diagnostic
