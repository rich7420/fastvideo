# Wan2.1-T2V-1.3B inference perf deep-dive (Phase 1.2)

720×1280 / 77f / 30-step, latent seq 72,000. 7 nsys profiles (H100 sp1 FA2/FA3, sp2, sp4; L40S sp1/2/4) + 4 A/B runs. Tooling: `nsys-ai` 0.2.2 + parquet cache + direct SQLite.
**Structure: Part I = inference/verdict (subjective, concise). Part II = objective-facts ledger (measured only). Every claim in Part I traces to a tagged fact in Part II.**

---
# PART I — Verdict & plan (inference)

## 1. Where the bottleneck is

| regime | bottleneck | evidence |
|--|--|--|
| **H100, 1 GPU** | attention **compute**, already at ceiling (72% MFU) — lever = *fewer* forwards, not faster | §II.3 |
| **H100, multi-GPU** | **fixed non-DiT tail**: VAE 19% e2e + text-enc 8% + inter-stage idle | §II.7, §II.9 |
| **L40S, multi-GPU** | **PCIe all-to-all** (38–42% kernel time, 0% overlap) | §II.5 |

- **FA3 was the one big win** — a pure attention-kernel swap (−46.5%/call, GEMM untouched, §II.2); e2e 219.5→116.6 s on H100×1 (§II.13).
- **Fabric, not software**: same all-to-all, NVSwitch is **53.7× faster per collective** than PCIe (§II.5). H100 SP scales **91%**, L40S **53%** (§II.13).
- **DiT is near-ceiling**: attention 72% MFU, GEMM 73% (§II.3–4); H100 comm trivial (3–7%); cross-attn 0.1 s.

## 2. Measured-dead — do NOT pursue (6 A/Bs + 1 structural)
CFG batching (~0, SSIM 0.90/0.95) · GEMM max-autotune (73% MFU already) · VAE-compile (~0 — wraps `forward`, not `decode()`) · VAE no-tile (~0) · vectorize-blend (no-tile already removed blend → ~0) · **VAE-SP** (sp4 decode 7.56 s vs 7.4 s) · text-enc step-cache (already per-gen). Also retired: **hybrid Ulysses-Ring** (attn scales exactly linearly → no occupancy gain, §II.3), async all-gather (<1% of kernel time), ncu-on-Modal (blocked), L40S NCCL-tuning.

⇒ **The VAE decode 7.4 s is a hard floor (intrinsic fp32 conv+norm, §II.7); on H100 the lossless *kernel*-level surface is exhausted.**

## 3. What's left (quality-preserving)
| # | lever | platform | type | status |
|--|--|--|--|--|
| 1 | **AG @ tg_0.5** (#1372) | both | VBench-validated <±0.011 | in flight — cuts the dominant attn work; **speedup still UNMEASURED** |
| 2 | wrap a2a as `torch.library` custom_op | L40S | lossless enabler | unblocks cross-a2a compile + overlap |
| 3 | compile ON for L40S sp≥2 (after #2) | L40S | lossless | fuse the 6–10× elementwise tax |
| 4 | compute–comm overlap of a2a (after #2) | L40S | lossless | exposed PCIe a2a |
| — | cap L40S SP at 2 | L40S | config | avoid sp4 H2D pathology |

Quality-trading (out of scope): fewer steps, bf16 VAE, sparse/VSA attention, layer-skip/DeepCache.

## 4. Open questions (unmeasured / tool-blocked)
1. **AG real speedup** — the one credited lever, never measured. *Highest priority.*
2. **L40S H2D anomaly** (§II.6: 154,800 H2D ops/rank) — `NCCL_P2P_LEVEL` test, ~30 min.
3. **L40S attention sp1→sp2 2.34× super-linear** — confounded by compile ON→OFF; deconfound run.
4. **H100 attention 72% — softmax-ceiling vs occupancy?** Needs ncu (blocked on Modal) or a counter-enabled host.
5. **H100 sp4 inter-stage idle** — sync-bound (§II.8) vs dispatch-bound; CPU-sampling segfaults on Modal, retry `--trace=osrt`.

---
# PART II — Objective-facts ledger (measured only, no estimates)

Tags: **[M]** measured · **[C]** computed (measured ÷ skill-FLOPs × peak) · **[T]** traced from source · **[O]** observed tool behavior.
**Sampling [M]:** kernel-trace numbers (II.1–II.8, II.13–15) = **1 nsys profile per config** — deterministic kernel-time sums, but **no run-to-run CI** (single Modal worker each). Multi-run numbers (per-generation spread shown): stage-wall II.9 & A/B II.10 = 1 warmup + 2 measured (e.g. VAE-SP decode 7701/7589/7559 ms). No variance bars on the single-profile metrics → treat sub-few-% deltas as noise.

## II.0 Workload, metadata & code provenance
**[T]** Wan2.1-T2V-1.3B: hidden 1536, 30 layers, ffn 8960, 12 heads×128, patch (1,2,2) → seq 72,000. **[O]** nsys-ai GPU auto-detect "unknown" (MFU peak supplied = 989 TFLOPS bf16 H100); profiler overhead 4.7%. **[T]** H100 ladder compile-consistent (all FA3+compile); L40S ladder compile-mixed (sp1 ON, sp2/sp4 OFF) — so any L40S sp1↔sp2 delta is confounded.

**Code provenance [T]:** all §II.12 code facts **verified against upstream/main @ `ba75ad82`** (match exactly). BUT profiles were **measured on the `inference-profile` staging branch**, which carries local perf deltas vs upstream in the profiled paths — FA3 custom-op + compile-enable (`82e2deb1`), RoPE-embed cache (`wanvideo.py`), per-step isnan-sync gating, UniPC `solve_ex`, hot-path cleanup. So **absolute numbers reflect the staging branch (≈ upstream + those wins), not pristine upstream**; the *controlled comparisons* (FA2-vs-FA3, NVSwitch-vs-PCIe, the SP ladder — all same-branch) hold as relative results. The AG/TGATE lever (`51ae1123`, stale-uncond reuse) is **present on this branch but gated OFF** (`FASTVIDEO_TGATE_STEP=1.0`) in every profile → measuring it = flip the env. `vae_sp` wiring in `decoding.py` is a **local A/B-only edit**; upstream's `vae_sp` is a no-op (commented `enable_parallel()`).

## II.1 Kernel-family table **[M]** (kernel-busy ms, summed across ranks, whole profile)
| config | attn | GEMM* | NCCL | VAE-conv | elemw | TOTAL | busy% dev0 |
|--|--:|--:|--:|--:|--:|--:|--:|
| H100 sp1 FA2 | 458,521 | 53,178 | 0 | 10,902 | 60,328 | 576,091 | 91% |
| H100 sp1 FA3 | 246,004 | 55,730 | 0 | 10,899 | 32,663 | 335,984 | 85% |
| H100 sp2 FA3 | 246,379 | 68,300 | 13,181 | 21,826 | 45,477 | 378,608 | 72% (steady ~94%) |
| H100 sp4 FA3 | 245,532 | 91,443 | 30,196 | 43,670 | 62,388 | 440,984 | 59% (steady ~85%) |
| L40S sp1 | 779,566 | 170,529 | 0 | 18,725 | 228,187 | 1,195,378 | 95% |
| L40S sp2 | 666,939 | 174,158 | 718,171 | 34,815 | 283,575 | 1,880,229 | 92% |
| L40S sp4 | 673,593 | 211,405 | 1,055,541 | 70,414 | 509,111 | 2,539,136 | 82% |

\* the "GEMM" column's `%gemm%` pattern also catches `implicit_gemm` (VAE conv), so it double-counts ~10.9k ms with VAE-conv; true DiT-GEMM (nvjet only) = §II.4. Whole-profile busy% is dragged down by one-time load+compile (gaps in §II.8); steady-state is the parenthetical.

## II.2 FA2→FA3 **[M]** (H100 sp1, both compile ON, 10,800 attn calls each — controlled)
attn kernel `flash_fwd_kernel` 458,521 → `FlashAttnFwdSm90` 246,004 ms; **per heavy call 83.9→44.8 ms (−46.5%)**; GEMM 53,178→55,730 (flat); elementwise 60,328→32,663; total 576,091→335,984 ms (−41.7%). Win is entirely the attention kernel.

## II.3 Attention **[M]/[C]**
- Bimodal (FA3 sp1): 5,400 heavy self-attn @ 44.8 ms (242,123 ms, 99%) + 5,400 cross-attn @ 0.43 ms (2,346 ms).
- Per-call scaling (attn_total ÷ ngpu×10,800): **H100 FA3 22,636 / 11,361 / 5,660 µs (sp1/2/4) = exactly 2.00×, 2.01×**; L40S FA2 71,810 / 30,677 / 15,525 µs (2.34×, 1.98×). H100 sum conserved (244k); L40S not (775k→663k, confounded by compile boundary).
- **[C]** self-attn MFU: FA2 **38%** → FA3 **72%** (= 3.185e13 FLOP/call ÷ time × 989 TFLOPS).

## II.4 DiT GEMM **[M]/[C]** (FA3 sp1, nvjet only, 44.7 s — the *true* GEMM, not §II.1's contaminated column)
square proj (qkv+out, self+cross) 6/layer @ 487 µs → **71%**; FFN-in 1/layer @ 2,562 µs → **78%**; FFN-out 1/layer @ 2,769 µs → **72%**; aggregate **73%**. (The earlier "52%" was an artifact — `%gemm%` swept in VAE conv + omitted cross-attn FLOP.)

## II.5 All-to-all **[M]** (same `ncclDevKernel_SendRecv`, counts 21,604@sp2 / 43,208@sp4, overlap_pct 0 on both)
| fabric | NCCL kernel-busy sp2/sp4 | per-collective avg (nccl_breakdown, per-rank) | kernel-busy/call |
|--|--|--:|--:|
| H100 (NVSwitch) | 13,181 / 30,196 ms | 0.446 ms | 688 µs |
| L40S (PCIe) | 718,171 / 1,055,541 ms | 23.96 ms | 23,826 µs |
→ per-collective ratio **53.7×** (kernel-busy/call ratio 34.6×). all-to-all = 97.6–97.8% of NCCL; allgather the rest (<1% of kernel time).

## II.6 Memcpy bandwidth **[M]** (per-rank, device 0)
| | D2D | H2D | D2H |
|--|--|--|--|
| H100 sp4 | 413 GB @ **1507 GB/s** (peak 2174) | 159 GB @ 12.69 GB/s (5,998 ops) | 26 GB @ 2.71 GB/s |
| L40S sp4 | 1.21 TB @ **392.6 GB/s** (peak 2244) | 675 GB @ 12.67 GB/s (**154,800 ops, 54.5 s**) | 28 GB @ 1.15 GB/s |

L40S-sp4 H2D anomaly: 154,800 ops/rank (26× H100's); D2H only 28 GB (asymmetric → not a clean host-bounce). Mechanism unconfirmed (open Q #2).

## II.7 VAE decode internals **[M]** (FA3 sp1, clean last-gen window): wall 7.3 s, GPU-busy 7.2 s, **idle 1%**
conv (fp32 `implicit_gemm`) 3.43 s (47%) · elementwise (blend+GroupNorm/SiLU) 2.46 s (34%, 2,730+ kernels) · NCHW↔NHWC transposes 0.62 s (8%) · VAE-attn 0.19 s (3%). Compute-bound, not idle-bound.

## II.8 Sync / idle **[M]**
sync_density: H100 sp4 **45.4%** (82,758 ms, ~all `cudaStreamSynchronize`, span 182,421 ms); L40S sp4 **71.4%** (555,277 ms, span 777,700 ms). gpu_idle_gaps (H100 sp4): largest are 13.3 / 11.9 / 11.5 / 9.5 / 8.9 s — all inter-generation / load / compile, **not per-step**.

## II.9 Stage wall **[M]** (per generation)
| | denoise | VAE decode | text-enc | e2e |
|--|--:|--:|--:|--:|
| H100 sp1 | 106.0 s | 7.3 s | 3.3 s | 116.6 s |
| H100 sp2 | 56.0 s | 7.4 s | 1.73 s | 65.1 s |
| H100 sp4 | 29.0 s | 7.4 s | 3.32 s | 39.7 s |
| L40S sp4 | 159.7 s | 19.3 s | 4.7 s | 183.7 s |
VAE decode fixed (7.4 s at sp2 *and* sp4); text-enc anti-scales (1.73→3.32 s, reproducible).

## II.10 A/B experiments **[M]**
| experiment | result |
|--|--|
| CFG batching | H100 121.6→120.6 s (−0.85%) SSIM 0.905; L40S 221.2→219.7 s (−0.69%) SSIM 0.949 |
| VAE-compile sp1 | decode 7.35 s (baseline ~7.40) |
| VAE no-tile sp1 | decode 7.34/7.37 s; e2e 115.4 s; peak mem 24.7 GB |
| VAE-SP sp4 | decode 7.56 s (7701/7589/7559); denoise 29.5 s |

## II.11 Tool capability on Modal **[O]**
ncu → `LibraryNotLoaded` (bundled 12.8 + 2026.1.0) · CUTracer → builds+runs, nvjet GEMM SASS unresolvable (cuBLAS symbols), FA3 Sm90 hangs (v1 SIGTERM / v2 3 h timeout) · nsys CPU-sampling → segfault rc=139 · `--trace=cuda,nvtx` works. Working fallback = trace-derived MFU.

## II.12 Code facts **[T]**
Per-block compile (`blocks.N.forward`, cond `is_blocks`) · GEMM layers = to_q/k/v/to_out (self+cross) + ffn.fc_in/fc_out · `enable_torch_compile_vae` compiles `vae.forward` but inference uses `vae.decode()` · VAE `parallel_tiled_decode` exists (SP tile-shard), routes on `use_tiling∧use_parallel_tiling∧sp>1` — but **upstream `decoding.py` leaves `enable_parallel()` commented out, so `vae_sp` is a no-op on upstream** (the A/B's `enable_tiling(use_parallel_tiling=vae_sp)` wiring is a local edit) · text-enc = separate per-generation stage (embeds reused across 30 steps).

## II.13 e2e wall & SP-scaling **[M]** (Phase-1.1 sweep, `perf_sweep.py` — e2e wall, distinct from the kernel-busy of II.1)
| config | e2e | denoise | per-forward |
|--|--:|--:|--:|
| H100×1 FA2 (no compile) | 219.5 s | 208.7 s | 3.48 s |
| H100×1 FA3 (no compile) | 148.9 s | 139.3 s | 2.32 s |
| H100×1 FA3+compile | 116.6 s | 106.0 s | 1.77 s |
| H100×2 FA3+compile | 67.0 s | 56.6 s | 0.94 s |
| H100×4 FA3+compile | 40.8 s | 29.0 s | 0.48 s |
| L40S×1 FA2+compile | 360.8 s | 336.8 s | 5.61 s |
| L40S×4 FA2+compile | 183.7 s | 159.7 s | 2.66 s |
**SP-scaling efficiency** (denoise sp1→sp4): H100 106.0→29.0 s = 3.66× = **91%**; L40S 336.8→159.7 s = 2.11× = **53%**. FA3+compile e2e win on H100×1 = 219.5→116.6 s (1.9×).

## II.14 Top individual kernels **[M]** (% of total kernel-busy)
- **H100 sp1 FA3**: `FlashAttnFwdSm90` 72.8% (244,469 ms, n=10,800) · `nvjet_192x192_2x1` 8.0% · `nvjet_192x192_1x2` 4.4% · `triton…gelu` 1.7% · `sm90_xmma_fprop`(VAE conv) 1.6%.
- **H100 sp4 FA3**: `FlashAttnFwdSm90` 55.4% · `ncclDevKernel_SendRecv` 6.7% · `nvjet_192x192_2x1` 6.5% · `sm90_xmma_fprop`(VAE) 4.7%+4.3%.
- **L40S sp4**: `ncclDevKernel_SendRecv` 40.5% (1,029,482 ms) · `flash_fwd_kernel`(FA2) 26.4% · `cutlass_80` GEMM 3.7% · `elementwise_kernel` 3.2%+3.0%.

## II.15 GPU-busy by decile **[M]** (device-0, span split into 10 — basis for II.1 steady-state)
H100 sp2: `[0,0,48,99,91,93,99,85,99,99]` → steady ~94%. H100 sp4: `[0,0,0,57,99,67,99,75,98,99]` → steady ~85%. (First 2–3 deciles = model-load + compile-warmup at 0% busy; confirms II.1's whole-profile busy% is load-dragged, not steady-state.)

---
## Artifacts & reproducibility
**Profiles** (`profile_results/h100_fa_compare/` + Modal vol `fastvideo-nsys-rep`): `perf_h100_sp1{,_fa3}`, `perf_h100_sp{2,4}_fa3`, `perf_sp{1,2}`, `perf.sqlite` (L40S sp4) — each .nsys-rep + .sqlite.
**Modal harness** (`fastvideo/tests/modal/perf_nsys_profile.py`, `PERF_GPU=`): `h100-sp1`, `h100-sp1-fa3`, `h100-sp2-nsys`, `h100-sp4-nsys`, `ncu-roofline`, `cutracer`, `cpu-trace`; benchmark via `PERF_BENCHMARK_ID=`.
**A/B configs**: `wan-t2v-1.3b-h100-sp4{,-vaesp}.json`, `-sp1-vaecompile.json`, `-sp1-notile.json`; CFG-batch results `cfgb{,_h100}_ab_report.json` (volume); CUTracer `cutracer_out/`; `decoding.py` carries the `vae_sp→use_parallel_tiling` wiring.
**Memory**: `phase1_2_nsys_compare.md`, `modal_profiling_tool_capabilities.md`, `constraint_quality_preserving.md`.
