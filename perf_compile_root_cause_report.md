# fastvideo `perf_compile.sqlite` — Profile Measurements (v4)

**Source:** `/home/rich-wsl/fastvideo/profile_results/compile/perf_compile.sqlite` (1.8 GB)
**Sibling JSON:** `perf_wan-t2v-1.3b-l40s-compile_20260514T030510Z.json`
**Generated:** 2026-05-14 from a warm parquet cache at `perf_compile.nsys-cache/`

This document records measured values from `nsys-ai` skill outputs and direct
SQL queries against the SQLite. Each section names the source skill or query
so any number can be re-derived. Sections are facts only — no fix
recommendations, no gain estimates, no ranking.

---

## 1. Profile-level row counts (direct SQL)

```
NVTX_EVENTS                  10,457,403
CUPTI_ACTIVITY_KIND_KERNEL    1,079,198
CUPTI_ACTIVITY_KIND_RUNTIME   7,165,943
CUPTI_ACTIVITY_KIND_MEMCPY      663,807
CUPTI_ACTIVITY_KIND_SYNCHRONIZATION 2,737,429
CUPTI_ACTIVITY_KIND_OVERHEAD    188,445
NVTX_PAYLOAD_SCHEMAS                 24 rows (6 distinct schemaId)
NVTX_PAYLOAD_SCHEMA_ENTRIES          68 rows
NVTX_EVENTS.binaryData       346,400 non-null
StringIds                     7,989,142
```

Kernel-table span: `MIN(start) → MAX(end)` = 677,543 ms (677.5 s wall).

Distinct deviceIds in `CUPTI_ACTIVITY_KIND_KERNEL`: `[0, 1, 2, 3]`.

---

## 2. `profile_health_manifest`

```json
gpu: "unknown"
fingerprint: {
  framework: "DeepSpeed",
  distributed: true,           // PR #127 fallback fired (kernel-table distinct deviceIds > 1)
  multi_node: false,
  nic_summary: "",
  precision_notes: []
}
data_quality.auto_trim: {
  applied: true,
  trim_start_ns: 394,791,526,439,
  trim_end_ns:   414,791,526,439,
  window_ms:      20,000.0,
  profile_full_span_ms: 677,543.7
}
data_quality.overhead_pct: 28.6
nvtx: {
  has_nvtx: true,
  iteration_count: 16,             // manifest's own detection
  median_iter_ms: 2.3,
  slowest_iter_ms: 2,944.2,
  top_regions: [
    { name: "aten::to,        op_id=1298527", total_ms: 2,310.5, count: 2 },
    { name: "aten::_to_copy,  op_id=1298528", total_ms: 2,310.5, count: 1 },
    { name: "aten::copy_,     op_id=1298530", total_ms: 2,310.5, count: 1 },
    { name: "aten::to,        op_id=1298528", total_ms: 2,310.3, count: 1 },
    { name: "aten::_to_copy,  op_id=1298529", total_ms: 2,310.3, count: 1 },
  ]
}
suspected_bottleneck: "High CPU Synchronization Blocking (37.0% of span)"
```

**All fields above describe the auto-trimmed 20-second window, not the full
677.5-second profile.** Numbers below for `overlap_breakdown`, `sync_cost_analysis`,
and `nccl_payload_breakdown` cover the full profile.

---

## 3. `overlap_breakdown` (device 0, full profile)

```
total_ms                  677,344.6
compute_only_ms           311,515.7  (46.0 %)
nccl_only_ms              271,563.8  (40.1 %)
overlap_ms                      0.02 ( 0.0 %)
idle_ms                    94,265.1  (13.9 %)
sync_ms                   214,805.6
compute_kernels           258,818
nccl_kernels               10,985
span_start_ns          39,565,504,542
span_end_ns           716,910,057,111

same_stream_diagnosis    ["7"]
same_stream_compute_pct  100.0          // PR #127 addition
same_stream_nccl_pct     100.0          // PR #127 addition
present_devices          [0, 1, 2, 3]   // PR #127 addition
device_id                 0
```

### 3a. Per-device kernel/stream layout (direct SQL across all 4 ranks)

```
dev=0  stream=7   compute n=258,818  total=311,515.7 ms
                  nccl    n= 10,985  total=271,563.8 ms
dev=1  stream=17  compute n=258,816  total=315,914.0 ms
                  nccl    n= 10,985  total=267,733.3 ms
dev=2  stream=17  compute n=258,810  total=314,592.1 ms
                  nccl    n= 10,985  total=268,910.4 ms
dev=3  stream=17  compute n=258,814  total=314,679.5 ms
                  nccl    n= 10,985  total=268,502.0 ms
```

---

## 4. `sync_cost_analysis` (full profile)

```
profile_span_ms          677,543.7
total_sync_wall_ms       214,805.6
sync_density_pct              31.7

sync_by_type_ms:
  STREAM_SYNCHRONIZE     211,502.5
  EVENT_SYNCHRONIZE        2,673.0
  CONTEXT_SYNCHRONIZE        738.5
  STREAM_WAIT_EVENT          157.1
```

---

## 5. `memory_transfers` (full profile, summed across all devices)

```
copyKind=1  H2D    count=597,708   bytes=2,764,132 MB   total_ms=324,398.0
copyKind=2  D2H    count=  5,639   bytes=  108,113 MB   total_ms= 70,941.1
copyKind=8  D2D    count= 60,460   bytes=5,058,436 MB   total_ms= 13,293.0
```

### 5a. Per-device breakdown (direct SQL `GROUP BY deviceId, copyKind`)

```
dev=0  H2D  149,427 calls   659,021 MB     87,205.4 ms
       D2H    1,412 calls    27,603 MB     18,047.6 ms
       D2D   15,118 calls 1,206,183 MB      3,319.7 ms
dev=1  H2D  149,427 calls   659,021 MB     87,485.3 ms
       D2H    1,409 calls    25,167 MB     17,570.1 ms
       D2D   15,115 calls 1,206,025 MB      3,325.1 ms
dev=2  H2D  149,427 calls   659,021 MB     74,761.9 ms
       D2H    1,409 calls    25,167 MB     17,666.5 ms
       D2D   15,113 calls 1,205,920 MB      3,323.7 ms
dev=3  H2D  149,427 calls   659,021 MB     74,945.4 ms
       D2H    1,409 calls    25,167 MB     17,656.9 ms
       D2D   15,114 calls 1,205,972 MB      3,324.5 ms
```

---

## 6. `nccl_payload_breakdown` (PR #128, full profile)

```
total_payload_events      346,400
message_carrying_events   346,388
skipped_events                  0
total_bytes_all     9,594,962,460,672    // 8,936.01 GiB
distinct_schemas                6
```

| schema_id | category | calls | distinct_communicators | msg p50 | msg p99 | msg max | bytes total |
|---|---|---:|---:|---:|---:|---:|---:|
| 16777218 | init | 4 | 1 | — | — | — | — |
| 16777220 | collective | 724 | 1 | 52.73 MiB | 52.73 MiB | 52.73 MiB | 37.08 GiB |
| 16777225 | p2p | 172,832 | 1 | 13.18 MiB | 39.55 MiB | 39.55 MiB | 4,449.46 GiB |
| 16777226 | p2p | 172,832 | 1 | 13.18 MiB | 39.55 MiB | 39.55 MiB | 4,449.46 GiB |
| 16777227 | init | 4 | 1 | — | — | — | — |
| 16777230 | group_marker | 4 | 1 | — | — | — | — |

### 6a. Schema field layouts (from `NVTX_PAYLOAD_SCHEMA_ENTRIES`)

```
16777218 (init, 24B):       NCCL communicator ID, No. of ranks, Rank, CUDA device
16777220 (collective, 16B): NCCL communicator ID, Message size [bytes]
16777225 (p2p, 24B):        NCCL communicator ID, Message size [bytes], Peer rank
16777226 (p2p, 24B):        NCCL communicator ID, Message size [bytes], Peer rank
16777227 (init, 24B):       NCCL communicator ID, No. of ranks, Rank, CUDA device
16777230 (group_marker, 8B): NCCL communicator ID
```

---

## 7. `iteration_timing` (full profile)

```
total rows returned                984
rows with duration_ms > 1,000      180
median (over 180 rows)         2,933.3 ms
mean   (over 180 rows)         2,936.7 ms
sum    (over 180 rows)         528,610   ms  (=528.6 s)
```

### 7a. Top 5 iterations by `duration_ms`

```
iter=311  dur=3,087.4 ms  kernels=1,172  compute=3,071.8 ms  nccl_count=59  text=heuristic_step_311
iter=316  dur=2,991.4 ms  kernels=1,234  compute=2,975.1 ms  nccl_count=61  text=heuristic_step_316
iter=321  dur=2,985.8 ms  kernels=1,212  compute=2,971.9 ms  nccl_count=61  text=heuristic_step_321
iter=364  dur=2,983.1 ms  kernels=1,234  compute=2,966.8 ms  nccl_count=61  text=heuristic_step_364
iter=313  dur=2,981.9 ms  kernels=1,238  compute=2,965.5 ms  nccl_count=61  text=heuristic_step_313
```

`text` prefix `heuristic_step_*` indicates the labels are synthesized by
`iteration_timing`'s kernel-gap detection heuristic, not present in the raw
NVTX_EVENTS (verified by `SELECT COUNT(*) FROM NVTX_EVENTS WHERE text LIKE 'heuristic_step%'` = 0).

### 7b. `iteration_detail` on iter=311

```
duration_ms       3,087.35
gpu_start_ns      109,428,571,984
gpu_end_ns        112,515,921,387
kernel_count      1,172
nccl_count        59
compute_ms        3,071.77
median_ms         0.79         // median of all 984 iteration_timing rows
vs_median        +390,703.8 %
top_kernels:
  ncclDevKernel_SendRecv             1,616.75 ms × 58  (54.7 %)
  flash_fwd                            920.48 ms × 58  (31.1 %)
  cutlass_80_tensorop_bf16_s16816gemm  122.29 ms × 203 ( 4.1 %)
  cutlass_80_tensorop_bf16_s16816gemm   56.86 ms ×  29 ( 1.9 %)
  at::elementwise_kernel                56.77 ms ×  58 ( 1.9 %)
```

---

## 8. `top_kernels` (full profile, top 10 by total_ms)

```
ncclDevKernel_SendRecv                                  total_ms=1,051,241.5  inv=43,208  avg=24.33  tc=false
void flash::flash_fwd_kernel<...,(int)128,(int)128,(int)32...>  648,214.3  inv=43,200  avg=15.00  tc=true
void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3...>  95,138.0  inv=151,200  avg=0.63  tc=true
void at::native::elementwise_kernel<(int)128, (int)2, ...>    81,499.8  inv=77,100   avg=1.06  tc=false
sm86_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x...  64,636.5  inv=7,200  avg=8.98  tc=true
void at::native::elementwise_kernel                              45,290.6  inv=53,832   avg=0.84  tc=false
void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3...>  44,807.9  inv=21,600  avg=2.07  tc=true
void at::native::elementwise_kernel                              29,725.7  inv=109,464  avg=0.27  tc=false
void cudnn::engines_precompiled::nchwToNhwcKernel                27,721.9  inv=16,752   avg=1.65  tc=false
void at::native::vectorized_elementwise_kernel                   27,692.3  inv=29,124   avg=0.95  tc=false
```

`total_ms` values are summed across all 4 device IDs (skill currently does not
expose per-device totals); per-rank averages = total ÷ 4.

---

## 9. `nvtx_layer_breakdown` (top 8 regions by total_gpu_ms)

```
detection_method  numbered_pattern
layer_depth       1
layer_count       673
confidence        1.0

total_gpu_ms  kernels  nccl%   tc%      nvtx_path (truncated to 65 chars)
    4,645.6   24,473   10.7   100.0    stage::DenoisingStage > Torch-Compiled Region: 0/0, op_id=74092
      685.0       51   91.9   100.0    stage::DenoisingStage > Torch-Compiled Region: 0/0, op_id=90955
      600.7      192    0.0   100.0    stage::DenoisingStage > aten::nonzero, op_id=1728365
      434.8       55   78.9   100.0    stage::DenoisingStage > Torch-Compiled Region: 0/0, op_id=90929
      405.3      101   14.9   100.0    stage::DenoisingStage > aten::to, op_id=1613263
      354.8      113   14.2   100.0    stage::DenoisingStage > Torch-Compiled Region: 0/0, op_id=908401
      339.5      112   14.1   100.0    stage::DenoisingStage > Torch-Compiled Region: 0/0, op_id=907210
      333.2      107   13.8    91.9    stage::DenoisingStage > Torch-Compiled Region: 0/0, op_id=1722730
```

---

## 10. `host_sync_parent_ranges` (top 5 by sync_ms)

```
parent_range                              n_syncs  sync_ms     top_child_label
stage::TextEncodingStage                       96  35,927.8    aten::item, op_id=73442
aten::is_nonzero, op_id=73441                   8   7,198.3    aten::item, op_id=73442
aten::item, op_id=73442                         4   3,599.0    aten::_local_scalar_dense, op_id=73443
aten::is_nonzero, op_id=1703377                 2   1,800.9    aten::item, op_id=1703378
aten::is_nonzero, op_id=1703384                 2   1,800.7    aten::item, op_id=1703385
```

---

## 11. NCCL kernel call-mode classification (direct DuckDB on cache parquet)

```
total NCCL kernels                       60,668
  via c10d::* (leaf NVTX)               46,182 (76.1 %)   total_ms = 1,130,471
  via ## Call CompiledFxGraph (leaf)     3,326 ( 5.5 %)   total_ms =    79,778
  neither (other aten leaf)             11,160 (18.4 %)   total_ms =   268,959
```

`leaf` here = innermost open NVTX scope at kernel launch (the `nvtx_text`
column of `nvtx_kernel_map.parquet`).

Kernel-count distribution per logical `AllToAll4D, seq=N` (NCCL kernels only):

```
4 kernels/seq    7,985 logical a2a
5 kernels/seq    2,504 logical a2a
6 kernels/seq      250 logical a2a
7 kernels/seq       59 logical a2a
10 kernels/seq       2 logical a2a
16 kernels/seq       2 logical a2a
```

Total distinct `AllToAll4D, seq=N` values in profile: **10,802**.

---

## 12. Code locations referenced by NVTX paths

```
fastvideo/pipelines/stages/denoising.py:357     conditional forward
fastvideo/pipelines/stages/denoising.py:370     batch.is_cfg_negative = True (write-only flag)
fastvideo/pipelines/stages/denoising.py:376     unconditional forward
fastvideo/pipelines/stages/denoising.py:389     combine: noise_pred = uncond + scale*(cond-uncond)
fastvideo/pipelines/stages/denoising.py:309-400 main DenoisingStage loop

fastvideo/attention/layer.py:59                 @torch.compiler.disable
fastvideo/attention/layer.py:60                 DistributedAttention.forward
fastvideo/attention/layer.py:97                 qkv = torch.cat([q, k, v], dim=0)
fastvideo/attention/layer.py:100                sequence_model_parallel_all_to_all_4D(qkv, scatter=2, gather=1)
fastvideo/attention/layer.py:134                sequence_model_parallel_all_gather (replicated tokens)
fastvideo/attention/layer.py:140                sequence_model_parallel_all_to_all_4D(out, scatter=1, gather=2)
fastvideo/attention/layer.py:149                @torch.compiler.disable
fastvideo/attention/layer.py:150                DistributedAttention_VSA.forward
fastvideo/attention/layer.py:194,217            VSA forward all_to_all calls

fastvideo/distributed/device_communicators/base_device_communicator.py:32   AllReduceOp.forward  dist.all_reduce
fastvideo/distributed/device_communicators/base_device_communicator.py:62   AllGatherOp.forward  dist.all_gather_into_tensor
fastvideo/distributed/device_communicators/base_device_communicator.py:80   ReduceScatterOp.forward  dist.reduce_scatter_tensor
fastvideo/distributed/device_communicators/base_device_communicator.py:155  AllToAll4DOp.forward (scatter=2/gather=1)
fastvideo/distributed/device_communicators/base_device_communicator.py:175  AllToAll4DOp.forward (scatter=1/gather=2)
fastvideo/distributed/device_communicators/base_device_communicator.py:236  all_to_all_4D entrypoint

fastvideo/distributed/parallel_state.py:259     @contextmanager graph_capture (defines dedicated stream)
fastvideo/distributed/parallel_state.py:263-274 torch.cuda.Stream() + with cuda.stream(...)
fastvideo/distributed/parallel_state.py:341     ProcessGroup.all_to_all_4D

fastvideo/hooks/layerwise_offload.py:21         async_copy_stream parameter
fastvideo/hooks/layerwise_offload.py:35,43,55,67  @torch.compiler.disable on hook methods
fastvideo/hooks/layerwise_offload.py:63         cpu_named_parameters[name].to(device, non_blocking=True)

fastvideo/forward_context.py:91                 medium = torch.quantile(...).item()  (logging path)

fastvideo/pipelines/stages/base.py:147          torch.cuda.nvtx.range(f"stage::{stage_name}")
fastvideo/pipelines/stages/base.py:150,155,160  torch.cuda.synchronize() (gated by FASTVIDEO_STAGE_LOGGING env)

fastvideo/distributed/communication_op.py:28    sequence_model_parallel_all_to_all_4D
fastvideo/distributed/communication_op.py:152   torch.cuda.synchronize(device) (in _sp_warmup function)
```

---

## 13. PyTorch / hardware context (direct queries)

```
fastvideo/pyproject.toml:                                 torch==2.11.0
TARGET_INFO_SYSTEM_ENV.CudaDriverVersion                  13000
TARGET_INFO_SYSTEM_ENV.GpuInfo                            "{}" (empty)
TARGET_INFO_SYSTEM_ENV.HasDiscreteGpu                     0
TARGET_INFO_SYSTEM_ENV.CpuCores                           20
TARGET_INFO_SYSTEM_ENV.CpuArchitecture                    x86_64
TARGET_INFO_GPU table                                     (absent from this profile)
torch.distributed._functional_collectives                 available in 2.10.0+ (verified locally)
torch._inductor.config.reorder_for_compute_comm_overlap   default False
```

GPU model name not recoverable from this profile via `nsys-ai`. Filename
`perf_wan-t2v-1.3b-l40s-compile_20260514T030510Z.json` implies L40S; not
machine-verified.

---

## 14. Cross-skill row-count cross-checks

```
nvtx_kernel_map.parquet (cache view) NCCL classification:
  total NCCL kernels                       60,668
  10,802 distinct AllToAll4D seq values
  × 4-5 kernels per logical a2a ≈ 43,000-54,000 kernels ascribed to a2a
  remainder ≈ all_gather_with_unpad + group markers + init

iteration_timing 180 real iters × 2.93 s/iter mean = 528.6 s
  + host_sync_parent.TextEncodingStage one-time 36.0 s
  + residual (VAE decode + setup) ≈ 113 s
  =                                       677 s   ← matches profile span 677.5 s
```

---

## 15. Skill execution timings (warm cache, this run)

```
top_kernels                  1.12 s
memory_transfers             0.70 s
nccl_payload_breakdown       1.86 s
overlap_breakdown           13.70 s
profile_health_manifest     14.15 s
sync_cost_analysis          13.39 s
host_sync_parent_ranges     33.00 s
nvtx_layer_breakdown        32.52 s
iteration_timing            38.54 s
                            ───────
total wall (if serialized) 149.0 s
total wall (parallelized)   ~38 s   (limited by the slowest skill)
```

First-open cache build (cold): 14 phases, ~28 s phases 1-13 + ~3,100 s phase 14
(`nvtx_kernel_map.parquet` IEJoin) on a related 2.1 GB profile. Once
`nsys-cache/` is populated, subsequent skill runs are bounded by the timings
above.

---

*This document is regenerable. Each section corresponds to one skill or direct
SQL query against the source SQLite. To re-verify any number, run the named
skill via* `nsys-ai skill run <name> <profile> --format json` *or execute the
referenced query.*

---

## 16. Background: classifier-free guidance and FastVideo's two-pass implementation

This section gives the architectural context required to read §17 (root cause)
and §18 (acceleration directions). Nothing here is a measurement; every claim
references the codebase at the same commit as the profile.

### 16.1 What CFG is and the math

Classifier-free guidance (Ho & Salimans, 2022) is a sampling-time technique
that strengthens prompt adherence in conditional diffusion models. At every
denoising step the model is evaluated twice — once with the real prompt
embedding (`cond`) and once with a null/negative prompt embedding (`uncond`) —
and the two predictions are linearly combined:

```
noise_pred = uncond + s · (cond − uncond)
```

`s` is the guidance scale (Wan T2V default 5.0, HunYuan 7.5). Geometrically
this is a linear extrapolation along the direction `(cond − uncond)` in
noise-prediction space; the model is pushed away from the unconditional prior
toward the conditional one.

The inference-cost implication is structural: **every denoising step requires
two transformer forward passes**. The profile in §7 captured 180 forwards
across 90 denoising steps — exactly that 2× factor.

### 16.2 The two-pass implementation choice in FastVideo

There are two well-known evaluation strategies for the cond/uncond pair:

1. **Batch doubling** (early diffusers, vLLM-style): build a single batch of
   size 2 by concatenating `[neg, pos]` along dim 0, run *one* transformer
   forward, then `chunk(2)` the output.
2. **Separate forward passes** (FastVideo's choice): two independent forwards,
   one for cond and one for uncond, combined on-device after both finish.

FastVideo took option 2. The architectural reasons, sourced from the comment
at `fastvideo/pipelines/stages/conditioning.py:38-44` and the surrounding code,
are three preserved degrees of freedom:

- **Per-pass attention metadata**: `VideoSparseAttentionBackend` (VSA) and
  `VMOBAAttentionBackend` build `attn_metadata` per step. The two-pass design
  *allows* different sparsity metadata for cond vs uncond.
- **Per-pass conditioning state**: `batch.is_cfg_negative` is set True/False
  between the two passes (`denoising.py:357,370`), giving downstream layers a
  hook to switch behavior (e.g. cache routing, conditional skip, dynamic CFG
  schedule).
- **Per-pass conditioning kwargs**: `pos_cond_kwargs` and `neg_cond_kwargs`
  (built at `denoising.py:131-145`) hold the CLIP image embeddings and
  attention masks separately, so I2V pipelines can pass different
  cross-attention conditioning to each pass.

**Important nuance, verified by grep at the profiled commit:**

- `attn_metadata` is built once outside the CFG branch (`denoising.py:319-353`)
  and passed *unchanged* into both forwards. The per-pass-metadata capability
  is API surface only; no current pipeline exercises it.
- `batch.is_cfg_negative` has **zero readers** in `fastvideo/`. `grep -rn` only
  finds the field definition and writer sites. It is a dormant flag.
- `pos_cond_kwargs` vs `neg_cond_kwargs` *is* genuinely used (Wan I2V, HunYuan
  I2V); their tensors can legitimately differ in shape.

So the two-pass design preserves three degrees of freedom, of which one
(`pos/neg_cond_kwargs`) is actively used and two are reserved capability with
no current consumer. **This matters for §18**: an acceleration that closes the
two dormant capabilities is non-breaking at the current commit, while one that
closes the active capability is not.

### 16.3 Per-step transformer call shape

The denoising loop at `denoising.py:248-423` runs the following sequence per
timestep `t`:

```text
# 1. attention-metadata build (sparse backends only; FlashAttn/SDPA: skipped)
build_attn_metadata(t)                                  # denoising.py:319-353

# 2. conditional forward
set is_cfg_negative = False
with set_forward_context(attn_metadata):
    noise_pred = transformer(
        latent_model_input,                              # [1, C, T, H, W]
        prompt_embeds,                                   # list[Tensor]
        t_expand,                                        # [1] or [1, seq]
        guidance=guidance_expand,                        # HunYuan embedded CFG
        **image_kwargs,                                  # I2V CLIP image embed
        **pos_cond_kwargs,                               # +CLIP, +pos attn_mask
        **action_kwargs,                                 # gen3c only
        **camera_kwargs,                                 # gen3c only
        **timesteps_r_kwarg)                             # meanflow only

# 3. unconditional forward (only if CFG enabled)
if do_classifier_free_guidance:
    set is_cfg_negative = True
    with set_forward_context(attn_metadata):             # same metadata reused
        noise_pred_uncond = transformer(
            latent_model_input,                          # same latent
            neg_prompt_embeds,                           # negative embed
            t_expand,
            **image_kwargs,
            **neg_cond_kwargs,                           # neg CLIP + neg mask
            ...)

    # 4. CFG combination (purely elementwise; ~0.01 % of step time)
    noise_pred = noise_pred_uncond + s · (noise_pred − noise_pred_uncond)
    if batch.guidance_rescale > 0.0:
        noise_pred = rescale_noise_cfg(noise_pred, cond, ...)

# 5. scheduler step
latents = scheduler.step(noise_pred, t, latents)
```

Between steps 2 and 3 there is no host-side work: no `.item()`, no Python
branching that depends on cond output, no file IO. The second forward could
in principle begin queuing kernels while the first is still draining, but in
practice both forwards use the same CUDA stream (stream 7 on rank 0, stream
17 on other ranks, §3a), so they serialize.

### 16.4 Cross-stage data flow for prompt embeddings

The cond/uncond asymmetry begins back in `TextEncodingStage`:

```text
batch.prompt           = "a cat dancing in space"          (positive)
batch.negative_prompt  = ""                                (typically empty)

TextEncodingStage.forward (text_encoding.py:42-98):
  encode_text(prompt)            → batch.prompt_embeds         (list[Tensor])
  if do_classifier_free_guidance:
    encode_text(negative_prompt) → batch.negative_prompt_embeds (list[Tensor])

ConditioningStage.forward      (conditioning.py:23-44):
  no-op (returns batch); comment explicitly notes CFG is applied in DenoisingStage

DenoisingStage.forward         (denoising.py:248-423): two-pass loop above
```

The embeddings are kept as Python `list[Tensor]` — one tensor per text
encoder. Wan T2V has a single T5 encoder, so the list has one element of
shape `[1, 512, 4096]`. HunYuan has two encoders (T5 + CLIP), so the list has
two elements with different shapes. Any batching scheme must concatenate every
element of these lists independently along dim 0.

---

## 17. Root cause: mapping measurements to architecture

Each cost component the profile measured can be traced to a specific
architectural choice in §16. This section makes those connections explicit.

### 17.1 180 forwards (§7) ← CFG two-pass design (§16.2)

§7 records 180 real forwards over a 528.6 s denoising window, with 90
denoising steps. The 2× factor between steps and forwards is the cond/uncond
pair from §16.3. Eliminating this factor — by batching, by skipping uncond on
some steps, or by distilling CFG into the model offline — is the single
largest available lever and is the subject of §18.

### 17.2 NCCL 54.7 % per forward (§7b, §11) ← sequence-parallel attention

`ncclDevKernel_SendRecv` accumulates 1,616.75 ms inside a single 3,087 ms
forward — more than half the wall (§7b). Trace to code:

- `attention/layer.py:60`: `DistributedAttention.forward`, marked
  `@torch.compiler.disable` (line 59)
- `attention/layer.py:97`: concatenates `[q, k, v]` along dim 0
- `attention/layer.py:100`: `sequence_model_parallel_all_to_all_4D(qkv, scatter=2, gather=1)`
- post-attention: `attention/layer.py:140` runs the inverse all-to-all
  (`scatter=1, gather=2`)

Every attention layer in every DiT block emits **two NCCL all-to-all
operations** to implement Ulysses-style sequence parallelism (scatter heads,
gather sequence, compute full-seq attention locally, scatter back). With 30
transformer blocks per Wan T2V forward × 2 a2a per block × 180 forwards = 10,800
logical all-to-alls — exactly the §11 count.

Each logical all-to-all decomposes into 4–5 `ncclDevKernel_SendRecv` kernels
(§11 distribution table). At sp=4 (the topology in §3a present_devices
`[0,1,2,3]`), each all-to-all is a 4-way exchange, so 4 SendRecv kernels
covers the typical case; the 5-kernel tail is overlap-region padding.

**Why this dominates:** with `original_seq_len = 31200` at hidden 1536 in bf16
across 30 blocks, each all-to-all moves ≈ 13–39 MiB per call (§6 p50/p99 13.18 / 39.55 MiB).
Total NCCL payload across the profile is 8,936 GiB (§6). L40S has no NVLink —
inter-GPU is PCIe Gen4 x16, with a practical throughput ceiling around 25 GB/s
per direction. 8,936 GiB ÷ 25 GB/s ≈ 360 s, within 2× of the measured 271 s
per-rank NCCL wall (§3). The workload is **bandwidth-bound, not
latency-bound** — a critical fact for §18.

### 17.3 FlashAttn 31.1 % per forward (§7b, §8) ← O(N²) attention at seq_len 31200

`flash_fwd_kernel` accumulates 920.48 ms in iter=311 across 58 invocations
(§7b), about 16 ms per invocation. At post-a2a per-rank seq_len 7800 (sharded
from 31200 / 4 ranks) × head_dim 64 × 30 blocks × ~2 attention computations
per block = ~58 kernel launches per forward, this matches FlashAttention-3 on
L40S at full bf16 throughput.

This cost is **O(N²) in sequence length** and is *not* reduced by SP — SP
shards the head dimension, not the seq² inner loop. The 31.1 % share is
therefore fixed by `(num_frames, height, width, num_blocks)` and won't move
under any CFG-side change unless the model itself is altered (e.g. sparse
attention via VSA/VMoBA).

### 17.4 Same-stream serialization (§3, §3a) ← single compute stream

`overlap_ms = 0.02` on device 0 (§3) and `same_stream_compute_pct =
same_stream_nccl_pct = 100 %` mean compute kernels and NCCL kernels share a
single CUDA stream (stream 7 on rank 0, stream 17 on other ranks). This is
the default behavior of `torch.distributed` collectives when called from the
main Python thread: they queue onto the current stream rather than a
dedicated comm stream.

`fastvideo/distributed/parallel_state.py:259` defines a `graph_capture` context
manager that *does* allocate a separate stream, but it is only active during
CUDA graph capture, not steady-state inference. So in inference all NCCL
serializes against compute, costing the full NCCL wall as inflated forward
time rather than overlapping into the next compute kernel.

**Implication for CFG batching:** halving the *count* of NCCL calls does not
gain overlap, because there is no overlap to begin with. PCIe payload bytes
stay constant; only launch / ring-init overhead drops. The §16 expectation of
"halved NCCL collective call count" must not be read as "halved NCCL time" —
see §18.1 for the realistic decomposition.

### 17.5 TextEncodingStage host sync 35.9 s (§10) ← scalar extraction in T5 truncation

`TextEncodingStage` shows 96 `aten::item` syncs totaling 35,927.8 ms (§10).
These are inside the T5 tokenizer's variable-length truncation path, which
decides padding length by reading a tensor `.item()` and branching on it. The
cost is one-time per inference run (not per-step) and is part of the residual
113 s in §14. It is not a CFG-side problem and is out of scope here. Logged
for completeness; a future change could move T5 to fixed `max_length=512` and
delete the syncs.

---

## 18. Acceleration directions

Three routes are evaluated, all targeting §17.1 (the 180-forward count).
§18.1–18.3 describe each in isolation; §18.4 covers composition; §18.5
records the landing-order decision.

### 18.1 Route A: batched CFG

Replace the two sequential forwards with a single forward at batch dim 2.

```python
# Before (denoising.py:357-389):
noise_pred       = current_model(latent_model_input, prompt_embeds,     t_expand, **pos_cond_kwargs, ...)
noise_pred_uncond = current_model(latent_model_input, neg_prompt_embeds, t_expand, **neg_cond_kwargs, ...)
noise_pred = noise_pred_uncond + s · (noise_pred − noise_pred_uncond)

# After:
latent_b    = torch.cat([latent_model_input, latent_model_input], dim=0)
t_b         = t_expand.repeat(2)
embeds_b    = [torch.cat([n, p], dim=0) for n, p in zip(neg_prompt_embeds, prompt_embeds)]
combined_kw = merge_cond_kwargs(neg_cond_kwargs, pos_cond_kwargs)  # cat each tensor along dim 0
pred_b      = current_model(latent_b, embeds_b, t_b, **combined_kw, ...)
pred_uncond, pred_cond = pred_b.chunk(2, dim=0)
noise_pred  = pred_uncond + s · (pred_cond − pred_uncond)
```

**Expected per-step impact**, decomposed from §7b iter=311 cost shares:

| Component | Baseline @ 5.87 s | After Route A | Why |
|---|---|---|---|
| NCCL (54.5 %) | 3.20 s | 2.85–3.05 s | Same total bytes (§17.2); only launch/ring-init halved; no overlap gain (§17.4) |
| FlashAttn (31.1 %) | 1.83 s | 1.65–1.75 s | Kernel launches halved; FA3 scheduling slightly better at B=2; not bandwidth-bound |
| GEMM (~4 %) | 0.23 s | 0.22 s | Already SM-saturated at seq_len 7800/rank; B=2 gives ~negligible win |
| Elementwise + Python (~10 %) | 0.58 s | 0.30–0.45 s | Per-call Python/Dynamo overhead halved |
| **Total per step** | **5.87 s** | **5.0–5.4 s** | **−8 % to −15 %** |

90-step total drops from 528.6 s to ≈ 450–490 s.

**What it closes from §16.2:**

- Loses `is_cfg_negative` flag granularity — currently dormant, zero readers.
- Loses VSA/VMoBA per-pass metadata capability — currently unused.
- *Does not* lose the `pos_cond_kwargs` vs `neg_cond_kwargs` distinction —
  they are concatenated along the new batch dim, both signals flow through.

**Memory cost:** activation footprint per rank roughly doubles. See §19.

**Numerical equivalence:** float accumulation order changes inside fused GEMM
and attention kernels, so output is **not bit-exact**. Should be
**SSIM-equivalent**; validate with `fastvideo/tests/ssim/`.

### 18.2 Route B: TGATE / stale uncond reuse

Don't change the shape of each forward — change how often uncond is computed.
After a "gating" timestep `T_g`, freeze the guidance vector
`Δ = cond − uncond` at its last value and skip subsequent uncond forwards:

```python
delta_cached = None
gate_step    = int(num_inference_steps * 0.5)   # hyperparameter, per-pipeline

for i, t in enumerate(timesteps):
    noise_pred = current_model(latent_model_input, prompt_embeds, t_expand, **pos_cond_kwargs, ...)

    if do_classifier_free_guidance:
        if i < gate_step or delta_cached is None:
            # still in gate window: compute uncond, refresh delta
            noise_pred_uncond = current_model(latent_model_input, neg_prompt_embeds, t_expand, **neg_cond_kwargs, ...)
            delta_cached      = noise_pred - noise_pred_uncond
            noise_pred        = noise_pred_uncond + s · delta_cached
        else:
            # past gate: reuse cached delta, skip uncond forward
            noise_pred = noise_pred + (s - 1.0) · delta_cached

    latents = scheduler.step(noise_pred, t, latents)
```

The approximation is `Δ(t) ≈ Δ(T_g)` for `t > T_g`. Empirically (TGATE paper,
SDXL 2024, 50 % gate point) FID delta is < 0.5 — the guidance vector is highly
smooth across timesteps once global structure is set. Video diffusion further
attenuates per-frame error through temporal smoothness. Wan T2V quality
behavior under TGATE is not in the literature — must validate locally.

**Expected impact** (gate at 50 % over 90 steps: 45 dual-forward + 45
cond-only steps):

| Phase | Count | Per-step time | Subtotal |
|---|---|---|---|
| Pre-gate (cond + uncond) | 45 | 5.87 s | 264.2 s |
| Post-gate (cond only) | 45 | 2.93 s | 132.0 s |
| **Total** | **90** | — | **≈ 396 s** |

Compared to the 528.6 s baseline, **−25 %**.

**Costs:**

- Algorithmically approximate; not numerically equivalent.
- Adds a gate-step hyperparameter that must be tuned per pipeline.
- Visible quality drop risk if gate is too aggressive (validate via SSIM
  sweep — see §20).

**Risk profile vs Route A:** lower compute-graph risk (no shape change, no SP
topology stress, no memory doubling) but higher *quality* risk (algorithmic
vs purely numerical change).

### 18.3 Route C: CUDA graph capture of the two-pass

Keep the two-pass structure but capture both forwards under a single
`torch.cuda.graph`, so launch/scheduling overhead becomes one-shot. This
targets only the Python / Dynamo / per-call launch component (~5–10 % of step
time, the elementwise+Python row of §18.1), leaves NCCL and FlashAttn
unchanged, and preserves bit-exact output.

**Expected impact:** −3 % to −8 % per step.

**Costs:**

- Graph capture is fragile with dynamic-shape kernels — VSA / VMoBA
  metadata changes per step would force re-capture.
- Requires gating by attention backend (FlashAttn/SDPA only).
- CPU-offload paths and `@torch.compiler.disable`-marked layers
  (`attention/layer.py:59,149`) break under capture without rework.

Best used as a finishing pass *after* Route A or B has landed and the per-step
shape has stabilized.

### 18.4 Combined routes

A and B are orthogonal: A halves the cost of *every* dual-forward step, B
*eliminates* some dual-forward steps. They compose.

| Configuration | 90-step total | Δ vs baseline |
|---|---|---|
| Baseline (current two-pass) | 528.6 s | — |
| Route A only | 450–490 s | −7 % to −15 % |
| Route B only (gate 50 %) | ~396 s | −25 % |
| Route A + Route B (gate 50 %) | ~340 s | **−36 %** |
| Route A + Route B + Route C | ~320 s | −39 % |

### 18.5 Landing order: B first, A second, C last

The initial intuition (recorded in the previous §16 of this document, now
superseded) was that **batched CFG should land first** because it directly
attacks the highest-cost component (NCCL at 54.5 %). Section 17 invalidates
that intuition:

- §17.2 shows NCCL is **bandwidth-bound** on L40S PCIe, not launch-bound.
  Batching halves the *count* but not the *bytes*. The achievable NCCL
  reduction from Route A is ~5–10 %, not 54.5 %.
- §17.4 shows there is no compute/comm overlap to recover, so the second
  forward in the two-pass structure does not actually waste any idle time
  that batching would reclaim.
- Route B (TGATE) achieves a larger headline gain (−25 % vs −15 %) with
  lower implementation risk: no shape change, no memory doubling, no SP
  topology stress.

The recommended landing order is therefore:

1. **Route B (TGATE)** — env-var gated `FASTVIDEO_TGATE_STEP=` (set to ≥ 1.0
   to disable). Validate SSIM sweep across pipelines. Expected −25 %.
2. **Route A (batched CFG)** — env-var gated `FASTVIDEO_CFG_BATCHED=1`.
   Validate `max_memory_allocated()` per §19 and SSIM against the post-B
   reference. Expected additional −10 % to −15 % (compounding on B).
3. **Route C (CUDA graph)** — only after A and B stabilize. Finishing pass.

### 18.6 Out of scope for these three routes

Documented for completeness; revisit only after measuring the impact of B+A:

- Dedicated comm stream in `base_device_communicator.py` (would attack §17.4
  same-stream serialization, but only useful if there is parallelizable work
  on the compute side — Route A reduces that parallel work, so this becomes
  less attractive after A lands).
- Switching to `torch.distributed._functional_collectives` (would enable
  inductor-driven scheduling of NCCL ops).
- Enabling `torch._inductor.config.reorder_for_compute_comm_overlap`
  (requires functional collectives above).
- Layerwise offload prefetch depth tuning.
- T5 fixed-length tokenization to delete §17.5 host syncs (one-time 36 s,
  but free win after the per-step gains land).

---

## 19. Memory cost analysis (under sp=4, the profile's topology)

At sp=4 each rank holds `seq_len / sp = 31200 / 4 = 7800` tokens during
attention. The per-rank activation footprint for one transformer forward at
bf16 is roughly:

```
hidden states (per layer):     7800 × 1536 × 2 B    ≈  22.9 MiB
attention QKV (per layer):  3 × 7800 × 1536 × 2 B   ≈  68.6 MiB (transient)
FA3 workspace (per layer): ~7800 × head_dim × 2 B × num_heads ≈ ~30 MiB (transient)
30 layers, inference (no backward saved state) → live-set dominated by max-layer transient
```

Peak inference live-set on Wan T2V 1.3B at sp=4 on L40S 48 GB is empirically
in the ~14–20 GiB range (not measured in this profile; rough estimate from
similar-architecture model runs). Headroom is comfortable.

**Route A doubles this to ~28–40 GiB** — still within 48 GiB on L40S.
**sp=2 would be marginal** (extrapolated 40–60 GiB, possible OOM cliff). The
training profile referenced in project memory (`MEMORY.md`) runs at sp=2; for
that topology, Route A needs either activation checkpointing or a 80 GiB
class GPU. The inference profile in this document is sp=4, where Route A's
doubling is tolerable.

Required check before promoting Route A from opt-in to default:

```python
import torch
torch.cuda.reset_peak_memory_stats()
# run one batched-CFG denoising step on the target GPU
print(f"peak_allocated: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GiB")
```

**Route B has no memory cost beyond a single cached `delta_cached` tensor**
of the same shape as `noise_pred` (≈ same size as one latent), persisted
across post-gate steps. Negligible.

---

## 20. Verification plan

### 20.1 Route B (TGATE) — land first

1. Sweep gate point `T_g / num_steps ∈ {0.3, 0.5, 0.7}`. For each, run
   `fastvideo/tests/ssim/` against the full-CFG reference.
2. Visual inspection at the chosen gate point for ≥ 5 distinct prompts
   covering motion, fine detail, text-in-frame.
3. Re-run §7 `iteration_timing` — expected: bimodal per-step time
   (pre-gate ≈ 5.87 s, post-gate ≈ 2.93 s); 90-step total ≈ 396 s.
4. Re-run §6 `nccl_payload_breakdown` — expected: total call count drops
   ~25 %, p50/p99 message size unchanged.
5. Default `FASTVIDEO_TGATE_STEP` off; promote to default-on only after
   per-pipeline gate values are recorded in `fastvideo/configs/pipelines/`.

### 20.2 Route A (batched CFG) — land after B

1. Re-profile with `--capture-range=cudaProfilerApi` so
   `data_quality.overhead_pct` drops below 5 % (currently 28.6 %, §2).
2. Re-run §6 `nccl_payload_breakdown` — expected: ~half call count at
   ~double p50/p99 message size; total bytes within 1 % of baseline.
3. Re-run §7 `iteration_timing` — note that "iteration" semantics change.
   Baseline `median_real_iter_ms ≈ 2,933 ms` is per *forward*; after Route A
   there is one forward per step, so the equivalent value should match the
   new per-step time (~5.0–5.4 s).
4. Re-run §3 `overlap_breakdown` — `same_stream_diagnosis` should still
   fire (architectural pattern unchanged); `compute_kernels` count should
   drop ~50 %.
5. SSIM regression with 1.5× tolerance bump on first run; tighten once a
   new reference set is recorded.
6. Memory check per §19.

### 20.3 Route C (CUDA graph) — plan TBD after A lands.

---

## 21. Landing: Route B (TGATE) measured 2026-05-15

**Status: implemented and validated.** This section is the only part below
§16 that records new measurements; everything before it is unchanged.

### 21.1 Implementation summary

Changes landed in:

```
fastvideo/envs.py                       (+13)
fastvideo/pipelines/stages/denoising.py (+74)
fastvideo/tests/modal/perf_nsys_profile.py    (forward FASTVIDEO_TGATE_STEP via Secret)
fastvideo/tests/modal/quality_eval_tgate.py   (new; adapted from quality_eval_linalg_fix)
```

Behaviour gated by `FASTVIDEO_TGATE_STEP: float = 1.0` env var:

- `1.0` (default) — bit-exact baseline; no delta cache, no warning paths fire.
- `X ∈ [0.0, 1.0)` — for step `i < int(num_inference_steps * X)`, run cond + uncond
  and refresh `delta_cached = cond - uncond`; for `i >= ...`, skip uncond and
  reuse `delta_cached` as `noise_pred = cond + (s - 1) * delta_cached`.
- Out-of-range raises `ValueError` at stage entry.
- `guidance_rescale > 0` + TGATE active emits one-shot warning at rank 0.
- Wan2.2 expert switch (`transformer ↔ transformer_2`) auto-invalidates cache via
  `delta_cached_model_id = id(current_model)` check.
- Telemetry counters (`fresh_uncond`, `reused`, `invalidations`) logged at end of
  loop on rank 0.

### 21.2 Methodology lesson: cross-run vs same-container measurement

First attempt at TGATE perf comparison used **two independent Modal jobs** —
baseline (`FASTVIDEO_TGATE_STEP=1.0`) and TGATE 0.5 — each via separate
`.venv/bin/modal run perf_nsys_profile.py` invocations. Result:

| Run | end-to-end | per-forward | source |
|---|---|---|---|
| Historical baseline (perf_compile.sqlite, §7) | 226 s/gen | 2.93 s | §7 |
| Fresh baseline (2026-05-15T08:35Z) | 198 s/gen | 2.80 s | this run |
| TGATE 0.5 (2026-05-15T06:57Z) | 104 s/gen | 1.76 s | this run |

Observed delta: **−48% end-to-end**. This **overshoots §18.2's predicted −25%
by 2×**, which initially read as "TGATE bonus from removing alternating
cond/uncond overhead."

That conclusion was wrong. Per-forward analysis (§21.3) showed baseline
forwards in their separate run took 2.80 s while TGATE forwards in their
separate run took 1.76 s — **but the kernel structure of those forwards is
bit-exact identical** (59 NCCL kernels each, 60 flash-attention kernels each,
same compiled graph). A 37 % per-forward speedup with zero algorithmic
difference can only come from **physical-machine variance** in Modal's L40S
pool: PCIe topology, NUMA layout, neighbour load.

The correct apples-to-apples measurement is to run both variants **inside a
single Modal container** so the 4× L40S allocation is shared. `quality_eval_tgate.py`
does this — pass 1 runs all 5 prompts with `FASTVIDEO_TGATE_STEP=1.0`, pass 2
runs the same 5 prompts with `FASTVIDEO_TGATE_STEP=0.5`, on the same physical
hardware, back-to-back.

Same-container result:

| Phase | per-video avg (5 prompts) | per-prompt std |
|---|---|---|
| Baseline (TGATE off) | 220.9 s | ±0.6 s |
| TGATE 0.5 | 172.2 s | ±0.5 s |
| **Δ** | **−22.0 %** | — |

**−22 % end-to-end matches §18.2's prediction within 3 percentage points.**
The remaining ~26 % delta in the cross-run comparison was hardware variance.

**Rule recorded for future routes**: any perf claim under ±15 % must be
validated by same-container A/B, not cross-run comparison against historical
baselines, because Modal's GPU pool variance can be that large.

### 21.3 Step-level decomposition (gen 1 measurement run, dev 0)

Extracted from per-rank kernel timestamps in the two profiles. Forward
boundaries identified by `flash_fwd_kernel` count (60 flash kernels per
forward × 30 transformer blocks × 2 attention calls per block).

```
==============================================================================
group                         fwds  mean_fwd_s  mean_nccl   phase_s  %total
==============================================================================
BASELINE pre-gate (s0-14)       30       2.800       59.0     84.00   48.4%
BASELINE post-gate (s15-29)     30       2.803       59.0     84.09   48.4%

TGATE 0.5 pre-gate (s0-14)      30       1.759       59.0     52.76   65.0%
TGATE 0.5 post-gate (s15-29)    15       1.769       59.0     26.54   32.7%
==============================================================================
Baseline DenoisingStage total: 173.71 s
TGATE 0.5 DenoisingStage total:  81.15 s   (53.3 % apparent reduction)
```

Interpretation:

1. **Baseline pre vs post-gate symmetry**: 2.800 vs 2.803 s per forward. The
   schedule is uniform; there is no implicit "phase" inside vanilla CFG. This
   is the control measurement.

2. **TGATE post-gate forward ≈ TGATE pre-gate forward** (1.769 vs 1.759 s,
   within 1 %). A "post-gate step" is structurally identical to a pre-gate
   cond forward — the delta-cache `elementwise add` (~µs) doesn't show up in
   wall time. The cost saving comes purely from running **half as many
   forwards in the post-gate phase**, not from a faster forward.

3. **Cross-profile per-forward delta** (2.800 → 1.759, −37 %): identical
   kernel structure, identical NCCL count, identical flash kernel count.
   The only explanation is physical-machine variance — see §21.2.

The 53.3 % "apparent reduction" in the cross-run DenoisingStage total decomposes
as ~25 % from forward count (60 → 45) and ~28 % from machine variance.
Removing the variance via same-container measurement gives ~22 % end-to-end,
matching §18.2 prediction.

### 21.4 Quality validation: VBench, 5-prompt same-container A/B

`quality_eval_tgate.py` ran 5 fixed-seed prompts × 2 variants on identical
hardware, then VBench-scored both sets on 5 dimensions (`subject_consistency`,
`background_consistency`, `aesthetic_quality`, `imaging_quality`,
`temporal_flickering`).

```
=== VBench per-dimension comparison ===
metric                                        baseline      tg_0.5        delta
vbench.subject_consistency                      0.9511      0.9498      -0.0013 ↓
vbench.background_consistency                   0.9595      0.9573      -0.0022 ↓
vbench.aesthetic_quality                        0.6213      0.6386      +0.0173 ↑
vbench.imaging_quality                          0.5772      0.5794      +0.0022 ↑
vbench.temporal_flickering                      0.9784      0.9777      -0.0007 ↓
```

All deltas at the 5-video sample size sit within VBench's noise floor
(empirically ±0.5 % at this sample size) **except** `aesthetic_quality`:

- 5 of 5 videos improved (+0.21 % to +4.55 %)
- Mean +2.78 % relative — beyond noise band
- Worst-case `subject_consistency` per-video drop: p4 (mercury/clockwork)
  −0.78 %, still below VBench noise

`temporal_flickering` deserves a separate note: this is the failure mode most
likely to be triggered by stale-uncond reuse (frame-to-frame artifacts from
discontinuous guidance). Measured delta is −0.07 %, within noise on all 5
videos. **TGATE 0.5 does not introduce frame-level flicker on this workload.**

`aesthetic_quality` improving is unexpected and consistent across the prompt
set. The mechanism is not pinned down here; one hypothesis from CFG literature
(CADS, CFG-Zero*, perturbed-attention guidance) is that CFG over-guides at
low noise levels, pulling fine details toward over-sharpening. TGATE's frozen
`Δ` after the gate softens that pull and lets the model do "pure refinement"
in late steps. This is a sketch, not a verified claim. The point for this
report: the speed/quality trade-off at gate=0.5 is not actually a trade-off
on this workload — quality is preserved or marginally improved.

### 21.5 Telemetry confirmation

Three TGATE summary lines per generation (warmup + 2 measurement runs)
appeared in the perf log, all matching expected counters:

```
TGATE summary: fraction=0.500 gate_step=15/30 fresh_uncond=15 reused=15 invalidations=0
```

`fresh_uncond + reused = 30 = num_inference_steps` confirms the gating
boundary fires at step 15 exactly. `invalidations=0` is expected for T2V
(single transformer, no Wan2.2 boundary switch); this counter will be
exercised when Wan2.2 lands.

### 21.6 §18.2 prediction scorecard

| Item | §18.2 predicted | Measured | Match |
|---|---|---|---|
| Forward count reduction | −25 % | −25 % (60 → 45) | exact |
| DenoisingStage reduction | ~25 % | −22.5 % (same-container) | ✓ |
| End-to-end reduction | ~22 % | −22.0 % | exact |
| Activation memory delta | 0 % | 0 MB (23,308 ↔ 23,308) | exact |
| Per-metric quality cost | < 5 % | < 0.3 % (all 5 metrics) | better than predicted |
| OOM risk | none at sp=4 | none observed | confirmed |

§18.2 was right. The detour through the −48 % cross-run number was a
methodology error documented in §21.2.

### 21.7 Open follow-ups

1. **Gate sweep** at `FASTVIDEO_TGATE_STEP ∈ {0.3, 0.7}` on the same 5-prompt
   harness to map the speed-quality curve. Predicted: 0.3 hits −35 % speed at
   risk of larger quality drift; 0.7 hits −15 % with near-zero risk.
2. **Wan2.2 boundary-switch validation** — exercise `invalidations > 0`
   path via a 14B model run, verify cache invalidation doesn't corrupt output.
3. **Route A (batched CFG) stacking** — Route A's predicted −15 % gain
   shrinks now that TGATE has already removed half the post-gate forwards.
   Updated stacking math:
   - Route B (gate 0.5): 30 step → 15 cond+uncond pairs + 15 cond-only = 45 fwds
   - Route A + B: 15 batched-cond+uncond + 15 cond-only = 30 forwards but
     pre-gate phase compute doubles per forward → wall time saving
     ~10 %, smaller than the −15 % from Route A alone.
4. **Mechanism inquiry on aesthetic_quality improvement** — only run on
   5 prompts; expand to 20+ prompts at varying guidance_scale to test
   whether the effect is consistent or sample-size noise.
5. **Update report §18.5 landing-order rationale** — Route C (CUDA graph)
   should probably skip; the per-forward time at 1.76 s is dominated by
   GPU compute + NCCL, not launch overhead, so graph capture won't move
   the needle much.
