# Chunked prefill (lab-prefill)

Owner: lab-prefill (`kernels/prefill/`, `models/flash_next/prefill.*`, `tests/prefill/`, `tools/prefill_*`, this file).
Status 15 Oct 2026: **the whole model prefills in chunks of up to C tokens (default 4096 since 16 Oct, section 4b; was 2048) as matrix work**, leaving
every cache / state exactly where n decode steps leave it (within the decode path's own 1-row-vs-4-row noise).
Numbers: M5 Max 128 GB, aligned pack `<packs>/Qwen3.8-Flash-Next-EXL3-K5-a16`, machine lock held, mtl4.

```
build/pf-check gemm PACK [time]                     NAX q8 GEMM vs CPU f64 (layer-0 weights), TFLOP/s
build/pf-check layers PACK TOKENS N K [C]           first K layers (< 15 GB): 1-row decode vs prefill, + the 4-row floor
build/pf-check lspeed PACK TOKENS N K [C] [profile] first K layers: prefill GPU ms, per-stage split
build/pf-check model PACK TOKENS N K [C]            whole model (~81 GB): decode-only vs prefill, states + logits KLD
build/pf-check speed PACK N [C] [profile]           whole model: prefill tok/s (NE_PF_TOKENS = token file)
build/pf-check attn PACK T                          lab-attn's QSA step at 8 rows, per dispatch
$PY tools/prefill_tokens.py OUT N [files]           real-text token ids (the pack's tokenizer) for the checks
```
(build with `make BUILD=build-prefill`; run GPU modes only under `LOCK_OWNER=lab-prefill tools/with_lock.sh`, whole
model with `NEED_GB=85`.)

## 1. Interface

`models/flash_next/prefill.h`: `PrefillPtr prefill_create(Model &m, int chunk, bool profile = false)`;
`prefill_tokens(p, tokens, n)` (positions m.T .. m.T + n - 1; advances m.T, the n-gram history and every state; no
head); `prefill_prompt(p, prompt, n)` = `prefill_tokens(prompt, n - 1)` + the decode step with the head on the last
token (its logits / sample, exactly the decode path's head). C API: `ne_fn_prefill_create(model, chunk, profile)`,
`ne_fn_prefill_tokens`, `ne_fn_prefill_prompt` (+ stats: wall, GPU, gather ms, chunks, 8 stage ms). The decode
path is untouched: after a prefill, `step()` / `generate()` continue from the states. `prefill_tokens` first runs
`gdn_materialise(m.st)` (lab-gdnlazy: decode steps before it may have deferred GDN state stores, docs/kernels.md 5d;
`pf_gdn_scan` reads ssm directly and leaves no pending rows).

One recording per chunk length n (cached; chunks containing rows with position <= 2050 are keyed by (n, T0) as
well, because their dense-attention split is recorded). Dynamic data (tokens, T0, the gathered PLE rows) go into
GPU buffers before each commit. Buffers ~0.55 MB per chunk token (C = 2048: ~1.2 GB, 16 attention-slice scratch
sets ~0.3 GB).

## 2. Pipeline per chunk (n rows)

| stage | kernels | notes |
|---|---|---|
| embed | `fn_embed_g64` (lab-forward's, n threadgroups) | |
| PLE (layer 1) | host: 16 n-gram ids per token, parallel `pread` of all n x 16 rows (overlapped with the previous chunk's GPU work); `pf_f2bf`, key / value GEMMs, `pf_ple_gate` (per row x stream, fn_ple_mix's math), `pf_ple_conv` (all rows at once), `pf_ple_state` | |
| HC read | `pf_hc_norm` (per row: stream rms in fn_hc_pre's order, normed bf16, inject gates in fn_hc_pre's order), down GEMM, `pf_hc_act`, up GEMM, `pf_hc_blockin` | since 16 Oct (`NE_PF_HCF=0` = before): the mixer's HC write is done by the MoE-side read's norm (`pf_hc_norm_wr`), `pf_hc_act` is the down GEMM's epilogue (`pf_q8_bb_g*_act`); bit-identical |
| trunk GEMMs | `pf_q8_*` (NAX, 64x64 tiles, half weight tiles dequantised from the MLX 8-bit layout in place, f32 accumulate) | GDN qkv / z / b / a, out_proj; attention q / k / v / indexer (f32 out), o_proj (f32 in); HC down / up; shared expert; PLE |
| GDN | `pf_gdn_conv` (every row at once from [state; raw rows]), `pf_gdn_gates`, `pf_gdn_convstate`, `pf_gdn_qknorm`, `pf_gdn_scan_pipe`, `pf_gdn_gnorm` | since 16 Oct the scan is Sushi's / oMLX's pipelined recurrence (8 lanes per value row, 12-token blocks staged, 4 q.S reductions batched; section 4b); `NE_PF_SCAN=tb` = the decode-order `pf_gdn_scan_tb` (same lane layout and op order as `fn_gdn_step`) |
| attention | lab-attn's `attn::record_chunk` (docs/attn.md section 8; with tests/attn/prefill_chunk.patch applied): one prep for every row (cache append, indexer queries, raw keys), the chunk's block keys, scores / select_local / select over every sparse row (one dispatch each); dense rows (positions <= 2050): `pf_attn_qcopy` + MLX NAX flash attention (`pf_attn_nax_d256`) + `pf_attn_gate`; the other rows: `qsa_chunk_attn_bf16`, one (row, kv group) per threadgroup on the NAX (no split-K / merge). Before the patch: lab-attn's step per 8-row slice, 16 slices concurrently | KV format = lab-attn's (bf16 default; kv8 / f32 and `NE_QSA_CHUNK=0`: every slice through lab-attn's step) |
| MoE (expert GEMM planes f16 since 16 Oct, `NE_PF_MOEH=0` = f32) | router `pf_bf16_bb` GEMM, `pf_moe_route` (fn_moe_router's top-10 + shared gate logit), shared gate/up GEMMs, `pf_shared_act`, shared down GEMM; GPU sort: `pf_moe_count` (per expert: count + rank in slot order), `pf_moe_scan` (offsets, <= 32-row windows), `pf_moe_place`; `pf_moe_prep`, `pf_moe_gemm_gu` x 2, `pf_moe_mid`, `pf_moe_gemm_dn`, `pf_moe_finish` (fixed slot order + gated shared + HC write) | no host sync anywhere in a chunk |

## 3. Correctness

Reference = the decode path itself: the same tokens fed one row per step (`step()`, lab-forward's 1-row kernels).
"Floor" = the decode path's own spread: the same tokens through lab-forward's 4-row prefill (`fn::prefill`, the
R = 4 decode kernels: same math, different staging precision / reduction order) vs the 1-row steps. Prompts: real
text (this repo's docs and sources, `tools/prefill_tokens.py`), chunk 2048, QSA bf16 KV, reference rounding ON.

**Whole model** (`pf-check model`; chunked prefill of N - 1 tokens + the decode step on token N - 1, then 32
teacher-forced decode steps; KL(decode-only || prefill) of the f32 log-softmax per position over the 33 positions;
states = rel L2 of every layer's state after N tokens, worst layer):

| N | path B | KLD mean | KLD max | top-1 | worst GDN ssm / conv | worst KV (k, v) |
|---|---|---|---|---|---|---|
| 64 | chunked prefill | 7.1e-2 | 0.46 | 28/33 | 0.19 / 0.14 | 0.18 |
| 64 | floor (4-row decode) | 5.1e-2 | 0.17 | 29/33 | 0.16 / 0.11 | 0.15 |
| 2048 | chunked prefill | 3.0e-2 | 0.18 | 29/33 | 0.18 / 0.22 | 0.14 |
| 2048 | floor (4-row decode) | 3.2e-2 | 0.19 | 29/33 | 0.18 / 0.20 | 0.14 |
| 8192 | chunked prefill | 2.6e-2 | 0.13 | 27/33 | 0.16 / 0.09 | 0.13 |
| 8192 | floor (4-row decode) | 2.8e-2 | 0.17 | 26/33 | 0.16 / 0.12 | 0.14 |

The chunked prefill is at the decode path's own floor at every length (8K: QSA sparse for rows past 2050, both
attention paths of the prefill exercised). The floor itself is large: a bf16-ulp difference in any rounding point
grows to ~15% rel in the late layers' states and ~3e-2 nats of logits KLD at this model's depth (the MLX
reference's own prefill-vs-decode spread was 2.4e-3 on lab-forward's 4 short prompts: a different measure, the
reference's GEMMs; what matters here is that prefill adds nothing beyond the decode family's spread).

**Per layer** (`pf-check layers`, first 4 layers = GDN, GDN + PLE, GDN, attention; rel L2 vs the 1-row decode):

| n tokens | hidden after layer 3: median / max over rows | L00 ssm | L01 ssm / PLE conv | L03 k / v / ikeys / bkeys | floor (4-row) hidden median, L03 k |
|---|---|---|---|---|---|
| 64 | 8.1e-3 / 7.9e-2 | 2.2e-3 | 6.6e-3 / 4.0e-3 | 1.7e-2 / 2.2e-2 / 1.2e-2 / 8.2e-3 | 6.9e-3, 1.4e-2 |
| 2048 | 6.3e-3 / 7.9e-2 | 2.8e-3 | 9.5e-3 / 3.3e-3 | 1.3e-2 / 1.5e-2 / 1.0e-2 / 7.4e-3 | 5.5e-3, 1.2e-2 |
| 8192 (last chunk's rows) | 6.0e-3 / 4.9e-2 | 2.4e-3 | 8.9e-3 / 5.2e-3 | 1.2e-2 / 1.4e-2 / 9.6e-3 / 7.0e-3 | 5.3e-3, 1.1e-2 |

Layer 0 alone: hidden 4.1e-3 median (decode vs the MLX golden chain: 5.7e-3). NAX q8 GEMM vs CPU f64 on the real
weights: 1.9e-4 - 2.0e-4 rel L2 (half weight tiles; MLX's bf16 tiles: 1.2e-3). Results are bit-identical run to run
(fixed-order reductions everywhere; the GPU sort ranks slots in slot order).

## 4. Speed

Whole model (`pf-check speed`, real-text tokens from position 0, chunk 2048, after a warm-up chunk; wall = host
time of `prefill_tokens`, incl. the n-gram gather of the first chunk; 15 Oct, lock held, nothing else loaded):

| prompt | wall | **tok/s** | GPU | chunks | vs Sushi (served path, docs/qwen4-perf.md) | vs oMLX port |
|---|---|---|---|---|---|---|
| 2048 | 1.01 s | **2028** | 0.97 s | 1 | Sushi ~1900-2100 at 2K: par | |
| 10240 | 6.25 s | **1637** | 6.13 s | 5 | Sushi ~1,900-2,100 at 10K (4bpw pipelined-GDN build: 2534 at 8K): 0.78-0.86x | 1649: par |
| 32768 | 21.6 s | **1516** | 21.5 s | 16 | Sushi-4bpw 2384 at 32K (TTFT ~13.7 s): 0.64x | |

(The model check's runs agree: 2047 tokens 1976 tok/s, 8191 tokens 1569 tok/s. The decode path's own prefill
options for comparison: 1-row steps 73 tok/s, lab-forward's 4-row steps 158 tok/s at 2K.)

Per-stage GPU time per 2048-token chunk, 10K prompt (profile mode: every stage its own command buffer; 6.10 s GPU
vs 6.13 unprofiled):

| stage | ms per chunk | share | what |
|---|---|---|---|
| MoE expert GEMMs (gate, up, down; NAX trellis decode) | 333 | 27% | 6.9 ms per layer: decode-bound (each <= 32-row window re-decodes its expert's tiles); 64-row windows measured 10% slower |
| attention core | 314 | 26% | per layer: dense rows (first 2051 positions) 3 ms per chunk on the NAX flash kernel; sparse rows ~30 ms per chunk on lab-attn's 8-row step (qsa_attn 86 + merge 17 us per 8 rows, flat in T) |
| trunk GEMMs (q8 NAX, ~48-53 TFLOP/s) | 297 | 24% | 6.2 ms per layer: GDN qkv 2.0 + z 1.2 + out 1.3, HC down / up 2 x 0.55, attention q 2.4 |
| GDN (conv, gates, q/k norm, scan, gated norm) | 102 | 8% | 2.8 ms per GDN layer (scan ~2 ms: the recurrence's latency chain) |
| MoE glue (router GEMM, routing, shared expert, sort, prep / mid / finish) | 102 | 8% | 2.1 ms per layer |
| HC glue (norms, act, block_in, writes) | 68 | 6% | bandwidth: 4 passes over the 84 MB hidden per layer |
| PLE (layer 1) + embedding | 4 | 0.1% | gather (CPU, ~18 ms per 1K tokens) overlaps the previous chunk |

Per-layer view at 2K (4-layer partial model): GDN layer 18.6 ms, attention layer 19.1 ms (dense), so 2K is ~48 x
19 ms = 0.9 s. At 32K every chunk after the first is sparse: attention layer ~44 ms -> 1.34 s per chunk.

Kernel-level: NAX q8 GEMM [10240 x 2560] at M = 2048: 2.06 ms = 52 TFLOP/s; [2560 x 6144]: 48; HC down
[320 x 10240]: 45 (M = 512: 15, too few column tiles). lab-attn step at 8 rows: prep 4.7, score 6-10, select_local
1-9, select 8-9, attn 86, merge 17 us (T = 2K / 8K / 32K).

(16 Oct, lab-attn2: with the multi-row core the attention per 10K chunk is 106 ms (10.1%), 137 ms at 32K (12.4%);
whole prefill 2,019 / 1,851 tok/s at 10K / 32K: docs/attn.md section 8. The union-tile premise of lever (1) does not
hold on real text: 8 neighbouring rows' selections cover 2.4-3.1x of 512 blocks; the core reads each row's own blocks
once per kv group instead.)

Levers, in order: (1) a multi-row sparse attention core (rows of a chunk share most selected blocks: a 64-row tile
over the union with per-row masks would read K/V once per tile instead of once per row; -250 ms per chunk at 10K,
~-30% of a long-context chunk); (2) MoE windows: decode each expert's tile once per projection for all its rows
(e.g. decode to f16 in threadgroup memory and loop the windows, or a 2-pass decode-then-GEMM for experts with > 32
rows); (3) fuse the HC glue into the GEMM prologues / epilogues; (4) GDN scan: chunked delta rule (intra-chunk
matmuls) instead of the token-serial chain.

### 4b. Low-risk levers from docs/research-prefill.md (lab-pflevers, 16 Oct)

K5 aligned pack, real text (`tok33k.bin`, this repo's docs + sources), `pf-check speed` wall tok/s, one with_lock job per
round, arms interleaved in the job (the box drifts 5-15% between runs: compare within a job only). 4-layer stage times
from `pf-check lspeed PACK TOKENS 4096 4 2048 1` (2 chunks, 8 layer-chunks, two reps per arm, identical to 0.1 ms).

**Result (new defaults vs all four off at chunk 2048, same binary, interleaved):** 10K 1,935 -> **2,093** tok/s (medians
of 5 pairs; +8%), 32K ~1,851 -> **~1,991** (4 pairs: 1,843 / 1,872 / 1,823 / 1,859 -> 2,014 / 1,984 / 1,998 / 1,984; +7.6%,
TTFT 17.7 -> 16.5 s). T6 pack 10K: 1,917 -> 2,082 (one pair). Sushi on K5 at ~33K: 1,775-1,933 (its default 4096).

| # | lever | indicative A/B | verdict |
|---|---|---|---|
| 1 | chunk 4096 (`kChunk`, model.h; server pieces 4097) | 32K all-on: 2048 1,856 / 1,861 -> 4096 1,938 / 1,936 (+4.2%); 10K 2,027 -> 2,093 (medians of 3; +3%). Profile at 32K: moe_gemm 342 -> 306 ms per 2048 tokens, other stages flat | **kept** (default). Memory: +~1.1 GB chunk buffers, attention scores 4 B x C x tmax/4 (+0.5 GB at 262K) |
| 1' | chunk 8192 | 32K 1,931 / 1,930 (= 4096); 10K 2,020 vs 2,113 (-4%: 2 uneven chunks); moe_gemm 272 ms per 2048 tokens but trunk / GDN / attention per token rose | dropped (`prefill_create(m, 8192)` works; no gain over 4096) |
| 2 | pipelined GDN scan `pf_gdn_scan_pipe` (Sushi `gdn_pipelined.metal` <- oMLX `gated_delta_pipelined`) | gdn stage 16.1 -> 14.9 ms (-7.5%, ~-0.2 ms per GDN layer-chunk, ~-7 ms per 2048 chunk); whole model within noise (10K 2,020 vs 2,004, 32K 1,961 vs 1,964). The research doc's prediction (scan <= 1.1 ms, -35..-45 ms) is falsified: the scan is not latency-chain bound here | **kept** (small, default; `NE_PF_SCAN=tb` = decode op order). Layers check: hidden 6.17e-3 vs 6.22e-3, L02 ssm 1.35e-2 vs 1.38e-2 |
| 3 | fused HC: write folded into the next norm (`pf_hc_norm_wr`), SiLU in the HC down GEMM epilogue (`pf_q8_bb_g*_act`, q8 trunk only; T6 keeps `pf_hc_act`) | hc_glue 11.1 -> 9.37 ms (-16%), 311 -> 287 dispatches; ~-10 ms per 2048 chunk (prediction -20..-30: the up-GEMM epilogue (gate x normed x mean) not done) | **kept** (default; `NE_PF_HCF=0` off). Bit-identical (layers check equal to every printed digit) |
| 4 | f16 expert GEMM output planes (`pf_moe_gemm_*_h`, `pf_moe_mid_h`, `pf_moe_finish_h`; Sushi's `nax_store` to half) | moe_glue 16.2 -> 13.6 ms, moe_gemm 54.7 -> 53.4 (4 layers); whole model at 4096: 10K 2,195 / 1,931 -> 2,263 / 2,005, 32K 1,908 / 1,922 -> 1,998 / 1,984 (+3..+4.7%, 4 of 4 pairs) | **kept** (default; `NE_PF_MOEH=0` off), see the KLD note below: the one lever that adds rounding |

**Correctness** (`pf-check model`, decode-only vs prefill, 32 teacher-forced positions, same prompt):

| N, chunk | new defaults | all four off | 4-row decode floor |
|---|---|---|---|
| 2048 (one chunk) | KLD mean 5.66e-2, max 0.62, top-1 32/33 | 3.54e-2, 0.54, 31/33 | 4.92e-2, 0.66, 30/33 |
| 2048, single levers | pipe + HCF (MOEH 0): 2.76e-2; HCF + MOEH (scan tb): 4.66e-2 | | |
| 8192 (C 4096 vs 2048) | 5.12e-2, max 1.10, 32/33, worst state 0.163 | 5.52e-2, 1.24, 32/33, 0.158 (= lab-attn2's) | |

At the floor (docs/forward.md 2c: any two valid arithmetics differ by 2e-2..9e-2 on teacher-forced repo text), but
MOEH moved the 2K mean up both times it was added (2.8 -> 5.7e-2, 3.5 -> 4.7e-2): **a teacher-KLD run (test-quality
A/B/C on prefilled contexts) should confirm it at the merge**; `NE_PF_MOEH=0` reverts it (-3..-4.7%).

Not tried here: A8 trunk GEMMs (quality risk, out of scope), the HC up-GEMM epilogue (gate / product / stream mean,
needs the 16 x 4-stream column regrouping: ~-10 ms per chunk left), 48-row windows, bf16 HC stream.

### 4c. Small chunks (lab-ttft, 17 Oct): what a 16-512-row chunk costs at long context

Server turns feed 20-500 new tokens per chunk (docs/server.md 3e). Profile mode, whole model, T6 pack, fresh real-text
tokens (synthetic repeated tokens touch far fewer experts: moe_gemm 45 vs 71 ms at 64 rows), GPU ms per chunk:

| rows | T | total | trunk GEMM | expert GEMM | MoE glue | attention | GDN | HC glue |
|---|---|---|---|---|---|---|---|---|
| 16 | 33K | 121 | 61 | 35 | 12 | 5 | 3 | 4 |
| 64 | 33K | 151 | 54 | 71 | 11 | 6 | 3 | 4 |
| 128 | 33K | 183 | 58 | 90 | 13 | 11 | 6 | 5 |
| 512 | 33K | 354 | 104 | 147 | 25 | 38 | 23 | 16 |
| 128 | 161K | 159 (synthetic) | 59 | 56 | 12 | 20 | 6 | 5 |
| 512 | 161K | 340 (synthetic) | 102 | 106 | 25 | 69 | 22 | 15 |

Removal experiments on a 4-layer model (`NE_PF_SKIP=<Linear name parts>`, `NE_T6_PFDBG=nodense|nogemm`: wrong outputs,
timing only), 16 rows: T6 trunk 4.84 ms per 4 layers = prep / post 0.2 + dense W' decode 1.0 + **GEMM 3.6**; HC down
(10240 -> 320, both HC reads) alone 1.4. lab-prefill's NAX GEMM gives a Linear (NP / 64) x ceil(M / 64) threadgroups,
each over the whole K: 1-160 per trunk Linear at M <= 64, latency-bound. Chunk-capacity-sized grids (the expert GEMMs'
maxwin = 1536 window rows at C = 4096) cost nothing measurable (C = 128 / 512 / 4096 alike).

**Split-K for <= 32 rows** (`kernels/trunk6/t6_splitk.metal`, `t6::record_prefill`; `NE_T6_SK` = max rows, 0 = off, <= 128):
after t6_dense, `t6_pf_sk{2,4,8,16}` (8 MT rows; simdgroup = 8 columns x all rows, x staged per 32-wide k block, W'
read once) over S k-slices (~1024 threadgroups), partials f32 in `PfBuf::part` (32 MB) summed in slice order by
`t6_pf_skr` (deterministic), then t6_pf_post unchanged. 4 layers, trunk ms per chunk, NAX -> split-K: 16 rows 5.02 ->
2.77, 32: 4.14 -> 3.24, 48: 4.35 -> 4.77, 64: 4.38 -> 4.85, 128: 4.93 -> 7.50 (simdgroup MMA compute-bound from ~48
rows: NAX keeps those). Whole model: 20-row chunk 125-131 -> 104-113 ms, 32 rows 127-133 -> 119-124, 45 rows (NAX
both) unchanged. Numerics: f32 reduction order only (4-layer logits KLD 1.6e-5 .. 6.4e-5; whole model docs/server.md 3e).
The kernels are self-contained (metal_stdlib): a build without the Metal toolchain compiles the file at run time
(`Pipelines::add_source`) when the metallib lacks `t6_pf_sk2`.

Recording a new chunk length costs ~2.5 ms on the host (3,481 dispatches; `NE_PF_RECLOG=1` logs every recording and each
prefill_tokens call's wall / GPU / gather / recording ms); the n-gram gather of a turn's chunk 5-16 ms when cold.

### 4d. kernels, Oct 18 (pfkern): where a chunk's GPU time goes, and the trunk on the tensor ops

T6 pack, chunk 4096, real text (`tok33k.bin`), one `tools/with_lock.sh` job per table; the box drifts 5-15% between
jobs (a 32K prefill also heats it: the same qkv GEMM costs 137 ms per chunk at 8K and 162 at 32K), so compare arms
within a job only.

**Profile tool.** `NE_PF_PROF=fine build/pf-check speed|lspeed ... 1`: one command buffer per kernel label (each T6
Linear split into prep / dense / gemm / post, or dense / fold / gemm), GPU ms per label and per chunk (GPU / attention).
`NE_PF_MOEDBG=nodec|nomma` (timing only, wrong outputs): the expert GEMM without the trellis decode / without the MMA.

**Profile before (master build, whole model, GPU ms per 4096-row chunk; fine mode, 1-7% above plain):**

| group | 8K (2 chunks) | 32K (8 chunks) | notes |
|---|---|---|---|
| trunk GEMMs, MMA (`*/gemm`, pf_h16_hf) | 565 | 654 | qkv 138 / 162, out_proj 97 / 105, z 82 / 100, HC down 65 / 69, attn q 54 / 64, HC up 47 / 55 |
| trunk prep + post (x * suh, H; H, svh) | 179 | 188 | HC up post 52 / 54, HC down prep 37 / 39, qkv post 18 / 19 |
| trunk trellis decode (`t6_dense`) | 15 | 16 | 2% of the trunk: the dense decode is amortised over 4096 rows |
| MoE expert GEMMs (gate, up, down) | 560 | 670 | 4-layer split: fused 89 ms, without decode 57, without MMA 57 (decode and MMA each ~64% alone; they overlap badly) |
| MoE glue | 189 | 208 | finish 53 / 56, prep 45 / 48, shared gate/up/down GEMMs 35 / 43, mid 14, router 10-12, sort 9-11 |
| attention core | 192 | 304 | per chunk 143 -> 240 (8K); 158 -> 367 (32K, chunk 8) |
| GDN | 195 | 237 | scan 156 / 192 (ALU-issue bound: 32 or 64 value rows per threadgroup instead of 16 -- q / k staged once per 32 / 64 rows, bit-identical -- 12.0 / 12.1 / 13.4 ms per 4-layer chunk: no gain, dropped) |
| HC glue | 108 | 113 | norm 72 / 76, block_in 35 / 36 |
| total | 1,955 | 2,333 | plain run: 2,180 / 1,920 tok/s |

**Tensor-op microbenchmarks** (M5 Max; scratch harness, random data; matmul2d per simdgroup, operands in registers
unless noted): f16 16x32x16 (our GEMMs' and MLX's NAX shape) 55-58 TFLOP/s; f16 with K = 32 (16x32x32, 32x32x32)
64-66; int8 16x32x16 65 TOP/s; **int8 with K = 32: 118-122 TOP/s**. Whole GEMMs with the operands as MPP tensors (the op
loads them): f16 x f16 128 x 64 tiles 62-65 TFLOP/s (64 x 64 over 8 simdgroups the same in isolation but 1.5-2x slower
in the model); bf16 x f16 65 (same rate); f32 x f16 16; int8 x int8 with a per-128-block scale epilogue ~100 TOP/s
(cooperative-tensor fragment loads instead: 64-70). In the model the expert GEMM on 16x32x32 ops was 55% slower
(register pressure; dropped).

**What changed (kept):**
1. **Folded W** (default; `NE_T6_FOLD=0` = before): per chunk and EXL3 Linear, `t6_dense` (W' -> half), `t6_fold_k`
   (rows: H_hi, * suh) and `t6_fold_n` (columns, through threadgroup memory: H_ho, * svh) build the dense
   Wf = diag(svh) H_ho W'^T H_hi diag(suh); `t6_fold_gemm` (bf16 x on the tensor op, Wf f16, y stored by the op:
   bf16 round-to-nearest-even, bit-identical to bf16r) writes y directly. No prep / post pass over the activations;
   f32-input Linears (attention o_proj) and < 64 rows keep the old path. 4-layer trunk stage 104 -> 93 ms (2 chunks).
2. **f16 tensor-op GEMM** `t6_h16_gemm` (128 x 64 tiles) for the remaining prep / post Linears (`NE_T6_TG=0` =
   pf_h16_hf); bit-identical outputs (prefill-path KLD equal to every digit).
3. **INT8 trunk, opt-in** (`NE_T6_A8=128` = chunks of >= 128 rows; `NE_T6_A8_SKIP=<name parts>`): `t6_pf_prep_a8`
   (H_hi(x * suh) -> int8, absmax per (row, 128-block)), `t6_dense_a8` (W' -> int8 per (output row, 128-block)),
   `t6_a8_gemm`, `t6_pf_post`; Linears with in >= 1024 only (the HC up Linear, 320 -> 10240 into the HC gates, carried
   most of the error: 4-layer hidden rel L2 vs decode 2.1e-2 with it, 9.4e-3 without, f16 4.8e-3). Idea from oMLX's
   oQ A8 (`omlx/patches/qwen35_oq_a8.py`, Apache-2.0; no code copied).

**Speed** (`pf-check speed`, whole model, one job, arms interleaved, tok/s):

| | 8K | 32K |
|---|---|---|
| before (`NE_T6_TG=0 NE_T6_FOLD=0`) | 1,876 / 1,880 | 1,807 / 1,815 |
| **new default (fold + tensor GEMM)** | **2,102 / 2,036** (+10%) | **1,971 / 1,992** (+9.3%) |
| + `NE_T6_A8=128` | 2,138 / 2,159 (+14%) | 2,072 / 2,074 (+14.5%) |

(A cooler job earlier: 32K before 1,967, tensor GEMM alone 2,017, A8 2,164 / 2,168.)

**Same client** (the same-client head-to-head harness, not in this repository: smoke, warm, prefill phases, REPS 3, port 8143, master's
harness with IJAH_ROOT, back to back, medians): master 10K **1,698** / 32K **1,624**; this branch **1,819 / 1,792**
(+7% / +10%). A8 in an earlier, hotter pair: 1,712 / 1,676 against the default's 1,759 / 1,589 in the same pair. Sushi
2,001 / 2,030, oMLX 2,260 / 2,808 (18 Oct night). The server's prefill (19.9-20.7 s at 32K for the default) still sits
~3 s above pf-check's 16.5 s for the same tokens: pfpath's part.

**Quality, prefill path** (`kld_teacher.py --prefill`: B 4 windows, prefill 1024; C windows 0-1, prefill 16384,
score 2K; paired vs this branch with all off):

| arm | B | C |
|---|---|---|
| off (f16 prep / post) | 0.00182 | 0.00502 |
| tensor GEMM | | 0.00502 (+0, bit-identical) |
| **fold (default)** | 0.00186 (+0.00004 [+0.00001, +0.00007]) | 0.00487 (-0.00015 [-0.00038, +0.00008]) |
| A8, every Linear | 0.00190 (+0.00008 [+0.00003, +0.00012]) | 0.00534 (+0.00032 [+0.00015, +0.00049]) |
| A8, in >= 1024 | 0.00184 (+0.00002 [-0.00001, +0.00005]) | 0.00585 (+0.00083 [-0.00007, +0.00173]) |

The decode-only control is unaffected (these kernels run only in the chunked prefill); decode-only C on these windows
was 0.00589 (17 Oct). A8 stays off until a larger C sample (>= 8 windows) shows it within noise.

**Next leads (numbers above):** the MoE expert GEMM (29%: decode and MMA each ~57 of 89 ms alone, so better overlap or
a decode shared by more rows, e.g. 64-row windows with the decoded tile in threadgroup memory, could take up to ~1/3
off); the fold's own cost (fold_k + fold_n ~2.3 ms per qkv per chunk of 3 layers: fuse into the dense decode); HC glue
(norm + block_in ~110 ms per chunk: fuse block_in into the up GEMM's epilogue); attention at long context (158 ->
367 ms per chunk from 4K to 32K).

### 4e. Server path (lab-pfpath, 18 Oct): a cold prompt through the server, end to end

The same-client harness (not in this repository; cold prompt = new text behind a nonce, thinking off, 16 tokens out,
sampled with a seed: the step path) measured Ijah 10K 1,674 / 32K 1,644 tok/s against pf-check's ~2,000 for the same
engine: the gap was the request path, not the kernels. Instrumented (no behaviour change; kept, env-gated):
`NE_SRV_TRACE=1` (server: a timestamped line per step -- POST, body, render + encode, admission, restore, every
prefill piece, checkpoint, first token, first SSE write), `NE_FEED_LOG=1` (engine: per chunk of an MTP feed: prefill
wall / GPU / n-gram gather / recording, the MTP head's rows, head catch-up ms), `NE_PF_RECLOG=1` (prefill_tokens).
In-process (the server's Template + Session + RealBackend, no socket): `tools/bench_cold.py` (arms interleaved in one
process: `env:K=V`, `B:NAME=V`, `noahead`; `--idle S`, `--ping`).

**Attribution, master 9245921 (T6, chunk 4096; the harness's server, traced; ms):**

| | 10K rep 0 (after the 3 s phase gap) | 10K rep 1 | 32K rep 0 |
|---|---|---|---|
| TTFT at the server (POST -> first SSE write) | 6,888 | 6,024 | 19,923 |
| HTTP body + template + tokenize | 12 | 11 | 30 |
| restore (park the previous conversation's KV rows) | 13 | 27 | 17 |
| chunk GPU (pf GPU time) | 4,772 | 4,883 | 16,724 (83.9%) |
| n-gram gather of every chunk (CPU, before its commit; 16 x 320 B random reads per token, SSD-bound) | 804 | 660 | 1,942 (9.7%) |
| MTP head catch-up (KV-only 8-row head steps after each chunk, ~29 us per row) + row copies | 314 | 323 | 1,017 (5.1%) |
| first command buffer after idle (see below) | ~860 | 0 | 0 |
| tail: < 16-token rest as steps, 1-token piece + 2 checkpoints, generation prompt steps | 89 | 84 | 155 |

**Fixes (one indicative in-process A/B each on 9245921's kernels, `tools/bench_cold.py`, tok/s = prompt / TTFT):**

1. **n-gram rows read ahead** (`prefill_tokens(.., ahead)`, `ne_fn_mtp_feed_ahead`; the server passes the next
   piece): the next call's first chunk is gathered while this call's last chunk runs (prefill_tokens overlapped
   chunks only within one call; the server feeds 4096-token pieces, one call each, so nothing overlapped). Checked
   against the history and tokens of the next call, so the rows are the same either way. 10K 1,679 -> 1,787, 32K
   1,590 -> 1,681. The overlapped reads cost the GPU ~2% (17.6 -> 18.0 s at 32K; 4 reader threads at utility QoS
   instead: slower, dropped). Off: `NE_SRV_AHEAD=0` / `NE_PF_AHEAD=0`.
2. **The MTP head's KV rows of a chunk in one batched pass** (`prefill_head_kv`: fuse_pre over the chunk's layer-47
   rows, fc_embedding / fc_hidden pf_q8 GEMMs, + the embedding row (bf16) via pf_hc_write, the head layer's HC read,
   k / v / indexer GEMMs, `attn::record_chunk_kv` = the chunk prep alone; ~16-20 ms per 4096 rows instead of ~120):
   10K 1,734 -> 1,850, 32K 1,770 -> 1,856. Recordings by (head cache, rows); also with `NE_MTP_HEAD_KV=full` (the
   test's full-layer head arm must match the KV-only one). `tools/check_headkv.py` (8K feed): target KV rows identical, head KV rel L2
   8.4e-4 vs the 8-row steps (79% of the bytes identical; the target's own chunked-vs-decode KV spread is ~1.4e-2),
   MTP greedy d2: 256 / 256 tokens identical, 1.656 tokens per round in both. Rows queued before a chunk (decode
   steps, the previous feed's last row) are drained first (8-row steps + 1-row tail), so the cache stays in order;
   the call's last row waits for its token as before. Off: `NE_MTP_HEAD_CHUNK=0`.
3. **GPU keep-warm while idle** (`ne_fn_gpu_ping` + `ne_fn_mtp_touch`, the engine thread every 0.5 s for 15 min
   after the last request: `NE_GPU_WARM_S`, `NE_GPU_WARM_EVERY`; 0 = off). After >= ~1 s without GPU work the next
   command buffer waits 0.6-0.7 s before it starts (a trivial one too: measured 565-732 ms after 4 s idle), and a
   recording's first replay ~0.1-0.15 s more (its residency set). Every request after a pause paid it -- the
   harness's first request of each phase, and in an agent client every turn after the user or a tool took a second. A trivial
   command buffer every 0.5 s removes the first part, `requestResidency` on every recording's set the second (an empty
   command buffer using the set does not): after 4 s idle, 4096-token chunk wall - GPU - gather 550-850 ms -> 2-5 ms.
4. **Short piece first** (`pieces(rest_first)`, a request's first segment; `NE_SRV_REST_FIRST=0` = last): the first
   chunk's gather is the only one the GPU waits for; with the rest (>= 16 tokens) first it is ~half the rows at 10K
   (gather 146-181 -> 71-89 ms), same number of pieces. **64 reader threads** for that gather (`NE_PF_GATHER_THREADS`;
   the SSD needs queue depth: 65K random 320 B reads per 4096 tokens, 280 -> 212 ms at 64 threads in isolation).
5. **No duplicate checkpoint**: a prompt whose first message is the only one (the end of that message and the
   generation prompt's `<|im_start|>` one `\n` apart) keeps only the second (one 1-token step piece and one 113 MB
   state export less, ~25-30 ms).

Dropped: the head catch-up on a second Metal queue next to the next chunk (the GPU does not co-schedule them: each
128-row head command buffer waited 0.4-2.6 s for the chunk, and the chunk got slower).

**Same client before / after** (`session.sh ijah ... smoke,warm,prefill default`, REPS 5, one with_lock job, back to
back, 18 Oct 10:17-10:21; prefill tok/s = prompt tokens / TTFT, mean +- SD):

| | 10K | 32K |
|---|---|---|
| master 9245921 (18 Oct night, the problem statement) | 1,674 +- 59 | 1,644 +- 16 |
| master 95de2a9 (+ pfkern's kernels, 4d) | 1,770 +- 122 (rep 0 after the phase gap: 1,554) | 1,796 +- 7 |
| **lab/pfpath on 95de2a9** | **2,048 +- 21** (TTFT 5.01 s) | **2,025 +- 15** (TTFT 16.19 s) |
| Sushi 4bpw (18 Oct night) | 2,001 +- 33 | 2,030 +- 12 |
| oMLX oQ4e (18 Oct night) | ~2,260 | 2,808 +- 22 |

Reused turns (`tools/bench_turn.py --levels 32768 --sizes 50,500 --reps 2`, same job, master vs this branch, MTP
path): 50 new tokens at 32K 183 / 186 -> 187 / 190 ms, 500 new 467 / 487 -> 466 / 459 ms (unchanged), the 28.7K-token
cold growth turn 15.5 -> 14.1 s. Checks: `make test` ALL PASS; `tests/mtp/test_mtp_cont.py --whole` all pass (its
`fresh run repeats [chunked d0]` -- adaptive depth from measured round times -- also fails now and then on master:
1 of 3 runs here); `tools/check_headkv.py` as in 2.

**Left on the path** (after): restore of the previous conversation 20-75 ms (a host copy of its KV rows; could run
on a thread during the first chunk, whose rows are the only ones it overwrites), the generation prompt's 7 tokens as
4-row + 1-row steps (~50-60 ms; one 8-row verify-style pass would be ~30), render + encode 10-30 ms, the read-ahead's
~2% GPU contention. **Kernel side (pfkern):** the chunk GPU time is now ~95% of a cold TTFT. The head's KV pass
(~16-20 ms per 4096 rows, not profiled per kernel) skips the q GEMM; its fc_hidden GEMM runs 4n rows with f32 input.

### 4f. kernels, round 2 (pfkern, 18 Oct): MoE overlap, sparse attention, fusions, INT8 quality

T6 pack, master with lab-pfpath merged, chunk 4096; 4-layer A/Bs with `NE_PF_PROF=fine pf-check lspeed`, whole model
with `pf-check speed` (one job, arms interleaved).

**Kept (default on):**
- `t6_fold_hcup` (`NE_T6_HCUP=0` = off): the HC up Linear (EXL3, >= 64 rows) with `pf_hc_blockin` in its epilogue.
  `t6_fold_n` writes Wf's rows permuted so that a 64-column tile holds 16 hidden columns x 4 streams (into yraw: the
  permuted rows cross threadgroups); the 64 x 64 tensor-op tile goes through threadgroup memory, then block_in's
  formula. The [n][10240] up plane is never written. 4 layers: hc_blockin -3.3 ms per chunk, HC up GEMM +1.3.
- `t6_dense_fold` (`NE_T6_DFOLD=0` = off): the trellis decode and the k-side fold (H_hi, suh) in one pass (16 rows x
  128 k per threadgroup through threadgroup memory, coalesced half stores). qkv fold + dense 2.07 -> 1.16-1.58 ms per
  chunk (4 layers).
- Numerics: 4-layer hidden rel L2 vs decode 4.67e-3 (round 1) -> 4.71e-3 (accumulation order of the 64-row tile).

**Whole model, tok/s** (round 1 = `NE_T6_HCUP=0 NE_T6_DFOLD=0`):

| | 8K | 32K |
|---|---|---|
| round 1 | 2,087 / 2,076 | 2,032 / 2,118 |
| **round 2 (default)** | **2,140 / 2,124** (+2.4%) | **2,202 / 2,150** (+4%) |
| + `NE_T6_A8=128` | 2,232 / 2,230 (+7%) | 2,251 / 2,214 (+7%) |

**Same client** (`session.sh` from this worktree with its own test key, port 8143, REPS 3, back to back, medians
10K / 32K): master (81ce807) **2,102 / 2,015**; this branch **2,179 / 2,047** and **2,256 / 2,051** (two sessions);
`NE_T6_A8=128` **2,248 / 2,118**. Sushi 2,001 / 2,030, oMLX 2,260 / 2,808.

**INT8 trunk quality, prefill path** (`kld_teacher.py`: A 16 windows with `--prefill 256`, which needed the scoring
loop to take a partial first 512-row block; B 16 windows `--prefill 1024`; C all 4 windows `--prefill 16384
--max-pos 18432`; paired A8 - default, 95% block bootstrap CI): A 0.02922 -> 0.02823 (-0.00099 [-0.00418, +0.00121]);
B 0.00239 -> 0.00245 (+0.00005 [-0.00000, +0.00011]); C 0.01541 -> 0.01500 (-0.00041 [-0.00069, -0.00020]). A and C
are not worse (C significantly better); B leans +2% with a CI that touches 0. Stays opt-in (rule: all three clearly
not worse); the main session decides.

**Tried, dropped (one A/B each, 4 layers, ms per chunk):**
- Warp-specialised expert GEMM (4 decoder simdgroups fill a double-buffered threadgroup slab of decoded fragments,
  4 MMA simdgroups run 64-row windows, one barrier per slab): bit-identical, gate 14.9 -> 20.0-23.6, down 15.0 ->
  23.2-24.4 (slab 1 / 2 / 4 k tiles). Sushi's +30% for a shared decode reproduced: the role split halves the MMA
  simdgroups per core and the barrier keeps them in lockstep.
- Expert GEMM on the MPP tensor op with the expert's tile decoded once per 128-row window into threadgroup memory
  (`t6_dec`; decode ~once per expert and chunk): identical outputs, 14.9 -> 27.5 (decode -> barrier -> op -> barrier
  serialises inside the threadgroup; 17 KB of threadgroup memory limits residency; 128-row ops on ~80 live rows).
- Sparse prefill attention: the core (`qsa_chunk_attn_bf16`) is ~220-245 ms per chunk at 32K and flat in the context;
  the growth (155 -> 305 ms from chunk 0 to 7) is the indexer score + select_local (15 -> 69 ms). Timing variants:
  without K / V loads 16.1 -> 13.2 ms, without MMAs -> 8.2 (1 attention layer, 32K): issue-bound, not gather-bound,
  and there are 8,192 threadgroups per chunk, so a split-K over keys does not add useful parallelism. P as half for
  P.V: 16.05 (no change). 16 x 32 x 32 ops with persistent destinations: 20.2 (+25%; correct).
  K = 32 cooperative-tensor ops have now lost three times in real kernels (MoE, attention) although they win in a
  register-only microbenchmark: only the MPP-managed (tensor-operand) path reaches the higher rate.
- GDN scan with 32 / 64 value rows per threadgroup: no gain (round 1).

**Leads left:** the sparse attention core as a multi-row kernel on tensor operands (8 rows x 12 heads = 96 query rows
per threadgroup over the union of their selections, K / V tiles staged in threadgroup memory: ~2x the FLOPs at a
much better MMA shape); the indexer score / select_local (linear in T: 69 ms per chunk at 32K, 262K would make it
dominant); the expert GEMM remains 29% with decode and MMA each ~2/3 of its time.

### 4g. kernels, round 3 (pfkern, 18 Oct): multi-row sparse attention, selection

**Profile at 32K** (whole model, fine labels per attention sub-step and chunk, `NE_PF_PROF=fine`; 12 layers): the
sparse core `qsa_chunk_attn_bf16` costs ~220-245 ms per chunk, flat from chunk 1 to 7; the growth with context is
the indexer: score + select_local + select 15 ms (chunk 1) -> 69 ms (chunk 7). On one attention layer: without the
K / V loads 16.1 -> 13.2 ms per chunk, without the MMAs -> 8.2. The core is issue-bound, not gather-bound.

**Tried, dropped (one A/B each, 4-layer `lspeed` at 32K, attn_sparse ms per chunk):**

| variant | before | after | correctness |
|---|---|---|---|
| multi-row core `qsa_mr_*`: 4 rows x 16 (12) heads = 64 query rows per threadgroup, union of the 4 selections (one-thread 4-way merge, 4 x 4-bit key masks per block), K / V 16-key tiles in threadgroup memory, S and P.V as MPP tensor ops over 8 simdgroups (64 x 16 x 256, 64 x 256 x 16), online softmax through threadgroup memory | 16.6 / 17.5 | 31.5 / 32.5 | NaN in the 4-layer check, not chased: 2x slower |
| P as half for P.V (current kernel) | 16.12 | 16.05 | 4-layer check equal to 4 digits |
| persistent cooperative tensors (16x32x16, no per-call fragment copies) | 18.0 / 18.9 | 22.8 / 23.4 | equal |
| 16x32x32 ops (round 2) | 16.1 | 20.2 | equal |

The multi-row kernel spends ~4 barriers per 16 keys and runs its MPP ops at shapes too small to be efficient, on a
union ~2x larger than one row's 2,048 keys. In this kernel MLX's `Frag::mma` helpers (fragment arrays, copies the
compiler removes) beat cooperative tensors held across the loop.

**Kept: direct selection** (`qsa_select_r8_d` / `qsa_select_local_r8_d`, chunk path only; `NE_QSA_DSEL=0` = off).
Rows with <= 8192 complete blocks (32K tokens) take their top 512 straight from the scores: the cached radix already
holds 8192 elements in registers (SCACHE x STG). The per-slice pass is skipped. The result is the same, since both
sequences ascend in block id and ties go to the lower id: 4-layer check at 24K equal to every digit, and set-C
prefill-path KLD (4 windows) 0.01541 = master's, diff +0.00000. Per attention layer and 32K chunk: select_local 1.5 ->
0.06 ms, select 0.65 -> 1.06 ms, i.e. ~-12 ms per chunk over 12 layers. pf-check speed at 32K, one job: 2,035 / 2,033 ->
2,056 / 2,044 (+0.8%). At 8K it is unchanged (2,080 / 2,127 vs 2,092 / 2,081: no rows past 2,048 blocks).

**Same client** (worktree's session.sh, own key, port 8143, interleaved master / lab / master / lab, medians 10K /
32K; box warm): master 1,827 / 1,945 and 2,067 / 1,980; lab 2,055 / 2,015 and 1,987 / 1,975. Within noise of each
other.

**Leads left:**
- The indexer score at long context (a 4-head ReLU-weighted GEMM over T/4 block keys) on MPP tensor operands, with the
  head combination in the epilogue. ~31 ms per chunk at 32K, linear in T.
- A sparse attention core on MPP tensor operands only at shapes the op runs efficiently: >= 64 x 64 x 128 per
  run, which needs 64-key tiles (K and V staged in turn).

### 4h. kernels, round 4 (pfkern, 18 Oct): INT8 experts, chunk 8192, tensor-op indexer score

**INT8 routed gate / up (`pf_moe_prep_a8` + `pf_moe_gemm_a8`, tried and dropped; oMLX's routed-A8 idea, no code).**
- Activations: the prep writes int8 rows with one scale per (slot, 128-block).
- Weights: each decoded trellis fragment is requantised in registers with one global scale. |w| < 4 by construction:
  a sum of two halves of magnitude < 2, so q = rint(w * 127 / 4).
- The MMA is int8 x int8 -> int32 per 128-block, scaled into f32.
- Correct (4-layer hidden rel L2 5.07e-3 against 4.71e-3), but slower. 4 layers, gate per chunk: f16 14.9 ms;
  int8 16x32x16 25.0; int8 16x32x32 22.2.
- The expert GEMM is decode-bound (round 1: decode alone ~2/3 of its time). Int8 only helps the MMA half, and the
  per-element requantisation, the int32 + f32 accumulators and the block epilogue cost more than the faster op saves.
  So the KLD gate was not run, and there is no combined NE_T6_A8 + NE_MOE_A8 number. Code removed.

**Chunk 8192 vs 4096** (`pf-check speed`, one job, interleaved): 32K 4096: 2,227 / 1,965 vs 8192: 2,009 / 1,993; 8K
4096: 2,080 / 1,975 vs 8192: 2,045 / 2,030. No gain within the box's drift, while peak RSS of the check rises
5.6 -> 9.5 GB. At 8192, attention per chunk is 491-648 ms, double the 4096 chunk's, as expected. Stays 4096.

**Kept: indexer score on the tensor op** (`qsa_chunk_score_t`, `NE_QSA_TSCORE=0` = the simdgroup-matrix kernel).
- Per 8-row slice: the 32 (row, head) queries as a threadgroup tensor, one matmul2d 32 x 64 x 128 per 64 blocks, the
  4 heads' ReLU sum in the epilogue.
- 4 layers at 32K, per attention layer and chunk: 1.46 -> 0.57 ms, i.e. ~-11 ms per chunk over 12 layers.
- 4-layer check at 24K equal to every printed digit. Set-C prefill-path KLD (4 windows) 0.01531 vs master 0.01541:
  -0.00010 [-0.00047, +0.00037]. Whole model at 32K: within the drift (2,198 / 2,149 off, 2,158 / 2,149 on).

**Same client** (worktree's session.sh, own key, port 8143, REPS 5, master / lab / master / lab back to back, median
[min-max]):

| session | 10K | 32K |
|---|---|---|
| master (8ecf0d2) | 2,120 [2,091-2,165] | 2,048 [2,043-2,095] |
| lab | 2,149 [2,082-2,198] | 2,062 [2,008-2,069] |
| master | 2,119 [2,083-2,284] | 2,057 [2,057-2,094] |
| lab | 2,109 [2,039-2,283] | 2,046 [2,039-2,087] |

- The two arms are within noise of each other.
- Box temperature: in every session the first 1-2 reps are the fastest (10K 2,165-2,284) and the later ones settle
  3-8% lower (2,039-2,099). Over a whole job, the same pf-check arm drifts 5-12% (2,227 -> 1,965 at 32K).
- Arms therefore have to run back to back and interleaved; the effects of round 3 / 4 (< 1%) are below this noise.

### 4i. The per-turn floor (pfast, 23 Oct): small chunks, the server's prefix match, the capped drafter

**Where a pi turn's time goes.** The user's serve.log, 3,065 requests since 19 Oct: 27% of turns add 16-128 tokens, 36%
add 128-1K. Median prefill at > 40K context: 16-40 new tokens 192 ms, 40-64 273, 64-128 328, 128-256 391, 256-512 485,
512-1K 689. In-process (`tools/bench_turn.py`, now with `--dflash DIR` = the served drafter, `--lib`, and a per-turn
split into chunk feeds / tail steps / checkpoint / MTP start), 50 new tokens at 65-90K: TTFT 182-203 ms = the 45-row
chunk 143-158 + the generation prompt's 4 tail tokens as steps 21-26 + checkpoint 1 + MTP start 16-19. Feeding the tail
inside the chunk (GDN / head / drafter state captured at the checkpoint row) would save <= ~20 ms: not done. Through
the server (`tools/bench_turn_http.py`, NE_SRV_TRACE) a turn also pays render + tokenize 1-8 ms (incremental already),
the admission wait (below) and the checkpoint restore ("prepared") 12-16 ms.

**Single chunks at 16K, whole model, GPU ms per stage** (`bench_turn.py --profile 1 --pf-sizes ...`; `--prof-env`
repeats the profile under engine-knob settings, each with new Prefills):

| rows | total | trunk | expert GEMM | MoE glue | attention | GDN | HC |
|---|---|---|---|---|---|---|---|
| 16 | 91 | 34 | 38 | 9 | 4 | 1 | 4 |
| 45 | 138 | 54 | 62 | 11 | 5 | 2 | 4 |
| 64 | 181 | 92 | 61 | 15 | 5 | 3 | 4 |
| 128 | 221 | 94 | 92 | 16 | 9 | 6 | 4 |
| 500 | 369 | 126 | 151 | 27 | 31 | 22 | 11 |
| 2000 | 975 | 304 | 325 | 84 | 125 | 95 | 37 |

At 90K the 2000-row chunk's attention is 184 ms (17%; +59 over 16K): the long-context growth is not the turn lever. The
expert GEMM at <= 128 rows reads every touched expert once (~200-300 per layer): a bandwidth floor.

**Kept, exact:**
1. **`prefix_cache.lcp` in blocks.** The old one compared the whole lists and binary-searched with full-length slices
   on any mismatch: every turn whose previous reply re-renders differently, once per candidate (the conversation, its
   checkpoints, SSD entries sharing the system prompt). 10.8 -> 0.65 ms per call at 190K tokens; equal to a reference
   on 20,000 random pairs. It was the admission wait in NE_SRV_TRACE: 2 / 12 / 26 ms at 21K / 92K / 188K.
2. **The DFlash2 drafter stays capped past its context cap on a state import** (`dfl_import`, mtp_loop.mm). Every turn
   past 8K restored its checkpoint, which un-capped the drafter: a context rebuild on the GPU, the feature rows of each
   chunk captured, drained and exported, all before the run re-capped it and drafted with the MTP head. Exact: the
   drafter cannot draft above the cap either way; below it dfl_sync restarts the context as for any capped state.

**Parked (env knobs, default = master):** `NE_T6_FOLD_MIN=1024` (fold only from 1K rows; chunk GPU ms at 16K, fold vs not:
64 rows 185 / 145, 128 216 / 181, 256 273 / 241, 500 383 / 346, 750 505 / 483, 1000 619 / 616, 1500 872 / 889, 4096 2142
/ 2352) and `NE_T6_SK_NARROW=128` (split-K for the narrow Linears -- HC down, shared gate / up -- up to 128 rows: trunk
45 / 64 / 128 rows 53 / 62 / 65 -> 41 / 45 / 51 ms). Same client, both on, client TTFT medians of 4 (old / new):
~21K 70 tokens 269 / 234, 550 519 / 503; ~92K 70-100 288 / 240, 530 562 / 518; ~188K 70-96 304 / 261, 530 578 / 560.
They change rounding points: old vs new logits after the chunk (`tools/check_pfpath.py`, 32 teacher-forced steps) KL
3.7e-3 (45 rows) .. 2.9e-2 (500), the same as old vs the 1-row decode path (5.1e-3 .. 3.3e-2). Paired teacher KLD
(`kld_teacher.py --prefill N --max-pos N+1024`, new - old): N 100: B +0.00009 [-0.00112, +0.00156], C -0.00228
[-0.00645, +0.00058]; N 500: B +0.00004 [-0.00004, +0.00010], C +0.00174 [+0.00079, +0.00239]; narrow split-K alone,
N 48: B +0.00179 [+0.00027, +0.00411], C -0.00035 [-0.00118, +0.00029]. A reduction-order change "failing" B says the
gate's per-window noise is as large as these effects: `tests/quality/pfast_null_gate.sh` (night job, not run) calibrates
it with null arms and judges both knobs against the null spread.

Also tried: the f16 tensor-op GEMM below 64 rows (`NE_T6_TG_MIN=17`: 45 rows trunk 61 vs 52 ms, kept off); split-K for
every Linear to 64 rows (`NE_T6_SK=64`: no gain).

**Same client, exact fixes only** (master tree vs lab/pfast, greedy, old / new / old / new):

| context (prompt tokens) | new tokens | master TTFT ms | lab/pfast TTFT ms | change |
|---|---|---|---|---|
| 16.5-19.2K | ~70 | 277 | 262 | -15 (-5.5%) |
| 17.0-19.7K | ~540 | 492 | 475 | -16 (-3.3%) |
| 19.1-21.7K | ~2,030 | 1,211 (1,066-1,359) | 1,254 | +44 (master's spread) |
| 66-69K | ~70 | 278 | 258 | -21 (-7.5%) |
| 66.5-69.2K | ~540 | 546 | 506 | -40 (-7.3%) |
| 68.6-71.3K | ~2,070 | 1,364 | 1,335 | -29 (-2.1%) |
| 90.3-92.9K | ~70 | 295 | 268 | -27 (-9.1%) |
| 90.8-93.4K | ~540 | 565 | 509 | -56 (-9.8%) |
| 92.8-95.5K | ~2,070 | 1,401 | 1,339 | -62 (-4.4%) |

(Medians of 4: 2 servers per arm x 2 reps, `tools/bench_turn_http.py --temperature 0`, max_tokens 16, server on :8143
with a test key, `--dflash-kmax 5 --dflash-bits 8`, RAM-only prefix cache.) Trace medians at 65-90K, master -> lab: admission
wait 8-12 -> 1 ms (lcp), "prepared" 12-15 -> 1.5 ms (no drafter rebuild); the rest is the chunk feeds' drafter capture.
Greedy outputs: 18 / 18 turns identical to master in both lab servers (and master's two). make test ALL PASS,
tests/server/run_mock.sh all passed.

## 5. Traps

- (pfkern) MPP tensor-op GEMMs: a kernel's last dispatch on the shared `PfBuf::wd` needs a barrier after it (the next
  Linear's `t6_dense` rewrites wd; without it the fold path read half-rewritten weights: hidden rel L2 6e2). A 64 x 64
  tile over 8 simdgroups ran 2x slower in the model than in isolation: keep 128 x 64 over 4. f32 x f16 tensor ops run
  at a quarter of bf16 x f16.
- (pfkern) `pf-check lspeed / layers` partial models need `vfeat` (fn_embed reads image rows since lab-vision): one
  dummy row.
- **A MachineLock inside a `with_lock.sh` job waits forever** on the lock with_lock holds for it (cost the machine
  ~10 minutes of lock on 15 Oct); with_lock now exports `NE_NO_LOCK=1`. pf-check takes no lock itself.
- MLX's own q8 loader dequantises the trunk weights to bf16: 1.2e-3 rel per GEMM vs the decode path's f32 weights
  (the CPU check). `PfQ8Loader` dequantises to half instead: 2e-4 (activations are bf16, exact in half).
- The QSA dense / sparse split depends on the chunk's start position (rows with (pos + 1) / 4 <= 512 are dense), so
  recordings of early chunks are keyed by T0.
- The GDN conv (and the PLE conv) can be fully parallel over rows: each output reads its 3 (9) predecessors from
  the raw rows or the old state; the new state is written by a separate kernel after a barrier (reading and writing
  the state in one dispatch races).
- lab-attn's step scratch is per cache (qr, scores, selections, partials): concurrent slices need their own sets
  (`p.slot`, shapes copied from `attn::alloc_cache`).

## 6. Requests

- **lab-forward** (`models/flash_next/model.mm`): (1) `ne_fn_model_prefill` / `generate()` should prefill through
  `ne::fn::prefill_prompt` (exact patch: in `prefill()`, `if (!m.pf) m.pf = prefill_create(m, 2048); return
  prefill_prompt(*m.pf, prompt, n);` with a `PrefillPtr pf;` member in `Model` and `#include
  "models/flash_next/prefill.h"` in model.mm; keep the 1-row path for n < 16 or behind `NE_FN_PREFILL=steps`);
  (2) move `struct ne_fn_model` into model.h (prefill.mm re-declares the identical struct to reach the Model from
  the C handle).
- **lab-attn**: (1) `alloc_step_scratch(fmt, tmax)` (prefill.mm mirrors `alloc_cache`'s scratch shapes for its 16
  concurrent slices); (2) the sparse attention core is the long-context cost of prefill: `qsa_attn` + `qsa_merge`
  take 86 + 17 us per 8 rows at any T >= 2K (each row streams its own 512 blocks: ~4 MB of K/V per row); rows of a
  chunk select mostly the same blocks, so a multi-row (64+) variant over the union of the rows' blocks (per-row
  mask) on the NAX would cut it several-fold.

## 7. Attribution

- `kernels/prefill/mlx/` (NAXTile / tile_matmad_nax, QuantizedBlockLoader, attention_nax_dsplit, AttnParams) and
  the GEMM body of `pf_qmm_t` are MLX 0.32.3 (MIT License, Copyright (c) 2023-26 Apple Inc.; `LICENSE.mlx`).
- The EXL3 MoE prefill scheme and its NAX GEMM body / trellis fragment decode (`pf_moe.metal`) are ported from the
  lab tree's `exl3/kernels/exl3_prefill.py` (NOTES.md step 3), which lifts them from
  sushi `src/exl3/expert_exl3_kernels.zig` (https://github.com/beamivalice/sushi, MIT License, Copyright (c) 2026
  Theinruj Toranavikrai, Copyright (c) 2026 David Dalcu; permission is hereby granted, free of charge, to any person
  obtaining a copy of this software ... THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND). The MCG
  codebook decode follows sushi / exllamav3 (turboderp, MIT). Sushi's chunked prefill and GPU routing metadata
  (docs/qwen4-perf.md) informed the design; no other Sushi code is used.
- Every pointwise kernel mirrors lab-forward's decode kernels in this repo (fn_trunk / fn_moe / fn_model.metal).
- `kernels/trunk6/t6_a8.metal` and the A8 / fold kernels in `trunk6.metal` (pfkern) are ours; the INT8-activation
  idea (per-row activation quantisation, int8 x int8 -> int32 tensor ops) follows oMLX's oQ A8
  (`omlx/patches/qwen35_oq_a8.py`, `m5_gather_qmm_a8.py`; jundot/omlx, Apache License 2.0); no oMLX code is used.
- `kernels/prefill/pf_gdn_pipe.metal` (`pf_gdn_scan_pipe`, lab-pflevers) is ported from Sushi
  `src/kernels/gdn_pipelined.metal` (https://github.com/beamivalice/sushi, MIT License, Copyright (c) 2026 Theinruj
  Toranavikrai, Copyright (c) 2026 David Dalcu), which ports oMLX `omlx/custom_kernels/qwen35_prefill/gdn.py`
  `gated_delta_pipelined` (jundot/omlx, Apache License 2.0; the Sushi file carries the Apache-2.0 SPDX header).
  The f16 expert GEMM planes follow Sushi's `nax_store` / `sortedGemmCfg` (.float16), the HC write fold Sushi's
  `hc_prefill_norm` WR=1 / oMLX's `norm_inject` (ideas; the kernels are ours).
