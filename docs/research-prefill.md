# How Sushi and oMLX / MLX prefill Flash-Next, and what would close our gap (research-prefill, 16 Oct 2026)

Owner: lab-research-prefill. Read-only survey: Sushi's source (branch dense-hc, HEAD 908f8ea),
oMLX 0.7.x (the app bundle's `omlx` package, its vendored `mlx_vlm` qwen4_exp, MLX 0.32.3 headers and
`libmlx.dylib`), an oMLX pull-request checkout, and
the lab notes in `the lab tree's exl3/{kernels,omlx}/NOTES.md`. Nothing was ported; any port
keeps the source licence (Sushi MIT; oMLX Apache-2.0; MLX MIT).

**Labels.** *Evidence* = read in source or a measured number in the named doc. *Inferred* = my reasoning, or read
from disassembly. Savings are per 2048-token chunk of a 32K prompt (our chunk today: ~1.11 s = expert GEMMs ~330 ms,
trunk GEMMs ~300, MoE + HC glue ~170, attention ~140, GDN ~100; docs/prefill.md 4, docs/attn.md 8).

## 0. Bottom line

- **Sushi's prefill is our prefill.** Same per-layer pipeline, same EXL3 NAX GEMM body (ours is a port of it), same
  <= 32-row run-aligned windows, same GPU-built window table and inverse-indexed finish, same per-(row, kv group) NAX
  sparse attention. Its 2,384 tok/s at 32K comes from three things: **a wider chunk** (4096 by default, 8192 when it
  fits, against our 2048), a **pipelined GDN recurrence**, and **a cheaper pack** (Sushi-4bpw's K4 experts take the
  one-funnel `nax_wfrag` decode; our K5 takes the generic `exl3_window` reader). Lighter
  precision helps as well: bf16 HC stream, f16 GEMM output planes.
- On our K5 pack at 10K with the chunk pinned to 2048, Sushi measured 1,890-1,918 tok/s (h2h, 14 Oct) and we now
  run 2,019-2,102: we are ahead at the same chunk width. **Sushi on our K5 pack at 32K (measured today, section 4):
  1,560-1,580 tok/s at chunk 2048 and 1,775-1,933 at its default (4096).** That puts us (1,851 at 2048) 1.18x ahead at
  the same width and at par with Sushi's served default. The 2,384 target is Sushi-4bpw, a cheaper pack.
- oMLX / MLX bring three techniques Sushi does not have: **INT8-activation GEMMs** on the NAX (`oQ A8`, opt-in);
  **HC-read epilogue fusion** (3 dispatches instead of 6); and an 8192 prefill step for qwen4_exp.
- Ranked levers (section 5): chunk 4096 / 8192 (-55..-80 ms per 2048 tokens), pipelined GDN scan (-35..-45), A8 trunk
  GEMMs (-70..-110, needs a quality check), fused HC read / write (-20..-30), f16 MoE planes (-10..-25), 48-row windows
  (0..-30, uncertain), bf16 HC stream (-30..-40, quality risk). Top five together: ~-190..-290 ms per 2048 tokens,
  i.e. 1.11 s -> ~0.82-0.92 s, ~2,200-2,500 tok/s at 32K (inferred, not additive-tested).

## 1. Sushi (Zig + MLX graph + custom Metal kernels)

**Chunking.** The default width is 8192 (`src/generate.zig:47`). The ladder is 8192 / 4096 / 2048 / 1024 / 512
(`src/server.zig:3344`). A per-request arch starts from `prefillStartWidth`, which is 4096 for qwen4_exp
(`src/model.zig:617-622`). `--prefill-chunk N` sets the top rung, and a long-context-gated MoE may take 8192 when its
memory bill fits (`src/generate.zig:247-251`, `chooseRequestPrefillChunk` `src/server.zig:6760-6787`). Evidence for
8192 over 4096 on Flash-Next: commit 1386382 (M5 Max, counterbalanced boots) measured prefill +8% at 16k/64k/128k and
+13% at 350k, for +2.7-3.3 GB of peak memory. oMLX's port uses the same 8192 (section 2). The h2h harness pinned Sushi
to `--prefill-chunk 2048` (`omlx_h2h/session.sh`), so Sushi's 10K numbers on our packs were taken at our width; its
llmprobe 2,384 at 32K ran at the default (>= 4096).

**Per layer** (`forwardQwen4With`, `src/qwen4_forward.zig:1898-2110`), all one lazy MLX graph, evaluated every 4
layers (`PREFILL_EVAL_CADENCE_DEFAULT`, `src/transformer.zig:16012`):
- **HC read**: `hc_prefill.norm` (`src/hc_prefill.zig:49-98`, kernel `src/kernels/hc_prefill_norm.metal`) does the
  per-stream RMS norm. With `WR=1` it first applies the *previous* block's deferred write (`stream += out * inj`)
  and stores the stream: Sushi folds the HC write into the next read's prologue (`hcWriteOrDefer` /
  `hcReadPending`, `src/qwen4_hc.zig:224-247`, `:116-128`). The inject projection is an MLX matmul. Down / up are
  MLX `qmatmul` (8-bit affine NAX), with a compiled SiLU between them, then `hc_prefill.mix`
  (`kernels/hc_prefill_mix.metal`: sigmoid table, stream product, mean). The **stream is bf16** (`hc_prefill.eligible`
  requires bf16, `src/hc_prefill.zig:12-14`).
- **GDN**: MLX q8 projections, conv / gates as graph ops, then the recurrence. On NAX hosts this is the **pipelined**
  kernel `gated_delta_pipelined_tb12` (`src/kernels/gdn_pipelined.metal`, ported from oMLX gdn.py; route
  `src/transformer.zig:534-596`). Layout: 8 lanes own one value row with 16 fp32 state channels each, 128-thread
  threadgroups cover 16 rows, 12-token blocks of q/k/v/g/beta are staged in threadgroup memory with the next block
  prefetched, the 12 steps are unrolled, and the q.S reductions of 4 steps are batched off the critical path.
  Measured (docs/qwen4-perf.md:260-270, M5 Max, Sushi-4bpw): pipelined against the older blocked kernel gives 8K 2,534
  against 2,415 tok/s (+4.9%) and 32K 2,384 against 2,296 (+3.8%, TTFT 13.7 against 14.3 s). The state is f32 by
  default (`--qwen-gdn 32`).
- **Attention** (`qwen4AttnProjected`, `gatedFullAttnProjected`): positions <= 2052 keys take stock fused SDPA (NAX).
  Past that floor (`qsaPrefillGatherMinKv`, `src/qwen4_qsa.zig:64-68`), selection is the indexer score sheet plus a
  radix top-512 (11/11/10-bit digits at prefill widths, `src/qwen4_qsa.zig:94`, `:393-570`). The core is the gather
  kernel `sushi_qsa_nax_precise` (`src/kernels/qsa_nax.metal`; dispatch `src/transformer.zig:3387-3500`): one
  threadgroup per (row, kv head), 2 simdgroups splitting D = 256, the 12 q heads loaded as rows of a 16-row NAX
  tile (4 padded, as in ours), K / V staged through threadgroup memory in 32-key tiles, online softmax.
  With a kv8 cache, a wide chunk dequantizes the cache into a bf16 rebuild once rather than gathering packed
  (`qsaPackedGatherServes`, `src/qwen4_qsa.zig:719-725`). "Packed reads for wide long-context chunks" were declined
  as 15-20% slower (docs/qwen4-perf.md:441). **Same design as lab-attn2's core.**
- **MoE** (`moePrefill`, `src/exl3/expert_exl3_kernels.zig:3080-3154`): `argsort` of the slots, a token-prepare
  (x * suh, H128 -> f16), the window table and inverse built **on the GPU in one threadgroup** (`buildMimoWindowTable`
  :7826, source :7798-7814; +2-5% measured, docs/qwen4-perf.md:523-540), then three sorted GEMMs (gate, up, down)
  with a fused SwiGLU mid between them and a finish that reads the down plane through the inverse. Windows are run
  aligned, **32 rows** (`GEMM_WINDOW_ROWS`, :1144). Body `GEMM_NAX_SOURCE` (:675-749) uses MPP `matmul2d` 16x32x16
  with a per-lane fragment decode (`nax_wfrag_k`). A threadgroup covers 128 output columns (4 simdgroups x 32), the
  k loop is unswitched on the second 16-row block and unrolled by 2, x row pointers are clamped (no bounds branch),
  and **the output plane is f16** (`nax_store` writes `Y = half`). Each window re-decodes its expert's whole tile set.
  Dead ends Sushi measured on this body (docs/engine-exl3-experts.md:136-142): 64-row windows (+7-18%, +9-13%),
  a threadgroup-shared double-buffered decode (+30%), decoding tile k+1 before the MMA of tile k (+27%), 256- or
  64-thread groups (+12%), a threadgroup LUT decode (+32%), unroll 4 (+20%), and a counting sort instead of the
  argsort (-0.03..-0.16 ms per layer, yet 5-10% slower end to end). "The kernel is register / occupancy bound: added
  live state loses."
- **K4 fast branch**: at N = 64 (K4) `nax_wfrag_k` returns `nax_wfrag` (one 64-bit funnel from 3 words per quad).
  K5 (N = 80) takes `exl3_window` with two windows. This is part of why Sushi-4bpw prefills faster than our K5
  pack (inferred: I have no per-rate prefill GEMM timing; `SUSHI_EXL3_LOWK_UBENCH=1` prints n24..n48 only).

**Sushi numbers (evidence, docs/qwen4-perf.md):** 3bpw ladder (quiet box): 2k 1,906 / 8k 1,940 / 33k 1,961 / 131k
1,900 tok/s. 4bpw pipelined: 8K 2,534, 32K 2,384. Sushi-2.6bpw: 4k 2,099-2,330, 32k 2,065-2,250. Box drift of
20-33% between hot and quiet runs is documented (same section), so only paired cells compare.

## 2. oMLX / MLX (Python + MLX 0.32.3 + oMLX native kernels)

**Chunking.** oMLX's step is 2048 by default (`omlx/scheduler.py:1689`), but qwen4_exp takes
`_QWEN4_WIDE_PREFILL_STEP = 8192` on NAX hosts with >= 64 GB (`scheduler.py:592`, `:3041-3066`). The first chunk is
kept narrow so that the next chunk's PLE gather can overlap GPU work (`scheduler.py:1988-1992`).

**MoE (affine experts).** mlx_lm sorts the slots when `indices.size >= 64` (`mlx_lm/models/switch_layers.py:178-195`)
and calls `gather_qmm(sorted_indices=True)`, which runs `affine_gather_qmm_rhs_nax` on M5 (`mlx/include/.../
quantized_nax.h:1463-1600`). Each threadgroup computes one single-expert BM x 64 tile, found by a per-threadgroup
simd scan of the expert offsets (`schedule_row_tile`, `kernels/utils.h:503-540`, no table). The weight tile is
dequantized **into threadgroup memory once per K step and shared by all 4 simdgroups** (2 x 2); the activations are
read straight from device memory. **BM = 32 when the mean rows per expert (M / E) is < 64, else 64** (inferred from
the disassembly of `mlx::core::gather_qmm_rhs_nax`, `libmlx.dylib` 0xd1b184-0xd1b198: `sdiv M, E; cmp #64;
csel 32, 64`). oMLX replaces this with its own kernel (`omlx/patches/m5_gather_qmm_nax.py:1-50`): a one-threadgroup
pre-pass cuts runs into BM = 64 / 96 / 128 tiles, there are two schedules (segmented, or double-buffered weight
tiles), empty 16-row fragments are skipped, and threadgroups are laid out so that a row tile's columns share an x
coordinate (+3-28% on M5 Ultra from 36 rows per expert). Affine dequant is cheap, so taller shared-weight tiles pay
there. For trellis decode, Sushi measured the shared-decode variant at +30% (above), so this transfers to EXL3 only
through more rows per expert, i.e. wider chunks.

**INT8 activations (`oQ A8`, opt-in, `qwen35_oq_a8_enabled = False` by default, `omlx/model_settings.py:407`).**
Dense Q4 / Q5 / Q8 GS64 projections (`omlx/patches/qwen35_oq_a8.py`) and the routed Q4 gate+up
(`m5_gather_qmm_a8.py:1-45`) quantize each activation row to INT8 once ("Stage A": scale = amax / 127, plus the code
sum of each 64-group). Weight codes are decoded straight into INT8 fragments, multiplied with MPP
`matmul2d<int8, int8, int32>` (`m5_gather_qmm_a8.py:165-190`), and the GS64 affine correction
`Sa * sum_g (Sw_g * acc_g + Bw_g * Ra_g)` is applied per group. For 8-bit weights the activation scale is per 64-group
instead of per row, "so outliers do not set a whole row's scale" (`qwen35_oq_a8.py:113-118`). Measured (comment, no
table on disk): crossover at 4-8 rows, **1.6x by 128 rows** on a 5120x8704 Q4 MLP block (`qwen35_oq_a8.py:352-357`).
The routed path lists qwen4_exp as "measured for speed and NLL" (`m5_gather_qmm_a8.py:722-724`); the numbers are
not shipped.

**HC prefill** (vendored `mlx_vlm/models/qwen4_exp/hc_prefill_nax.py:1-30`). Six dispatches become three:
`norm_inject` (the stream norm with the pending residual write, plus the inject partial sums), `down_silu` (MLX's NAX
q8 tile loop for 10240 -> 320 with the /4 and SiLU in the epilogue), and `up_tail` (the up GEMM over column tiles
holding 16 hidden columns of all 4 streams, with sigmoid gate x normed x 4-stream mean in the epilogue, so only
[rows, 2560] is written). Bit-identical to the six-dispatch path. No speed number on disk.

**GDN** (`omlx/custom_kernels/qwen35_prefill/gdn.py`, `omlx/patches/qwen35_gdn_chunked.py:1-20`). The production path
is `gated_delta_pipelined` (Sushi's port, above). Measured at T = 8191 with 16 / 48 heads on M5 Ultra: **3.0 ms
against 4.9 ms per layer call** for `gated_delta_blocked_seq`. The FLA chunked WY kernels (A: intra-chunk
(I + A)^-1 per 64 tokens; B: state across chunks) are "accuracy-validated but slower than the stock kernel E2E; kept
for future iteration". That is evidence against our lever (4), the chunked delta rule.

**Other**: an opt-in ANE + GPU hybrid prefill for *dense* Qwen MLPs (`patches/qwen35_ane_prefill.py`, private ANE
runtime, fixed shapes, >= 1024 rows); it does not cover MoE. oMLX's QSA prefill attention is a compiled native op
(`qwen4_qsa_sparse_gqa_attention`, `custom_kernels/glm_moe_dsa/fast.py:318-337`: 512 block ids, causal tail in kernel,
fp32 softmax). The source is not on disk, only the metallib. PR 4027 (`omlx-pr4027/src/omlx/quantization/
exl3_prefill.py`, `docs/exl3-backend.md`) adapts Sushi's cooperative simdgroup-matrix prefill to MLX gathers,
1.04-1.19x over its predecessor per projection. Nothing there beats our NAX body.

## 3. Ours vs theirs, stage by stage (2048-row chunk at 32K)

| stage | ours (ms) | Sushi | oMLX / MLX | gap source |
|---|---|---|---|---|
| expert GEMMs | ~330 | same body; 32-row windows; f16 out; chunk >= 4096 | affine, BM 32/64 (MLX), 64-128 (oMLX) | chunk width (fill of windows), K4 vs K5, f32 out |
| trunk GEMMs | ~300 (48-53 TFLOP/s) | MLX q8 `qmatmul` (same NAX kernel we ported) | same; A8 opt-in | A8 only |
| HC glue | ~68 | 2 kernels + MLX GEMMs; write fused into next norm; bf16 stream | 3 dispatches, epilogue-fused | f32 stream (2x bytes), separate write pass, up plane written |
| MoE glue | ~100 | argsort + same glue | | parity (ours has no host sync) |
| attention | ~140 | same per-(row, kv head) NAX gather | native op (closed) | parity |
| GDN | ~100 (scan ~2 ms/layer) | pipelined recurrence | pipelined (3.0 ms / 8191 tok on Ultra) | scan lane layout + pipelining |

## 4. Sushi on our K5 pack at 32K (measured)

16 Oct 07:38-07:41, one private instance (`sushi` dense-hc build, zig-out/bin, 127.0.0.1:8097, isolated HOME),
inside one `tools/with_lock.sh` job (NEED_GB=85), stopped inside the same job. Flags: `--no-mtp --kv-quant 8
--ctx-size 65536 --prefix-cache-entries 0 --prefix-cache-disk 0`; pack `<packs>/Qwen3.8-Flash-Next-EXL3-K5`.
Each boot: a warm-up, a cold 3.2K prompt, then two cold ~33K prompts (different slices of this repo's docs + Sushi's
docs + our .metal sources, behind a nonce), `max_tokens` 1. tok/s is the server's own `[prefill: ...]` line (wall
minus decode; the client's prompt / wall agrees to <0.5%). One boot per arm, in the order below (no A B B A; the box
was otherwise idle). Scripts: one-off, not kept.

| arm | chunk the log reports | rep 0 (32,928 tokens) | rep 1 (33,852 tokens) | 3.2K |
|---|---|---|---|---|
| default | "per request, up to 4096" | **1,933** | **1,775** | 1,888 |
| `--prefill-chunk 2048` | S = 2048 | 1,581 | 1,560 | 1,486 |
| `--prefill-chunk 8192` | 8192 rung | 1,834 | 1,688 | 2,514 (one chunk) |

- Sushi on K5 at 32K is **~1,560-1,580 at our chunk width** and **~1,780-1,930 at its default**. Ours on the same pack
  at 2048 (aligned copy, bf16 KV) is 1,851 (1,747-1,854 across the 16 Oct pairs). So we are ~1.18x Sushi at the
  same width and at par with its default. Sushi-4bpw's 2,384 is the pack: K4 experts on the one-funnel decode, and fewer
  expert bits (inferred: Sushi-4bpw's per-tensor rates were not checked).
- Sushi gains **+14-22% from 2048 to 4096**. That is more than the -55 ms per 2048 tokens (+5%) I expect for us
  from lever 1. Sushi's MLX graph pays per-chunk host costs that we do not (graph build, eval cadence, QSA rebuild;
  our wall is within 0.5% of GPU), so its chunk gain overstates ours (inferred). 8192 read 5% below 4096 here, against
  +8% in Sushi's counterbalanced measurement (commit 1386382). This boot ran last, so drift cannot be excluded, and a
  single boot does not resolve it.
- Rep 1 (a code-heavier slice) is 1-9% slower in every arm: compare our numbers only on the same text.

## 5. Ranked techniques we do not have

Each entry gives the source, the expected saving per 2048-token chunk at 32K (from section 3), the cost, and one
falsifiable A/B. One indicative run per arm, interleaved (the lab's rule).

### 1. Wider chunks: 4096, then 8192 (Sushi default 4096 / 8192 rung; oMLX 8192)
- **Where**: `src/model.zig:617-622`, `src/generate.zig:247-251`, commit 1386382 (+8% at 8192 over 4096, measured);
  `omlx/scheduler.py:592, 3041-3066`.
- **Why it saves**: the expert GEMMs are decode-bound per window (each window re-decodes its expert). Windows per
  token fall as rows per expert rise: a Poisson / Zipf model of 512 experts x top-10 gives 0.42-0.47 windows per token
  at 2048, 0.37 at 4096 and 0.34 at 8192, i.e. -11..-22% and -18..-27% (inferred). The lab's measured NAX GEMMs per
  MoE layer: 6.8 ms at T = 2048 and 11.5 ms at T = 4096 (5.75 per 2048 rows, -15%; kernels/NOTES.md step 3, skewed
  random routing; our real 2048 number is 6.9 ms). The HC down GEMM [320 x 10240] also gains row tiles (45 TFLOP/s
  at M = 2048, 15 at 512).
- **Expected**: expert GEMMs 330 -> ~280 (4096) / ~255 (8192); trunk -5. Total **-55 ms (4096) / -80 ms (8192)**,
  +5-8% tok/s (Sushi on K5 measured +14-22% from 2048 to 4096 today, section 4; its per-chunk host costs inflate that).
- **Cost**: low. `prefill_create(m, 4096 | 8192)`; buffers ~0.55 MB per chunk token (2.3 / 4.5 GB); attention
  scores scratch 4 B x C x tmax/4 (256 MB at C = 8192, T = 32K; 2 GB at 262K: cap C by context as Sushi does);
  check maxwin and the scratch sizing in the MoE and QSA recordings.
- **A/B**: `pf-check speed PACK 32768 4096` vs `... 2048` (interleaved, profile). Prediction: >= +4% tok/s and
  moe_gemm <= 290 ms per 2048 tokens. Falsified if < +2%.

### 2. Pipelined GDN recurrence (oMLX gdn.py "Kernel P", Sushi `kernels/gdn_pipelined.metal`)
- **Where**: `omlx/custom_kernels/qwen35_prefill/gdn.py:597-623` (design), Sushi `src/kernels/gdn_pipelined.metal`
  (Apache-2.0 header), route `src/transformer.zig:583-596`.
- **Why it saves**: our `pf_gdn_scan_tb` gives each value row all 32 lanes (4 dk each). Every step does two 32-lane
  `simd_sum`s (5 shuffle levels each) on the dependency chain, for 4 rows serially per lane
  (kernels/prefill/pf_trunk.metal:421-470). Kernel P gives a row 8 lanes x 16 channels: the k.S reduction takes
  3 xor levels, and q.S comes off the chain (batched 4 steps). Rough per-token rates: ours ~0.98 us/token/layer
  (2.0 ms per 2048); oMLX's pipelined 0.37 us and blocked 0.60 us on M5 Ultra (cross-machine, so inferred). Sushi's
  measured blocked -> pipelined swap on the M5 Max saved 0.6 s per 32K, i.e. ~37 ms per 2048 tokens (evidence),
  and our kernel is slower than their blocked one.
- **Expected**: scan 2.0 -> ~0.9-1.1 ms per GDN layer, **-35..-45 ms**.
- **Cost**: low-medium (one kernel; adapt to our f32 normalized qk layout, bf16 conv v, gb). The summation order
  changes against fn_gdn_step, so the prefill state no longer follows the decode op order. Check `pf-check layers`
  against the floor.
- **A/B**: `pf-check lspeed PACK TOKENS 2048 4 2048 profile`. Prediction: GDN stage <= 1.6 ms per GDN layer (from
  2.8) and scan <= 1.1 ms. Falsified if the scan is > 1.5 ms.

### 3. INT8-activation trunk GEMMs (oMLX oQ A8)
- **Where**: `omlx/patches/qwen35_oq_a8.py` (routing, GS64 activation mode for 8-bit weights :113-118, crossover
  :352-357), kernel pattern `omlx/patches/m5_gather_qmm_a8.py:1-45, 165-190`.
- **Why it saves**: our trunk is MLX's q8 NAX kernel (half tiles, 48-53 TFLOP/s). The MLX 8-bit codes are already
  integers, so A8 x W8 -> INT32 removes the dequant from the loop and may run the tensor op at a higher rate.
  oMLX says 1.6x at 128 rows on Q4 (evidence). The M5 INT8 tensor-op rate and the q8 gain at M = 2048 are not
  documented anywhere on disk (inferred).
- **Expected**: trunk 300 -> 190-230, **-70..-110 ms**, the largest single lever. **Quality risk**: activations
  quantized per 64-group in every trunk projection (q/k/v, GDN in_proj, HC down/up, shared expert) change the KV
  and states written by prefill.
- **Cost**: medium-high. A Stage-A kernel, an A8 variant of `pf_qmm_t` with the code-sum correction (uint8 code
  -> code - 128, folded into the bias term), and a quality run (test-quality A/B/C on prefilled contexts).
- **A/B (cheap first step)**: an MPP int8 `matmul2d` microkernel at [10240 x 2560], M = 2048, against the half
  version, both without dequant. Prediction: int8 >= 1.4x. If it is < 1.15x, drop the lever. Then the full GEMM:
  >= 70 TOP/s-equivalent and rel L2 vs f64 <= 3e-3; then KLD within +0.0005 of the A16 prefill on A/B/C.

### 4. Fused HC read / write (oMLX `hc_prefill_nax.py`, Sushi `hc_prefill_norm.metal` WR)
- **Where**: vendored `qwen4_exp/hc_prefill_nax.py:1-30`; Sushi `src/hc_prefill.zig:49-98`,
  `src/qwen4_hc.zig:224-247`.
- **Why it saves**: per layer we run `pf_hc_write` after the mixer (reads h f32 84 MB, writes 84 MB), and per HC read
  `pf_hc_norm`, down GEMM, `pf_hc_act`, up GEMM (writes [n, 10240] bf16, 42 MB), and `pf_hc_blockin` (reads it back).
  Folding the mixer's write into the next norm saves one 84 MB pass per layer (~8 ms per chunk). SiLU in the down
  epilogue and gate / product / mean in the up epilogue save the 42 MB write plus read per HC read, 96 per chunk
  (~15 ms). This is our own lever (3), with a worked design and bit-identity recipe.
- **Expected**: HC glue 68 -> ~40-48, **-20..-30 ms**.
- **Cost**: medium (2 GEMM epilogues in pf_qmm_t; the up GEMM needs the column-tile regrouping of 16 hidden columns
  x 4 streams).
- **A/B**: profile stage hc_glue + trunk_gemm(HC). Prediction: >= 20 ms less per 2048 chunk; outputs bit-identical
  to today's path (same rounding points). Falsified if < 10 ms.

### 5. f16 output planes for the expert GEMMs (Sushi `nax_store` to half)
- **Where**: Sushi's sorted GEMM output is `.float16` (`src/exl3/expert_exl3_kernels.zig:1384`, `sortedGemmCfg`);
  ours stores f32 (`kernels/prefill/pf_moe.metal:236-248`, `nax_store`).
- **Why it saves**: per MoE layer at 2048 rows the f32 planes are gate / up [20480 x 640] 52 MB each and down
  [20480 x 2560] 210 MB, written and read once each (~630 MB of traffic). Half halves it: ~0.3-0.6 ms per layer when
  not hidden under the decode-bound GEMM.
- **Expected**: **-10..-25 ms**. Range: f16 has 11 mantissa bits (more than the bf16 the finish rounds to). Sushi
  serves Flash-Next with it; its MiMo note puts the f16 down-plane ceiling ~2x above the measured peak.
- **Cost**: low (template Y; `pf_moe_mid` / `pf_moe_finish` read half).
- **A/B**: profile moe_gemm + moe_glue. Prediction: >= 10 ms less; `pf-check layers` hidden rel L2 within 5% of
  today's. Falsified if < 5 ms.

### 6. 48-row windows (NB = 3) — not tried by Sushi (it tried 64)
- **Why**: at 2048 the mean is 40 rows per expert, so `ceil(c/32)` = 2 windows where `ceil(c/48)` = 1. Window count
  falls ~25-35% (inferred). Per-window live state rises by one destination tile (64-row windows with 4 lost 9-18% in
  Sushi and 10% here). MLX's own rule moves to taller tiles only from 64 rows per expert (section 2).
- **Expected**: 0..-30 ms, uncertain; less on top of lever 1.
- **Cost**: low (`pf_moe_gemm<..., NB = 3>` instantiation + `WIN = 48` in `pf_moe_scan`).
- **A/B**: moe_gemm stage at chunk 2048. Falsified if not >= 3% faster.

### 7. bf16 HC stream in prefill (Sushi)
- **Why**: halves every HC / finish / write pass over the hidden (84 -> 42 MB). Expected **-30..-40 ms** on top of
  lever 4.
- **Risk**: the decode path keeps f32. The prefill would write KV / states from a bf16-rounded stream. Our
  rounding-ON arm cost +0.0035 KLD on set A (STATUS 15 Oct 17:35), so expect a measurable quality cost. Only with
  a teacher-KLD check. Falsified (drop) if A > +0.001.

### Checked and not worth porting (evidence)
- Expert GEMM body, run-aligned windows, GPU window table, inverse-indexed finish: already ours
  (`pf_moe.metal:252-300` = `GEMM_NAX_SOURCE`; `pf_moe_count/scan/place`).
- Threadgroup-shared decoded weights (MLX / oMLX style for affine): +30% for trellis (Sushi).
- kv8 packed gathers for wide chunks: 15-20% slower (Sushi), which rebuilds bf16 once per chunk instead.
- FLA chunked WY GDN: slower end to end (oMLX). Prefer lever 2 over our docs/prefill.md lever (4).
- Counting sort instead of argsort: slower in Sushi's MLX graph; ours is already a GPU count / scan / place with no
  host sync.
- ANE hybrid prefill: dense MLPs only, private runtime. Out of scope for the MoE.

## 6. What could not be verified
- MLX's C++ dispatch sources are not on disk. BM 32 / 64 is read from the disassembly (inferred). oMLX's native QSA
  and A8 Stage-A kernels are compiled only (metallib).
- The M5 INT8 tensor-op rate, and oMLX's A8 speed / NLL numbers for qwen4_exp (the comment says they were measured,
  the tables are not shipped).
- oMLX's GDN timings are from an M5 Ultra; ours are from the M5 Max.
- Sushi's per-rate (K4 vs K5) prefill GEMM time; Sushi-4bpw's prefill stage split (no per-stage prefill
  attribution in its docs beyond the old "trellis GEMMs ~520 ms per 2048 chunk").

## Attribution
Sources read: Sushi (https://github.com/beamivalice/sushi, MIT, Copyright (c) 2026 Theinruj Toranavikrai, Copyright
(c) 2026 David Dalcu); oMLX (jundot/omlx, Apache-2.0); MLX 0.32.3 (MIT, Copyright (c) 2023-26 Apple Inc.). No code
copied into this repo by this document; ports of levers 2-5 must carry the respective licence.
