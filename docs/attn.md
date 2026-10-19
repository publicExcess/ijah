# QSA full attention (lab-attn)

Owner: lab-attn (`kernels/attn/`, `tests/attn/`, `tools/attn_*`, this file). Status 15 Oct 2026: **the attention of
the 12 full-attention layers (3, 7, ..., 47) with the reference's QSA sparse selection, at 1..8 rows, on f32 / bf16 /
kv8 caches, with rollback; matches lab-ref's long goldens (selection exact, block keys and caches bit-exact,
attn.gated 1.4e-3 / 2.3e-3).** lab-forward records it in every attention layer of the forward (default; `NE_FN_ATTN=dense`
= the old dense path). Math: docs/reference.md 3.4; reference code: mlx-vlm 0.7.7 `qwen4_exp/language.py`
(`Qwen4ExpQSAIndexer.select_from_projected`, `build_mask`).

16 Oct (lab-attn2): **the prefill-chunk attention core** (section 8): every row of a 2048-token chunk in one NAX
dispatch over its own selection, the selection batched (one dispatch per stage for the whole chunk); per layer-chunk
30.9 -> 9.7 ms (8K) / 32.8 -> 11.4 ms (30K), caches and selections bit-identical to the step path; whole-model prefill
10K 1,669 -> 2,019 tok/s, 32K 1,542 -> 1,851 (Sushi ~1,900-2,100 / 2,384). Needs the prefill.mm patch in Requests.

## 1. Files

| file | what |
|---|---|
| `kernels/attn/qsa.metal` | `qsa_prep`, `qsa_score` / `qsa_score_mma`, `qsa_select_local`, `qsa_select`, `qsa_attn` (SIMT) / `qsa_attn_mma` (simdgroup matrices), `qsa_merge`; cache import / export, block-key rebuild, pos add / rollback. Rows R = 1..8 and the formats are template parameters (`_f32 / _bf16 / _kv8`, `_r1 .. _r8`) |
| `kernels/attn/qsa_chunk.{metal,h,mm}` | prefill chunks: `record_chunk` (batched prep / block keys / scores / selection, dense rows on lab-prefill's NAX flash kernel, `qsa_chunk_attn_bf16` for the rest), `alloc_chunk_scratch`, `alloc_step_scratch`; C API `ne_attn_chunk*` (section 8) |
| `tests/attn/test_chunk.py`, `tests/attn/prefill_chunk.patch` | chunk core vs the step path (24 checks); the models/flash_next/prefill.mm patch that routes the prefill through `record_chunk` |
| `kernels/attn/qsa.{h,mm}` | `ne::attn`: `QsaCache` (caches + step scratch), `alloc_cache`, `record_step` / `record_one`, `record_advance`, `record_rollback`, `import_cache` / `export_cache` / `truncate`, `load_qsa_weights`; C API `ne_attn_*` for the tests |
| `kernels/attn/attn.mk`, `tests/attn/attn_tests.mk` | `qsa.mm` into libengine (one `-include` line in `kernels/kernels.mk`); `make test-attn`, `make bench-attn` |
| `tests/attn/test_qsa.py` | 266 checks: goldens, R = 2..8, synthetic contexts to 140K, rollback, kv8 (section 4) |
| `tools/attn_bench.py`, `tools/with_lock.sh` | timing at 2K..128K (section 5); the lock wrapper starts the job only once the (optional) machine lock is ours |

## 2. Interface and layouts

```cpp
ne::attn::QsaCache c = ne::attn::alloc_cache(ne::attn::KvFmt::BF16, tmax /* % 64 == 0 */, &first_layer_cache);
ne::attn::record_step(r, w /* load_qsa_weights(pack, layer) */, c, R, q, kv, iqk, out, /*advance=*/false);
```
In: fn_attn_proj's outputs `q` f32 [R][12288] (24 heads x [256 query | 256 gate]), `kv` f32 [R][1024] (k_proj | v_proj),
`iqk` f32 [R][640] (index_qk_proj: 4 query heads + the raw key, x 128). Out: `out` f32 [R][6144] = attention *
sigmoid(gate), what fn_gdn_outproj (o_proj + HC write) reads. `c.pos[0]` = T = tokens in the cache before the step (a
GPU buffer: recordings replay; the forward shares one pos buffer across its 12 caches).

| buffer | layout | per token, per layer |
|---|---|---|
| `kc`, `vc` | f32 / bf16 [2 kv heads][tmax][256] (keys stored roped); kv8: u8 codes, same shape | 4096 / 2048 / 1024 B |
| `ksb`, `vsb` (kv8) | ushort2 [2][tmax][4] = (bf16 scale, bf16 bias) per 64-dim group; w = scale * code + bias | 64 B |
| `ikeys` | bf16 [tmax][128] raw indexer keys (un-normed, un-roped, as the reference keeps them) | 256 B |
| `bkeys` | bf16 [tmax/4][128] block keys: rope(k_layernorm(bf16(mean_f32 of 4 raw keys))) at position 4b | 64 B |
| step scratch | `qr` [8][24][256], `iq` [8][4][128], `scores` [8][tmax/4], `sel` [8][512], `nsel` [8], `cand` [8][tmax/8192][512], `part` [2][8][65][16][258] f32 | ~7 MB + 36 B x tmax, shareable (`alloc_cache(.., &other)`) |

Per layer and token: bf16 2368 B, kv8 1408 B, f32 4416 B; 12 layers x 128K tokens: bf16 3.7 GB, kv8 2.2 GB.

## 3. Kernels (one step = 6 dispatches, barriers between)

| # | kernel | grid x threads | does |
|---|---|---|---|
| 0 | `qsa_prep` | 30R + ceil(R/4) x 32 | one simdgroup per task: q head norm + rope + bf16, * 1/16 -> `qr`; k norm + rope + bf16 and v bf16 -> cache row T + t (kv8: MLX affine quantize per 64 dims = 8 lanes); indexer query q_layernorm + rope -> `iq`; raw key -> `ikeys`; the block keys of blocks completed by this step, b in [T/4, (T+R)/4) (tokens < T from `ikeys`, the others straight from `iqk`: no race) |
| 1 | `qsa_score_mma` | min(tmax/128, 320) x 256 | scores[t][b] = sum_h relu(iq[t,h] . K_b) / sqrt(128), b < (T+R)/4: S = IQ [4R x 128] . K^T on simdgroup matrices, bfloat operands (both bf16 values: exact products), f32 accumulation. (`NE_QSA_SCORE=simt`: the SIMT kernel, 8 lanes per block) |
| 2 | `qsa_select_local` | R x tmax/8192 x 1024 | rows with comp > 2048 complete blocks: top 512 of each slice of 2048 blocks -> `cand` |
| 3 | `qsa_select` | R x 1024 | comp = (T+t+1)/4. comp <= 512: blocks 0..comp-1 (= causal, as the reference's dense path); comp <= 2048: top 512 of the scores; else top 512 of the candidates. Exact radix select (below), output ascending |
| 4 | `qsa_attn` / `qsa_attn_mma` | 2 x R x 65 x 384 / 128 | split-K: TG (kv group, row, chunk); chunk c < 64 = blocks sel[8c, 8c+8) (32 keys), chunk 64 = the tail [4 comp, T+t]; partial (acc[256], m, l) per head |
| 5 | `qsa_merge` | 24R x 256 | per (row, head): 8 simdgroups merge chunks s, s+8, .. in order, then the 8 in order (deterministic); * sigmoid(gate) with the reference's bf16 op chain |

**Selection (exact, tie rule = lower block id).** Monotone f32 -> u32 ordinal; MSD radix select in 8-bit digits
(threadgroup histogram with run-length adds, early stop when a bin holds exactly what is still needed), then a
compaction in index order (selected iff ord > threshold, or == threshold among the first `need_eq`), so the list is
born ascending. Up to 8192 elements per thread group stay in registers across the passes. Two stages past 2048 blocks:
the union of per-slice top-512s contains the global top 512, and the slices are visited in index order, so ties
still go to the lower id. Follows Sushi's `sushi_qsa_select` (below). The reference uses `argpartition` (arbitrary on
ties); the goldens have no ties at the cut (512th vs 513th score differ by 1e-3..1e-2 relative).

**Attention.** `qsa_attn` (f32, bf16): 12 simdgroups = the 12 q heads of the kv group, each walks the chunk one block
(4 keys) at a time: 4 dots, a butterfly reduction of the 4 sums (10 shuffles), one online-softmax update; the next
block's rows load while the current one computes; K / V rows hit L1 for 11 of the 12 heads. `qsa_attn_mma` (kv8): S
[16 x 32] = Q [12 heads + 4 zero rows] . K^T, softmax per row (lane = key), O = P . V, on simdgroup matrices with K
staged as half (dequantized: 11 bits, finer than the codes) and P, V as half, f32 accumulation.

**Rollback (MTP).** Only `pos` is state to undo: rows >= T' and block keys of blocks with 4b+3 >= T' are dead and get
rewritten by the step that re-appends them (block keys are written by the step that completes a block, from the raw
keys then in the cache), kv8 groups are per row. `truncate(c, T')` on the host or `record_rollback(r, c, keep, R)`
(pos = pos - R + keep[0], keep = u32 GPU buffer) in a recording. With the forward's shared pos buffer, roll back once.

## 4. Correctness (`make test-attn`, 266 checks, ~3 s)

Fed the goldens' own projections (`attn.q_proj`, `k_proj`, `v_proj`, `idx_qk`) and caches (`*_in`). rel L2 vs the
golden, bf16 cache, reference rounding on (default):

| tensor | L03_decode1 (48) | L03_verify4 | L03_long_decode1 (2304) | L03_long_verify4 |
|---|---|---|---|---|
| `attn.idx_q`, `attn.q_rope`, raw index keys, block keys, new K / V rows | 0 (bit-exact) | 0 | 0 | 0 |
| `attn.idx_scores` (f32) | 7e-8 | 9e-8 | 7e-8 | 7e-8 |
| `attn.selected_blocks` | causal list = all 12 | 13/13 | 512 of 576, exact | 4 rows exact |
| **`attn.gated`** | **8.4e-4** | **1.2e-3** | **2.3e-3** | **1.4e-3** |

Rounding off (`NE_FN_ROUND=f32` / `ne_fn_set_ref_rounding(0)`): gated 3.6e-3 / 3.5e-3 on the long cases. f32 cache =
bf16 cache (the reference stores bf16). kv8 vs the bf16 goldens: gated 7.3e-3 / 7.6e-3 (decode / verify, long),
1.0e-2 / 9.0e-3 short (the quantization); the kv8 cache equals a numpy port of MLX `affine_quantize` on 99.97% of codes (the rest one step: codes
use a precise reciprocal instead of a division), and the kv8 attention matches numpy over the same dequantized cache
at 8.6e-4. R = 2, 3: prefixes of L03_long_verify4 (row t depends on rows <= t only), same numbers. R = 5, 8 and
synthetic caches at 9K / 20K tokens (the two-stage select) vs the numpy reference (`tools/opref.py` + the MLX rope
table): selection exact, gated <= 7.3e-5 (bf16), <= 1.2e-3 (kv8: half staging in the MMA kernel); 140K tokens (merge stage not
register-cached): selection exact. Rollback: verify4 -> truncate -> decode1 is bit-identical to a fresh decode1;
accept 2 of 4 (block 576 half-written), re-append 4 other rows: block keys, selection and output match numpy.
Repeated runs are bit-identical.

## 5. Speed (M5 Max, `tools/attn_bench.py`, synthetic cache, machine lock held)

One full-attention layer's step (all 6 dispatches; o_proj not included), us. *flushed* = a 256 MB streaming read
between steps (SLC evicted; its cost subtracted): what a step inside the forward sees, with its KV cold. *GB/s* =
the bytes the step must read (block keys T/4 x 256 B + the selected K/V rows) / flushed time.

| T | R | bf16 warm / flushed | GB/s | kv8 warm / flushed | GB/s | f32 flushed |
|---|---|---|---|---|---|---|
| 2048 | 1 | 27 / **47** | 92 | 32 / **51** | 46 | 77 |
| 2048 | 4 | 79 / 109 | 156 | 83 / 118 | 77 | 156 |
| 2048 | 8 | 130 / 175 | 192 | 141 / 205 | 88 | 276 |
| 8192 | 1 | 37 / **46** | 103 | 38 / **62** | 45 | 79 |
| 8192 | 4 | 88 / 116 | 150 | 93 / 131 | 72 | 185 |
| 32768 | 1 | 50 / **70** | 90 | 47 / **65** | 66 | 94 |
| 32768 | 4 | 91 / 124 | 152 | 98 / 138 | 80 | 194 |
| 131072 | 1 | 66 / **87** | 144 | 71 / **94** | 113 | 113 |
| 131072 | 4 | 121 / 154 | 164 | 127 / 169 | 103 | 233 |
| 131072 | 8 | 192 / 233 | 181 | 200 / 263 | 100 | 378 |

Per dispatch (SLC-warm, 64 vs 320 in one command buffer), bf16, 1 row: prep 2.9, score 2.7 (2K) .. 9.5 (128K),
select_local 1.1 / 7.1, select 1.2 (2K) / 6.7 (8K, 32K) / 12.8 (128K), attention 16.6, merge 3.4. At 4 rows:
attention 50, score 15 (128K), selects 17 + 18 (128K). kv8 attention (MMA) 16.2 / 51 (1 / 4 rows), f32 39.

- 12 layers at 1 row: ~0.55 ms (2K) to ~1.05 ms (128K) per token on bf16. Sushi (same model, kv8): "attention +
  indexer is 6-13% of decode" at ~18 ms per forward = ~90-190 us per layer, and its S = 16 wide verify 3.8-4.1 ms
  per 12-layer forward (8K..256K keys, ~330 us per layer at 16 rows); ours: 47-94 us (1 row), 233-263 us (8 rows,
  128K). docs/qwen4-perf.md and qwen4-qsa-long-context.md have no per-kernel decode numbers to compare closer.
- The step is latency- / issue-bound, not bandwidth-bound (90-190 GB/s of necessary bytes): the attention over
  2048 keys costs the same with every TG reading one block (all in L1), and the per-dispatch costs add up to the
  warm step; flushed adds ~15-25 us of cold dependent loads (pos -> nsel -> sel -> rows) over 6 dispatches.
- kv8 halves the KV bytes but not the time (decode is ALU-bound): its value is RAM (2.2 vs 3.7 GB at 128K x 12
  layers) and matching Sushi's serving default in our tests (`--kv-quant 8`).

## 6. Traps

- **Rope table.** MLX computes inv_freq = 1 / 1e7^(2d/64) on the GPU; 19 of the 32 f32 values differ by one ulp from a
  correctly rounded table (11 from numpy's). At position 131072 one ulp of inv_freq[0] moves the angle by 0.008 rad.
  `qsa.mm` carries MLX's table bit for bit (`kInvFreqBits`); with it q_rope, block keys and k rows are bit-exact at
  2304 (with a double-precision table: 1.4e-5 / 4.7e-5 / 4e-6). Angles = f32(pos * inv_freq), `precise::cos/sin`
  (MLX's cos is accurate to 1.3e-7 at 1e5 rad). `tools/opref.py rope` uses numpy's table (lab-ref: see Requests).
- **Gate.** The reference's `output * sigmoid(gate)` is an uncompiled bf16 op chain: 1 / (1 + exp(-x)) with every step
  rounded to bf16, then the product rounded. One rounding of an f32 sigmoid left attn.gated at 3.4e-3; the chain gives
  0.8e-3..2.3e-3.
- **kv8 quantizer.** MLX starts w_max at 0 (so w_max >= 0) and w_min at +max, computes codes with the f32 (not the
  bf16-stored) scale and bias, and rounds half away from zero. GPU fast-math division moved 0.5% of codes by one step
  (and the scale of some groups); precise division made it exact, a precise reciprocal is 99.97% exact and 3 us
  cheaper per step.
- **Block keys and decode.** The reference skips block-key upkeep at S = 1 while <= 512 blocks are complete and
  catches up (all missing blocks) on the next call that needs them; block keys depend only on raw keys and positions,
  so writing each block when it completes is equivalent (checked: 576 / 577 keys bit-exact).
- **Timing.** Short chains of small dispatches run at low GPU clocks: 8 vs 40 repeats of a 3 us kernel read 15-40 us
  and moved between runs; per-dispatch numbers need 64 vs 320 repeats. Per-dispatch *flushed* numbers are wrong for
  kernels whose inputs the previous dispatch just wrote (the merge read 38 us flushed, 3.4 us in the step): only the
  whole step is timed flushed. `tools/with_lock.sh`'s first version started the job before the lock was granted
  (with_lock.sh's pattern relies on acquire failing fast); one 46 s bench ran beside lab-forward's MLX job before
  the fix (the lock now gates the job's start).
- Divergent `simd_sum` is undefined: every simd reduction sits in uniform control flow (the compaction loop runs a
  fixed trip count per threadgroup and breaks on a threadgroup-uniform total).
- Rejected shapes (T = 8192, bf16, 1 / 4 rows, attention alone): 2 or 4 heads per simdgroup (fewer simdgroups in flight:
  18-30 us), 2 or 4 blocks per chunk (more partials, merge 8-32 us), 16 blocks per chunk (fewer TGs) all within noise or
  worse than 1 head x 8 blocks; the simdgroup-matrix attention ties it on bf16 (15.7 vs 16.6 us) and wins on kv8. The
  MMA kernel with 4-block chunks: 21 vs 17.5 us (1 row), 61 vs 55 (4 rows); loading all 8 block ids up front in the
  SIMT kernel: no change. Both kernels sit at ~16-17 us per 2048 keys at 1 row whatever the shape: the next step is
  NAX (section 7), not tuning.

## 7. Integration and what is left

- lab-forward records `record_step` in every attention layer (fe44a5e, default, bf16, `NE_KV` selects the format);
  the step is now 6 dispatches (select_local added) and reads `ne::fn::g_ref_round` (their request, done).
- Prefill: `record_chunk` (section 8) once lab-integrate applies `tests/attn/prefill_chunk.patch`.
- Faster: (1) the attention core is issue-bound: NAX matmul2d (lab-mtp's `kernels/nax`) for QK^T / PV over 16-row tiles
  (Sushi's matmul2d prototype: -30..45% on its attention); (2) verify rows share most blocks: a union-of-blocks kernel
  reads each K / V row once for R rows (Sushi's `qsaVerifyGatherAttn`); (3) fold merge into the attention (last
  arriving TG per (row, group)) and select_local into select below 2048 blocks: -2 dispatches, ~5-8 us cold;
  (4) skip score + select while every row is dense (<= 512 blocks): ~4 us at short context.

## 8. Prefill chunks: the multi-row core (lab-attn2, 16 Oct)

**The premise did not hold: neighbouring rows do not select mostly the same blocks.** Layer 3's selections of every
row of a real 32K prompt (this repo's docs and sources, 4-layer partial model, chunk 2048), distinct blocks in the union
of P consecutive rows, x 512 (the per-row count):

| chunk start | P = 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| 4096 | 1.71x | 1.95 | 2.15 | 2.29 | 2.40 |
| 10240 | 2.40x | 3.04 | 3.72 | 4.30 | 4.78 |
| 20480 | 2.78x | 3.84 | 5.12 | 6.54 | 7.81 |
| 30720 | 3.09x | 4.39 | 6.11 | 8.19 | 10.27 |

A union tile over 64 rows with per-row masks would do 4-8x the work of the rows' own blocks (and 2.4-3.1x at P = 8),
so the core does **one (row, kv group) per threadgroup over the row's own 512 blocks + tail**, with the 12 q heads of
the group as the rows of a 16-row NAX tile: each selected K / V row is read once per (row, group) instead of once per
head (qsa_attn's simdgroup per head), on the matrix units, and with no split-K there is no partial / merge pass.

**Kernels** (`kernels/attn/qsa_chunk.metal`, `qsa.metal`; `record_chunk`):

| # | dispatch | grid | does |
|---|---|---|---|
| 1 | `pf_set_pos_multi` | 1 | slice positions T0 + 8 j |
| 2 | `qsa_chunk_prep_{fmt}` | 30 n x 32 | qsa_prep's tasks for all n rows at once (same code: `prep_body`): q / k norms + rope, cache append, indexer queries, raw keys |
| 3 | `qsa_chunk_blocks` | n/4 + 1 x 32 | block keys of the blocks the chunk completes, from the cached raw keys (after a barrier: no ordering between slices needed) |
| 4 | `qsa_chunk_score` | 16 x slices x 256 | qsa_score_mma<8> per 8-row slice at its own T (`score_mma_body`), every sparse slice in one grid; scores f32 [C][tmax/4] |
| 5, 6 | `qsa_select_local_r8`, `qsa_select_r8` | rows x slices / rows | the step's select kernels over every sparse row (their body depends only on the row index and T) |
| 7 | `qsa_chunk_attn_bf16` | (sparse rows, 2) x 128 | S [16 x 32] = Q [12 heads + 4 zero rows] . K^T per 8 selected blocks (keys gathered: 4 cache rows per lane), online softmax (exp2), O [16 x 256] += P . V; 4 simdgroups split the head dim (64 each), S partials summed in a fixed order through 8 KB of threadgroup memory; the tail [4 comp, pos] is a 513th, partial block; gate in the store (qsa_merge's bf16 chain) |
| 7' | `pf_attn_qcopy`, `pf_attn_nax_d256`, `pf_attn_gate` | | rows with <= 512 complete blocks (positions < 2051, bf16, T0 known): lab-prefill's dense NAX flash attention, concurrently with 7 |

kv8 / f32 caches (or `NE_QSA_CHUNK=0`) keep the old per-slice path (16 slices of qsa_prep .. qsa_merge at a time).
Scratch (`alloc_chunk_scratch`, C = 2048): qr 50 MB, scores 4 B x C x tmax/4 (64 MB at 32K, 256 MB at 128K), cand
4 B x C x 512 x tmax/8192 (16 / 64 MB), sel 4 MB, qb / ob 50 MB, + 16 step-scratch sets for the fallback.

**Exactness** (`tests/attn/test_chunk.py`, 24 checks; synthetic caches T0 = 1024 / 8192 / 30720, 2048 and 1001 rows,
indexer queries as a random walk; reference = the step path, which test_qsa.py ties to the goldens): K / V rows, raw
and block indexer keys and every row's selection **bit-identical**; gated output rel L2 1.19e-3 .. 1.33e-3, worst row
1.71e-3 with reference rounding ON (89% of the elements bit-equal, the rest one bf16 step: P . V runs bf16 V x f32 P
on the NAX vs f32 SIMT); rounding OFF 2.9e-4 (the dense NAX path, which stores O as bf16: 1.7e-3). Real text, 4-layer
partial model, 8192 tokens (`pf-check layers`): hidden after layer 3 vs 1-row decode rel 6.08e-3 median / 5.25e-2 max
(old path 6.10e-3 / 5.25e-2; decode's own 4-row floor 5.33e-3); whole model (`pf-check model`, 8192 + 32 positions):
KLD mean 5.52e-2 vs 5.44e-2 for the old path on the same prompt, worst state 0.133 vs 0.134 (this prompt's decode
floor is high: lab-integrate's docs/forward.md 2c).

**Speed** (synthetic, one layer, chunk 2048 rows, GPU ms, `test_chunk.py --time`):

| T0 | step path (16 slices x 6 dispatches) | chunk core | of which attention kernel |
|---|---|---|---|
| 1024 | 26.7 | 7.8 | 7.3 |
| 8192 | 30.9 | 9.7 | 8.5 |
| 30720 | 32.8 | 11.4 | 8.5 |

Whole model (`pf-check speed`, real text, chunk 2048, same binary with `NE_QSA_CHUNK=0 / 1`, interleaved runs; on
a quiet machine, 16 Oct ~05:45, before the batched selection):

| prompt | old path tok/s | multi-row core | attention per chunk (profile) | share of the chunk |
|---|---|---|---|---|
| 10240 | 1,669 / 1,672 | **2,006 / 2,032** (2,009 / 2,084 with batched selection) | 310 -> 114 -> **106 ms** | 25.6% -> 10.1% |
| 32768 | 1,539 / 1,544 | **1,848 / 1,854** (1,779 / 1,822 batched, noisier run) | 380 -> 157 -> **137 ms** | 28.6% -> 12.4% |

vs Sushi (served path, docs/prefill.md 4): ~1,900-2,100 at 10K (now par), Sushi-4bpw 2,384 at 32K (0.78x). Runs at
other times of the morning read 5-15% lower for both paths alike (another agent's oMLX job running); compare paired:
final build at 06:35, two pairs each: 10K 1,626 / 1,525 -> 2,102 / 1,949, 32K 1,453 / 1,457 -> 1,747 / 1,805.
Dispatches per 10K prefill 86,351 -> 8,963.

**What did not work** (attention kernel, T0 = 8192): prefetching the next tile's K into registers and loading V with
K at the tile start: 16.5 ms vs 11.6 (register pressure); the block list staged in threadgroup memory: no gain; two
key streams per threadgroup (8 simdgroups, fixed-order combine): 9.4 vs 8.5 ms. What did: K / V as 4-element vector
loads per 16-dim slice used at once, and the mask computed without a device load (11.6 -> 8.5 ms). The kernel runs at
~16 TFLOP/s of padded matrix work (25% of the tile is the 4 zero heads) and reads ~8.4 GB of K / V per layer-chunk
from L2 / SLC (~1 TB/s).

**Next**: (1) 2 rows per threadgroup over the union (P = 2: 24 matrix rows of 32, union ~1.3-1.5x) or the 4 zero heads
filled with a second row's heads when their block lists coincide; (2) the attention kernel for kv8 (dequantize in the
gather) so kv8 caches leave the per-slice path; (3) dense chunk 0 through the same kernel when T0 is unknown (today
the recording is keyed by T0 for chunks with dense rows); (4) scores at 128K: 256 MB of chunk scratch, process the
chunk's selection in 512-row quarters if RAM matters.

## Requests

- **lab-integrate / lab-prefill** (from lab-attn2, 16 Oct; docs/attn.md section 8): route the prefill's attention
  core through `attn::record_chunk` (10K 1,669 -> 2,019 tok/s, 32K 1,542 -> 1,851; caches and selections bit-identical
  to today's path). Exact patch: `tests/attn/prefill_chunk.patch` (`git apply` on 5147b0e's
  models/flash_next/prefill.mm, 8 lines in / 93 out): `#include "kernels/attn/qsa_chunk.h"`; in `Prefill` replace
  `qb, ob`, `pslot` and `slot` by `attn::ChunkScratch ac;`; the body of `record_attn` between the q / k / v / indexer
  GEMMs and the o_proj GEMM becomes `Recorder &r = sg.r(PfStage::Attn); attn::record_chunk(r, w.qsa, st.qc, p.ac, n,
  p.no_dense ? -1 : T0, p.posbase, p.aq, p.akv, p.aiqk, p.aout);`; `prefill_create` allocates `p->ac =
  attn::alloc_chunk_scratch(fmt, tq, C)` instead of the 16 slot sets; `prefill_tokens` keys recordings by T0 when
  `attn::chunk_dense_rows(p.ac, m.T, cn) > 0` (was `dense_rows(m.T, cn) >= 8`, the same rows); `dense_rows`,
  `kSlots`, `PfAttnParams` go. `NE_QSA_CHUNK=0` = the old per-slice path at run time (A/B). Checked with it applied
  (a scratch worktree): pf-check layers / model / speed / lspeed. lab-prefill's request (1) `alloc_step_scratch`:
  done (kernels/attn/qsa_chunk.h).
- **lab-forward**: (1) share the step scratch across the 12 caches: `attn::alloc_cache(fmt, tmax, &first.qc)` for
  layers 7..47 (each cache otherwise carries ~7 MB + 36 B x tmax of scratch: `part` alone is 6.3 MB). (2) On the QSA
  path `GdnState.ikeys` (f32 [tmax][128], written by fn_attn_proj and copied by `ne_fn_layer_set_kv`) is never read:
  `qsa_prep` keeps the raw keys in `qc.ikeys` (bf16). At tmax = 131072 it is 64 MiB per layer, 0.75 GiB for 12: bind
  a 16-byte dummy there (or drop the store) when `g_attn_qsa`. (3) MTP: with the shared pos buffer, roll back once per
  round (`record_rollback(r, any qc, keep, R)` or the host `pos[0] = T0 + keep`), not once per layer.
- **lab-ref**: `tools/opref.py`'s `inv_freq` (numpy's f32 table) differs from the MLX reference's GPU table by one ulp
  in 11 of 32 entries: fine at 2304, ~1e-2 rad at 128K positions. `tests/attn/test_qsa.py` patches in `MLX_INV_FREQ`
  (the bits are in `kernels/attn/qsa.mm`); the attention gate is `bf16(core * sigmoid_bf(gate))` (1.2e-3 vs 3.4e-3
  on the goldens with `bf16(core * bf16(sigmoid))`).

## Attribution

- `qsa_chunk_attn_bf16` uses MLX's steel NAX fragment helpers (BaseNAXFrag: get_coord / matmul2d mma / row_reduce)
  vendored by lab-prefill in `kernels/prefill/mlx/gemm_nax.h`, and follows the loop of MLX's `attention_nax_dsplit`
  (head-dim split, fixed-order S exchange) with gathered K / V rows and QSA masks (MLX 0.32.3, MIT, Copyright (c)
  2024-26 Apple Inc.; `kernels/prefill/mlx/LICENSE.mlx`). Sushi's prefill gather (`gatherQsa256`: one (query, KV
  head) per threadgroup, the 12 grouped heads as rows of a 16-row tile, docs/qwen4-qsa-long-context.md) informed the
  shape; no Sushi code is used in it.

- The radix select follows Sushi's `sushi_qsa_select` (`src/qwen4_qsa.zig`: ordinal map, 8-bit MSD digits with
  run-length histogram adds, early stop, index-ordered compaction with the lower-index tie rule, per-slice split +
  merge), github.com/beamivalice/sushi, MIT, (c) 2026 Theinruj Toranavikrai, David Dalcu. Written anew for this engine
  (one row per threadgroup, register-cached elements, fixed slices of 2048 blocks).
- The kv8 quantizer reproduces MLX's `affine_quantize` (mlx/backend/metal/kernels/quantized.h, MIT, (c) 2023 Apple
  Inc.); the cache format (8-bit, group 64, bf16 scales / biases) is Sushi's / MLX's `QuantizedKVCache` default.
- MIT notice: permission is hereby granted, free of charge, to any person obtaining a copy of this software ... THE
  SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND (full text in each upstream LICENSE).
- Everything else in kernels/attn (prep, MMA scores, split-K attention, merge, caches, rollback) was written for this
  repo; the reference semantics are mlx-vlm 0.7.7 `qwen4_exp` (read only).
