# Kernels and the Flash-Next layer graph (lab-layer, now lab-forward)

Owner: lab-speed since 15 Oct evening (the layer kernels in `kernels/` incl. `kernels/nax/`, not `kernels/attn/` or
`kernels/prefill/`; this file; section 5b is the decode-step profile). Before: lab-layer (`kernels/`, `models/flash_next/`); since 15 Oct afternoon lab-forward (the whole model:
docs/forward.md). lab-forward additions: `fn_gdn_inproj` with separate qkv / z-a-b groups, PLE in the layer (layer 1),
`kernels/fn_model.metal`, lab-attn's QSA step as the attention core (default). Original status: Status 15 Oct 2026: **layer 0 (GDN) and layer 3
(full attention, dense path) run at 1, 2 and 4 rows on the real K5 pack and match lab-ref's goldens per sub-block.**
Numbers are indicative (one machine, best of 5, machine lock held).

## 1. Files

| file | what |
|---|---|
| `kernels/ne_common.h` | shared MSL: 8-bit affine dots (MLX layout, 1 and R rows), bf16, H128 FWHT (simd shuffles), EXL3 trellis lane reader + MCG w15 decode (attribution inside), the reference-rounding switch `NE_REFR` / `rr()` |
| `kernels/fn_trunk.metal` | `fn_hc_pre_g{64,128}_r{1,2,4}`, `fn_hc_mix_*`, HC read v2 `fn_hc_pre2_*` / `fn_hc_pre3_*` + `fn_hc_mix2_*` (lab-speed), `fn_gdn_inproj_*`, `fn_gdn_step_r*`, lazy store `fn_gdn_step_lz{K}_r*` + `fn_gdn_materialise` (lab-gdnlazy, 5d), `fn_gdn_outproj_*` (also the attention out_proj) |
| `kernels/fn_attn.metal` | `fn_attn_proj_g*_r*`, `fn_attn_core_r*`, `fn_attn_merge_r*` |
| `kernels/fn_moe.metal` | `fn_moe_router_r*`, `fn_moe_gateup_k{4,5,6}_r*`, `fn_moe_down_k*_r*` (lab lead 4's 3-dispatch block on real tensors) + deduped `fn_moe_gateup_u_*` / `fn_moe_down_u_*` (R > 1, opt-in) |
| `kernels/util.metal` | `ne_stream_read` (SLC flush for timing), `ne_empty` (boundary cost) |
| `kernels/fn_rec.{h,mm}` (lab-speed) | recording helpers for the v2 kernels (`krec::hc_pre`, `hc_pre3`, `hc_mix`: drop-ins for record_stage's HC stages) |
| `kernels/bench/fn_prof.mm` (lab-speed) | `build/fn-prof`: the decode-step profile of section 5b (leave-one-out, A/B, bit checks, micro experiments) |
| `kernels/fn_model.metal` (lab-forward) | `fn_embed_g*`, `fn_ple_proj_g64_r*` + `fn_ple_mix_r*` (layer 1 PLE), `fn_lm_head_g*_r*`, `fn_sample_p1` / `fn_sample_p2` (docs/forward.md) |
| `models/flash_next/model.{h,mm}` (lab-forward) | the whole model: load, A/B recordings, token loop, C API `ne_fn_model_*` |
| `models/flash_next/layer.{h,mm}` | weights (`load_layer`), state, scratch, `record_stage` / `record_gdn_layer` (core Recorder; any layer type), `time_layer`, C entry points `ne_fn_layer_*` |
| `models/flash_next/fn_layer_check.mm` | `build/fn-layer-check PACK CASE_DIR [chain\|stage\|all\|time] [layer]` (layer and rows from the case): every sub-block vs the goldens; `time` = GPU time (take the machine lock) |
| `models/flash_next/ctypes_smoke.py` | the C API through `tools/engine.py`, vs `L00_decode1`, + determinism |

Build: `make` (`models/flash_next/model.mk` adds `layer.mm` to libengine and `build/fn-layer-check`).
lab-ref's `tests/test_fn_layer.py` (`make test-layer-ref`) drives the same C API.

## 2. The layer step: 10 dispatches (GDN) / 11 (attention), one recording

Layouts: hidden `h` f32 [R][4][2560] (4 hyper-connection streams); GDN conv state f32 [3][10240] (previous 3 qkv rows,
oldest first), recurrent state f32 [48][128 Dv][128 Dk] (MLX layout); attention KV cache f32 [2][tmax][256] (keys
roped), indexer raw keys f32 [tmax][128], `pos` u32 (tokens in the cache; a GPU buffer so the recording replays;
`ne_fn_layer_step` advances it). Every activation is [R][...] (R rows = consecutive tokens).

| # | kernel | grid x threads | does | bytes, 1 row (K5 L0) |
|---|---|---|---|---|
| 0 | `fn_hc_pre` (attn) | 41 blocks x 4 streams x 256 | per stream: stage x_s = h_s * hc_norm_s, rms_s; simdgroup = one row's stream slice (q8 g64 down rows; bf16 block_inject rows) * 1/rms_s; last of the 4 stream TGs sums in fixed order -> `mix = silu(d/4)`, `inj = 2 sigmoid(d/4)`, `inj[4+s] = 1/rms_s` | 3.4 MB |
| 1 | `fn_hc_mix` (attn) | 320 x 256 | simdgroup = output column, 8 lanes per stream row: `x[c] = mean_s sigmoid(up_s . mix) * normed_s[c]` | 3.5 MB |
| 2 | `fn_gdn_inproj` | 2060 x 256 | qkv / z / b / a rows (q8 g128); epilogue: conv1d K=4 with state (in place) + silu over the R tokens; beta; g | 43.5 MB |
| 3 | `fn_gdn_step` | 48 x 1024 | per v-head, state rows in registers, R tokens in order: l2norm q,k; S *= g; S += k (v - S.k) beta; y = S.q; gated rmsnorm * sigmoid(z) | 6.3 MB r+w |
| 4 | `fn_gdn_outproj` | 320 x 256 | out_proj (q8) + HC write `h_mid = h + y * inj_s` | 16.2 MB |
| 2' | `fn_attn_proj` (attn layers) | 1744 x 256 | q (+gate) / k / v / indexer qk rows (q8 g64); raw indexer key -> its cache | 36.5 MB |
| 3' | `fn_attn_core` + `fn_attn_merge` | 2 x tmax/32 x 384; R x 24 x 256 | TG = (kv group, 32 positions), simdgroup = q head: q/k rmsnorm, partial rope (64/256, native), new k/v -> cache, online softmax over the chunk; merge chunks in order, * sigmoid(gate) | KV 2 x T x 2 KB |
| 5,6 | `fn_hc_pre`, `fn_hc_mix` (mlp) | as 0, 1 | -> MoE input | 6.9 MB |
| 7 | `fn_moe_router` | 225 x 256 | router (512 bf16 rows + shared_expert_gate row) + shared gate/up (q8 g128); last TG: top-10 + renormalised softmax per row (+ unique-expert table at R > 1) | 9.2 MB |
| 8 | `fn_moe_gateup_k5` | R*400 + 320 x 256 | per (row, slot, gate/up, 2 tiles): x*suh -> H128 -> trellis GEMV; + shared down (q8) | 22.2+1.6 MB per row |
| 9 | `fn_moe_down_k5` | R*200 x 512 | per (row, slot, 128 cols, 2 k-splits): H128/svh, SwiGLU, *suh, H128, down GEMV, H128*svh*score; last of 10 slots: fixed-order sum + sigmoid(gate) * shared + HC write | 10.3 MB per row |

~118 MB per 1-row GDN step (the pack's q8 trunk, larger than lead 4's K6 model) -> floor 199 us at 594 GB/s.

## 3. Correctness vs lab-ref's goldens (the bf16 MLX reference)

*stage* = the sub-block alone on golden inputs; *chain* = whole step from the golden layer input + states.
rel L2 = ||ours - golden|| / ||golden||. Reference rounding ON (default, section 4).

| sub-block (golden name) | L00 decode1 stage / chain | L00 verify4 stage | L03 decode1 stage / chain |
|---|---|---|---|
| HC mix / inject gates | 1.1e-3 / 4.5e-3 | 1.6e-3 / 4.9e-3 | 3.4e-3 / exact |
| HC read output (`attn_hc.block_in`) | 3.2e-3 | 2.9e-3 | 2.5e-3 |
| GDN conv out / z / g / beta | 2.6e-3 / 1.9e-3 / 7.6e-4 / 1.7e-3 | 3.1e-3 / 1.7e-3 / 1.6e-3 / 1.6e-3 | |
| GDN gated norm / ssm state | 2.9e-3 / **exact** (stage); 6.5e-3 / 2.8e-3 (chain) | 3.8e-3 / 2.8e-4 | |
| attn q_proj / k / v / indexer qk | | | 1.7e-3 / 1.7e-3 / 1.8e-3 / 1.6e-3 |
| attn gated output / k cache / v cache | | | 3.3e-3 / 1.8e-4 / exact |
| out_proj / HC write (`attn_hc.out`) | 1.7e-3 / 3.0e-3 | 1.7e-3 / 2.9e-3 | exact / exact |
| mlp HC read (`mlp_hc.block_in`) | 3.0e-3 | 3.2e-3 | 2.7e-3 |
| router logits / top-10 ids / weights | 1.7e-3 / 10 of 10 / 1.1e-3 | 1.6e-3 / 40 of 40 / 5.3e-3 | exact / 10 of 10 / 3.2e-3 |
| shared gate / up / down | 1.6e-3 / 1.5e-3 / 3.4e-3 | 1.8e-3 / 1.5e-3 / 3.2e-3 | exact / exact / 4.9e-3 |
| MoE branch (`moe.out`) / layer output (`hc_out`) | 3.9e-3 / 3.5e-3; chain hc_out **5.7e-3** | 4.6e-3 / 3.9e-3; chain 6.8e-3 | 4.9e-3 / 7.6e-4; chain 1.5e-3 |

L03_verify4: 39/40 ids (lab-ref documents a bf16 router-probability tie in that case), hc_out 5.3e-3. Long context
(2304 tokens, the reference's QSA selects 512 of 576 blocks, ours is dense): gated output 6.2e-3 (decode) /
9.6e-3 (verify), hc_out 5.2e-3 / 6.8e-3. lab-ref's mode 3 (kernels vs f32 op references on the engine's own
inputs) found every f32 kernel at 2e-8..7e-7 (EXL3 GEMV 2.2e-4: f16 codebook / f16 prep). Repeated runs are
bit-identical (fixed-order finishers). `tests/test_fn_layer.py L00_decode1`: every stage check passes; the chain
checks that fail are cosine thresholds on compounding intermediates (`mlp_mix`, shared gate, ssm state, topk
weights), `hc_out` passes.

## 4. Reference rounding points (policy: emulate, default ON)

Function constant 0 (`NE_REFR`, `kernels/ne_common.h`; ON when not supplied). The host sets it from
`ne::fn::g_ref_round` (env `NE_FN_ROUND=f32` or `ne_fn_set_ref_rounding(0)` before creating a layer = plain f32).
ON rounds to bf16 where the MLX reference stores bf16: HC down / inject logits, mix, normed, sigmoid products,
block_in; qkv rows (= the conv state), conv out, z, the gate chain bf16(a) -> bf16(a + dt_bias) -> bf16(softplus),
bf16(b) -> beta; the GDN q/k l2-normalisation in bf16 ops (sum, rsqrt, products, scale); y, gated norm; out_proj
branch, injection and HC sums; router / shared-expert outputs, SwiGLU act, routed sum, gated shared term; attention
roped q/k, attention output and gate. Not emulated: the EXL3 expert internals, accumulation orders. Effect on
L00_decode1: inject gates 9.3e-3 -> 4.5e-3, GDN state on golden inputs 1.0e-3 -> bit-exact, chain hc_out
7.2e-3 -> 5.7e-3. Cost: none measurable (271 us ON vs 285 us f32, noise). The f32 path stays for the KLD judgement
against the bf16 teacher.

## 5. Speed (M5 Max, `fn-layer-check CASE time`, mtl4 backend, machine lock held)

Differential GPU time per step (16 vs 64 steps in one command buffer, same layer); "flushed" = a 256 MB streaming
read between steps evicts the SLC (its own cost subtracted).

| case | dispatches | SLC-warm | **flushed** |
|---|---|---|---|
| L00 GDN, 1 row | 10 | 264 us | **271 us** (~118 MB: 435 GB/s, 73% of peak, 1.36x the 199 us floor) |
| L00 GDN, 4 rows | 10 | 503 us | **505 us** (1.87x one row; routed experts 4x bytes) |
| L03 attention, 1 row, T = 48 | 11 | 255 us | **259 us** |
| L03 attention, 4 rows, T = 48 | 11 | 485 us | **488 us** |
| L03 attention, 1 row, T = 2304 | 11 | 310 us | **325 us** (attention core 66 us) |
| L03 attention, 4 rows, T = 2304 | 11 | 591 us | **600 us** (attention core 140 us) |

Backends, L00 1 row: mtl4 269, icb 279, classic 276, serial 286 us (flushed). Lead 4's model: ~228 us per layer for
101.7 MB (K6 trunk, 446 GB/s); we run the same efficiency on the pack's bigger q8 trunk, so the per-layer gap to
the model is bytes, not kernels. 48 layers at 1 row today: ~36 x 271 + 12 x 259 = 12.9 ms + head (q8 lm_head
~1.1 ms) ~ 14 ms GPU, vs Sushi 15.9 ms; with lead 5's K6/K8 trunk (-0.98 GB/token) ~ 12.3 ms.
Per dispatch alone after a 128 MB flush (noisy, +-5 us; L00 1 row): hc_pre 8-14, hc_mix 4-7, gdn_in 74 (590
GB/s), gdn_step 14, gdn_out 27, router 11, gateup 39 (610 GB/s incl. shared down), down 29 (355 GB/s).
Fixed on the way: `fn_hc_pre` with 41 TGs re-reading h per row took 55 us (now 164 TGs: 8); the first attention
core (24 TGs walking all positions) took 151 us at T = 2304, an all-heads-per-simdgroup chunked version 402 us
(register spill), the head-per-simdgroup chunked one 66 us. Expert dedupe across verify rows (opt-in,
`NE_FN_MOE_DEDUPE=1`) is correct but 6% slower on L00_verify4: the 4 consecutive tokens share 1 of 40 experts.

## 5b. Decode-step profile (lab-speed, `build/fn-prof`)

`build/fn-prof PACK [reps T rounds layers]` (lock; ~8 GB: layers 0-3 loaded and replayed in the model's 48-layer
pattern, attention at l % 4 == 3, PLE on virtual layer 1; each loaded layer is re-read every 4 layers, ~0.5 GB apart,
so no SLC reuse). Method: **leave-one-out in context**: the 48-layer body is recorded in full and once per kernel kind
with every instance of that kind left out (with its barrier); delta / count = what one instance costs in the step,
boundary included. Medians of 15 replays x 3 interleaved rounds. MB = weights / states touched per instance;
GB/s against lead 1's 594 GB/s read peak. Options: `NE_PROF_KINDS=a,b` (subset), `NE_PROF_V2=mask` (A/B of
lab-speed's v2 recordings + bit checks), `NE_PROF_MICRO=1` (chained single-kernel timings, runtime-compiled
experiments via `NE_PROF_SRC` / `NE_PROF_SRC_K`).

Baseline, 15 Oct (T = 64, rounding ON, ba9b934 kernels): **body 12.95 ms + head 1.15 ms = 14.10 ms** (the model's
own number: 14.07); 542 body dispatches; an extra empty dispatch + barrier costs **0.78 us** (548 boundaries =
0.43 ms: dispatch count alone is not where the time is).

| kernel | count | us each | ms / step | % body | MB each | GB/s | % of 594 | floor us (bytes) |
|---|---|---|---|---|---|---|---|---|
| gdn_inproj (qkv z a b + conv) | 36 | 79.7 | 2.87 | 22.2 | 44.4 | 557 | 94 | 74.7 |
| moe_gateup (EXL3 K5 + shared down) | 48 | 45.4 | 2.18 | 16.8 | 22.3 | 492 | 83 | 37.5 |
| moe_down + HC write | 48 | 31.8 | 1.53 | 11.8 | 10.3 | 324 | 55 | 17.3 |
| hc_pre (attn + mlp) | 96 | 14.3 | 1.38 | 10.6 | 3.58 | 251 | 42 | 6.0 |
| gdn_outproj + HC write | 36 | 29.5 | 1.06 | 8.2 | 16.2 | 549 | 93 | 27.3 |
| hc_mix (attn + mlp) | 96 | 9.2 | 0.88 | 6.8 | 3.50 | 380 | 64 | 5.9 |
| attn_proj (q k v indexer) | 12 | 62.6 | 0.75 | 5.8 | 38.0 | 606 | 102 | 63.9 |
| moe_router (+ shared gate/up, top-10) | 48 | 14.1 | 0.68 | 5.2 | 6.0 | 425 | 72 | 10.1 |
| gdn_step (state r+w) | 36 | 18.2 | 0.66 | 5.1 | 6.3 | 346 | 58 | 10.6 |
| QSA step (lab-attn, T = 64) | 12 | 29.8 | 0.36 | 2.8 | - | | | |
| o_proj + HC write | 12 | 27.1 | 0.33 | 2.5 | 16.7 | 616 | 104 | 28.1 |
| PLE (layer 1) | 1 | 52.8 | 0.05 | 0.4 | 35.6 | 674 | | 60 |
| head: lm_head q8 (248320 rows) | 1 | 1103 | 1.10 | | 662 | 600 | 101 | 1114 |
| head: mixer (hc_pre + hc_mix) / sampler p1 + p2 | 1 / 1 | 11.4 / 10.3 | 0.02 | | | | | |

(Sum of the leave-one-out deltas 12.72 of 12.95 ms; > 100% of 594 = partly SLC-resident or measurement noise.)
Reading: the big GEMVs (in_proj, out_proj, attention projections, lm_head) stream at 93-104% of the read peak: their
3.9 ms are bytes. The excess over the bytes is in the small latency-bound kernels: hc_pre + hc_mix 2.26 ms against
a bytes floor of 1.15 ms, gdn_step +0.27 ms, router +0.19 ms, and the EXL3 MoE kernels, which are
near their **ALU** floor rather than bandwidth: the K5 trellis decode is ~7.5 ALU ops per weight (funnel shift,
16-bit window, 32-bit MCG multiply, mask, xor, 2 half adds), i.e. ~16 us for moe_down's 16.4 M weights and ~32 us
for gateup's 32.8 M at the GPU's ~7.7 Tops: deeper prefetch made both slower (measured below).

**After (lab-speed, 15 Oct evening)** -- same tool, same partial model:

| | body | + head | whole model (`ne_fn_model_time`, T = 64 / 2048) |
|---|---|---|---|
| baseline (ba9b934 kernels) | 12.95 ms | 14.10 ms | 14.03-14.09 / 14.27-14.32 ms |
| + finisher / epilogue loads hoisted (in the graph now: no signature change) | 12.74-12.75 | 13.89 | **13.86-13.89 / 14.10-14.15 ms** |
| + HC read v2 (`fn_hc_pre3` + `fn_hc_mix2`; needs the layer.mm patch in Requests) | **12.45** | **13.60** | ~13.55 expected |

Per kernel after both (us each, in context; +-2 us between runs): hc_pre 14.3 -> **11.0**, hc_mix 9.2 -> 8.5,
moe_down 31.8 -> **27.5-29.4**, gdn_outproj 29.5 -> 29.3-31.0, gdn_inproj 79.7 -> 73.7-81.6 (noise-level), the rest
unchanged. What worked and what did not:

- **Loads that wait behind stores / fences are the small kernels' tail.** fn_moe_down's finisher (the last of the
  10 experts per 128-column block) loaded h, inj, the shared-expert output and the gate logit only after its atomic
  stores + device fence + counter: one more dependent round trip at the end of the MoE block. Loading them first
  (every threadgroup, 2 KB from L2; only the finisher uses them): moe_down 31.9 -> 27.5 us, body -0.22 ms,
  bit-identical with rounding ON. Same hoist for fn_gdn_outproj's HC write inputs and fn_gdn_inproj's conv state /
  conv weights / A_log / dt_bias (neutral within noise; kept).
- **HC read v2**: fn_hc_pre's cross-threadgroup finisher (atomics, fence, counter, a second load chain) goes away:
  `fn_hc_pre2` / `fn_hc_pre3` plain-store the per-stream partials (+ 1/rms_s in the inject block's unused slots of
  `hc_part`), `fn_hc_mix2` sums them in the same fixed order in its prologue (its weights are already in flight) and
  threadgroup 0 stores mix and all of inj. `fn_hc_pre3` also drops the threadgroup staging (each simdgroup reads its
  stream slice of h + hc_norm from L1/L2 and sums its own squares): no 32 KB limit, so R = 8 needs no 2 x 4 split.
  pre2 + mix2: -0.29 ms, every intermediate of every layer bit-identical to the old pair except where the compiler
  contracts differently (below); pre3 + mix2: -0.30..-0.35 ms, 1/rms in the last bit (2 of 8 inj entries).
- **Deeper prefetch in the EXL3 GEMVs is slower** (lab-forward saw the same for moe_down): 2 or 4 k-steps in flight
  per simdgroup instead of 1, clamped to the simdgroup's own k range: body 12.96 (1) / 13.15 (2) / 13.17 ms (4).
  The trellis decode is ALU-heavy (above), extra registers cost occupancy. Reverted.
- **gdn_step** (48 TGs x 1024): chained alone it costs 24 us SLC-warm, 4.4 us if the gated norm's z / weight loads are
  issued before the 64 KB state store (store -> load ordering), but in context the same change gives 18.5 vs 19.0 us:
  the in-context cost is the cold 3 MB state read + write at 48 threadgroups, not that stall. Not kept. Next try:
  more, smaller threadgroups (2-4 per head) with the gated norm in a second pass, or the norm in out_proj's prologue.
- Dispatch count alone: 0.77 us per extra empty dispatch + barrier, 548 boundaries = 0.42 ms: fusing neighbours pays
  only where it removes a dependent latency chain (as the HC finisher did), not per dispatch.

**Rounding (f32 path).** With rounding ON (tests, goldens) the 48-layer output hash is unchanged by the hoists. On the
plain-f32 generation path the compiler's FMA contraction of `h + branch * gate` and the conv sums moves with the
hoisted code (last-bit f32 differences; no combination of pinned fma / contract(off) per site reproduced the old
hash, so other sites move too). Greedy `tools/generate.py`-style continuations, 64 tokens, f32: 2 of 4 prompts
token-identical ("17 x 23" -> "391", the primes list), 2 diverge late in the text (palindrome function, sky blue: 50 vs
51 tokens), from f32 rounding only. The HC v2 first-order difference is the same kind: at virtual layer 9 of the
check chain one element of the HC read output (1 of 2560) lands one bf16 step apart (0.94921875 vs 0.9453125) from a
last-bit f32 difference in fn_hc_mix's dot, then the chain amplifies it.

## 5c. lab-fuse attempts (16 Oct, 1-row decode step)

Method: `NE_PROF_AB=1 NE_FUSE=<bit> build/fn-prof PACK 15 64 5` with `NE_PROF_KINDS=none` (4 loaded layers in the
48-layer pattern, T = 64, rounding ON): the candidate is guarded by `(g_fuse & bit)` in `record_stage`
(`models/flash_next/layer.h`) and timed against the old recording in the same process, rounds interleaved; both
48-layer output hashes are printed. Same-process noise ~ +-0.04 ms on a ~12.3 ms body; one A/B per variant (light
testing). "probe" = a deliberately wrong kernel that only bounds what a change could gain. The code of every
attempt is in commit 4308cba (`kernels/fn_fuse.metal`, the `fz(bit)` branches in layer.mm); the next commit keeps
only the win.

| # | change | body before -> after (ms) | kept? |
|---|---|---|---|
| 1 | **`fn_hc_pre3`: the q8 rows' and the bf16 inject rows' weight registers in separate branches** (one shared `uint4 wq[10]` sized every simdgroup for the 4 inject rows; the q8 rows need 5) | 12.402 -> 12.245, 12.428 -> 12.298, 12.353 -> 12.243 (**-0.11..-0.16**, bit-identical) | **yes** |
| 2 | no barrier between `fn_hc_pre3` and `fn_hc_mix2`, mix waits on a counter (spin, coherent / plain / back-off loads) | +0.08..+0.19 (probe without barrier and without wait: -0.39) | no |
| 3 | `gdn_inproj` rows regrouped per key head, `gdn_step` without a barrier waiting per head group | +1.03 (fence in all 8 simdgroups of 2064 threadgroups: +1.56 alone; 1 simdgroup fences: +0.06); with the cheap signal +0.05 | no |
| 4 | `gdn_step` signals after `o`, then stores the state; `out_proj` without a barrier waits (weights prefetched, `o` staged) | +0.32; the same kernels with a barrier +0.08..+0.51 | no |
| 5 | one-dispatch HC read (pre + grid barrier + mix; 82 x 512 / 164 x 256 / 41 x 1024 threads) | +2.22 / +1.55 / +3.65 (without the grid barrier, wrong: +0.64) | no |
| 6 | SLC prefetch of the next kernel's weights next to latency-bound kernels (`ne_prefetch3`, no barrier between) | HC mix weights +0.04; out_proj 3 / 6 / 12 MB during `gdn_step`: +0.08 / +0.27 / +0.67 | no |
| 7 | `fn_moe_router` finisher: one simdgroup fences instead of all 8 | -0.01 | no (noise) |
| 8 | router's shared-expert threadgroups first in the grid | +0.03 | no |
| 9 | shared gate/up moved router -> gate/up kernel, shared down -> moe_down (joins the block counter; bit-identical) | +0.21..+0.25; shared down alone -> moe_down: +0.07 (with the finisher loads hoisted; +0.18..+0.32 without) | no |
| 10 | `fn_moe_gateup` 16 / 32 simdgroups per threadgroup; 1 / 4 / 8 output tiles per threadgroup; prep/hsh threadgroup memory aliased | -0.04 / +0.04; +0.04 / +0.19 / +0.46; -0.03 | no |
| 11 | `fn_moe_down` 4 k splits; float4 coherent partials in the finisher; its SwiGLU prologue done once per expert by a gate/up finisher | +0.04; -0.03; -0.05 (hash differs) | no |
| 12 | `fn_hc_mix2` 2 / 4 columns per simdgroup; 2-32 simdgroups per threadgroup (both HC kernels) | +0.10 / +0.32; -0.04..+0.14 | no |
| 13 | `fn_hc_pre3` 2 / 4 rows per simdgroup (h read once for both) | +0.18 / -0.16 (= #1's gain, nothing more) | no |
| 14 | `fn_gdn_inproj` 2 rows per simdgroup (x read once) | +0.00 | no |
| 15 | `gdn_step`: each row's state stored right after its update; + z / norm loads hoisted | -0.02 / -0.03 | no |
| 16 | router shared rows through a fixed-trip-count q8 GEMV with every load issued first | +0.02 | no |
| 17 | the same fixed-trip q8 GEMV (`q8rows_fx`) in in_proj / out_proj / attn_proj / o_proj | +0.61 (busy machine: +1.95) | no |
| 18 | staged `fn_hc_pre2` with #1's register split instead of `fn_hc_pre3` | +0.11 (busy: +0.34) | no |
| 19 | `fn_gdn_inproj` epilogue inputs loaded after the GEMV (undo lab-speed's hoist) | +0.01 | no |

Whole model (`tools/bench_forward.py --ctx 64,2048 --gen 64`, f32, same build with `NE_METALLIB` = master's kernels
vs this branch, back to back): **T = 64: 13.57 -> 13.31 ms GPU** (min 13.47 -> 13.29), T = 2048: 13.71 -> 13.57 ms;
greedy 72.6 vs 73.6 tok/s wall (host-side noise: the n-gram gather was 351 vs 92 us in that run). Greedy
`tools/generate.py` (64 tokens, "17 x 23", palindrome function): identical text on both. Rows 17-19 ran while the
machine was slower (bodies 14.5-18 ms instead of 12.3), so only their sign is reliable.

Probes (what the time is made of; wrong outputs):

- **No barrier anywhere** (`Recorder::barrier` disabled): body 9.21 vs 12.77 ms in the same build. ~3.5 ms of the body is
  serialisation of dependent kernels, but every software replacement for the barrier tried here (spin on a counter
  after early launch, grid barrier inside a dispatch) costs more than the hardware barrier on this GPU: waiting
  threadgroups launched early slow the producer (#4: plain kernels +0.08, the same without barrier +0.42), a grid
  barrier costs ~10 us. A fence in every simdgroup is very expensive in a many-threadgroup kernel (#3), one per
  threadgroup is ~free.
- **MoE decode ALU is not the limit**: the MCG multiply replaced by an xor in both expert kernels: -0.08 ms. Microbench
  (scratch `alu.metal`): the decode loop is ~4 ALU slots per weight (imul ~2.2), 1.8 T weights/s on the whole GPU,
  i.e. 18 / 9 us for gate/up / down against 46 / 29 us measured. Cheaper trellis decode has a ceiling of ~0.1 ms.
- `gdn_step` without its state store: -0.37 ms (~10 us of its 19); without the norm or with 2-4 threadgroups per head
  only -0.05. The 3 MB f32 state write is the step's cost; avoiding it means a lazily applied rank-1 update or a
  narrower state (lead below).
- `moe_down` without its SwiGLU / H128 prologue: -0.16 (3.4 us); without the finisher's cross-threadgroup sum -0.08.
- `fn_hc_pre3` reading h for one chunk only: -0.19 (2 us per hc_pre spent re-reading h through L1/L2 per simdgroup;
  #13 shows the 2-row restructure does not recover it). The router without its shared-expert threadgroups: -0.27
  (5.7 us per layer: its 3.3 MB run as the kernel's tail, but moving them elsewhere costs as much, #9).
- A plain 3.5 MB streaming dispatch costs ~4.2 us in context (hc_mix 9, hc_pre ~10 after #1); co-residency
  (scratch `occ.metal`): 3072 threads per core (122,880 on the GPU), so the small kernels run in one wave.

Leads not done: ~~lazy GDN state store~~ (done by lab-gdnlazy, section 5d); a narrower GDN state (quality question).

## 5d. Lazy GDN state store (lab-gdnlazy, 16 Oct)

`fn_gdn_step_lz{K}_r{R}` (kernels/fn_trunk.metal) replaces `fn_gdn_step` in `record_stage` (env `NE_GDN_LAZY=K`,
default **3**; 0 = the old kernel). At R = 1 it stores the 3 MB f32 state only every K-th step; in between it keeps
the step's rank-1 inputs (normalised k, v, g, beta; 1 KB per value head) as pending rows in `GdnState::pend`
f32 [48][15][260] with the count per head in `GdnState::lz` u32 [48]. The next step (or `fn_gdn_materialise`) loads
the stored S, re-applies the pending rows with the same expressions (`gdn_upd`) and continues: the outputs and the
materialised state are **bit-identical** to storing every step (fn-prof 48-layer hash with K = 1 vs 2..16 equal;
the lazy kernel vs old `fn_gdn_step` differs in the last bit from the norm hoist below). Each threadgroup owns its
head's lz / pend (reads before its threadgroup barrier, writes after): no cross-threadgroup sync. R > 1 (verify,
4-row prefill, layer API at 4 rows) applies the pending rows, runs the R rows and always stores.

The second half of the gain is a hoist: the gated norm's z / weight loads now go before the state store (simdgroup 0
did store -> load; lab-speed's attempt on the old kernel was neutral, here K = 1 alone = -0.11 ms), and the pending
row is written by simdgroup 31, which has no norm to do.

Readers of ssm call `gdn_materialise(states)` first (host, one command buffer, no-op when nothing is pending):
`ne_fn_model_snapshot`, the server shim's `ne_srv_snapshot`, `prefill_tokens` (chunked prefill entry: pf_gdn_scan),
`mtp_generate` / `mtp_verify_forced` after the prefill (the verify's lazy-commit kernels read ssm), the layer API
(`get_state`, `read("ssm_state")`), fn-layer-check, pf-check, mtp-nax-check. Writers call `gdn_clear_pending`:
`reset`, restores, `set_state`, `write("ssm_state")`, test state writes.

fn-prof A/B (`NE_PROF_AB=1 NE_PROF_AB_LAZY=1 NE_GDN_LAZY=K NE_PROF_KINDS=none build/fn-prof PACK 15 64 5`: old body =
`fn_gdn_step`), body ms, one run each (+-0.04 noise):

| K | before norm hoist | with hoist + sg 31 writer |
|---|---|---|
| 1 (no deferral) | | -0.113 |
| 2 | -0.048 | |
| 3 | -0.112 | **-0.249** |
| 4 | -0.153 / -0.130 (pending inputs prefetched: -0.124) | -0.235 |
| 6 | -0.099 | -0.174 |
| 8 | -0.093 (prefetched: +0.170, registers) | |
| 16 | +0.050 | |

Leave-one-out gdn_step: 17.3 us (old) -> 14.7 (K = 4, no hoist) -> 13.8 us (K = 4 with hoist) per layer.
Re-applying pending rows costs more than the 1 KB they read suggests (each is a dependent simd_sum chain per state
row), so the optimum is K = 3-4, not "never store".

**Whole model** (`tools/bench_forward.py --ctx 64,2048 --gen 64`, f32, same build, `NE_GDN_LAZY=0` vs default, back
to back): **T = 64: 13.45 -> 13.22 ms GPU** (min 13.41 -> 13.07), **T = 2048: 13.67 -> 13.45 ms**; greedy 72.3 vs
73.1 tok/s wall (the n-gram gather was 304 vs 85 us in that run). Greedy `tools/generate.py` (64 tokens, "17 x 23"
+ palindrome) and bench_forward's greedy continuation: identical text. MTP d2 greedy (`bench_mtp.py --modes greedy
--depths 2`, 3 thinking prompts x 1024): **94.9 tok/s median** (94.9 / 103.7 / 76.8; 2.50 tok/round), vs 95.7 on the
same K5 pack in lab-t6wire's run this morning: unchanged within run-to-run spread, as expected (the verify path keeps
lab-mtp's lazy commit, which stores once per round already; a store-every-other-round variant would save ~0.1 ms of
a ~22 ms round). No-MTP greedy in that run 73.0 / 70.0 / 66.4.

Correctness: `tests/forward_gpu.sh` (= make test-forward: test_fn_forward, test_fn_layer on 8 cases) PASS;
fn-layer-check L00_decode1 ssm chain 2.79e-3 / stage exact (as before), L00_verify4 chain 3.27e-3; mtp-nax-check
lazy and rollback = the docs/mtp.md 5.4 numbers; mtp-loop-check all passed; test_fn_api (snapshot round trip, plus
a new check: 16 greedy steps after a materialising snapshot = 16 steps without one, bit for bit) PASS, incl. the
restore round trip (16 greedy steps identical), chunked vs steps prefill (24 / 24) and the MTP callback;
`tests/mtp/test_verify_forced.py --rows 2,3` (verify after the materialised prefill vs the lazy 1-row decode,
with rollback patterns) all OK: KL mean 1.6e-4..4.9e-4 (max 2.1e-2), top-1 99.2-100%, every flip at a gap of 0.000;
rollback patterns = all-kept, as before. (The 1-row reference itself moved in the last f32 bit with the norm hoist,
so its greedy continuation and the per-row numbers are not directly comparable with docs/mtp.md 10.2's.)

## 6. Traps

- **Misaligned pack tensors.** safetensors does not pad its header: every `model-exl3-L*` shard of the K5 pack has
  an odd header length (trellis / suh / svh at odd offsets); 193 trunk U32 weights sit at 2 mod 16, some BF16 at
  8 mod 16, all layer-3 attention weights at 2 mod 16. Metal loads at such addresses return garbage (NaN from f16
  suh). `load_layer` takes `Pack::aligned()` copies for any tensor not 16-byte aligned (1.58 GB for layer 0, 1.66 GB
  for layer 3): fine for one layer, not for 48 (the aligned pack via `engine-cli align-pack` is the fix).
- **Cross-threadgroup partials need atomics** inside one dispatch: a last-arriving finisher read stale values from
  plain stores + fence; atomic store/load (relaxed) fixed it (lead 4's pattern). Across dispatches plain stores are
  fine (`fn_attn_core` -> `fn_attn_merge`). Finishers reset their counters, so recordings replay.
- **The MoE input has no extra rmsnorm** (lead 4's prototype normed it): the mlp hc_norm is the norm.
- hc_norm / q_norm / k_norm weights are ones-centred in the pack: multiply directly. `linear_attn.norm.weight` is
  plain. The shared expert gate is its own tensor (router row 512 here).
- Group sizes vary per tensor (HC g64; GDN / shared g128 on layer 0; attention g64; in_proj_qkv g64 on most
  layers): `load_layer` solves G from the shapes; instantiations exist for 64 and 128. **qkv and z/a/b differ** on
  29 of the 36 GDN layers (qkv g64, z/a/b g128): `fn_gdn_inproj_q64_g128_r*` (lab-forward; template <GQ, G, R>).
- PLE (layer 1, lab-forward): `record_gdn_layer` records `fn_ple_proj` + `fn_ple_mix` first (h_in + PLE -> s.hp),
  then the layer on s.hp. The 16 n-gram rows are gathered by the host into `s.ple_emb` (`ne_fn_layer_write
  "ple_emb"` in the layer API; the model gathers them itself), state `ple_conv_state` f32 [9][10240].
- Attention (lab-forward, default): `GdnStep` of an attention layer = lab-attn's `ne::attn::record_step` (QSA,
  bf16 KV, `GdnState::qc`, its pos = the layer's / model's shared pos); `NE_FN_ATTN=dense` = `fn_attn_core` +
  `fn_attn_merge` as before. `ne_fn_layer_set_kv` / `get_kv` go through lab-attn's import / export then.
- `fn_attn_core` must fit 12 online-softmax states per lane if one simdgroup serves all heads: it spills. One
  simdgroup per head, K/V shared through L1, is 6x faster.
- Build: test binaries' `.d` files are not included by the core Makefile, so `fn-layer-check` was not rebuilt after
  a header change (a struct layout mismatch looked like a kernel bug); `model.mk` includes its `.d` now. A SIGBUS
  (exit 138) once came from another agent relinking `libengine.dylib` under a running binary.
- Timing: per-dispatch numbers must be flushed (SLC-warm repeats of one 40 MB GEMV read at 880 GB/s); the full step
  is the number to trust. `fn-layer-check all` does not time.

## 7. What is left

- **QSA sparse path** (layers 3, 7, ... past 512 complete blocks = 2048 tokens): pooled block keys (k_layernorm of the
  mean of 4 raw keys, rope at block start), block scores sum_h relu(q_h . K_b) / sqrt(128), top-512 selection, attend
  to selected blocks + tail. The reference updates block keys only on S > 1 calls; the engine must mirror that.
  Today the long cases run dense (6e-3..1e-2 off on the gated output).
- **PLE (layer 1)**: n-gram row ids, table gather (95 GB table, row views), key/value projections, signed-sqrt gate,
  dilated conv with a 9-row state. Then 48 layers + final HC mixer + lm_head + sampling; MTP head.
- **Speed**: KV cache in bf16 (halves attention bytes; the reference stores bf16, so exact), gdn_step with more TGs,
  moe_down at 355 GB/s, fuse hc_mix into the next GEMV's prelude (-2 boundaries per layer), out_proj k-split; NAX
  bodies (lead 6) for the 4-row step, where routed experts dominate (152 + 83 us of 524).
- Per-layer weights for 48 layers need the aligned pack (no RAM copies).

## Requests

- **Status 16 Oct (lab-integrate, the forward / MTP-loop owner)**: done -- lab-prefill's (1) `prefill()` / `generate()` /
  the MTP loop's start through the chunked prefill (from 16 tokens; `NE_FN_PREFILL=steps` = the old path; pf-check's
  `NE_PF_FLOOR=1` sets it) and (2) `struct ne_fn_model` in model.h; lab-speed's HC read v2 in `record_stage`, the head
  mixer and the MTP head / verify mixers (`NE_FN_HC=v1` = old; R = 8 keeps the old pair because `nax::record_layer` still
  splits `fn_hc_pre` there: **lab-speed / kernels/nax owner**, drop that split and I remove the R = 8 exception in
  `fn::hc_v2`); `mtp.mm` `alloc_head` uses `fn::alloc_scratch` (any R) instead of `nax::alloc_scratch_rows`; router
  `uq` at R > 4 was done by lab-speed (ba9b934); `-include tests/quality/quality.mk` in tests/tests.mk. From lab-server:
  (1) `ne_fn_mtp_generate_cb` / `mtp_generate(..., cb, ctx)` as patched (callback per emitted run, nonzero = stop after
  the round); (3) official `ne_fn_model_snapshot` / `_restore` / `ne_fn_snap_T` / `_bytes` / `_free` in model.h (the
  shim's semantics; the shim can go -- note model.h's `Model` gained a member, so a shim built against older headers
  reads wrong offsets); the prefill LRU cap (32 recordings, `NE_PF_MAX_RECS`). lab-server's (2) MTP from the model's current state: done by lab-mtpcont (docs/mtp.md 10.9).

- **lab-loop / lab-integrate** (from lab-server, 16 Oct; the OpenAI server, docs/server.md section 6): (1) **an
  abortable, streaming MTP loop**: today `ne_fn_mtp_generate` cannot stream (the server polls `out` from another
  thread), cannot stop on a stop string, a second EOS (`<|endoftext|>` 248044) or a client disconnect, so a run the
  client no longer wants continues to max_new = the context window. Patch: in `mtp_loop.h`
  `using MtpTokenCb = int (*)(void *ctx, const int *tokens, int n);` and add `MtpTokenCb cb = nullptr, void *cb_ctx =
  nullptr` to `mtp_generate`; in `mtp_generate` after `out[k++] = t1;` add `if (cb && cb(cb_ctx, out, 1)) done = true;`,
  and in the round loop wrap the emission: `const int k0 = k;` before `for (int tkn : res.emitted)`, then after it
  `if (cb && k > k0 && cb(cb_ctx, out + k0, k - k0)) done = true;`; C entry `NE_API int ne_fn_mtp_generate_cb(ne_fn_mtp
  *p, const int *prompt, int n, int max_new, int stop, int depth, uint64_t seed, int *out, double *stats, int *hist,
  int (*cb)(void *, const int *, int), void *ctx)` = `ne_fn_mtp_generate` passing them through. The server uses it as
  soon as the symbol exists (server/backend.py `RealBackend.mtp`). (2) **MTP from the model's current state**
  (`prompt[0, T)` already in the caches, prefill only the rest): the server's prefix reuse (checkpoints, live state)
  works only on the 1-row step path, so a multi-turn conversation must choose between MTP (+46% decode) and reuse
  (a 20K-token agent prompt re-prefilled every turn otherwise); needs the head KV kept per position (or rebuilt for
  the suffix from layer 47's hidden rows) and the lazy pending rows flushed at the end of a run instead of leaving
  the model dirty. (3) **official `ne_fn_model_snapshot` / `_restore` / `_snap_free`** in model.h: copies of every
  GDN conv + ssm, the PLE conv, T and the n-gram history (113 MB), restore = memcpy back + `pos = T` + `dirty =
  false`; the code is `server/shim/ne_serve_state.mm` (built outside libengine against model.h's `Model` layout,
  guarded by mtime + a T / tmax cross-check); moving it into model.mm removes that coupling.
- **lab-prefill** (from lab-server): `Prefill::recs` caches one recording per (chunk length, T0 for dense rows)
  without bound; a server sees every suffix length. The server feeds canonical pieces (2049, 1025, ..., 17 tokens)
  to bound it; an LRU cap (e.g. 32 recordings) in `get_rec` would make that unnecessary.
- **lab-loop** (from lab-speed, 15 Oct evening): **HC read v2, -0.30..-0.35 ms per step** (section 5b; kernels and
  recording helpers committed, `kernels/fn_rec.h`, linked into libengine). In `models/flash_next/layer.mm`:
  `#include "kernels/fn_rec.h"`, then in `record_stage`
  ```
  case Stage::AttnHCPre: krec::hc_pre3(r, w.hca, R, h_in, w.hca.inject, s.inj_a, s.hc_part); break;
  case Stage::AttnHCMix: krec::hc_mix(r, w.hca, R, h_in, s.mix_a, s.inj_a, s.xa, s.hc_part); break;
  case Stage::MlpHCPre:  krec::hc_pre3(r, w.hcm, R, s.h2, w.hcm.inject, s.inj_m, s.hc_part); break;
  case Stage::MlpHCMix:  krec::hc_mix(r, w.hcm, R, s.h2, s.mix_m, s.inj_m, s.xm, s.hc_part); break;
  ```
  (`krec::hc_pre` = fn_hc_pre2 if you prefer the threadgroup-staged, bit-identical 1/rms; `s.hc_ctr` is not used).
  The head mixer in `model.mm record_head` (same threadgroup counts, indirect as now):
  `fn_hc_pre3_g{G}_r{R}` with `{hf, x.norm, x.down.w, x.down.s, x.down.b, m.dummy_inj, s.inj_a, s.hc_part}` and
  `fn_hc_mix2_g{G}_r{R}` with `{hf, x.norm, x.up.w, x.up.s, x.up.b, s.mix_a, s.inj_a, m.x, s.hc_part}`; the same swap
  for the mixers in `mtp.mm` / `mtp_loop.mm` (and `nax::record_layer`'s R = 8 split becomes unnecessary with pre3:
  I will drop it in kernels/nax once the graph uses v2). Semantics: hc_pre3 alone no longer produces mix / inj (the
  pair does: `ne_fn_layer_run_stage` block 0 / 4 run both, fine; `run_dispatch(0)` alone is not). Checked at R = 1
  (every intermediate of layers 0-3 vs the old pair, rounding ON); R > 1 compiled, not run: please run
  `make test-forward` (verify4 cases) and a greedy generate after the switch.
- **all** (from lab-speed): `fn_moe_router`'s `uq` slots are now 4-bit fields of `e[1]` (R <= 8; lab-mtp's (5)); 1, 2, 3,
  4, 5, 6, 8-row instantiations live in fn_trunk / fn_moe (fn_attn: 1, 2, 4, 6, 8); `rows68_*` / `rows35_*` are gone
  (lab-loop's and lab-mtp's requests: done in ba9b934).


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

- **from lab-prefill** (details: docs/prefill.md section 6): (1) lab-forward: route `prefill()` / `generate()` through
  the chunked prefill: in model.h `#include "models/flash_next/prefill.h"` is circular, so forward-declare
  `struct Prefill; struct PrefillDeleter { void operator()(Prefill *) const; };` and add a member
  `std::unique_ptr<Prefill, PrefillDeleter> pf;` to `Model`; in model.mm `#include "models/flash_next/prefill.h"` and
  at the top of `prefill(m, prompt, n)`: `if (n >= 16 && !getenv("NE_FN_PREFILL_STEPS")) { if (!m.pf) m.pf =
  prefill_create(m, 2048); return prefill_prompt(*m.pf, prompt, n); }` (2048-token prompt: 13.0 s with the 4-row
  path -> 1.05 s). (2) Move `struct ne_fn_model` into model.h (prefill.mm re-declares the identical struct).
  (3) lab-attn: `alloc_step_scratch(fmt, tmax)` (prefill.mm copies alloc_cache's scratch shapes for 16 concurrent
  slices) and a multi-row (64+) sparse attention core over the union of the rows' selected blocks: qsa_attn +
  qsa_merge = 86 + 17 us per 8 rows at any T >= 2K, the long-context cost of prefill.

Status 15 Oct (lab-forward): lab-core's aligned pack exists (`IJAH_PACK`) and the Makefile items are done
(`.DEFAULT_GOAL`, `LIB` before the includes, and now every `build/obj/**/*.d` is included); lab-ref's (2)-(4) are
done in `tests/test_fn_layer.py` (mode 3 on a rounding-off layer, cosine slack scaled by the chain scale squared,
every golden case incl. verify4 / PLE / attention / long via the rows + KV API). New:
- **lab-core** (from lab-forward, FYI): two recorders over the 81 GB pack cost 300-500 ms per switch (each builds
  its own residency set); the forward now adds one queue-level residency set with the whole text model
  (`[queue4 addResidencySet:]`, model.mm), after which switches are free. A core option for this (or recorders
  sharing a set) would help other multi-recording users (MTP verify vs decode).
- **lab-attn** (from lab-forward): `qsa.mm`'s `ref_round()` read only `NE_FN_ROUND`; please use
  `ne::fn::g_ref_round` (done in caf5436). The forward calls `record_step(..., advance = false)` with its own shared
  pos buffer.
- **lab-forward** (from lab-attn, details docs/attn.md Requests): (1) share the QSA step scratch:
  `attn::alloc_cache(fmt, tmax, &first.qc)` for the attention layers after the first (~7 MB + 36 B x tmax each);
  (2) on the QSA path `GdnState.ikeys` (f32 [tmax][128]) is written by fn_attn_proj and `set_kv` but never read
  (`qsa_prep` keeps bf16 raw keys in `qc.ikeys`): 64 MiB per layer at tmax 131072, bind a dummy when `g_attn_qsa`;
  (3) the step is 6 dispatches now (`qsa_select_local`); MTP with the shared pos: one rollback per round.
  -> lab-forward: (1) and (2) done (`alloc_state(w, tmax, rows, share)`: the model's attention layers after the
  first share its QSA step scratch and one f32 indexer-key sink; the sink cannot be a small dummy, fn_attn_proj
  writes at pos); (3) noted (one shared pos = one rollback per round for the 12 layers).

- **lab-mtp** (their docs/mtp.md section 7, status by lab-forward): (1) done: `record_stage` uses `nax::record_gdn_in`
  / `record_attn_proj` / `record_out_hc` at R >= 4 (default; `NE_FN_NAX=0` = SIMD), prefill 145-159 -> 167 tok/s,
  verify4 goldens pass. (2) not done: moving the 6/8-row instantiations into fn_trunk / fn_moe / fn_attn means
  deleting lab-mtp's `kernels/nax/rows68_*.metal` in the same change; the session's permission check refused
  deleting another lab's files, so lab-mtp please make that change (append `NE_INST_ALL(6) NE_INST_ALL(8)` /
  `NE_INST_MOE(6/8)` / `NE_INST_ATTN(6/8)` and remove the three includes in one commit). (3) done:
  `alloc_scratch(rows)` takes 1..16, `alloc_attn_state(tmax, rows)` / `alloc_state(w, tmax, rows)` size the dense
  partials for `rows`. (4) done: `load_layer_prefix(pack, "language_model.mtp.layers.0.")` (QSA weights from the same
  prefix). (5) open (dedupe is opt-in and R <= 4 here). (6) open: the model keeps layer 47's output in
  `h[48 % 2]` per step (1 row); a verify recording with R rows, rollback v1/v2 buffers and the head is the MTP
  token-loop work. Acceptance measured for lab-mtp: docs/forward.md section 3b.

- **lab-ref / tests owner** (from lab-quality, 15 Oct): please hook `make test-quality` / `make quality-report`
  (tests/quality/quality.mk, docs/quality.md) into the build by appending one line to `tests/tests.mk`:
  `-include tests/quality/quality.mk`

- **lab-speed** (from lab-loop, 15 Oct; the MTP token loop, docs/mtp.md section 10): (1) verify widths 3 / 5 / 6 / 8
  in fn_trunk / fn_moe / fn_attn and the stop-gaps deleted (`kernels/nax/rows68_*`, my `models/flash_next/rows35_*`):
  done by lab-speed (thanks); the loop verifies at R = D + 1 = 2..5 rows, so 3 and 5 are on the hot path. (2)
  lab-mtp's open (5): `fn_moe_router` packs row r's slot byte into `uq` `e[1]` byte r, so rows >= 4 overwrite `e[2]`
  (only the opt-in dedupe kernels read it; the loop runs R = 5). -> lab-speed: (1) and (2) done in ba9b934 (rows35
  deleted with your go-ahead; uq slots are 4-bit fields, R <= 8). (3) Optional, open: an R <= 3 policy inside
  `nax::record_layer` (SIMD out_proj / attention projections at R <= 2-3, where lab-mtp measured SIMD 9-23 us per
  layer faster; the lazy GDN in_proj must stay NAX: epilogue 6 exists only there). (4) `nax::alloc_scratch_rows(R)` for
  R = 3, 5..8 starts from `alloc_scratch(4)` and re-allocates only the fields lab-mtp needed: `ple_emb` / `ple_kv` /
  `hp` stay 4 rows, so a 5-row PLE step overflows them (found by the loop's verify-forced check: R = 5 KLD 1.0 until
  the loop switched to `fn::alloc_scratch(R)`, which handles any R since 95e1344). Suggest
  `alloc_scratch_rows(rows) { return fn::alloc_scratch(rows); }` (the MTP head's 8-row scratch uses it; no PLE there).

Original requests (lab-layer):
- **lab-core**: (1) an aligned pack (your `engine-cli align-pack`): until then 48 layers would copy ~75 GB. (2) The
  Makefile includes `models/*/model.mk` before `all:` / `LIB` exist (a rule there becomes the default goal; I set
  `.DEFAULT_GOAL := all`) and does not include the `.d` files of `TEST_BINS` objects; please set
  `.DEFAULT_GOAL := all`, define `LIB` before the includes, and `-include` every `build/obj/**/*.d`.
- **lab-ref**: (1) `ne_fn_layer_run_stage(block)` is again the 6 sub-blocks your `STAGES` uses (I had briefly
  renumbered it per dispatch: that was the "not isolated" finding); single dispatches are `ne_fn_layer_run_dispatch`.
  Stage checks all pass now. (2) With reference rounding ON, your mode-3 GDN-state check (f32 op reference, tol 1e-5)
  fails by design (bf16 q/k): use the `bf=True` op references when rounding is on, or create the layer after
  `ne_fn_set_ref_rounding(0)` for mode 3. (3) Chain-mode cosine thresholds on intermediates (`mlp_mix`, shared gate,
  ssm state, topk weights) fail at 5e-3..6e-3 rel from compounding; `hc_out` passes. (4) New C API: rows
  (`ne_fn_layer_create_rows`), attention layers (`ne_fn_layer_set_kv` / `get_kv`; intermediates `attn_q`, `attn_kv`,
  `attn_iqk`, states `k_cache`, `v_cache`, `index_keys`), `ne_fn_set_ref_rounding` — all in `models/flash_next/layer.h`.
  An L03 / verify4 entry in `test_fn_layer.py` would cover them.
- **from lab-mtp** (details and measurements: docs/mtp.md sections 3 and 7): (1) multi-row trunk on NAX at R >= 4
  (`nax::record_gdn_in` / `record_attn_proj` / `record_out_hc` are drop-ins for the GdnIn / GdnOut dispatches, same
  buffers; L00 4 rows 530 -> 494 us, 8 rows 917 -> 754 us; SIMD stays at R <= 2), exact patch in docs/mtp.md;
  (2) instantiate 6 / 8 rows in fn_trunk / fn_moe / fn_attn and delete `kernels/nax/rows68_*.metal` in the same commit;
  (3) `alloc_scratch` / `alloc_attn_state` for any R; (4) `load_layer_prefix` (the MTP layer is at
  `language_model.mtp.layers.0.`); (5) router `uq` slot bytes overflow `e[1]` at R > 4 (dedupe path only);
  (6) the MTP token loop needs every verify row's layer-47 output (pre-mixer, [R][10240]) and the GDN layers recorded
  with `nax::alloc_layer_extra(0, true, &keep)` (rollback v2, lazy commit: free; reads the same keep buffer as
  lab-attn's rollback); (7) please run `build/mtp-accept-eval $PACK tests/golden/data 256 4 [4]` under your lock (full
  model): measured greedy MTP acceptance on the engine + the tok/s it implies (docs/mtp.md section 6).

- **from lab-trunk6** (16 Oct; details docs/trunk6.md): EXL3 K6/K7/K8 + bf16 trunk kernels and the T6 pack
  (`<packs>/Qwen3.8-Flash-Next-EXL3-K5-T6`, layers 0-5 + bf16 mixer now, all 48 after the night run).
  (1) **lab-speed** (kernels/kernels.mk): add
  ```
  ENGINE_SRCS += kernels/trunk6/trunk6.mm  # lab-trunk6: EXL3 / bf16 trunk Linears (docs/trunk6.md)
  -include tests/trunk6/trunk6.mk
  ```
  (trunk6.mk then links build/t6-check against libengine only; until then it builds with
  `make -f Makefile -f tests/trunk6/trunk6.mk BUILD=build-t6 build-t6/t6-check`.)
  (2) **lab-integrate** (models/flash_next/): per-Linear format dispatch. Load every trunk Linear with
  `t6::detect(pack, prefix)`: `Fmt::Q8` -> the existing `fn::Q8` path unchanged; `Fmt::EXL3` / `Fmt::BF16` ->
  `t6::load(pack, prefix)` (kernels/trunk6/trunk6.h). In `record_stage`, for a non-Q8 Linear record
  `t6::record(r, lin, R, x, y_raw, t6scratch); r.barrier();` into an f32 [R][out] buffer, then the stage's epilogue
  on y_raw: GDN in_proj (conv1d K4 + silu / z / beta / g: nax_q8_gemm EPI 1's code as an epilogue-only kernel),
  out_proj / o_proj (+ HC write, EPI 2), attention projections (EPI 3 stores), shared gate/up/down (SwiGLU, gated
  shared term), HC down (x = concat_s(h_s * hc_norm_s / rms_s), one 10240 -> 320 GEMV: the engine's per-stream
  1/rms_s factor is a row scale, so it folds into x), HC up (logits [10240] -> fn_hc_mix's sigmoid-mean epilogue),
  head mixer the same with the bf16 Linears. One `t6::Scratch` (alloc_scratch()) serves the whole recording. Unfused
  first (K6 in_proj_qkv alone: 39 us vs ~49 us for the q8 qkv rows at 557 GB/s), fuse in_proj / out_proj after.
  Checks: fn-layer-check on the T6 pack (layers 0-3 converted; its goldens are the 8-bit reference, so expect the
  quantization difference, docs/trunk6.md section 4), then `make test-quality` on the 48-layer pack (lead 5 predicts
  lower KLD than the K5 pack's 8-bit trunk).
  **Done by lab-t6wire (16 Oct, branch lab/t6wire; docs/trunk6.md section 8)**: (1) as written; (2) the format switch
  per Linear (`Q8::t6`), all stages incl. nax::record_layer (MTP verify, rollback / lazy GDN), the head and verify
  mixers and the chunked prefill; the in_proj / out_proj / HC-act / iqk epilogues fused in the t6 finisher.
