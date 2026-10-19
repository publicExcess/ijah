# EXL3 trunk (lead 5's K6 / K8 / bf16 recipe) in the engine (lab-trunk6)

Owner: lab-trunk6 (`kernels/trunk6/`, `tools/convert_trunk*`, `tests/trunk6/`, this file). 16 Oct 2026. Numbers are
indicative (one machine, medians of 7, machine lock held).

## 0. Status

- **Kernels** for EXL3 dense Linears, K6 / K7 / K8 (MCG window 15, Hadamard 128 / 64 / 16 + suh / svh folded in):
  1-8 rows SIMD (`t6_simd`), 1-16 rows NAX matmul2d (`t6_prep` + `t6_nax`), plus bf16 dense (`t6_bf16`, the final
  mixer). **Exact** vs numpy on every real encoded matrix of layers 0-3 (K6 / K8) and a K7 encode of layer 0:
  342 cases, rel <= 2.3e-7 against the kernel's own arithmetic in numpy, <= 5.8e-4 against f64 x @ W_hat (the f16
  prepared input), deterministic (two runs bit-identical).
- **Converter** `tools/convert_trunk.py`: full-H capture (layer-sequential, bit-identical to lead 5's whole-forward
  capture), lead 5's encode (bit-identical proxies), aligned per-layer shards, symlinks to the K5 experts / n-gram /
  MTP / head, resumable, RAM + disk watchdogs. Pack: `<packs>/Qwen3.8-Flash-Next-EXL3-K5-T6` (layers 0-3
  converted + bf16 mixer today, and layers 4-5 by the night path's smoke run; layers 6-47 still the 8-bit trunk).
- **4-layer proof**: slice KLD vs the bf16 teacher with layers 0-3 from the pack: A -0.00019 [-0.00025, -0.00012],
  B -0.00016 [-0.00017, -0.00014], C -0.00022 [-0.00023, -0.00020] vs Sushi's 8-bit trunk = lead 5's row. On real
  layer activations (lab-ref's goldens) the engine kernels' Linear outputs are 0.81x the 8-bit trunk's error
  (geomean of 74 Linear x case pairs; section 4).
- **Integrated (lab-t6wire, 16 Oct, section 8)**: the layer graph dispatches every trunk Linear on its pack format
  (8-bit affine unchanged; EXL3 / bf16 through these kernels + fused finisher epilogues) at 1 row, R rows (MTP
  verify incl. R = 8 and the rollback / lazy-commit GDN epilogues, NAX from 4) and in the chunked prefill (dense W'
  decode + NAX GEMM). The full 48-layer pack was encoded 06:49-07:36 (night command, section 5).

## 1. Format (per Linear; y = x @ W, W [in, out] = HF weight^T)

| tensor | dtype / shape | meaning |
|---|---|---|
| `<prefix>.trellis` | U16 [in/16, out/16, 16 K] | W' tiles, tensor-core slot order, Sushi / exllamav3 bit layout (MCG, window 15) |
| `<prefix>.suh` | F16 [in] | input scales (random signs, channel scales, global scale) |
| `<prefix>.svh` | F16 [out] | output scales |
| `<prefix>.weight` (BF16, no `.scales`) | BF16 [out, in] | dense bf16 (the final hyper-connection mixer) |

`W = diag(suh) . H_hi . W' . H_ho . diag(svh)`, H_n block-diagonal normalised Sylvester Hadamard, n = 128 if it divides
the dimension, else the largest power of two <= 64 that does (HC 320 -> 64, GDN a/b 48 -> 16). K = last dim / 16.
**Format rule for the layer code** (`t6::detect`): `.trellis` -> EXL3, `.scales` -> 8-bit affine (existing kernels),
BF16 `.weight` -> dense bf16. Per-layer shard `model-t6-Lxx.safetensors` (data 64-byte aligned), mixer
`model-t6-mixer.safetensors`; source shards that still hold replaced tensors are rewritten as
`model-t6-rest-<shard>` with only the tensors the index uses (no stale 8-bit copies, no duplicate names: core's Pack
maps every tensor of every referenced shard).

Recipe t6 (lead 5): K6 for GDN in/out, attention q/k/v/o + indexer, shared expert, HC down-mixes; K8 for the HC
up-mixes (`input_mix_weight_up`) and the PLE key/value projections; final mixer bf16: 2.80 GB/token for the trunk vs
3.78 (Sushi 8-bit). Recipe t7: K7 bulk, same K8 / bf16 parts (3.22 GB/token). MTP layer, routers, norms, conv,
block_inject, embed, lm_head: unchanged.

## 2. Kernels (`kernels/trunk6/trunk6.metal`, host `trunk6.{h,mm}`)

- **Reader for any K <= 8.** A lane (tensor-core thread) owns ring slots 8l..8l+7; only the low 15 bits of each
  codeword matter (window 15), so its 8 codewords span 15 + 7K bits: <= 64 for K <= 7 (one 3-word funnel, then
  compile-time extractions), 71 at K = 8 (3 words at a lane-independent offset 25, extractions from (w0, w1) or
  (w1, w2)). The engine's existing `dec8` (16 + 7K bits in a 64-bit window) only reaches K6.
- **Hadamard of any size 16..128** on the (lane, +32, +64, +96) chunk layout: shuffles for index bits below
  min(N, 32), register pairs for bits 5 / 6; blocks never straddle a chunk, so partial chunks (320, 48) just carry
  zeros.
- **t6_simd** (R 1-8): threadgroup = 8 simdgroups = 128 output columns x one k split; the split's x span is prepared
  (x * suh -> H_hi -> half, k-permuted per tile) in threadgroup memory; simdgroup = one output tile; S > 1 combines
  through per-split partials (atomics) and a last-arriving finisher that applies H_ho * svh (fixed order:
  deterministic, counters reset). Split rule: smallest S with >= 320 threadgroups and >= 32 k tiles per split.
- **t6_prep + t6_nax** (R 1-16): `t6_prep` writes the prepared input to scratch (one dispatch + barrier), then
  matmul2d 16x32x16 per k step on two decoded tiles (lab lead 6's remap: lane l decodes tensor-core thread
  4 c0 + ((l & 1) | ((l >> 3) & 1) << 1), whose slots are its right-operand elements). KS = 8 simdgroups split k inside a
  threadgroup, S across threadgroups (>= 640 threadgroups, >= 4 steps per simdgroup); finisher as above. Preparing x
  inside every NAX threadgroup (as t6_simd does) cost 2x at 8 rows (28 KB threadgroup memory, 5 k steps per
  simdgroup): 104.6 -> 45.9 us on in_proj_qkv.
- **t6_bf16** (R 1-8): one simdgroup per row; when out < 2048, 8 simdgroups split a row (mixer down 10240 -> 320:
  14.8 -> 12.8 us).
- **Interface** (`t6::Linear`, `t6::load(pack, prefix)`, `t6::plan`, `t6::record(r, lin, R, x, y, scratch)`): x f32
  [R][in] -> y f32 [R][out], no trailing barrier; SIMD below 4 rows, NAX from 4 (`T6_PATH=simd|nax`, `T6_S`, `T6_KS`,
  `T6_TGMEM` override). One `t6::Scratch` (8 MB partials, 4096 counters, 512 KB prepared input) serves every Linear of
  a recording as long as they are serialised by barriers.

### Exactness (`tests/trunk6/check_kernels.py` + `build/t6-check cases`)

x = N(0,1) rows with 4 outlier channels (x20); references from an independent numpy decode of the pack.

| path | K | cases | max rel vs kernel arithmetic (f32 FWHT, f16 input) | max rel vs f64 x @ W_hat |
|---|---|---|---|---|
| SIMD R 1/2/4/8 | 6 / 7 / 8 | 120 / 40 / 38 | 1.8e-7 / 1.9e-7 / 2.2e-7 | 2.5e-4 / 2.8e-4 / 5.8e-4 |
| NAX R 4/8/16 | 6 / 7 / 8 | 80 / 30 / 26 | 1.6e-7 / 1.7e-7 / 1.7e-7 | 2.5e-4 / 2.2e-4 / 3.3e-4 |
| bf16 R 1/4 | - | 8 | 2.3e-7 | 2.3e-7 |

The f16 input preparation costs ~3e-4 relative output error, 1-5% of the K8 / K6 weight error on real activations.

### Speed (`build/t6-check bench`; weights cycled through >= 320 MB of copies, so every unit reads DRAM)

us per Linear (GB/s); q8 = the 8-bit pack's same Linear through kernels/nax's q8 GEMM (the engine's fused 1-row q8
kernels stream at 93-104% of 594 GB/s in context, docs/kernels.md 5b: e.g. qkv ~49 us at 557 GB/s).

| Linear (K, MB t6 / q8) | t6 SIMD R1 | t6 SIMD R4 | t6 NAX R1 | t6 NAX R4 | t6 NAX R8 | q8 NAX R1 / R8 |
|---|---|---|---|---|---|---|
| GDN in_proj_qkv 2560->10240 (K6, 19.7 / 27.0) | **39.4 (500)** | 48.4 | 40.9 | 42.2 | **45.9** | 55.2 / 56.5 |
| GDN in_proj_z 2560->6144 (K6, 11.8) | 27.5 | 32.2 | 27.7 | 28.3 | 30.9 | 37.0 / 37.6 |
| GDN out_proj 6144->2560 (K6, 11.8) | 28.2 | 32.4 | 27.6 | 28.5 | 30.2 | 34.6 / 35.5 |
| attn q_proj 2560->12288 (K6, 23.6) | 46.9 (504) | 54.1 | 47.9 | 48.6 | 53.1 | 65.1 / 65.0 |
| PLE key_proj 2560->10240 (K8, 26.2) | 54.1 | 58.3 | 56.3 | 56.4 | 59.6 | 56.1 / 56.3 |
| HC down 10240->320 (K6, 2.5) | 11.2 | 16.0 | 10.6 | 11.3 | 11.7 | 40.8 / 41.6 |
| HC up 320->10240 (K8, 3.3) | 9.0 | 11.7 | 10.9 | 11.8 | 14.8 | 8.2 / 8.3 |
| shared gate 2560->640 / k_proj 2560->512 (K6, ~1) | 10-12 | 10-15 | 7-11 | 8-11 | 8-12 | 8 / 8-9 |
| mixer down / up, bf16 (6.6 MB each) | 12.8 / 12.9 | 14.6 / 13.8 | | | | (q8: 41 / 8.3) |

- 1 row: big K6 GEMVs read 84-90% of peak; per byte the 8-bit q8 kernels are faster (94-100%), so in time K6 gains
  only ~10-20% on the big shapes (lead 4's finding again): roughly -12 us per GDN layer, ~-0.5 ms per token if the
  layer used these kernels as separate dispatches; the HC mixes and small Linears are latency-bound (7-12 us) either
  way. The trunk change is about quality, not speed.
- Rows: NAX is near-flat (qkv 40.9 -> 45.9 us from 1 to 8 rows) and 0.81-0.85x the q8 NAX GEMM time at 8 rows on
  the big shapes. SIMD grows ~3 us per row; crossover 2-3 rows. Small Linears: noisy at +-2 us.
- bf16 mixer: 25.7 us per token for both matrices vs ~11 us for the q8 mixer kernels (+~14 us per token, one-off).

## 3. Converter (`tools/convert_trunk.py`, encode library `tools/convert_trunk_enc.py`)

- **Hessians: full H = X^T X / rows per Linear input, captured, not a proxy.** Calibration = lead 5's (and the
  imatrix's) text: mixedweb-v1 + mlx_lm sources, 279.5K tokens, 137 windows of 2048 from a fresh cache, through the
  K5 pack in fn_exl3 (MLX). Layer-sequential: each window runs one layer at a time exactly as
  `Qwen4ExpModel.__call__` does (tiled embedding, the layer's own masks and fresh cache); the residual stream entering
  the next group is stored (`WORK/hstate-Lxx.npy`, 5.7 GB bf16) so each layer is loaded and run once; hooks only on
  the group's Linears; shared inputs captured once (GDN z / a / b = qkv; attention k / v / indexer = q; shared up =
  gate). Checked: every H of layers 0-3 is **bit-identical** to lead 5's whole-forward capture, including across a
  group hand-off (L0-1 | L2-3), and the aliases are bit-identical when captured separately (`--verify-aliases`).
  Cost ~10 s per layer (L0-1 with the PLE layer and warm-up: 59 s). A diagonal-H proxy was not used: lead 2 found
  77-99% of the trunk H's energy off the diagonal, and full H is cheap at this cost.
- **Encode** = lead 5's `enc.encode` keeping the codewords (signs seed 0, channel scales with the 1e-3 x median floor,
  Hadamards, g-scale search, block LDL + LDLQ with the damped full H (sigma 0.025), Metal Viterbi over the 2^15 MCG
  table, 2-round refit, f16 suh / svh), packed with exl3_codec.pack_codewords; every Linear is decoded again and must
  match the encoder's W_hat (< 1e-6). The proxies equal lead 5's results.jsonl to the printed digits (e.g. L00 HC down
  K6 2.54e-07 / rel 0.0201, HC up K8 2.45e-06 / 0.0127). ~61 s per layer (HC down 10240x320 13 s, the rest < 11 s).
- Stages (all resumable; a layer is done when its shard exists): skeleton (symlinks to the real files of
  `Qwen3.8-Flash-Next-EXL3-K5-a16`), mixer (bf16 from the base globals), per group of 8 layers: capture, then encode
  layer by layer (H deleted after its layer unless `--keep-h`), the index rewritten after every layer (a partial pack
  is valid: unconverted layers keep the 8-bit trunk), at the end the slim of the source trunk shards.
- Safety: watchdog thread (exit 3 if system free memory < 8% or MLX memory > `--mem-limit` 100 GiB), disk check
  (stop if < `--min-disk-gb` 150 GB free), plus tools/with_lock.sh's own RAM / swap watchdog. Base weights:
  `BASE` = lab-exl3-pack's fetch in a scratch directory (all 48 layers present today); a missing layer is fetched
  by HF range reads (exl3/pack/fetch_base.py --no-experts) into `WORK/base`.
- `--status`: encoded layers, Hessians on disk, mean encode time and the 48-layer estimate (CPU only).

## 4. The 4-layer proof (layers 0-3 encoded with the converter, 16 Oct 05:13-05:24)

| | A wikitext | B reasoning | C 32K |
|---|---|---|---|
| base: Sushi 8-bit trunk (lead 5) | 0.00084 | 0.00056 | 0.00061 |
| **T6 pack, decoded (tests/trunk6/slice_kld.py)** | **0.00065** (-0.00019 [-0.00025, -0.00012]) | **0.00040** (-0.00016 [-0.00017, -0.00014]) | **0.00040** (-0.00022 [-0.00023, -0.00020]) |
| lead 5's K6/HCu8/PLE8 + fm (lab encodes) | 0.00064 (-0.00020) | 0.00040 (-0.00016) | 0.00040 (-0.00021) |

(4-layer slice of the K5 pack through fn_exl3, the 52 Linears of layers 0-3 + mixer as dense f32 from the pack,
kld_core vs `RUN/teacher/ref-*-L04`, paired 95% bootstrap CIs over windows; rerun 0; top-1 A/B/C 0.917 / 0.929 /
0.933 vs base 0.902 / 0.907 / 0.915; C flat over position bins.)

In the engine, on real activations (`tests/trunk6/layer_goldens.py`: lab-ref's L00 / L01 / L03 decode1 + verify4
goldens; a pair is used only if the 8-bit weight reproduces the golden output, < 2e-3; 74 pairs), Linear output error
vs the bf16 base weight, engine kernel (T6 pack) / 8-bit trunk, geomean: HC up K8 0.33, GDN out 0.74, HC down 0.78,
PLE value K8 0.77, GDN in 0.93, shared expert 1.00, **attention 1.48** (K6 loses to 8-bit there, as lead 5 found for
q_proj); all 0.81. Kernel vs the numpy decode of the same pack on these inputs: <= 4.1e-4 (the f16 input).
The full-layer engine graph on the T6 pack: section 8.

## 5. Night run (the full 48-layer encode)

```
cd <repo>
LOCK_OWNER=lab-trunk6 NEED_GB=40 LOCK_WAIT=60 tools/with_lock.sh "trunk6 night: H capture + EXL3 encode L6-47 (~35 GB)" <START+2h> -- \
  .venv/bin/python -I -B tools/convert_trunk.py --layers 0-47 \
  > <packs>/trunk6-work/night.log 2>&1
.venv/bin/python -I -B tools/convert_trunk.py --status   # any time
```

- **Expected duration ~55 min** (45-70): capture ~8 min (42 layers x ~10 s from the stored hand-off
  `hstate-L06.npy`, 6 group loads, 5 hand-off writes of 5.7 GB), encode 42 layers x ~59 s = 41 min, final slim
  < 1 min. Smoke run of exactly this path (16 Oct 06:10, `--layers 4-5`): layers 0-5 run from the embedding (0-3
  without hooks) in 55 s, 2 layers encoded in 2 min, index + slim correct; 2.9 min in all. Resumable: rerun the same
  command after any stop (a stopped capture group is redone; finished layers are skipped).
- RAM: ~35 GB peak (8-12 layers of the K5 pack at ~1.7 GB each, their H ~8 GB, MLX temporaries; not a whole-model job, so
  NEED_GB=40 only keeps it off a loaded 78 GB server's last 10%). Disk: +2.6 GB shards; transient <= 8 GB H +
  11.4 GB hand-offs under `<packs>/trunk6-work` (delete the directory afterwards; keep `encode.jsonl`).
- Output: `<packs>/Qwen3.8-Flash-Next-EXL3-K5-T6` with all 48 layers' trunk in EXL3 + bf16 mixer
  (`model.safetensors.index.json` metadata `trunk6.layers` lists them). Checks after: `--status`; the slice KLD of any
  4 layers is the same script; the whole-model KLD needs the engine's format dispatch (or fn_exl3 with the dense swap
  of slice_kld.py over 48 layers: 48 x 52 dense f32 Linears ~15 GB extra).
- Base weights are read from a scratch directory (`exl3-pack/run/base`, /private/tmp); if it was cleaned, the
  converter fetches each layer's trunk by range reads (~0.15 GB per layer).

## 6. Traps

- K7 / K8 do not fit the engine's 64-bit lane window (16 + 7K bits); 15 + 7K is enough because window 15 masks the
  top bit, and K8 needs 3 words.
- In-threadgroup input preparation is fine at 1 row (span x 16 halves) but at 8 rows its threadgroup memory (28 KB)
  cut occupancy and the k split left 5 steps per simdgroup: 2x slower. Separate prep dispatch for NAX.
- Timing with a 256 MB streaming flush between units left +-5 us of noise per Linear (the flush's own variance);
  cycling the weights through >= 320 MB of copies is stable.
- The engine's Pack maps every tensor of every shard the index names: the bf16 mixer `.weight` clashed with the
  8-bit `.weight` in the source trunk shard ("appears twice"). Hence the rest shards.
- `mlx-vlm` decoder layers need `position_ids=None` passed explicitly; GDN `in_proj_a/b` never pass through a hook
  (gather_qmm), their H is in_proj_qkv's (lead 5).
- Someone else's `kernels/attn/qsa_chunk.metal` did not compile at 05:30; t6-check was built with a metallib linked
  from the AIRs that did (`build-t6/`). trunk6.metal compiles clean in the shared metallib.

- (lab-t6wire) The layer type was detected by `linear_attn.in_proj_qkv.weight` / `self_attn.q_proj.weight`: a T6
  layer has `.trellis` instead ("neither GDN nor attention"). zsh does not word-split `$PY` in scripts (`${=PY}`).
- (lab-t6wire) The golden layer cases cannot validate a T6 layer (other weights): check against numpy on the T6
  weights from the engine's own inputs (tests/trunk6/layer_t6.py); whole-model drift between jobs (+-1.5 ms per
  step for minutes) makes cross-job tok/s comparisons meaningless: K5 and T6 in one job.

## 7. Attribution

The encode pipeline re-implements exllamav3's quantize_exl3 (turboderp, MIT) via the lab's `exl3/convert/encode.py`,
`exl3_codec.py` and `engine/trunk/viterbi_lut_metal.py` (imported read-only); the MCG codebook, bit layout, tensor-core
slot order and the simd-shuffle FWHT follow sushi (github.com/beamivalice/sushi, MIT, (c) 2026 Theinruj Toranavikrai,
David Dalcu) and exllamav3; the lane funnel reader and dense GEMV shape are lab leads 1 / 4, the NAX operand layout
lab lead 6 (from sushi's nax_wfrag). MIT License: permission is hereby granted, free of charge, to any person
obtaining a copy of this software ... THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND (full text in each
upstream LICENSE).

## 8. Integration in the layer graph (lab-t6wire, 16 Oct)

**Format switch per Linear** (`models/flash_next/layer.mm` `q8()`): `t6::detect` at load; an EXL3 / bf16 Linear
becomes a `Q8` with `Q8::t6` set (w / s / b empty, group 0), so the layer structs, shape checks and every caller stay
as they are. A layer whose trunk is EXL3 (`LayerW::t6`; a T6 pack converts whole layers, mixed layers are refused)
records its stages in `models/flash_next/layer_t6.mm` (`t6_record_stage`, one hook at the top of `record_stage` and
one in `nax::record_layer`); q8 layers run the old code untouched. Unchanged kernels: GDN step, attention core
(QSA), router GEMV, routed experts (the router / gateup dispatches skip their fused shared-expert threadgroups).

| stage | EXL3 path (each Linear `t6::record_multi`: a stage's Linears concurrently, one scratch slot each) |
|---|---|
| HC read (x2 per layer) | `t6e_hc_prep` (per-stream 1/rms, x = h * hc_norm / rms_s: the q8 path's per-stream row scale folded into one 10240 -> 320 input; inject partials, bf16) -> down K6 with **fused** silu(/4) ‖ `t6e_hc_act` (inject gates) -> up K8 -> `t6e_hc_out` (sigmoid-mean) |
| GDN in | qkv / z / b / a K6, epilogues **fused** in the four finishers: conv1d K4 + silu (state; rollback history EPI 5, lazy commit EPI 6), rr, beta, decay |
| attention proj | q / k / v / indexer K6 straight into aq / kv (row stride 1024) / iqk (+ raw key sink) |
| out_proj / o_proj | K6, branch + HC write **fused** |
| shared expert | gate / up K6 (alongside the router) -> `t6e_swiglu` -> down K6 (alongside the routed gate / up) |
| PLE (layer 1) | key / value K8 -> `t6e_copy` (rr) into ple_kv |
| final mixer (head, MTP verify) | the HC read with the bf16 Linears (`t6_bf16`), no inject, unfused epilogues |
| chunked prefill | `gemm()` -> `t6::record_prefill`: `t6_pf_prep` (x * suh, H) ‖ `t6_dense` (W' -> half [out][in]), `pf_h16_hf` (lab-prefill's NAX GEMM with half operands), `t6_pf_post` (H, svh, f32 / bf16 store); 200 MB of buffers; since 18 Oct (pfkern, docs/prefill.md 4d) bf16-input Linears from 64 rows: `t6_dense` -> `t6_fold_k` / `t6_fold_n` (Wf = svh H W'^T H suh, half) -> `t6_fold_gemm` (MPP tensor op, y stored directly); the rest `t6_h16_gemm`; opt-in INT8 (`NE_T6_A8`) |

Fused epilogues = `t6_finish` with `T6_EPI` (FC 20; the finished 128-column block goes back to threadgroup memory,
then one thread per column runs the q8 path's epilogue over the R rows, same rounding points). `NE_T6_FUSE=0` = the
separate `t6e_*` epilogue kernels (`kernels/trunk6/t6_epi.metal`), kept for A/B.

### Checks

- **Per op vs numpy on the T6 weights** (`tests/trunk6/layer_t6.py`; lab-ref's golden cases only supply inputs:
  their weights are the 8-bit trunk's; tools/opref.py's PackNP now decodes EXL3 Linears): one engine step, every t6
  stage recomputed from the engine's own input buffers. L00 / L01 (PLE) / L03 (attention) at 1 and 4 rows (SIMD and
  NAX paths), rounding off vs f32 references: worst rel L2 over the 6 cases 1.1e-3 (HC block_in: the up Linear's f16
  input), GDN in <= 8.7e-5, out_proj 2.9e-4, attention projections <= 1.6e-4, shared expert <= 2.0e-4, PLE <= 1.9e-4,
  whole MoE out (routed + gated shared) 2.6e-4, HC writes <= 4e-8. Unfused epilogues: the same (<= 1.1e-3).
  Rounding ON vs the bf16 references: <= 5.9e-3 (bf16 flips; inject gates). All PASS.
- **Prefill slice** (`build/pf-check layers PACK TOKENS N 4`: first 4 T6 layers, 1-row decode vs the chunked prefill;
  rel L2 of the hidden after layer 3): n 64: median 1.02e-2 (K5 pack: 6.4e-3), 4-row floor 7.0e-3 (K5 6.0e-3), max
  9.1e-2 = the floor's max 1.05e-1 range; n 2048 (chunk 2048): median 6.7e-3 (K5, docs/prefill.md: 6.3e-3), L03 k
  1.3e-2 (1.3e-2). The prefill adds nothing beyond the decode family's spread.
- **Layer step GPU time** (`ne_fn_layer_time`, SLC flushed, f32): L00 1 row K5 256 / **T6 262 us** (unfused 271),
  4 rows 463 / **427**; L02 257 / 262, 461 / 426; L03 (attention) 249 / 254, 455 / 423.

### Whole model (all 48 layers T6)

Measured 16 Oct 07:43-08:10 (K5 and T6 back to back in one job per row group, K5 = `Qwen3.8-Flash-Next-EXL3-K5-a16`, f32, QSA, one run each):

| | K5 (8-bit trunk) | **T6** | |
|---|---|---|---|
| teacher KLD set B (`tools/kld_teacher.py`, 16 windows) | 0.00995, top-1 0.9739, NLL 1.0611 | **0.00906**, top-1 0.9733, NLL 1.0590 | paired -0.00090 [-0.00202, +0.00005] |
| logits vs the MLX K5 reference (`compare_logits.py`, 289 positions) | KLD 2.96e-3, top-1 99.7% | 3.17e-3, top-1 98.6% | floor 2.42e-3 |
| **engine K5 vs engine T6** (same tokens, `tests/trunk6/engine_kld.py`) | | KL mean **3.0e-3**, median 1.7e-4, top-1 99.0% | |
| decode GPU ms / step, T 64 / 2048 (`bench_forward.py`) | 13.43 / 13.70 | 13.74 / 14.01 (+2.3%) | 1178 dispatches vs 548 |
| decode tok/s greedy / sampled | 73.2 / 72.3 | 70.7 / 69.8 | |
| MTP verify GPU ms, T 64, R 2 / 3 / 4 / 5 (`bench_mtp.py --time`) | 17.94 / 20.83 / 23.67 / 30.47 | **16.34 / 19.90 / 22.11 / 24.92** | NAX forced at R 2-3: no gain |
| MTP d2 tok/s greedy, median of 3 h2h prompts x 1024 (one job; tok/round) | 95.7 (95.7 115.1 91.6; 2.41) | **97.9** (97.9 120.8 92.4; 2.38) | no-MTP 71.8 / 71.2 |
| prefill 10K tok/s (`pf-check speed`, chunk 2048) | 1918 | 1848 (-3.6%) | +8.7K dispatches |

- Quality: lower teacher KLD on B (-9%; the CI just touches 0 over 16 windows), NLL closer to the teacher's; the
  logits move by about the engine's own noise floor vs K5 (KL 3e-3 mean, 1.7e-4 median). Sets A / C not run (light
  testing).
- An earlier MTP run (T6 alone, 07:45) gave 85.9 / 86.3 tok/s greedy / sampled with no-MTP at 64.3 (GPU 15.2 ms per
  token): the documented between-job drift, not the pack.
- Speed: the trunk reads 0.98 GB less per token (-1.8 ms at 550 GB/s), but the 1-row step is +0.3 ms: the HC read
  (x2 per layer) is 5 dispatches + 4 barriers instead of 2 (HC down 10240 -> 320 and up are latency-bound t6 GEMVs,
  ~11 / 9 us), the shared expert 3 phases instead of 2 fused ones. From 2 rows up T6 is faster (verify -1.6 ms at
  R 2, -5.6 ms at R 5; layer step 4 rows 427 vs 463 us). Next: fold the HC prep (per-stream 1/rms x hc_norm) into
  the down GEMV's in-threadgroup x preparation and the shared SwiGLU into the down Linear's prep (-2 barriers per HC
  read / MoE, ~-0.4 ms), then the HC up + sigmoid-mean.

## 8b. Fewer dependent phases in the T6 decode path (lab-t6fuse, 16 Oct)

**Folds** (`NE_T6_FUSE2=0` = section 8's recording, kept for A/B; `models/flash_next/layer_t6.mm`, `kernels/trunk6/trunk6.metal`):

| stage | section 8 | now |
|---|---|---|
| HC read part 1 | `t6e_hc_prep`, barrier, down ‖ `t6e_hc_act`, barrier | **down** with `T6_PRE` 1 (FC 23): each threadgroup computes its stream's 1/rms (the whole 2560-stream from L2, same sums and order as t6e_hc_prep) and prepares x = h * hc_norm * inv in its own prep; column block 0's threadgroups also take the 4 inject rows' 256-chunk partials (atomics), its last arrival sums them into inj (t6e_hc_act's order); EPI 2 writes mix. 1 dispatch, 1 barrier |
| HC read part 3 | up, barrier, `t6e_hc_out`, barrier | **up** with EPI 8: each finished 128-column block (one stream) is published (atomics) and counted per column group (`ctr[128 + cb % 20]`); the 4th arrival computes the sigmoid-mean for those columns. 1 dispatch, 1 barrier |
| shared expert | sg ‖ su ‖ router, barrier, `t6e_swiglu`, barrier, sd ‖ routed gate/up | sg ‖ su ‖ router, barrier, **sd** with `T6_PRE` 2 (SwiGLU in its prep; block 0 also writes shared_gu) ‖ routed gate/up |

SIMD path (R <= 3) only for the two `T6_PRE` folds (the NAX path keeps the separate kernels); EPI 8 on both. Per
layer 15 -> 10 barriers (= K5's), dispatches per step 1178 -> 842. `tests/trunk6/layer_t6.py`: all PASS with the
same errors as section 8 (worst HC block_in 1.10e-3). Not bit-identical to the unfused recording (fn-prof 48-layer
hash differs; f32 contraction of h * w * inv in one expression), greedy text unchanged.

**Decode k splits.** The small T6 Linears are a serial chain of `span` k tiles per simdgroup (1-deep prefetch), so
the split rule (>= 32 tiles per split) left the shared gate/up (span 32, 25 threadgroups) and shared down (span 40,
20 threadgroups) latency-bound. `t6::plan` now uses, for R <= 3: shared gate / up 2560 -> 640 S 10, shared down
640 -> 2560 S 5, out_proj / o_proj 6144 -> 2560 S 24 (in-process A/B below: -0.11 / -0.11 / -0.08 ms, together -0.33).
Neutral (within +-0.05 ms): HC down S 40, a / b S 20, k / v S 10, qkv S 2 / 5, z S 10, q S 2, out_proj S 6 / 48.
Worse: HC up S 5 (+0.19), shared down back at S 1 (+0.13). A 2-8 deep prefetch ring in t6_simd (FC, all small
Linears): +0.2 ms whole model (as lab-speed found for the EXL3 MoE kernels), dropped. `T6_S_<in>x<out>` overrides a
shape's split (A/B runs).

**Method.** `build/fn-prof` now runs T6 packs (bf16 mixer head) and has `NE_PROF_ABENV="K=V,..."`: the new body with
the env settings (t6::plan reads them at record time; `FUSE2=0|1` -> g_t6_fuse2, `X=bits` -> g_t6_x), the old body
without, interleaved in one process (4 T6 layers in the 48-layer pattern, T 64, rounding ON; noise ~+-0.04 ms).
`NE_PROF_KINDS=none NE_PROF_ABENV=FUSE2=1 NE_T6_FUSE2=0 build/fn-prof T6PACK 15 64 5`: folds -0.33 ms body. The
per-stage leave-one-out table was too noisy on T6 to use (negative entries on a busy run). Whole model:
`tools/bench_forward.py --ctx 64,2048 --gen 32`, every variant its own process, two rounds in one job:

| GPU ms / step, T 64 / 2048 (two rounds) | round 1 | round 2 |
|---|---|---|
| K5 (`Qwen3.8-Flash-Next-EXL3-K5-a16`) | 13.36 / 13.61 | 13.53 / 13.76 |
| T6, section 8 recording | 13.77 / 14.02 | 13.76 / 14.04 |
| T6 + folds | 13.54 / 13.92 | 13.57 / 13.82 |
| **T6 + folds + splits** | **13.24 / 13.47** | **13.30 / 13.56** |

Greedy tok/s in the same runs: K5 74.1, T6 section 8 71.3-71.4, now 73.6-74.3.

**Quality vs the bf16 teacher, sets A and C** (`tools/kld_teacher.py --name t6fuse-T6-f32 --sets A,C --pack T6`, this
build, f32, bf16 KV; paired with the K5 engine run `kld-engine-f32-bf16kv.json`; rerun control 0):

| set | K5 | **T6** | paired diff [95% CI] | top-1 K5 / T6 | NLL K5 / T6 (teacher) |
|---|---|---|---|---|---|
| A (16 windows) | 0.03609 | **0.03324** | -0.00285 [-0.00435, -0.00141] | 0.9278 / 0.9326 | 1.8335 / 1.8282 (1.8267) |
| B (section 8) | 0.00995 | 0.00906 | -0.00090 [-0.00202, +0.00005] | 0.9739 / 0.9733 | 1.0611 / 1.0590 (1.0505) |
| C (4 windows) | 0.01625 | **0.01476** | -0.00149 [-0.00204, -0.00094] | 0.9680 / 0.9693 | 0.7081 / 0.7078 (0.7067) |

C by position (0 / 2K / 8K / 16K+): K5 0.0232 0.0177 0.0148 0.0156, T6 0.0217 0.0171 0.0133 0.0137 (no growth with
context). T6 is 8-9% closer to the teacher on every set (A and C significant). B was scored on section 8's build.

**Checks.** `make test` ALL PASS; `tests/trunk6/layer_t6.py` PASS after the split change (shared down at S 5 with
the SwiGLU prep: 2.04e-4). Greedy `tools/generate.py` (64 tokens, "17 x 23", palindrome function): identical text
for section 8's recording, the folds, and the folds with MTP d2 (verify rows 2-3 take the same SIMD folds).

**Left.** T6 1-row decode is now ~0.1-0.2 ms under K5, not 13.0: every remaining split / prefetch variant is within
noise. The NAX path (R >= 4: MTP verify at depth >= 3, prefill is separate) still runs the separate HC prep /
SwiGLU kernels (`T6_PRE` is SIMD-only; EPI 8 is on both). Next levers are not T6-specific: the router stage
(router ‖ shared gate / up, ~19 us vs K5's fused 14) and docs/kernels.md 5c's lazy GDN state store.

## 9. Swift-1.5 Flash-Next on the engine (lab-swift, 16 Oct)

Swift-1.5 (ukisai/Swift1.5-Qwen3.8-Flash-Next @ 0bd4fe2) changed exactly 300 trunk Linears of the base: GDN
in_proj_qkv / in_proj_z / out_proj (36 layers), attention q / k / v / o (12 layers), shared-expert gate / up / down
(48 layers). Routed experts, routers, GDN a / b, attention indexer, HC mixes, PLE, final mixer, embed, lm_head, MTP
and the n-gram table are the base's. The 300 bf16 tensors are on disk: `<packs>/swift15-bf16-delta` (5.4 GB,
fetched by the fine-tune's packaging script).

**Packs.**
- `Swift-1.5-Flash-Next-EXL3-K5-a16` (Swift K5, aligned): `engine-cli align-pack SRC DST --align 16` after seeding DST
  with hard links to `Qwen3.8-Flash-Next-EXL3-K5-a16`'s rewritten shards wherever Swift-K5's file *is* the base K5
  file (same inode: the 129 expert / small shards, 53.9 GB; headers + 64 KB of every tensor's start and end checked).
  align-pack keeps an aligned regular file (its resume rule), so it rewrote only Swift's two trunk shards + the MTP
  file: 4.8 GB in 4 s instead of 58.7 GB, and base and Swift share the page cache for the experts.
- `Swift-1.5-Flash-Next-EXL3-T6`: `tools/convert_trunk.py --src <Swift K5-a16> --out <Swift T6> --work
  <packs>/swift-t6-work --delta <packs>/swift15-bf16-delta`. `--delta DIR`: a Linear's weight is
  `DIR/<pack prefix>.bf16` when present (shape checked against the base, relative change vs the base logged, must
  be > 0; per layer the delta files must all be used), else the base bf16 as before. Recipe, mixer (base bf16 =
  Swift's), skeleton and slim unchanged.

**Hessians: captured through the Swift model** (its 8-bit pack), not reused from the base. Reuse would not have
been cheaper: the base night run kept only L0-3's H (the converter deletes H after each layer), so either way
42+ layers need a capture pass (~8 min of the ~55). Measured on L0-3 (`--keep-h`, vs the base's kept H): H of every
Linear downstream of the first Swift module differs by 0.4-8.5% (rel Frobenius; L0 out_proj 0.4%, L3 q_proj 7.6%),
while the Linears upstream of any Swift change are **bit-identical**: L0 HC down / up, GDN a / b and L1 PLE key /
value have H rel diff 0 and their trellis equals the base T6 pack's bit for bit (end-to-end check of the pipeline
on the Swift source). Proxy errors are within 1-2% of the base encode's on every Linear (K6 rel L2 ~2%, K8 0.5-1%).
Swift's weight change per Linear is small: rel L2 vs base 0.3-0.5% (GDN / attention), 0.7-2% (shared expert), i.e.
below the K6 quantization error (~2%) per matrix; the change is structured, the quantization error is not.

**Full encode** (16 Oct 13:09-13:58, one with_lock job, NEED_GB=40; L0-3 from the proof run 12:52-13:00): 48 layers,
578 Linears (300 from the delta, 278 base), 57 s per layer, captures 73-76 s per 8-layer group, decode checks 0.
Proxy error vs the base T6 encode, Linear by Linear: geomean 0.998 (0.94-1.15). Delta change vs base: median 0.62%
(0.21-2.4%). Pack `<packs>/Swift-1.5-Flash-Next-EXL3-T6`: 3.6 GB of own files (T6 shards + mixer + the slimmed
rest shards), everything else symlinks to Swift K5-a16's real files; index metadata `trunk6.layers` 0-47,
`mixer_bf16`, `delta`. Per-Linear rows: `tests/trunk6/results/swift-t6-encode.jsonl`. Transients deleted.

**Engine A/B** (`tools/pack_ab.py`, packs one after the other in one job; f32, QSA; teacher-forced on the 289
golden WHOLE positions (chat / code / prose / facts / long-2.6K, tokens = the MLX base K5's greedy continuation),
greedy 128 tokens on 4 chat prompts; JSON `tests/trunk6/results/swift-ab-{k5,t6}.json`):

| pair | KL mean | KL median | p99 | top-1 | greedy common prefix (mul / palin / explain / story) |
|---|---|---|---|---|---|
| base K5 vs Swift K5 | 1.07e-2 | 6.5e-4 | 0.111 | 0.976 | 3/4, 128/128, 7/63, 3/128 |
| **Swift K5 vs Swift T6** | **3.5e-3** | **1.3e-4** | 0.037 | **0.983** | 21/21, 128/128, 7/63, 21/128 |
| base T6 vs Swift T6 | 9.2e-3 | 7.1e-4 | 0.121 | 0.983 | 21/21, 128/128, 7/63, 1/128 |
| (section 8: base K5 vs base T6) | 3.0e-3 | 1.7e-4 | | 0.990 | |

- Swift moves the logits ~3x more than the K5 -> T6 trunk change does (base vs Swift 9-11e-3 on both formats; T6 vs
  K5 3.0-3.5e-3 on both models): the fine-tune survives both quantizations and the T6 step costs Swift what it
  costs the base. Text is coherent on every pack (17 x 23 = 391; the same palindrome function; Rayleigh
  scattering; the lighthouse story) and Swift's answers are recognisably its own (mul: Swift adds "The product of
  17 and 23 is 391." on both packs, the base K5 answers "391").
- Decode, one run each: Swift K5 72.6-73.7 tok/s greedy, GPU 13.15 ms/step at T 64; Swift T6 68.3-74.2 tok/s, 14.69
  ms (base T6 in the same job 13.61, 69.6-75.1 tok/s): within section 6's between-run drift (+-1.5 ms); the Swift
  T6 pack has the base T6's shapes and dispatches (842 per step), so no speed difference is expected. In the
  server e2e job (below) Swift T6 ran 13.15 ms/step, 73.4 tok/s direct, 94.5 tok/s with MTP d2.
- **No Swift teacher exists on disk**, so "T6 is closer to Swift's bf16 than K5" (true for the base, section 8b:
  -8 to -9% teacher KLD) is not measured for Swift. A Swift teacher needs the Swift bf16 model: the HF repo is
  360 GB (131 shards). It does not need a 360 GB download: Swift's experts and every other non-delta tensor are
  byte-identical to the base, and the 300 changed Linears are already in swift15-bf16-delta, so the base teacher
  pipeline (the lab tree's exl3/pack/teacher.py, layer-sequential, ~15 GB RAM) with the delta patched over each
  layer's trunk is the Swift teacher: ~335 GB of base bf16 range reads again (the base's per-layer fetch was deleted
  after its run; fetch-bound ~85-90 s per layer = ~70 min), +~10 s compute per layer, ~10 GB of teacher states on
  disk, then `tools/kld_teacher.py` per pack (~50 min for sets A-C per pack, whole model).

**Serving**: `--model-id Swift-1.5-Flash-Next-native` (docs/server.md 1b). E2E (`E2E_MODEL=Swift-1.5-Flash-Next-native
tests/server/e2e_job.sh`, 16 Oct 15:38, ctx 32768): 23 / 23 client checks incl. the reported id; HTTP 68.9 tok/s step,
83.7 tok/s MTP d2 (512 tokens); prefix reuse 409 / 452 tokens on the tool turn.

**Traps.** A Swift K5 pack cannot be aligned by align-pack alone without duplicating ~54 GB of base experts: seed the
destination with hard links first (above). `convert_trunk.py --keep-h` on a proof run leaves those layers' H on
disk for later comparison; delete the work directory's `hessians/` and `hstate-*.npy` afterwards (9.3 GB).
