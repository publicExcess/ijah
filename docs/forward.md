# Whole-model forward (lab-forward)

Owner: lab-forward until 15 Oct afternoon, then lab-loop; since 16 Oct **lab-integrate** (`models/flash_next/`: forward,
layer, mtp, mtp_loop, the prefill integration; `tools/generate.py`, `tools/bench_*`, `tests/tests.mk`, this file,
docs/mtp.md); the layer kernels in `kernels/` are lab-speed's, the chunked prefill's kernels lab-prefill's.
**16 Oct: prefill() / generate() / the MTP loop's start run through lab-prefill's chunked prefill** (TTFT 2K 14.1 ->
1.03 s, 8K 5.1 s, 32K 23.9 s; MTP start 8K 52 -> 7.0 s; section 3c), **HC read v2** is wired (decode step 13.8 -> 13.4-13.6
ms GPU), and section 2c settles the "floor" discrepancy. **MTP in the token loop: docs/mtp.md section 10** (101 tok/s sampled, 92 greedy at depth 2 on
the thinking prompts vs 69 here without MTP). Status 15 Oct 2026: **the whole Flash-Next model generates on the native engine**
(aligned pack `<packs>/Qwen3.8-Flash-Next-EXL3-K5-a16`, no RAM copies): 14.1 ms GPU per decode step (14.5 at
8K), 69-71 tok/s end to end (Sushi on the same pack: 17.4 ms forward, 15.9 ms GPU), prefill 160 tok/s (4-row
chunks), whole-model logits at the MLX reference's own noise level (KLD 2.8e-3 vs its prefill-vs-decode 2.4e-3;
greedy continuations token-identical), checked to 8K context. Numbers: one machine (M5 Max 128 GB), machine lock held.

```
$PY tools/generate.py "What is 17 multiplied by 23? Answer briefly." [--greedy | --temp 1 --top-p 0.95 --top-k 20]
make test-forward      # n-gram ids, PLE layer, head, sampler + every layer golden (GPU, lock, ~1 min)
make ref-logits        # MLX whole-model reference logits -> tests/golden/data/WHOLE (~76 GB, lock; never with the engine)
make test-logits       # engine teacher-forced on them: KLD / top-1 (81 GB wired, lock)
$PY tools/bench_forward.py   # GPU ms per step at T = 64 / 2K / 8K, tok/s (lock)
```
(`$PY` = `.venv/bin/python -I -B`; wrap GPU jobs in
`LOCK_OWNER=lab-forward tools/with_lock.sh "JOB" HH:MM -- ...`.)

## 1. Pipeline (one decode step = 2 command buffers, 548 dispatches)

| # | what | kernels | buffers |
|---|---|---|---|
| A | embedding: 8-bit g64 row of `tok[0]`, bf16 values, tiled to the 4 HC streams | `fn_embed_g64` | `tok` u32 (GPU) -> `h[0]` f32 [4][2560] |
| A | layer 0 (GDN) | 10 (layer.mm `record_gdn_layer`) | `h[0]` -> `h[1]` |
| B | layer 1 PLE: key/value projections of the 16 gathered n-gram rows, then per stream the norms, signed-sqrt gate, `sigmoid(gate) * values`, norm, dilated (3) K=4 conv + silu, `h + gv + conv` | `fn_ple_proj_g64`, `fn_ple_mix` (4 TGs x 1024) | `s.ple_emb` f32 [2560] (host-written), `st.ple_conv` f32 [9][10240] -> `s.hp` |
| B | layers 1..47 (36 GDN x 10 dispatches, 12 attention x 15: proj + lab-attn's QSA dispatches + o_proj ..) | layer.mm | `h[l % 2]` -> `h[(l + 1) % 2]` |
| B | final mixer = HC read without inject (`fn_hc_pre` with a zero inject matrix, `fn_hc_mix`) | 2 | `h[0]` -> `x` [2560] |
| B | lm_head 8-bit g128, 248320 rows, bf16 logits | `fn_lm_head_g128_r1` (31040 TGs) | `x` -> `logits` f32 [248320] |
| B | sampler pass 1: logit_bias, penalties, token mask (all optional, docs/server.md sections 9-10), then per 1024-logit block top-K (value desc, lowest id on ties) + (max, sum exp) | `fn_sample_p1` (243 TGs) | `cand`, `lse` (+ `bias`, `cnt`, `mask`) |
| B | sampler pass 2: global top-K, logsumexp, temperature, top-k, min_p, top-p, draw (PCG hash of seed, step), counts the sampled token, candidate list for logprobs | `fn_sample_p2` (1 TG) | `res` [token, logprob, argmax, K], **`tok`** (+ `top`, `cnt`) |

- **Positions**: one `pos` u32 buffer shared by the 12 attention layers (the layer states and lab-attn's
  `QsaCache.pos` point at it); the host writes `pos[0] = T` before committing B (A has no attention).
- **Head skipped for 1-row prefill tokens** by indirect dispatch: the 5 head dispatches read their threadgroup
  counts from `ind` u32 [5][4]; the host writes the counts (decode) or zeros (prefill).
- **Residency**: every text-model pack tensor is in one `MTLResidencySet` added to the MTL4 queue
  (`requestResidency` at load: ~10 s cold from the SSD, 0.4 s from the page cache); each recorder's own set then
  only adds references. Without it, switching between recorders over the 81 GB costs 300-500 ms each time (measured;
  `NE_FN_QUEUE_RESIDENCY=0`), see Traps.
- **PLE n-gram rows (CPU)**: the 16 ids from (p2, p1, t) with mlx-vlm's math (`ngram_ids`: shift with EOS
  boundaries = `s1 = p1, s2 = p1 == EOS ? EOS : p2`, int64 wrapping products, XOR, floor mod, + offsets; exact vs
  `ple.ngram_ids` on all 53 golden rows), 16 rows x 320 B read with `pread` in parallel (`dispatch_apply`, fd with
  `F_RDAHEAD` 0) from `ngram_table.bin`, which is never bound to the GPU. Warm: 70-80 us per token, cold rows
  ~0.3 ms; the first version (page faults on the mmap) took 20 ms per token.
- **Token loop** (`ne_fn_model_generate`): the sampler writes the next token into `tok`, so A(t+1) is committed
  right behind B(t) before the host has seen the token; the host waits for B(t), reads `res`, gathers the n-gram
  rows of the new token while A(t+1) (embedding + layer 0, ~0.3 ms) runs, commits B(t+1). Wall per token = GPU
  + ~0.1 ms. After a generate() that stopped with A ahead, the model is marked dirty (layer 0 is one token ahead):
  reset() before reuse.
- **Prefill** (`prefill()`, used by generate, `ne_fn_model_prefill`, the MTP loop): from 16 prompt tokens (`kChunkedMin`)
  lab-prefill's chunked prefill (`Model::pf`, created on first use, chunks of 2048; docs/prefill.md) for the first n - 1
  tokens + the decode step with the head on the last; shorter prompts or `NE_FN_PREFILL=steps` (read per call): 4-row chunks (A4 = embed + layer 0 at R = 4, B4 = PLE + layers at
  R = 4 through the existing r4 kernels and lab-attn's QSA at R = 4; own scratch / hidden / token buffers, the same
  states, no head) while >= 4 tokens remain before the last, then 1-row steps, the last token with the head:
  **160 tok/s** at 64 / 512 / 2600 tokens (1-row steps: 76 tok/s). `NE_FN_PREFILL4=0` disables it. Real prefill
  kernels (NAX, longer chunks) are a later lead.
- **States** (all zero at reset): GDN conv f32 [3][10240] + ssm f32 [48][128][128] x 36 layers (113 MB); PLE conv
  [9][10240]; attention: lab-attn's `QsaCache` per layer (bf16 KV [2][tmax][256] x 2, indexer keys / block keys,
  scratch), tmax a multiple of 64. Activations: one `Scratch` (rows = 1) shared by all layers.
  **Lazy GDN state store** (lab-gdnlazy, docs/kernels.md 5d): the 1-row step stores each GDN layer's ssm only every
  3rd step (`NE_GDN_LAZY=K`, 0 = every step) and keeps the deferred rank-1 rows in `GdnState::lz` / `pend`; ssm alone
  is the state only after `gdn_materialise` (snapshot, chunked prefill entry, MTP start, layer API reads and tests
  call it; reset / restore / state writes call `gdn_clear_pending`).
- **Memory**: the referenced pack tensors (81.2 GB: experts 75.5, trunk, embed 0.64, lm_head 0.64) are wired by the
  recorders' residency sets at the first commit (0.5 s when in the page cache), plus ~0.2 GB of states at tmax 8K.
- **Attention**: default lab-attn's QSA step (`kernels/attn/qsa.h`, KV format `NE_KV`, default bf16; causal = dense
  until 512 complete blocks, then top-512 blocks + tail = the reference). `NE_FN_ATTN=dense` = lab-layer's
  `fn_attn_core` + `fn_attn_merge` (f32 KV, dense at any length: wrong past 2048 tokens vs the reference).
- **Rope** native (no YaRN), as the reference and Sushi for these packs.

C API (`models/flash_next/model.h`, Python `tools/engine.py: FnModel`): `ne_fn_model_create(pack, tmax)`, `reset`,
`set_sampler(temp, top_p, top_k, seed)`, `set_mask(bits | NULL)` (allowed-token bits for the sampler, lab-api: docs/server.md section 9),
`set_sampler_ex(min_p, presence, frequency, repetition, logprob_k)` / `set_seen` / `set_logit_bias` / `set_rng_step` / `top` (lab-sampler: docs/server.md section 11; the MTP loop applies bias / penalties / min_p to its verify rows (`mtp_logit_xform`) and refuses logprobs),
`step(token, head, logits*, next*, logprob*, argmax*, gpu_ms*)`,
`generate(prompt, n, max_new, stop, out, logprobs, stats[6])`, `time(T, reps, ...)`, `info`, debug `run_head(h)`,
`ne_fn_ngram(pack, p2, p1, t, ids, emb)`; state `ne_fn_model_snapshot` / `ne_fn_model_restore` / `ne_fn_snap_T` /
`ne_fn_snap_bytes` / `ne_fn_snap_free` (lab-server's prefix reuse: GDN conv + ssm, PLE conv, T, n-gram history, 118 MB;
KV not copied, restore truncates to pos = T; `tests/test_fn_api.py`: 16 greedy steps after a restore bit-identical).
`struct ne_fn_model` (the handle) is defined once in model.h (prefill.mm / mtp_loop.mm used to re-declare it).

## 2. Correctness

**(a) Layer goldens** (`make test-forward`; chain = one engine step from the golden `hc_in` + states, tolerances
x2 (block outputs) / x3 (compounding intermediates), cosine slack x scale^2; rows whose top-10 differs from the
golden only by an exact bf16 tie are left out of the MoE-output comparisons and reported):

| case | hc_out rel L2 (rows) | notes |
|---|---|---|
| L00_decode1 / verify4 | 5.7e-3 (1/1) / 6.8e-3 (4/4) | + stage checks; mode 3 (kernels vs f32 op refs, rounding-off layer): GDN state 4e-8 |
| L01_decode1 (PLE) | 1.7e-3 | `hc_after_ple` 9.2e-4, PLE values exact, PLE conv state 1.5e-3, n-gram ids/rows exact |
| L01_verify4 | 3.2e-3 (3/4: 1 tie row) | |
| L03_decode1 / verify4 | 1.3e-3 / 1.8e-3 (3/4) | QSA: k cache 5e-4, gated core 4.8e-3 |
| L03_long_decode1 / verify4 (2304, sparse) | -/1.7e-3 (1/4; others tie rows) | QSA: top-10 0 rows differ (dense: 1 row) |
| HEAD_decode1 | logits 4.6e-3, argmax equal | mixer mix 4.4e-3, block_in 3.9e-3; sampler greedy = argmax, logprob to 1e-4 |

**(b) Whole model vs the MLX reference** (`tools/ref_logits.py`: fn_exl3 = mlx-vlm 0.7.7 qwen4_exp + EXL3 experts,
native rope, same pack; the prompt fed one token per call = the decode path, greedy continuation to 64
positions; the engine teacher-forced on the same tokens, `tools/compare_logits.py`). KL(ref || engine) per
position from f64 log-softmax of the bf16 logits; "floor" = the reference's own prefill (one call) vs its decode
path on the same tokens:

| prompts (4 x 64 positions) | KLD mean | KLD max | top-1 | top-5 overlap |
|---|---|---|---|---|
| **engine, QSA, plain f32 (generation default)** | **2.8e-3** | 0.18 | **99.6%** | 0.95 |
| engine, QSA attention, rounding points ON (engine default, tests) | 3.8e-3 | 0.19 | 99.2% | 0.94 |
| engine, dense attention, rounding ON | 4.6e-3 | 0.19 | 98.4% | 0.95 |
| engine, dense attention, plain f32 (`NE_FN_ROUND=f32`) | 3.3e-3 | 0.14 | 99.2% | 0.95 |
| reference prefill vs reference decode (floor) | 2.4e-3 | 0.05 | 99.6% | |

Per prompt (QSA + f32): chat 4.6e-4, code 1.2e-3, prose 7.3e-4, facts 9.0e-3 (floor 6.9e-4 / 9.8e-4 / 1.1e-3 /
6.9e-3; medians 5e-6 .. 9e-4): the outliers are high-entropy positions ("The capital of X is ..."), where the reference's two paths
disagree as well. On the generated parts top-1 is 100% for all 4 prompts (the 2 disagreements of 256 are at
prompt positions), so the engine's greedy continuation is token-identical to the reference's in all 4 windows
(prose: " make their own food.\n\nA. True\nB. False\n\n<think>\nWe are given a statement: ...").

Past 2048 tokens: a 2600-token prompt (mlx-vlm source; the reference prefills it in chunks of 2048, its serving
path, QSA-sparse with 650 blocks), then 32 greedy steps; 33 positions: QSA engine KLD median 1.4e-4, mean 7.2e-3
(one position 0.14), top-1 100%; dense engine median 1.4e-4, mean 3.6e-3, top-1 97%. Both within noise of each
other there; QSA is the reference's attention and the faster one at long context. After the engine's 4-row prefill
of the same prompt: KLD mean 3.9e-3, top-1 97%. **8K**: an 8192-token prompt (the source repeated: a copy-heavy
context), reference chunked prefill, 32 greedy steps: engine (4-row prefill, QSA) KLD mean 7.9e-3, median ~2e-4,
top-1 100% (1-row prefill: 1.05e-2, top-1 100%).

Rounding: the reference-rounding points (ON only when asked for since 17 Oct: `NE_FN_ROUND=ref` or
`ne_fn_set_ref_rounding(1)`, fixed per model at load; the library default is plain f32) make the per-layer goldens tight but, at whole-model
level, the plain f32 path is closer to the reference (QSA: 2.8e-3 vs 3.8e-3; dense: 3.3e-3 vs 4.6e-3): the emulated
bf16 points add their own rounding noise wherever MLX fuses differently. `tools/generate.py` and
`tools/bench_forward.py` therefore default to `NE_FN_ROUND=f32` (`--ref-round` keeps the emulation); the golden tests
request rounding ON explicitly. The bf16-teacher KLD question is open.

### 2c. The decode path's "floor": 1-row vs 4-row (lab-prefill's 3-5e-2) vs verify rows (lab-loop's 5.5e-5..3e-4)

Both numbers are right; they measure different positions. `tests/forward_floor.py` (`make forward-floor`, 16 Oct):
reference A = N + K 1-row decode steps; then the same tokens through A again, lab-forward's 4-row prefill (B4), the
chunked prefill (BC), each followed by the same K = 48 1-row steps; KL(A || path) over the K + 1 positions:

| stream (positions) | ref entropy | A again | B4 f32 / ON | BC f32 / ON | B4 vs BC f32 | A: rounding ON vs f32 | logit diff rms (B4 f32) |
|---|---|---|---|---|---|---|---|
| real64: repo text, N = 64 (= pf-check model) | 2.1 | 0 | 4.1e-2 / 8.9e-2 | 7.1e-2 / 9.1e-2 | 6.8e-2 | 9.1e-2 | 0.28 |
| real2k: repo text, N = 2048 | 1.5 | 0 | 1.7e-2 / 4.6e-2 | 3.1e-2 / 3.0e-2 | 3.4e-2 | 3.1e-2 | 0.21 |
| greedy: chat prompt + 32 own tokens, next 48 own tokens (lab-loop's kind) | 0.3 | 0 | 4.5e-4 / 1.5e-3 | 1.3e-3 / 1.5e-3 | 9.0e-4 | 1.2e-3 | 0.03 |

(first run, K = 32: greedy B4 f32 2.2e-4, ON 1.3e-4.) So: (1) every path is deterministic (A vs A = 0 exactly); (2) the
spread is the same for any two valid arithmetics -- 4-row vs 1-row, chunked vs either, and even the *same* 1-row kernels
with the reference-rounding emulation ON vs plain f32 (9.1e-2 on real64) -- so it is not a 4-row (or chunked) path bug;
(3) it depends on the positions: on the model's own greedy continuation (entropy 0.3, the logits' perturbation rms 0.03)
it is 1e-4..1.5e-3, lab-loop's range; on teacher-forced repository text (entropy 1.5-2.1, markdown tables / code the model
did not write) the perturbation itself is ~7x larger (rms 0.2-0.4) and KLD 2e-2..9e-2, lab-prefill's range; KLD at the
low-entropy positions of the real-text streams stays 4e-4..3e-3. (4) Reference rounding ON roughly doubles it on real
text (bf16 rounding points turn tiny f32 differences into whole-ulp steps). These 0.02-0.09 mutual KLDs are of the order
of the engine's KLD to the bf16 teacher (docs/quality.md: 0.010-0.036) yet the two rounding modes' teacher KLDs differ by
only 0.0035: the perturbations are noise orthogonal to the quantization error, not a quality loss. A meaningful
regression check of a new path therefore compares at matched positions with the same reference arithmetic (and
judges the real-text numbers against these floors, not against lab-loop's).

## 3. Speed (M5 Max, mtl4 backend, lock held)

**16 Oct, HC read v2** (`fn_hc_pre3` + `fn_hc_mix2`, lab-speed; wired in `record_stage`, the head mixer, the MTP head
and verify mixers; `NE_FN_HC=v1` = the old pair; not at R = 8 where `nax::record_layer` still splits `fn_hc_pre`):
decode step GPU (`bench_forward.py`, f32, median of 20) T = 64 / 2K / 8K: v2 13.40-13.60 / 13.65-13.72 / 13.81-13.88 ms vs
v1 13.79-13.83 / 14.02-14.03 / 14.18 ms, -0.2..-0.4 ms (two runs each, same build). Layer goldens (`make test-forward`)
pass; hc_out at 4 rows 6.80e-3 vs 6.79e-3 (v1). MTP verify-forced (R = 2..5, rollback patterns): KLD 5.6e-5..2.5e-4,
every flip at a tie (gap 0.000). Replays of the same build vary by up to +3 ms for minutes at a time on 16 Oct (verify
R = 2 at T = 2048: 18.1 vs 21.3 ms in one job): compare runs from one job.


GPU time of one full decode step (command buffers A + B, median of 20 replays; positions faked to T):

| T | QSA (default), f32 | QSA, rounding ON | dense (f32 KV), rounding ON |
|---|---|---|---|
| 64 | **14.07 ms** | 13.98 ms | 14.15 ms |
| 2048 | **14.32 ms** | 14.21 ms | 14.68 ms |
| 8192 | **14.46 ms** | 14.36 ms | 16.72 ms |

(+-0.1 ms between runs; the QSA columns were measured before / after lab-attn's later select changes.)

- **End to end** (single stream, no MTP, pipelined loop, short context): greedy **69.3-70.8 tok/s** (14.1-14.4
  ms/token wall), temp 1 / top-p 0.95 / top-k 20: 68.8-69.0 tok/s (14.5 ms). Sushi on the same pack: 17.4 ms
  forward (15.9 GPU) -> 57.5 tok/s: the engine is ~1.2x faster per token (GPU 0.89x of Sushi's).
- **Dispatches**: 548 per step with QSA (A 11, B 537), 500 dense. Body without the head 12.9 ms; the head (mixer
  + 636 MB lm_head + sampler) ~0.95 ms.
- Expected from the layer numbers: 36 x 271 + 12 x 259 us = 12.9 ms + head 1 ms = 13.9 ms; measured 14.0: the 48
  layers in one recording run at the per-layer speed (no extra cost from chaining).
- Prefill: 167 tok/s with 4-row chunks (one R = 4 step ~24 ms for 4 tokens; GDN in_proj / attention projections /
  out_proj on lab-mtp's NAX GEMMs at R >= 4, `NE_FN_NAX=0` = the SIMD kernels: 145-159 tok/s), 76 tok/s with 1-row
  steps.
- Tried on `fn_moe_down` (the slowest GEMV per byte, ~355 GB/s): a 4-deep register prefetch ring (+14 us per
  layer, worse: register pressure) and KS = 4 k-splits / 1024 threads (no gain); reverted.
- Host: gather 70-80 us warm (overlapped with A), encode ~0.13 us x 525 dispatches, sync ~0.1 ms not hidden.

## 3c. Time to first token (16 Oct, `tools/bench_ttft.py`, repository text, f32, second run; first run in brackets)

| prompt | `generate()` TTFT (chunked prefill + head step) | before (4-row + 1-row path, `NE_FN_PREFILL=steps`) | MTP loop start (d2; head prefill part) |
|---|---|---|---|
| 2048 | **1.03 s** (1.39: recordings + pipeline compiles), 1985 tok/s | 14.13 s (145 tok/s), same first token | 1.49 s (0.31) |
| 8192 | **5.10 s** (4.82), 1605 tok/s | ~52 s (lab-loop's MTP start) | **6.95 s** (1.28) |
| 32768 | **23.9 s** (23.7), 1370 tok/s | ~3.6 min (est., 150 tok/s) | |

A 60-token chat prompt: 187 ms vs 371 ms (steps), greedy identical over 24 tokens. The MTP loop feeds the head from
the chunked prefill's layer-47 rows (`prefill_hidden`) chunk by chunk (8-row head steps, 0.16 ms per position; a
chunk-sized head prefill through the prefill machinery is the next lead: -1.2 s at 8K). Recordings per chunk length
are LRU-capped at 32 (`NE_PF_MAX_RECS`; lab-server's request).

## 3b. MTP acceptance on the engine (run for lab-mtp)

`build/mtp-accept-eval PACK tests/golden/data 256 6 [4]` (lab-mtp's tool on this model: target = `ne_fn_model`,
greedy from the 52-token golden prompt, 256 new tokens; drafts = lab-mtp's head from layer 47's pre-mixer hidden;
teacher-forced over 252 start positions), 15 Oct:

| depth | P(accepted >= d) | conditional a_d | tokens / round | projected tok/s, full-head draft (1.58 ms) | 4-bit draft readout (1.11 ms) |
|---|---|---|---|---|---|
| 1 | 0.782 | 0.782 | 1.78 | 92.2 (1.31x) | 94.0 (1.34x) |
| 2 | 0.552 | 0.706 | 2.33 | **94.9 (1.35x)** | **98.2 (1.40x)** |
| 3 | 0.401 | 0.727 | 2.73 | 91.6 | 95.6 |
| 4 | 0.266 | 0.663 | 3.00 | 84.7 | 89.0 |
| 6 | 0.115 | 0.690 | 3.28 | 73.2 | 77.8 |

(serial 70.1-70.5 tok/s at 14.2 ms GPU; the projection uses lab-mtp's measured verify-cost ratios, docs/mtp.md.)
One prompt (the golden English prompt + its greedy continuation), so the per-depth numbers are indicative.

## 4. Traps

- **Two recorders over the same 81 GB = a 300-500 ms stall at every switch.** Each Recorder builds its own
  MTLResidencySet; alternating `rec_full` / `rec_body` (head / no head) made every switch cost 300-500 ms (prefill
  of 26 tokens: 1.4 s); the first use of a big set ~12 s cold. Fixed by a queue-level residency set with the whole
  text model (`addResidencySet` on the MTL4 queue + `requestResidency`): switches cost nothing measurable then
  (4-row chunk + full step: 38 ms = 24 + 14). The head-indirect trick (one B recording for both) stays.
- **mmap faults on the n-gram table: 20 ms per token** (16 random rows, each fault reads a cluster); `pread` of
  exactly 320 B per row in parallel with read-ahead off: 70-80 us.
- **qkv group != z/a/b group** on most GDN layers (qkv g64, z/a/b g128; all g128 on 7 layers incl. 0):
  `fn_gdn_inproj` takes both (`fn_gdn_inproj_q64_g128_r*`). The goldens (layers 0, 1, 3) never had the g64 qkv +
  g128 z case except layer 1.
- **RAM: whole-model runs page other processes out.** The engine wires 81 GB, the MLX reference holds 76 GB. With
  an idle Sushi server (pid 60371, port 8137, 78 GB) resident on 15 Oct, the first engine run pushed ~45 GB of it
  into swap (swap 11 -> 56 GB, later 75 GB) before `with_lock.sh`'s swapout watchdog (new: `MAX_SWAPOUT_MB`)
  stopped the job; SIGTERM took ~25 s to land (the process was inside the residency commit). Check `top -o mem`
  before a whole-model job.
- **The reference's rounding emulation is not free at whole-model level** (section 2).
- The mixer has no `block_inject_weight`: `fn_hc_pre` is reused with a zero bf16 [4][10240] matrix (its inject
  outputs are ignored; the 1/rms entries it writes are what `fn_hc_mix` needs).
- Sampler order is temperature -> top-k -> min_p -> top-p (HF's warper order, min_p before top-p as vLLM); top-p is applied to the
  top-k / min_p distribution (smallest prefix with mass >= p). K <= 64. The RNG counter counts sampled tokens (head steps),
  not fed tokens.
- `ne_fn_model_time` fakes positions (KV rows past what was written are zeros) and leaves the model reset.

## 5. What is left

- **Prefill**: done (lab-prefill's chunked prefill behind `prefill()`, section 3c). Next: the MTP head's prefill as a
  chunk (1.28 s of the 8K MTP start), lab-attn's multi-row sparse core (long prompts), MTP continuing from the current
  state (lab-server's request (2)).
- **Speed**: the per-layer list in docs/kernels.md (moe_down at 355 GB/s, hc_mix fusion, gdn_step TGs, K6/K8 trunk
  -0.98 GB/token); the head's lm_head is 1 ms of 14.
- **MTP**: done (lab-loop, docs/mtp.md section 10: `ne::fn::mtp_generate` on this model; the forward gained
  `gather_ple_rows` (R rows' n-gram gather in one parallel batch) and `add_queue_residency` (the head's weights on the
  MTL4 queue's residency, so draft / verify recorder switches cost nothing)).
- f32 vs reference-rounding default for generation (section 2); teacher (bf16) KLD.
- Longer whole-model references (16K+); a non-repetitive 8K text.
