# MTP: NAX multi-row kernels, the draft head, verify / accept, the token loop

Owners: sections 0-9 lab-mtp (15 Oct, finished); 10.9 lab-mtpcont (16 Oct); since then **lab-loop** owns `models/flash_next/mtp*`, `tests/mtp/`,
`tools/bench_mtp*` and this file, **lab-speed** owns `kernels/nax/`. Section 10 = the MTP token loop (lab-loop).
Section 11 = drafting at real settings (lab-draft, 18 Oct): pruned draft vocabulary + copy rule, **same-client server
108.9 vs 105.6 tok/s (1.031 [1.028, 1.035], 18 / 18), whole-file edits +23%**.
**Headline (section 10): MTP generates end to end on the engine: 101 tok/s sampled, 92 greedy at depth 2 (thinking
prompts, median of 3 x 1024 tokens; no MTP 69, Sushi 90.5 on this pack); 91 / 91 at 8K; sampled output checked
against the target distribution.**
Status 15 Oct 2026.
Pack: `<packs>/Qwen3.8-Flash-Next-EXL3-K5-a16` (aligned; no RAM copies). M5 Max, mtl4 backend, machine lock
held, GPU times **flushed** (a 256 MB streaming read between steps, its own cost subtracted) unless noted;
run-to-run noise ~+-5-10% on single kernels, ~+-3% on layer steps.

## 0. Summary

| item | status |
|---|---|
| NAX 8-bit affine GEMM (1-16 rows, MLX layout read in place, per-segment g64/g128, 5 fused epilogues) | done: 2.9e-4 vs f64 on real weights (= half rounding of random f32 x), 1-16 rows |
| NAX layer step (GDN in_proj, attention projections, out_proj + HC write on NAX; the rest lab-forward's kernels) | done: matches the verify4 goldens at 2 and 4 rows exactly as the SIMD path does (L00, L03); faster from 4 rows |
| 6- and 8-row versions of lab-forward's kernels | done without touching their files (`kernels/nax/rows68_*.metal`) |
| NAX routed-expert block over the row union | **not built, on evidence**: real verify rows share 1 of 40 experts; the per-row MoE kernels already stream the union at 509-558 GB/s (section 3) |
| MTP head (fuse + QSA/MoE layer + mixer + shared lm_head + argmax, chained depth d in one recording) | done: matches `tools/mtp_goldens.py`; every greedy draft equal (draft 1-3, chain, 4-row step); attention through lab-attn's QSA step (the forward's default) |
| draft-only readout: lm_head requantised to 4 bits + top-8 shortlist re-scored exactly through the 8-bit head | done: argmax = full head on 60/60 golden rows; 732 vs 1192 us per draft step |
| verify / accept (greedy, exact sampled), commit plan, adaptive depth | done (host C++, `models/flash_next/mtp.h`), Monte-Carlo checked |
| GDN rollback v1 (per-row state snapshots + conv history) | done: state after each verify row = `gdn.ssm_state_rows[a]` (2.7-3.4e-3, the chain level); +24 us per GDN layer at 4 rows |
| GDN rollback v2 (lazy commit, zero extra bytes) | done: two-round golden test (keep 2 / keep 1) matches; step 504 vs 506 us plain: **free** |
| measured acceptance on the engine (target = lab-forward's model, drafts = the head) | measured by lab-forward (docs/forward.md 3b) and now in the token loop itself (section 10) |
| **MTP token loop** (draft -> R-row verify -> greedy / exact sampled accept -> lazy commit, end to end) | **done (lab-loop, section 10)**: 101 tok/s sampled, 92 greedy (depth 2, thinking prompts) vs 69 no-MTP |

Measured verify cost (whole forward from measured layers, section 6): **1.24x / 1.49x / 1.75x / 2.28x / 2.62x at
2 / 3 / 4 / 6 / 8 rows**. Projected decode at Sushi's observed acceptance (71-92% per draft, 2.2-2.8 tokens per
round): **~94-135 tok/s GPU vs 71 serial (1.33-1.90x)** with the 1.15 ms draft step (4-bit draft readout), ~91-127
with the full-head draft step (1.55 ms). Section 6.

## 1. Files

| file | what |
|---|---|
| `kernels/nax/nax_q8.metal` | `nax_q8_gemm`: NAX matmul2d 16x32x16 over MLX 8-bit (or 4 / 2-bit) affine weights, up to 4 segments, epilogues 0 store / 1 GDN in_proj (conv + silu, z, beta, g) / 2 out_proj + HC write / 3 attention projections (+ index key cache) / 4 row bias (MTP fuse) / 5 GDN in_proj + rollback history / 6 GDN in_proj with lazy commit |
| `kernels/nax/nax.{h,mm}` | recording interface: `record_q8`, `record_q8_rowbias`, `record_gdn_in`, `record_out_hc`, `record_attn_proj` (drop-ins for lab-forward's dispatches, same buffers), `record_layer` (a layer step at 1-8 rows), `alloc_scratch_rows` / `alloc_state_rows` (any R), `LayerExtra` (8-row hc_pre split, rollback buffers), `record_gdn_rollback`, `load_q8` |
| `kernels/nax/rows68_{trunk,moe,attn}.metal` | lab-forward's `fn_*.metal` at 6 and 8 rows: their file is included with its own instantiations renamed (`nax_dup_*`), then `NE_INST_*(6/8)` give the standard names |
| `kernels/nax/nax.mk` | build hook (included from `kernels/kernels.mk`): engine sources + `build/mtp-nax-check`, `mtp-head-check`, `mtp-accept-test` |
| `models/flash_next/mtp.{h,mm,metal}` | the draft head (`load_mtp`, `alloc_head`, `record_step`, `record_chain`, `set_draft_head` / `requantize` / `record_draft_readout`), kernels `mtp_fuse_pre`, `mtp_argmax`, `mtp_pos_add`, `mtp_topk_local` + `mtp_shortlist`, rollback v1 `mtp_gdn_step_rows_r{2..8}` / `mtp_gdn_rollback`, rollback v2 `mtp_gdn_step_lazy_r{1..8}` / `mtp_gdn_pending_copy`; verify / accept (`accept_greedy`, `accept_sampled`, `make_dist`, `plan_commit`, `DepthPolicy`) |
| `tools/mtp_goldens.py` | MTP-head goldens from the MLX reference (head only, 2.6 GB peak, 1 s) |
| `tools/with_lock.sh` | lab-mtp copy of `tools/with_lock.sh` (machine lock with the job's PID + RAM watchdog) |
| `tests/mtp/nax_check.mm` | `mtp-nax-check gemm PACK [time]`, `layer PACK CASE ROWS [time]`, `rollback PACK CASE` (v1), `lazy PACK CASE` (v2) |
| `tests/mtp/mtp_check.mm` | `mtp-head-check PACK GOLDEN_DATA [time]` |
| `tests/mtp/accept_test.mm` | `mtp-accept-test` (CPU only) |
| `tests/mtp/accept_eval.mm` | `mtp-accept-eval PACK GOLDEN_DATA N_NEW DEPTH [bits]`: greedy target generation (lab-forward's model) + the head's draft chain at every position -> measured per-depth acceptance and tok/s; `selftest` = head only on the golden stand-in |
| `tests/mtp/common.h` | npy / tensor helpers, `import_kv` / `export_kv` (lab-attn's QSA caches or the dense buffers) |

Build: `make` (or `make BUILD=build-mtp` for a private build dir). Checks (take the lock):
`tools/with_lock.sh "job" HH:MM -- build/mtp-nax-check layer $PACK tests/golden/data/L00_verify4 4`.

## 2. NAX GEMM

One simdgroup = one 32-column block x a run of 64-wide k chunks. Per chunk a lane loads 16 bytes of each of its 4
weight rows straight from the MLX layout (the 4 lanes sharing a row read 64 contiguous bytes) and one (scale, bias)
per row, dequantises in f32 (w = s q + b) to half, and feeds 4 matmul2d steps. The k order inside a chunk is
permuted (logical (step j, k0 + e) = physical 16 (k0 / 4) + 4 j + e) and x is read with the same permutation, so
both operands are plain vector loads: **no repack of the pack's trunk**. Rows >= R of the 16-row M are zero; R <= 8
uses only rows 0-7 of the destination. KS simdgroups split k and reduce through threadgroup memory in fixed order
(deterministic); 32 threads then run a column-wise epilogue over the R rows (conv1d over the rows, HC write, ...).
Lane layout of the operands: lab lead 6 (`the lab tree's engine/nax/NOTES.md`), derived there
from Sushi's `nax_wfrag`. K splits per shape: lm_head 16, 6144-input out_proj 4, else 8 (`NE_NAX_KS` overrides).

Correctness (`mtp-nax-check gemm`, f32 output vs f64 CPU on the real L0 / lm_head weights, random x): rel L2
2.3e-4 - 3.6e-4 for qkv / z / b / a (4 segments, mixed groups), out_proj and lm_head at 1, 2, 3, 4, 6, 8, 12, 16 rows.
That is the half rounding of random f32 inputs; the engine's activations are bf16-rounded, which half holds exactly.

Time per GEMM (us, flushed; SIMD = lab-forward's q8 R-row kernel with the same epilogue):

| rows | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| GDN in_proj 18.5 MB (qkv/z/b/a + conv/gates), NAX | 84-88 | 80 | 79-85 | 84 | 76-90 |
| same, SIMD | 77-87 | 77 | 90-91 | | |
| out_proj 6144 -> 2560 + HC write, NAX | 31-41 | 42 | 41-49 | 48 | 43-58 |
| same, SIMD | 29-38 | 33 | 41-44 | | |
| lm_head 248320 x 2560 g128 (674 MB), NAX KS16 | 1131 (580 GB/s) | | 1138 | | 1148 (571 GB/s) |

The NAX body is flat in rows (lm_head +1.5% from 1 to 8 rows, 1176 -> 1194 us at KS 8). On the trunk shapes the
SIMD kernels are not FMA-bound at <= 4 rows either, so the two tie there; the out_proj (80 column blocks only) is the
weak NAX shape.

## 3. Layer step at 1-8 rows (`mtp-nax-check layer ... time`, verify4 inputs, T = 48)

| rows | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| L00 GDN, NAX trunk | 270 | 365 | **494** | **655** | **754** |
| L00 GDN, SIMD trunk (lab-forward's kernels) | 270 | **342** | 530 | 715 | 917 |
| L03 attention, NAX trunk | 274 | 340 | **478** | **637** | **740** |
| L03 attention, SIMD trunk | **260** | **329** | 498 | 684 | 885 |
| L03 with lab-attn's QSA step (the forward's default since fe44a5e), NAX / SIMD trunk | 293 / **268** | 352 / **331** | **476** / 489 | **612** / 674 | **725** / 874 |

(us per layer step. 1-4 rows SIMD = lab-forward's instantiations, 6 / 8 rows = theirs via rows68; fn_hc_pre at 8
rows runs as 2 x 4 rows because its staging needs 40 KB of threadgroup memory at 8.) Lab-forward's own number for
L00 at 4 rows was 505 us; the NAX trunk takes it to 494 and wins 7% at 6 rows, 18% at 8. Policy: **NAX trunk at
R >= 4, SIMD at R <= 2** (the request below).

Correctness (`mtp-nax-check layer CASE R`, chain from the golden hc_in + states; R = 2 = the first 2 rows of the
verify4 case, the GDN state against `gdn.ssm_state_rows[1]`): NAX and SIMD agree to the 4th digit on every tensor,
e.g. L00 R=4 hc_out 6.79e-3 / 6.76e-3, ssm 3.27e-3 / 3.25e-3; L03 R=4 hc_out 5.56e-3 / 5.55e-3 (L03 R=4 mlp_out
7.5e-2 on both: lab-ref's documented bf16 router tie, 39/40 experts).

Where the 4-row time goes (lab-forward's per-stage numbers, L00, 1 -> 4 rows): routed experts gate/up 44 -> 147 us
and down 31 -> 81 us (+153 of the +246), hc_pre 14 -> 32 (x2), router 15 -> 25, gdn_out 31 -> 42, gdn_in 78 -> 89,
gdn_step 20 -> 30. The 4 verify tokens route to 39 distinct experts of 40, so the MoE reads ~3.9x the bytes; it does
so at 558 GB/s (gate/up) and 509 GB/s (down), i.e. already near the 594 GB/s read peak. **A union-dedupe NAX MoE
cannot buy more than the shared experts' bytes (~2.5% here) plus ~20 us of efficiency**; lead 6's 1.48x-at-4-rows
estimate assumed 30% sharing between consecutive tokens, which real routing does not show. The MoE union bytes are
the verify cost: about 26% of a 1-row layer per extra row (L00: (754 - 270) / 7 = 69 us per row).

## 4. The draft head

### 4.1 Math (mlx-vlm `speculative/drafters/qwen4_exp_mtp`, MIT-licensed reference; Sushi's design notes agree)
Config: `mtp_num_hidden_layers` 1, `mtp_use_dedicated_embeddings` false (the target's `embed_tokens`), the
target's `lm_head` (not tied). Weights `language_model.mtp.*` (8-bit g64 trunk, **K4 experts**, own QSA attention,
own attention / MLP hyper-connections and its own final `hyper_connection_mixer`).
```
e  = embed(token)                               8-bit g64 (bf16 out)
pe = fc_embedding(rmsnorm(e) * w_e)             2560 -> 2560, 8-bit g64
hn = rmsnorm over ALL 10240 of hidden * w_h     one global norm over the 4 streams (not per stream)
h_in[s] = pe + fc_hidden(hn[s])                 same 2560 -> 2560 weight for each stream; bf16 sum
h_out = layer(h_in)                             QSA attention + MoE decoder layer at the head's own KV position
logits = lm_head(mixer(h_out))                  mixer = HC read without inject
```
`hidden` = the target's **pre-mixer** 4-stream state after layer 47 (`capture_hidden`, [.., 10240]), never the
mixed 2560. Norm weights are ones-centred in the pack (fn_exl3's NORM_FOLD lists `pre_fc_norm_*`): multiply as stored.

Lifecycle (greedy, `AutoregressiveMTPDraftModel`): prefill runs the head over the prompt's (hidden at r, token r+1)
pairs; its last row's argmax is the seed draft d_1 and its hc_out the seed hidden. A round drafts d_2..d_D by
chaining (token = previous draft, hidden = previous head hc_out), verifies [t1, d_1..d_D], then
`accept_verified_tokens` trims the head rows appended for rejected drafts, keeps those of accepted drafts (computed
from the head's own hidden, not recomputed) and runs the head over the remaining accepted drafts + the new token
with the verify rows' target hidden: that forward's last row gives the next seed. So **D head forwards per round**,
D - 1 of them 1-row, the last one (accepted - kept + 1) rows.

Positions: mlx-vlm puts head row r at position = the head cache length (row r of the prefill at r); Sushi's notes
put "head row r = (stream at r, token r+1) at query position r+1" (a `pos_base`). Rope is relative, so only the QSA
4-token block boundaries differ; the engine follows the reference (pos = head cache length).

### 4.2 Goldens (`tools/mtp_goldens.py`, `tests/golden/data/MTP_*`)
Head only (2.6 GB peak): the instrumented step (lab-ref's `layer_fwd` / `head_fwd` + the fuse) is asserted
bit-exact against the stock drafter `_forward_hidden` on a cache rebuilt from the saved state. Synthetic hidden:
layer 3's real hc_out on the real prompt (L03_prefill / L03_verify4) stands in for layer 47's (as lab-ref's HEAD
cases). Cases: `MTP_prefill` (48 rows: tokens ids[1..48], positions 0..47), `MTP_draft1..3` (the greedy chain from
the prefill seed: drafts 220, 17, 15), `MTP_rows4` (4 rows from the post-prefill cache: the accept-style multi-row
head forward). Extra tensors: `mtp.token_ids`, `mtp.hidden_in`, `mtp.embed`, `mtp.emb_normed`, `mtp.fc_embedding`,
`mtp.hidden_normed`, `mtp.fc_hidden`, `mtp.argmax`; the rest as docs/reference.md.

### 4.3 Engine vs goldens (`mtp-head-check`)

Attention through lab-attn's QSA step (bf16 KV, the forward's default):

| case | fc_embedding | hidden norm | h_in | q_proj | attn gated | hc_out | logits | argmax |
|---|---|---|---|---|---|---|---|---|
| draft1 (1 row, pos 48) | 5.9e-4 | exact | 6.9e-4 | 2.5e-3 | 4.3e-3 | 9.1e-3 | 7.5e-3 | equal |
| draft2 (golden inputs) | 6.5e-4 | exact | 7.4e-4 | 2.3e-3 | 4.7e-2 | 4.9e-2 | 6.0e-2 | equal |
| draft3 (golden inputs) | 7.3e-4 | exact | 6.9e-4 | 2.4e-3 | 4.1e-3 | 8.7e-3 | 6.9e-3 | equal |
| rows4 (4 rows) | 6.9e-4 | exact | 7.3e-4 | 2.5e-3 | 5.2e-3 (rows 4.3-7.1e-3) | 6.7e-2 | 6.1e-2 | 4 / 4 equal |
| chain of 3 in one recording | | | | | | 8.1e-2 (step 3) | 5.5e-2 | 220 17 15 = golden |
| serial 1-row head over the 48 prefill pairs (`mtp-accept-eval selftest`) | | | | | | | | 46 / 48 = the 48-row reference |
| prefill in 8-row chunks, readout on the last chunk only | | | | | | 1.1e-1 (48 rows) | | seed 279 = golden; KV 48 |

Two findings. (1) The head's attention logits reach ~750 (the target layers' ~11-13), so q / k precision matters far
more than in the trunk: with lab-forward's dense `fn_attn_core` (rope frequencies from `pow` on the GPU) rows 1-3 of
the 4-row case were 1.9e-2..6.9e-2 off; lab-attn's QSA step (the MLX inverse-frequency table, docs/attn.md) gives
4.3e-3..7.1e-3. (2) What remains (draft2: 4.7e-2) is the reference's own conditioning: perturbing only the newest key
by bf16-level noise (3e-3) moves the reference's own attention output by 1e-3..1e-1 (numpy, `opref.attention` on the
golden tensors), and the engine's new k / v there are 3e-3 off (one bf16 ulp on some elements); `-fno-fast-math`
changes nothing. The 48-row prefill shows the same: the reference perturbed by 3e-3 on q and k moves by 1.1e-1
(per row 3e-2..2e-1), exactly the engine's 1.1e-1. (The stand-in hidden is layer 3's, not layer 47's: real target
states may condition the head better.) Consequences: judge MTP heads by argmax / acceptance, not rel L2; keep the head's KV at bf16 or
better (Sushi also defaults to dense head KV, `--mtp-head-kv-quant` opt-in) and measure acceptance before quantising.

### 4.4 Head cost and the draft readout
Head step (SLC-warm): **1.55 ms at 1 row**, 1.66 ms at 2, 1.76 ms at 4 (the accept-step forward); a depth-3 chain
4.72 ms in one recording (no host round trip). Head prefill over the prompt (`record_prefill_chunk`: 8 rows per
step, no mixer / lm_head except on the last chunk): **0.38 ms per prompt token** (48 tokens 18.2 ms) instead of
1.55 ms one row at a time. The target lm_head (674 MB, 1.13 ms) is ~73% of the 1-row step; the
layer + fuse + mixer ~0.42 ms.

Draft-only readout (Sushi's design: a requantised low-bit copy for drafting only, its top candidates re-scored
through the full head; verification always reads the full head, so only acceptance can change): `set_draft_head(w,
bits, K)` requantises lm_head at load (CPU, min/max affine g64, 70-90 ms), drafts run the coarse NAX GEMM
(`nax_q8_gemm` at 4 / 2 bits) and a 2-stage top-K (`mtp_topk_local`: 128 slices per row; `mtp_shortlist`: merge,
exact 8-bit re-score, argmax). Argmax agreement with the full 8-bit head on the 60 golden mixer outputs (55 head rows
+ 5 target rows; the full head itself = the reference on 54 / 55):

| coarse head | bytes | K = 1 | K = 4 | K = 8 | K = 16 | readout (1 row, flushed) |
|---|---|---|---|---|---|---|
| 4-bit g64 | 358 MB | 54 / 60 | 60 / 60 | **60 / 60** | 60 / 60 | **732-744 us** (K 8; the GEMM 722-739) |
| 2-bit g64 | 199 MB | 28 / 60 | 47 / 60 | 55 / 60 | 59 / 60 | 567-575 us (K 16; the GEMM 527-536) |
| full 8-bit g128 + argmax | 674 MB | | | | | 1187-1192 us |

So a draft step is **~1.15 ms** with the 4-bit readout (-0.45 ms). The low-bit GEMMs run at 490 / 380 GB/s and are
ALU-bound, not load-bound: giving every lane 16 bytes per row per chunk at any width (128-k chunks at 4 bits, 256 at
2) changed nothing; they issue the 8-bit head's matmul2d count over half / a quarter of the bytes. A 1-row SIMD 4-bit
GEMV at ~560 GB/s would be ~640 us, ~0.1 ms better (not built).

## 5. Verify / accept (`models/flash_next/mtp.h`, host C++)

### 5.1 Interface for the token loop (lab-forward)
Round: verify input [t1, d_1..d_D] (t1 = the last committed token, in no cache yet), D + 1 rows; row i's logits
predict the token after it. Then:
- greedy: `accept_greedy(drafts, D, target_argmax)` (D + 1 ids from a GPU argmax; `mtp_argmax` or lab-forward's
  sampler) -> accepted a, `next_token` = argmax of row a, `emitted` = d_1..d_a, next.
- sampled: `accept_sampled(drafts, D, target_logits, draft_logits or nullptr, V, SampleParams, Rng)`, or
  `accept_sampled_q(..., const Dist *q, ...)` with sparse proposals, e.g. `shortlist_dist(head.sl row, K, sp)` when
  the drafts were sampled from the draft readout's shortlist.
- `plan_commit(T0, D, a, head_appended)` -> what to keep (5.3).
- `DepthPolicy` (5.5).

### 5.2 Acceptance rules
Sushi (`src/mtp_acceptance.zig`, MIT): modes `exact` (default), `typical`, `tokenv3`; temperature < 0.01 always takes
the argmax check, so greedy output is identical under every mode and equals serial byte for byte. **`exact` =
speculative sampling (Leviathan et al. / Chen et al.)**: accept d_i with probability min(1, p_i(d_i) / q_i(d_i)),
stop at the first rejection and draw the correction from norm(max(p_i - q_i, 0)); all accepted -> a bonus from p_D.
With argmax-drafted rows (Sushi's `--mtp-greedy-tail`; q = a point mass) the accept probability is p_i(d_i) and the
correction is p_i without d_i. p and q are the sampler-filtered (temperature, top-k, top-p) distributions. The
engine implements `exact` (both q forms); `typical` (accept if p(d) > min(eps, delta e^-H(p)), delta 0.2: +15-25%
tok/s in Sushi's runs, not distribution-preserving, loops seen) is left out on purpose. `mtp-accept-test` checks
it: 400k rounds, first emitted token vs p_0 worst |z| 1.24 (sampled q) / 0.87 (argmax q), acceptance 0.5525 vs
1 - TV = 0.5530 and 0.0915 vs p_0(argmax q) = 0.0918; with a 4-id sparse proposal (the shortlist case, q missing part
of p's support) |z| 1.65, acceptance 0.7664 vs 0.7659.

### 5.3 What a round commits (`plan_commit`)
keep_rows = a + 1 verify rows (t1 + accepted drafts). Attention layers: KV and QSA block keys truncated to
T0 + keep_rows (lab-attn: `attn::record_rollback(r, cache, keep, rows)` with keep a u32 GPU buffer, or the host
`attn::truncate`). GDN layers: state after verify row a (5.4). The head: trim `head_appended - min(a, head_appended)`
rows, keep the rest, run `head_rows` rows next (mlx-vlm's `accept_verified_tokens`).

### 5.4 GDN rollback
- **v2 (built, the one to use): lazy commit.** `alloc_layer_extra(0, true, &keep)` gives a GDN layer pending
  buffers; `record_layer` then runs in_proj epilogue 6 (conv over [committed conv state; P pending raw rows; the R new
  rows], conv state <- committed only, new raw rows -> pending), `mtp_gdn_step_lazy_r{R}` (state <- committed; apply
  the P = keep[0] pending rows without outputs; store the state; run the R new rows from registers for the outputs,
  never storing their state) and `mtp_gdn_pending_copy` (the R new rows become the pending set). The accept only
  writes keep[0] = rows kept (the same u32 buffer lab-attn's `attn::record_rollback` reads); nothing is copied or
  undone. Check (`mtp-nax-check lazy`, L00_verify4, 2-row rounds): keep 2 -> round 2 (rows 2, 3) hc_out 6.0e-3,
  committed ssm = `gdn.ssm_state_rows[1]` 2.70e-3, conv 1.5e-3; keep 1 -> round 2 (rows 1, 2; row 1 re-verified)
  hc_out 7.2e-3, ssm = rows[0] 2.79e-3, conv 1.1e-3; round 1 leaves the state bit-identical. Cost at 4 rows with 2
  pending rows: **504 vs 506 us** for the plain step (+2 dispatches, P extra row-updates of ALU): free.
- **With the lazy decode store (lab-gdnlazy, docs/kernels.md 5d)**: the 1-row decode step may leave deferred GDN
  rows (`GdnState::lz` / `pend`, separate from v2's pq / pg / keep). `mtp_generate` and `mtp_verify_forced` run
  `gdn_materialise(m.st)` after the prefill (its last token is a 1-row step) before the first verify; the verify's
  kernels never see pending decode rows. A store-every-other-round variant of `mtp_gdn_step_lazy` is not built
  (est. ~-0.1 ms of a ~22 ms round).
- **v1 (built, measured; superseded by v2)**: `alloc_layer_extra(R)` gives a GDN layer snapshot buffers; `record_layer` then runs
  `mtp_gdn_step_rows_r{R}` (lab-forward's `fn_gdn_step` + the state after each row t < R - 1 stored) and the
  in_proj epilogue 5 (also writes [conv state in; the R raw qkv rows]). `record_gdn_rollback(r, st, ex, R, keep)`
  (same `keep` buffer as lab-attn) restores ssm <- snapshot[a], conv <- history rows a+1..a+3. Checked on L00_verify4
  for a = 0..3: ssm vs `gdn.ssm_state_rows[a]` 2.79e-3 / 2.70e-3 / 3.36e-3 / 3.27e-3 (the chain-mode level: the
  4-row chain's final state is 3.27e-3), conv 1.1e-3..1.9e-3. Cost at 4 rows: step 522 vs 502 us (+20-24 us: 3 x
  3 MB of snapshots per GDN layer) = **+0.8 ms per verify** over 36 GDN layers; the rollback dispatch 17 us per layer
  when the state is copied, 3 us when only conv moves (a = R - 1) -> +0.6 ms on a partial rejection.

### 5.5 Adaptive depth hook
`DepthPolicy(cost_ms)`: per-position acceptance EMAs (acc[i] = P(d_{i+1} accepted | d_1..d_i accepted)) and a round
cost table cost[d] (ms of a round at depth d = d head steps + the (d+1)-row verify; seeded from section 6, updated
by `observe(drafted, accepted, round_ms)`); `next_depth()` maximises E[tokens](d) / cost[d] with E = 1 +
sum_k prod_{i<k} acc[i] (Sushi's EV planner in miniature; its `--mtp-min/max-depth` bounds = `min_depth`/`max_depth`).

## 6. Verify cost model and projected tok/s

Forward at R rows from the measured layers (best of NAX / SIMD per R; 3, 5, 7 interpolated; L01 counted as GDN;
attention = lab-attn's QSA step at T = 48; head = mixer + lm_head ~1.17 ms, flat in R on NAX):
F(R) = 36 L00(R) + 12 L03(R) + 1.17 ms.

| rows | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| F(R) ms | 14.11 | 17.45 | 21.06 | 24.67 | 28.38 | 32.09 | 34.55 | 37.01 |
| verify cost ratio | 1.00 | **1.24** | 1.49 | **1.75** | 2.01 | **2.28** | 2.45 | 2.62 |

Sushi's meter: 1.24 / 1.69 / 2.27 at 2 / 4 / 6 rows (lead 6). Real routing makes the engine's ratio track Sushi's:
the MoE bytes per row are the cost (section 3). (lab-forward measures the real 1-row forward, docs/forward.md;
`mtp-accept-eval` reads it live.)

Round time(D) = F(D + 1) + D x t_draft + 0.15 ms host sync (rollback v2 adds nothing); tokens per round
E(D) = (1 - a^(D+1)) / (1 - a) for a per-draft acceptance a (i.i.d. model; real acceptance decays with depth, so the
deep-D entries are optimistic). Serial: 1 / F(1) = 70.9 tok/s (GPU time only).

| a | t_draft 1.15 ms (4-bit draft readout): best D, tok/s, x serial | D = 2 | D = 3 | t_draft 1.55 ms (full head): best D, tok/s |
|---|---|---|---|---|
| 0.60 | D 1, 85.3, 1.20x | 83.4 | 77.0 | D 1, 83.5 |
| 0.71 | D 2, 94.2, 1.33x | 94.2 | 91.0 | D 2, 91.1 |
| 0.80 | D 3, 104.4, 1.47x | 103.8 | 104.4 | D 2, 100.4 |
| 0.85 | D 3, 112.7, 1.59x | 109.4 | 112.7 | D 3, 108.1 |
| 0.90 | D 7, 126.0, 1.78x | 115.3 | 121.7 | D 7, 118.6 |
| 0.92 | D 7, 134.6, 1.90x | 117.7 | 125.4 | D 7, 126.7 |

At Sushi's logged 2.2-2.8 tokens per round (its K4 head, 71-92% acceptance) this is ~94-113 tok/s at depth 2-3.
Remaining levers: the measured acceptance itself (request 7), the low-bit draft GEMM's efficiency (490 GB/s), the
head's own layer (0.42 ms: a 1-row attention + MoE K4 step), then MoE bytes per verify row.

## 7. Requests

- **lab-forward** (`models/flash_next/layer.mm` `record_stage`): (1) switch the multi-row trunk dispatches to NAX at
  R >= 4 (SIMD stays at R <= 2), exact patch:
  ```cpp
  // record_stage, case Stage::GdnIn, before the attention / GDN dispatch:
  if (R >= 4) { if (w.linear) nax::record_gdn_in(r, R, w, st, s); else nax::record_attn_proj(r, R, w, st, s);
                r.barrier(); break; }
  // case Stage::GdnOut:
  if (R >= 4) { nax::record_out_hc(r, R, w.linear ? g.out : w.attn.o, s.o, h_in, s.inj_a, s.h2, s.br_a);
                r.barrier(); break; }
  // + #include "kernels/nax/nax.h"
  ```
  (or call `nax::record_layer(r, w, st, s, ex, h_in, h_out)` for the whole step; PLE layers are not covered by it).
  (2) Instantiate 6 and 8 rows in `fn_trunk.metal` / `fn_moe.metal` / `fn_attn.metal` (and delete
  `kernels/nax/rows68_*.metal` in the same commit, or the metallib link fails on duplicate names); fn_hc_pre at 8 rows
  needs a 2 x 4-row split or half the staging (`record_layer` does the split). (3) `alloc_scratch(rows)` for any R
  (`nax::alloc_scratch_rows` is the stop-gap) and `alloc_attn_state(tmax, rows)` (apart is sized for 4 rows).
  (4) `load_layer_prefix(pack, prefix)` (the MTP layer lives at `language_model.mtp.layers.0.`; `mtp.mm` duplicates
  the attention-layer loader, incl. `w.qsa`). (5) The router's unique-expert table (`uq`) packs the slot bytes of row r into
  `e[1]` byte r: rows >= 4 overwrite `e[2]` (row 0's score) - only the opt-in dedupe kernels read it. (6) For MTP the
  token loop needs the target's pre-mixer hidden of every verify row (`[R][10240]` h_out of layer 47) kept for the
  head's accept step, and the verify GDN layers recorded with `nax::alloc_layer_extra(0, true, &keep)` (rollback v2:
  one shared u32 keep buffer, also lab-attn's). (7) **Measure acceptance** (the full model is over lab-mtp's RAM cap):
  under your lock, `build/mtp-accept-eval $PACK tests/golden/data 256 4` and the same with ` 4` appended (4-bit draft
  readout) print the per-depth greedy acceptance on the engine and the tok/s it implies with section 6's table.
- **lab-attn**: none blocking; the GDN rollback reads the same `keep` u32 buffer as `attn::record_rollback`.
- Status 15 Oct evening (lab-loop): (1)-(4) done by lab-forward; (2)'s 6 / 8-row instantiations (+ 3 / 5) done by
  lab-speed, rows68 deleted; (5) still open (lab-speed, docs/kernels.md); (6) done = the token loop (section 10);
  (7) done (docs/forward.md 3b).

## 8. Traps

- My first golden-input KV import wrote the dense `kc` / `vc` after lab-forward switched attention layers to lab-attn's
  caches (those became 16-byte dummies): a silent heap overwrite. Import KV through `attn::import_cache` when
  `st.qc.tmax > 0` (`tests/mtp/common.h import_kv`).
- Two labs editing the same files in one working tree: never `git add` a file with someone else's uncommitted hunk;
  instantiate other labs' templates from your own file (rows68 pattern) instead of appending lines to theirs.
- Function constants are not `constexpr` in MSL (`constexpr uint X = FC ? ..` fails); use `const`.
- matmul2d destination elements for R <= 8: only e in {0..3, 8..11} hold live rows (rows c0); store only those.
- The head's attention is ill-conditioned (logits ~750): bf16-level noise moves its output by percent; judge by
  argmax / acceptance.

## 9. Attribution

- NAX operand layout and the decode-into-cooperative-tensor body: lab lead 6 (`the lab tree's 
  engine/nax/kernels.py`), whose lane layout derives from Sushi's `nax_wfrag` (sushi, github.com/beamivalice/sushi,
  MIT License, Copyright (c) 2026 Theinruj Toranavikrai, Copyright (c) 2026 David Dalcu). The 8-bit body here reads
  the MLX layout instead of lead 6's fragment-order repack.
- Acceptance modes, the greedy-tail point-mass rule, the head-KV / draft-head notes and the verify-row-cost
  observations follow Sushi's `docs/engine-mtp.md` and `src/mtp_acceptance.zig` (MIT, same copyright; permission is
  hereby granted, free of charge, to any person obtaining a copy of this software ... THE SOFTWARE IS PROVIDED "AS
  IS", WITHOUT WARRANTY OF ANY KIND; full text in the upstream LICENSE). No Sushi code is copied.
- The head math and lifecycle follow mlx-vlm 0.7.7 `speculative/drafters/qwen4_exp_mtp` and `mtp_base.py` (MIT).
- `mtp_gdn_step_rows` is lab-forward's `fn_gdn_step` (this repo) plus the snapshot stores.

## 10. The MTP token loop (lab-loop)

`models/flash_next/mtp_loop.{h,mm}` (C API `ne_fn_mtp_*`; `ne_fn_mtp_generate_cb` streams every emitted run of tokens
to a callback whose nonzero return stops the run after that round -- lab-server's stop strings / disconnects), kernels in `mtp.metal` (section "MTP token loop"). Python:
`tools/bench_mtp.py` (tok/s, acceptance, greedy identity; `--time` verify / draft GPU ms), checks
`tests/mtp/test_verify_forced.py`, `tests/mtp/test_sampled_dist.py`, `build/mtp-loop-check PACK`.
```
NEED_GB=85 LOCK_OWNER=... tools/with_lock.sh "bench_mtp (~83 GB)" HH:MM -- $PY tools/bench_mtp.py [--ctx 8k]
    [--modes greedy,sampled] [--depths 1,2,3,4,0] [--n 1024]          # 0 = adaptive depth
NE_MTP_DRAFT=argmax|sample (sampled mode; default sample)
```

### 10.1 One round (depth D, R = D + 1 verify rows [t1, d_1 .. d_D]; t1 = the last emitted token)

| step | where | what |
|---|---|---|
| draft | 1 command buffer | head seed step over n = 1 (or 2 when every draft of the last round was accepted) rows: the target's pre-mixer hidden of the kept verify rows hk..a + the tokens after them (mlx-vlm's `accept_verified_tokens`; hk = min(a, D - 1) chain rows of accepted drafts stay in the head's KV) -> d_1; D - 1 chained 1-row steps -> d_2..d_D (4-bit draft readout + exact top-8 re-score; sampled mode: `mtp_draft_sample` draws each draft from that shortlist under the target's sampler and stores the proposal q); `mtp_copy_u32` puts the drafts behind t1 in the verify token buffer |
| A_R | committed right behind the draft | embedding of the R tokens + layer 0 (lazy GDN commit) |
| host | waits for the draft only | reads the D drafts, gathers the R x 16 n-gram rows (one parallel pread batch, `gather_ple_rows`) while A_R runs, commits B_R |
| B_R | | PLE (`mtp_ple_mix_lazy_rR`: the conv state commits keep[0] pending rows first) + layers 1..47 (`nax::record_layer`: GDN layers lazy-commit, attention layers lab-attn's QSA at pos = T, MoE per (row, slot)) + mixer + NAX lm_head at R rows + `mtp_argmax` (greedy) or `mtp_topk_local` + `mtp_topk_merge` (sampled: the K = top_k <= 32 best (id, logit) per row) |
| host | waits for B_R | greedy: R ids -> `accept_greedy`; sampled: R x K pairs -> `shortlist_dist` per row = the exact filtered p (temperature -> top-k -> top-p over the top 20) -> `accept_sampled_p` with the GPU's q (or point masses for argmax drafts). Commit = keep[0] = a + 1 (every GDN layer and the PLE conv apply those pending rows at the start of the next step), pos = T + a + 1 (KV truncation: one shared pos for the 12 attention layers), n-gram history, head KV length + next seed inputs (hidden rows copied from the verify's layer-47 output, 40 KB per row) |

Nothing is undone or copied back on a rejection: the lazy commits make rollback free (KV: just pos; GDN / PLE: keep).
The host reads per round D draft ids and R ids (greedy) or R x 20 (id, logit) pairs (sampled). Measured host costs per
round: gather 0.2-0.9 ms (hidden behind A_R: the round's wall is 3-4% over its GPU time), accept + bookkeeping 5-15 us.
Every verify / draft variant is recorded and replayed once in `mtp_create` (without it, the first round at each depth
paid 8-21 ms of pipeline compiles and residency).

### 10.2 Correctness

- **Verify vs the 1-row decode, teacher-forced** (`test_verify_forced.py`: the greedy no-MTP continuation, 128 tokens
  x 3 thinking prompts; the verify at R rows keeping all rows, or the pattern 1, R, 2, R-1, R, 1, R = rollbacks of
  KV, GDN and PLE): R = 2..5, KL(1-row || verify) mean 5.5e-5 .. 3.0e-4 (max 6.1e-3), identical with and without
  rollbacks; top-1 98.4-100%, every flip at an exact tie of the 1-row logits (gap 0.000) or one bf16 step (0.125).
- **Greedy MTP vs greedy no-MTP** (bench, 3 x 1024 tokens): identical until the first such tie, then the texts fork:
  first mismatches at 38-348 tokens, every one at a no-MTP logit gap of 0.000 / 0.125 / 0.250 between the two tokens
  (bf16 logits: exact ties are common). The multi-row verify (NAX GEMMs, other reduction orders) breaks ties
  differently from the 1-row SIMD kernels; token identity past ties would need batch-invariant kernels (one
  arithmetic for 1 and R rows). Depth 1 on the short prompt: identical over all 64 tokens.
- **Sampled = exact speculative sampling**: `mtp-accept-test` (CPU Monte Carlo, section 5.2, unchanged after the
  `accept_sampled_p` refactor) and end to end `test_sampled_dist.py` (section 10.4).
- `mtp-loop-check`: `mtp_ple_mix_lazy` over two rounds (4 rows, keep 2, 3 rows) bit-identical to serial
  `fn_ple_mix_r1` (outputs and committed state); PLE projections at 4 rows vs the 1-row decode's (below); top-20
  merge exact vs a CPU sort with bf16 ties.
- `mtp-loop-check` (2) on the T6 pack (lab-loopfix, 18 Oct): it used to stop with "dispatch: argument with neither
  buffer nor bytes" (and so never ran (3) / (4) there). Not a loop bug: the check predates T6 (1a70c52, 15 Oct) and
  bound the q8 PLE tensors (`key.w / .s / .b`) to `fn_ple_proj_g64_r4`; the T6 pack holds `ple.key_proj` /
  `value_proj` as EXL3 K8 (lab-t6wire, 7116a66, 16 Oct), whose `Q8` has those tensors empty, and the loop's own T6
  branch was never checked. Now both sides are the engine's functions: `nax::record_ple_proj` (the loop's projection,
  factored out of `mtp_loop.mm`: q8 = NAX GEMM, T6 = K8 t6 Linears rounded in place) at 4 rows vs
  `fn::record_ple_proj` (factored out of `record_ple`: the decode's projection) at 1 row, row by row; same weights,
  so the 5e-3 rel L2 bound is unchanged and pack-independent. Measured: T6 2.2e-7 (K8 NAX path vs SIMD path;
  `T6_PATH=simd` gives exactly 0, so the two paths really differ), K5 2.0e-4; (3) / (4) pass on both packs.

### 10.3 Speed (M5 Max, lock held, `NE_FN_ROUND=f32`, 4-bit draft readout, max depth 4)

Prompts: the oMLX / Sushi head-to-head's three thinking prompts (`omlx_h2h/h2h.py` THINK: code_lru, text_trains,
text_crisis; chat template, thinking on, reasoning_effort medium), 1024 new tokens each, no stop; tok/s = tokens after
the prefill's first / decode wall; median of the 3 prompts (per prompt in brackets). Sampled = temperature 1, top-p
0.95, top-k 20 (the harness's), drafts sampled from the head's shortlist. "8K" = ~7.9K tokens of repository docs in
front of each question (7957-8029 prompt tokens).

| config | short greedy | short sampled | 8K greedy | 8K sampled |
|---|---|---|---|---|
| no MTP (`ne_fn_model_generate`) | 68.5 (68.6 68.2 68.5) | 69.3 (69.7 67.8 69.3) | 65.3 | 65.9 |
| MTP depth 1 | 89.0 (89.0 95.0 80.6) | 90.7 (90.7 92.8 83.5) | | |
| MTP depth 2 | **92.2** (92.2 110.8 86.0) | **101.2** (101.2 109.3 82.9) | **91.2** (91.2 103.1 77.9) | **90.8** (90.8 103.4 80.2) |
| MTP depth 3 | 91.1 (91.1 113.0 79.5) | 93.6 (93.6 120.0 81.9) | | |
| MTP depth 4 | 91.4 (91.4 111.6 67.8) | 89.1 (89.1 117.3 72.0) | | |
| MTP adaptive (`DepthPolicy`, v2) | 90.3 (90.3 108.8 78.3) | 93.9 (93.9 111.4 84.4) | 88.8 (88.8 105.0 78.3) | 90.2 (90.2 104.6 80.1) |
| MTP adaptive, first policy (stuck-depth bug) | 95.6 (95.6 112.2 81.0) | | | |
| sampled, argmax drafts (`NE_MTP_DRAFT=argmax`) d2 / adaptive | | 88.6 / 80.5 | | |
| **Sushi on this pack, its MTP** (h2h: 3 thinking prompts x 1024, sampled, median) | | **90.5** | | |

So MTP takes the engine from 69 to **101 tok/s sampled at depth 2 (1.46x; 1.12x Sushi's 90.5)** and from 68.5 to 92
greedy; at 8K 66 -> 91 sampled, 65 -> 91 greedy. Depth 2 is the best fixed depth on the median; deeper depths win only
on the predictable prompt (text_trains: 120 tok/s at d3) and lose on text_crisis (acceptance ~0.7). The adaptive
policy is within 1-7% of the best fixed depth per prompt but not better on the median. Drafts sampled from the
head's shortlist accept 1 - TV(p, q) (0.68-0.93 at position 1) vs p(d) for argmax drafts (0.64-0.91): +12.6 tok/s at
d2. (An earlier sampled table was wrong: see 10.5, the proposal-store bug.)

Acceptance (P(draft i accepted | drafts < i accepted), positions 1..D) and tokens per round:

| run | code_lru | text_trains | text_crisis |
|---|---|---|---|
| greedy d2 | 0.80 0.73 / 2.39 | 0.95 0.92 / 2.83 | 0.76 0.69 / 2.28 |
| greedy d4 | 0.83 0.80 0.75 0.73 / 3.36 | 0.93 0.92 0.87 0.86 / 4.19 | 0.68 0.63 0.63 0.62 / 2.53 |
| sampled d2 | 0.81 0.83 / 2.48 | 0.91 0.90 / 2.74 | 0.68 0.64 / 2.11 |
| sampled d3 | 0.81 0.73 0.74 / 2.83 | 0.93 0.92 0.88 / 3.55 | 0.71 0.65 0.65 / 2.48 |
| sampled d4 | 0.78 0.76 0.75 0.82 / 3.20 | 0.92 0.89 0.90 0.84 / 4.10 | 0.73 0.59 0.66 0.58 / 2.60 |
| sampled d2, 8K | 0.81 0.77 / 2.42 | 0.95 0.90 / 2.80 | 0.70 0.65 / 2.16 |
| greedy d2, 8K | 0.84 0.77 / 2.49 | 0.95 0.90 / 2.81 | 0.71 0.66 / 2.18 |

GPU time per round (`bench_mtp.py --time`, median of 10, positions faked): verify (A + B, greedy readout) at R rows and
the draft recording (1-row seed + D - 1 chained steps):

| T | R = 2 | R = 3 | R = 4 | R = 5 | draft per step |
|---|---|---|---|---|---|
| 64 | 18.27 ms (1.28x) | 21.72 (1.52x) | 24.89 (1.75x) | 28.75 (2.02x) | 1.13-1.14 ms |
| 2048 | 18.69 (1.31x) | 22.21 (1.56x) | 25.65 (1.80x) | 29.70 (2.08x) | 1.15 |
| 8192 | 18.83 (1.32x) | 22.35 (1.57x) | 25.79 (1.81x) | 29.83 (2.09x) | 1.16-1.17 |

(ratios vs the 14.25 ms 1-row step; R pending lazy rows add 0.3-0.4 ms; lab-mtp's projection was 1.24 / 1.49 / 1.75 /
2.01x.) Round wall = draft + verify + ~3-4% (the exposed part of the n-gram gather and the two host wake-ups).

### 10.4 Distribution check (sampled)

`tests/mtp/test_sampled_dist.py`: model sampler seed fixed (so the prefill's x_P is the same every run), MTP seed
varied, N runs; the first round's first token must follow the target's filtered p(. | x_P) exactly (1-row decode
logits), and the next token p(. | x_P, out[1]). The position is chosen on the greedy continuation (text_crisis: 65
tokens in, x_P = '-' certain, then 6 ids, top 0.45). Depth 3, 400 runs: out[1] vs p TV 0.058 (sampling noise alone:
median 0.035, 95% 0.063; chi2 8.1 over 6 ids); out[2] | out[1] = ' Explain' TV 0.043 (noise 95% 0.091), | ' Avoid'
TV 0.097 (noise 95% 0.086, P 0.03, chi2 6.3 over 3 ids); argmax drafts, depth 1, 300 runs: TV 0.052 / 0.065 / 0.039,
all within noise (P 0.26 / 0.41 / 0.59). Before the fix in 10.5 the sampled-draft run gave TV 0.19 and a constant
out[2] (0.62): this test is what caught it; the CPU Monte Carlo of the accept could not.

### 10.5 Traps

- **`nax::alloc_scratch_rows(R)` for R = 3, 5..8 leaves the PLE buffers at 4 rows**: a 5-row verify overflowed
  `ple_emb` (host write) and `hp` (GPU write): KLD 1.0, 26% top-1 flips, while R = 2..4 were clean. The loop uses
  `fn::alloc_scratch(R)` (request (4) to lab-speed in docs/kernels.md). The teacher-forced check found it at once.
- **First rounds at a depth paid 8-21 ms** of recording, pipeline compiles and residency: record and replay every
  variant at create.
- **Adaptive depth stuck at depth 1**: the EMA of position i is only updated when a round drafts i + 1 deep; one dip
  of acc[1] while at depth 2 made the policy pick depth 1 for good (text_crisis: 403 of 538 rounds at d1, 81 tok/s
  vs 86 at fixed d2). Unreachable positions now relax toward the deepest drafted one.
- **Greedy identity breaks at exact bf16 ties** (section 10.2): expected with different 1-row / R-row arithmetic;
  judge by the gap at the first mismatch, not by "identical".
- A with_lock job's NEED_GB check runs before it waits for the lock: queue whole-model jobs behind a poll for a free
  lock (my `waitrun.sh` pattern), or they exit 76 while another whole-model job holds 81 GB.
- **The sampled-draft proposal came back with q = 0 for every entry up to and including the drawn token** (the kernel
  wrote q after its draw loop; the stored values before the pick were zeros, most likely a miscompile around the
  early `break`), so the host accepted every in-support draft with probability 1: fast (114 tok/s) and wrong. Now
  the proposal is written first and the draw reads the same probabilities; `mtp-loop-check` (4) checks the stored q
  against a CPU softmax and the draw frequencies against q, `test_sampled_dist.py` checks the loop's output.
  Lesson: an exactness claim needs an end-to-end distribution test at a position with entropy.
- Speculative sampling with argmax drafts is exact but slower under temperature 1 (accept = p(d)); draw drafts from
  the head's shortlist and pass that q.

### 10.6 What is left

- Verify cost: R-row MoE bytes are the cost (section 3); 1.28x at 2 rows vs lab-mtp's 1.24 projection: the SIMD
  out_proj / attention projections at R <= 3 (request (3) to lab-speed) and the lazy step's pending-row ALU (+0.3-0.4
  ms) are the visible extras.
- Draft step 1.13-1.17 ms (the 4-bit readout 0.73 ms of it): a 1-row SIMD 4-bit GEMV (~0.1 ms), 2-bit + K 16.
- GPU-side greedy accept (the host wake-ups, ~0.1-0.2 ms per round), and a larger shortlist for sampled drafts.
- Prefill: done 16 Oct -- the loop prefills through lab-prefill's chunked prefill one chunk per call and feeds the
  head from its layer-47 rows (`prefill_hidden`): 8K start 52 -> 6.95 s (head prefill 1.28 s of it, 8-row steps at
  0.16 ms per position; since 10.10 KV-only head steps at ~0.03 ms per position). docs/forward.md 3c.
- MTP from the model's current state (prefix reuse across turns; lab-server's request (2)): done 16 Oct, section 10.9.
- Token identity past ties: batch-invariant kernels (one reduction order for 1 and R rows), if wanted.

### 10.8 Re-measured 16 Oct (lab-integrate: chunked-prefill start, HC read v2; HEAD 0ef6a9c built from a clean tree)

Same 3 thinking prompts, f32, max depth 4, d2; 1024 new tokens short, 512 at 8K; tok/s median (code_lru, text_trains,
text_crisis):

| config | short greedy | short sampled (seed 500 / 501 / 502) | 8K greedy | 8K sampled |
|---|---|---|---|---|
| no MTP | 68.2 (72.1 67.0 68.2) | 66.8 (71.6 66.8 66.0) | 65.3 | 65.4 |
| MTP d2 | **100.7** (100.7 101.9 81.1) | **94.7** (94.7 99.3 80.2) / 89.5 (89.5 104.9 80.1) / 91.0 (91.0 99.5 79.9) | **96.2** | **88.4** |
| tokens / round | 2.50 / 2.80 / 2.19 | 2.47 / 2.72 / 2.19 (s500) | 2.50 / 2.75 / 2.28 | 2.33 / 2.73 / 2.20 |

8K prompts now start in 6.95 s instead of 52 (docs/forward.md 3c). **101.2 (15 Oct) vs lab-server's 84.0 on code_lru
sampled**: the acceptance reproduces exactly (code_lru sampled d2 2.38-2.48 tokens / round over 3 seeds vs lab-loop's
2.48; per-position accepts 0.78-0.82 / 0.75-0.78 vs 0.81 / 0.83), so neither the q-fix (780ba07, in both builds: build/
was linked 18:24 after it) nor a correctness difference explains it. The spread is the round time and the seed: verify
per round in the loop 22.9-23.4 ms here (R = 3) vs ~21.7 when lab-loop measured, and replays of one build vary by up to
+3 ms for minutes at a time today (R = 2 at T = 2048: 18.1 vs 21.3 ms in one job; 21.8 vs 18.4 between jobs) -- with
other labs' builds and small GPU jobs between the whole-model runs. code_lru's sampled tok/s over 3 seeds: 89.5-94.7. So
the sampled d2 number to quote is ~90-95 tok/s short (1.35-1.4x no-MTP), 88 at 8K; 101 was a fast-state run, 84 a slow
one (or one seed). Greedy d2 ~100 short, 96 at 8K. A quiet-machine re-run (night) would pin the absolute numbers.

### 10.9 MTP from the model's current state (lab-mtpcont, 16 Oct)

The loop used to start only from a fresh prefill and leave the model dirty, so the server's prefix reuse (snapshots,
live state) forced the 1-row step path (~70 tok/s) on every reused turn. Now:

- **The head's state is kept in step with the model** by the loop's own calls. State = head KV rows [0, Lh) (row j =
  the pair (H_j, x_{j+1}), H = layer 47's pre-mixer hidden) + the hidden rows of positions [Lh, T) still pending (the
  last one's token = the next one fed; at most 8 known + 1). `mtp_feed` (prefill, no sample: chunked from 16 tokens
  over all n, else 4-row / 1-row body steps) and `mtp_step` (one decode step) push every position's hidden and run the
  head (KV only) in 8-row steps at head positions [8i, 8i + 8), whatever the feed split: a prefix fed in pieces gives
  the head the same steps as one prompt. `mtp_generate_from(toks)` = feed toks[0, n - 1), the decode step with the
  head on the last, the remaining pending pairs 1 row at a time (the fresh prefill's order), then the rounds;
  `mtp_generate` = reset + the same. So the head is **rebuilt for the suffix exactly as a fresh prefill** would
  build it (no separate "rebuild" pass: the suffix's hidden rows only exist during its prefill).
- **Validity**: `Model::gen` is bumped by every state change (step, step4, prefill_tokens, reset, restore); the head is
  valid while `head_gen == gen` (or T = 0). A plain `ne_fn_model_*` call in between makes it stale:
  `mtp_feed` / `mtp_step` / `mtp_generate_from` refuse (state untouched), the server falls back to the step path.
- **Snapshots**: `ne_fn_mtp_snapshot` = `ne_fn_model_snapshot` (113 MB) + Lh + the pending rows (<= 360 KB; 118 MB
  measured in all). Same validity rule as the model's: the head's KV rows [0, Lh) are only truncated by position,
  like the target's, and they are a function of tokens [0, Lh] (prefix check = the server's).
- **Clean end of a run**: the last round's kept rows are lazy (keep[0]); a 2-row verify over dummy tokens at pos T
  commits them (GDN ssm / conv, PLE conv) and its own rows are dropped (keep[0] = 0; their KV rows are past pos).
  ~17 ms per run (the bench's "rest", incl. host work). A round cut by `stop` / max_new now commits only the emitted
  tokens before the last one (`c = k - k0` rows instead of a + 1): the state is always prompt + out[0, k - 1)
  (before, it could hold accepted drafts past the stop). The seed rows stay pending, chain rows of accepted drafts
  stay in the head's KV (mlx-vlm semantics: their hidden is the head's own, not the target's -- so a live
  continuation's head differs slightly from a fresh prefill's over the same tokens; only acceptance can change).
- C API: `ne_fn_mtp_head_ok`, `_head_state`, `_feed`, `_step`, `_generate_from` (with the callback),
  `_snapshot` / `_restore` / `_snap_T` / `_snap_bytes` / `_snap_free` (mtp_loop.h). Debug: `NE_FN_LAYERS=k` loads a
  k-layer partial model (k >= 2, ~1.7 GB per layer; the unloaded layers are not made resident): meaningless text,
  real state machinery -- `NE_FN_LAYERS=2 NE_FN_QUEUE_RESIDENCY=0 $PY tests/mtp/test_mtp_cont.py` runs in 10 s, 1.5 GB.

Correctness (`tests/mtp/test_mtp_cont.py`, greedy d2, 404-token prompt, 48 new tokens; whole model and 2-layer
partial model, all checks pass on both):
- **exact**: fresh `mtp_generate(P)` vs feed(P[:100]) + feed(P[100:k]) + snapshot + a run continuing past k
  (overwrites target and head KV rows >= k) + restore + `mtp_generate_from(P[k:])`, k = 200 / 256, with
  `NE_FN_PREFILL=steps` (same target arithmetic): **tokens and the per-(depth, accepted) round histogram identical**
  (whole model: 23 / 23 rounds), i.e. the head state is bit-identical. With the chunked prefill (other chunk
  boundaries): first fork at token 24 (exact tie, gap 0.000) / token 4 (1-row logit gap 0.250 = 2 bf16 steps at
  ~17.5): the prefill's reduction order, as on the step path.
- `mtp_step` x 40 then MTP == 1-token feeds then MTP (tokens + histogram); a plain step makes the head stale and
  `generate_from` refuses; runs cut at max_new 2..7 leave T = n + k - 1 and Lh + pending = T; a snapshot of the
  model right after an MTP run works (it was refused as dirty before); live continuations fork only at gaps
  0.000 / 0.062 (bf16 ties; the verify's multi-row arithmetic built that state).
- Fresh runs repeat exactly (tokens + histogram).

Speed (whole model, `tools/bench_mtp_cont.py`: the server's Session + RealBackend in-process, a ~7.9K-token system
prompt of repository docs + code_lru, 3 turns x 512 tokens, sampled T 1 / top-p 0.95 / top-k 20, f32, one run each):

| turn | prompt | cached | MTP-cont prefill ms | MTP-cont decode tok/s | step prefill ms | step decode tok/s |
|---|---|---|---|---|---|---|
| 1 | 7959 | 0 | 7011 (first use in the process) | 96.5 | 5766 | 70.7 |
| 2 | 8498 | 7954 | 698 | **100.0** | 723 | 70.5 |
| 3 | 9035 | 8493 | 684 | **102.0** | 676 | 70.5 |

So reused turns now decode at ~100 tok/s instead of ~70 (1.42x) with the same reuse and the same prefill time on
both paths. The price of keeping the head in step: **prefill +30%** (plain `ne_fn_model_prefill` vs `mtp_feed` +
`mtp_step` from an empty state, best of 2 after a warm-up: 2049 tokens 1068 -> 1391 ms, 4097 2164 -> 2805, 8193
4457 -> 5813; 157-165 us per token = the head's 8-row steps, as in the fresh MTP prefill before), i.e. ~80 ms on a
500-token turn suffix; decode: `ne_fn_mtp_step` 13.80 vs `ne_fn_model_step` 14.10 ms per token (noise; one 8-row head
step per 8 tokens). Next lever there: a chunk-sized head prefill through the prefill machinery (10.6). Sampled
exactness re-checked after the change (`test_sampled_dist.py`, text_crisis d3, 400 runs): out[1] TV 0.040 (noise 95%
0.060), out[2] | ' Explain' 0.035 (0.092), | ' Assume' 0.012 (0.035): OK.

### 10.10 Lazy head and KV-only head steps (lab-headlazy, 16 Oct)

Two changes to 10.9's head bookkeeping (`mtp_loop.mm`; `mtp::record_kv_step`, `nax::record_layer(.., kv_only)`):

- **KV-only head steps.** The head's state between rounds is only its KV cache, and a head row's KV depends only on
  the fuse, fc_embedding / fc_hidden, the attention HC read, the q / k / v / indexer projections and lab-attn's
  `qsa_prep` (norm + rope + cache append + indexer / block keys). Head prefill and catch-up now record exactly those
  dispatches (the same kernels at the same 8 / 1 rows as the full step, so the cache rows are bit-identical) and stop:
  no attention core, o_proj, MLP HC, MoE. 16 groups of 8 rows per command buffer over two host-filled staging buffers
  (the next fills while one runs). **26-30 us per head row instead of 130-148** (4.4-5x). `NE_MTP_HEAD_KV=full` keeps
  the old full-layer recordings (A/B).
- **Lazy head.** `mtp_step` and `mtp_feed` only queue each position's layer-47 hidden row (40 KB host copy of the
  shared buffer); the head runs over the queued rows (`head_drain`: whole 8-row groups at head positions
  [Lh + 8i, Lh + 8i + 8), the same partition the eager head used, so the result does not depend on when it runs) when
  `pend_cap` rows are pending (default 1024 = 40 MB; `NE_MTP_PEND_CAP`, `ne_fn_mtp_set_pend_cap`; <= 8 = the eager
  head of 10.9), at the start of every MTP run (`generate` / `generate_from`, as before), before an MTP snapshot and
  before `ne_fn_mtp_state_io` export (lab-srvcache's parking): those catch up first, so a snapshot / state buffer
  still holds <= 8 pending rows (118 MB; catching up is cheaper than carrying up to 40 MB of rows in every
  checkpoint, and the fixed 16-row state buffer stays valid). `ne_fn_mtp_catch_up(p, &ms)` runs it on demand.
  Validity rules (Model::gen, stale head) unchanged.

Correctness (`tests/mtp/test_mtp_cont.py`, new `lazy` checks, whole model and 2-layer model, both prefill modes,
all pass): feed(P) + 100 `mtp_step`s (40 forced + 60 own greedy) + MTP d2 x 48 with the lazy head (504 rows pending
at the MTP start) == the eager head (cap 0) -- **tokens and the per-(depth, accepted) histogram identical**, also
through a snapshot, an explicit catch-up, a state / KV export + import with both caches overwritten in between (the
server's parking), cap 64, and against the full-layer head recordings eager and lazy (= the KV-only rows are
bit-identical in effect). The 10.9 checks pass unchanged. `test_sampled_dist.py` (text_crisis d3, 400 runs): out[1]
TV 0.040 (noise 95% 0.060), out[2] | ' Explain' 0.035 (0.092), | ' Assume' 0.012 (0.035): OK. Server `test_units.py`
and `run_mock.sh`: all pass.

Speed (whole model, `tools/bench_headlazy.py`, one job; "old" = this build with `NE_MTP_HEAD_KV=full` and cap 0 =
lab-mtpcont's eager head):

| measurement | plain (no head) | old: eager, full-layer head | new: lazy (cap 1024), KV-only |
|---|---|---|---|
| step path tok/s, 512 sampled tokens after a 1K feed (run order plain, old, lazy, kv-eager, plain, lazy, old) | 72.3 / 74.2 | 71.4 / 73.5 | 71.1 / **74.1** (KV-only eager: 73.4) |
| catch-up, 50 / 500 / 5000 pending rows | | 7.4 / 65 / (~650, extrapolated) ms | **1.7 / 13.9 / 131 ms** (30 / 28 / 26 us per row) |
| `mtp_feed` + catch-up + step vs `ne_fn_model_prefill`, 2048 tokens | 1073-1078 ms | 1290-1354 ms (+20-26%) | 1059-1088 ms (-1..+1%) |
| same, 8192 tokens | 4178-4190 ms | 5333-5468 ms (+27-31%) | 4429-4459 ms (+6%) |
| fresh MTP d2 greedy, 512-token prompt, 512 tokens | | 89.7 / 90.7 tok/s, head prefill 69-73 ms | 90.6 / 90.1 tok/s, head prefill **18 ms**; tokens identical (235 rounds each) |

So the step path is back to the plain step's speed (74.14 vs 74.22 tok/s in the second half of the run; the drift
between the two halves, +2.5% on every variant, is larger than the differences). The eager head cost ~1% here
(73.50, one full 8-row head step = ~1.06 ms per 8 tokens), not the 6% of docs/server.md 9c's HTTP sample; the rest of
that gap was not the engine. The bigger gains are the KV-only steps: keeping the head in step during a prefill now
costs +1..6% instead of +20-31% (8K: -0.9 s), a 5000-token catch-up 131 ms, and a fresh MTP start's head prefill a
quarter of before. MTP decode is unchanged (the rounds do not use these recordings).

### 10.11 Image prompts (lab-mtpimg, 16 Oct)

MTP now runs after image prompts (server). The head follows the reference: token ids (`<|image_pad|>` embeddings for
image rows) at 1-D positions = its cache length; the target's verify rows use the MRoPE table (row + delta). Greedy
d2: chart prompt 117.7 tok/s (step 73.6), 2.73 tokens / round; Mermaid diagram 103.0 (73.3), 2.38; text code_lru 2.46.
`NE_MTP_HEAD_POS=mrope` (head on the target's multimodal positions) gives the identical acceptance histogram, so it is
off by default. Details and the server's stale-image-rows fix: docs/vision.md section 7.

### 10.12 First token before the head's catch-up (lab-ttft, 17 Oct)

`generate_impl` streams t1 (the prefill step's sample) through the callback right after that step, before
`head_drain(L, true, 1)` (the drafts of round 0 need the head's KV, the first token does not); a nonzero return still
ends the run after the bookkeeping. On a reused server turn the drain is small (the checkpoint export already caught
the head up: <= 13 rows, 1-4 ms); on a fresh MTP prompt it is the whole head prefill (KV-only, ~28 us per row).
`tests/mtp/test_mtp_cont.py` (4-layer model): all pass.

### 10.7 Attribution

The round structure (draft chain, R-row verify, exact acceptance with the draft's proposal distribution, head
`accept_verified_tokens` semantics) follows mlx-vlm 0.7.7's MTP drafter and Sushi's `docs/engine-mtp.md` /
`src/mtp_acceptance.zig` (MIT, see section 9); no code copied. `mtp_ple_mix_lazy` is lab-forward's `fn_ple_mix`
(this repo) with the lazy commit added.

## 11. Drafting at real settings: profile, pruned draft vocabulary, copy rule, gated verify (lab-draft, 18 Oct)

Why: Ijah's plain decode is the fastest of the three engines on the same client (72.7 tok/s) but MTP gained least
(72.7 -> 98.9, 1.36x; Sushi 1.49x, oMLX oQ4e 105.2). Block drafters for this MoE model gained only +3.9% over native
MTP k = 4 elsewhere (PixelML's Qwen3.8-Flash-Next DFlash on vLLM), so the drafting stays the native head and gets
cheaper and smarter. Settings throughout = the same-client harness's: temperature 1, top-p 0.95, top-k 20, thinking on
(medium), the six head-to-head THINK prompts (the same-client harness, not in this repository), seeds 500 + 10 i, 1,024 tokens; T6 pack; in-process
`tools/bench_draft.py` (one model load, configs alternated) unless marked "server".

### 11.1 Where a round's time goes (sampled, `NE_MTP_TRACE` per round + `ne_fn_mtp_prof` per part)

Parts on their own (`ne_fn_mtp_prof`, GPU ms, median of 20, head / target at T 512):

| part | full draft vocabulary | pruned (11.2) |
|---|---|---|
| head step, 1 row (fuse + fc + QSA / MoE K4 layer + mixer + draft readout) | 1.115 | **0.730** |
| ... without the readout / KV-only step (the lazy head's catch-up) | 0.333 / 0.159 | 0.333 / 0.160 |
| ... the readout: 4-bit coarse GEMM, top-K stage 1, shortlist + exact re-score, draft sample | 0.740, 0.011, 0.009, 0.016 | 0.352, 0.011, 0.010, 0.020 |
| (the full 8-bit lm_head + argmax instead) | 1.175 | |
| seed step at 2 rows (all drafts of the last round accepted) | 1.236 | 0.846 |
| draft recording, 1-row seed, depth 1 / 2 / 3 / 4 | 1.13 / 2.25 / 3.38 / 4.50 | 0.74 / 1.48 / 2.22 / 2.96 |
| verify (A + B, sampled top-K readout), R = 2 / 3 / 4 / 5 rows | 15.86 / 19.45 / 22.49 / 25.27 | (the same) |

1-row decode step 13.08 ms: the verify costs **1.21x / 1.49x / 1.72x / 1.93x** at 2-5 rows, +3.0-3.6 ms per row (the
MoE union bytes, section 3). A head chain step is 0.74 ms pruned, flat in depth (one command buffer).

In the loop (sums over the six prompts, ms per round; gap = round wall - draft - A - B: the host wake-ups, the exposed
part of the n-gram gather, the commits):

| config | rounds | draft | A | B | gap | gather | tokens / round | ms / token | tok/s (median of 6) |
|---|---|---|---|---|---|---|---|---|---|
| full, d2 | 2449 | 2.39 | 0.41 | 20.14 | 0.68 | 0.44 | 2.51 | 9.49 | 108.4 |
| pruned, d2 | 2439 | 1.63 | 0.41 | 20.29 | 0.36 | 0.14 | 2.52 | 9.09 | **112.2** |
| full, d3 | 2131 | 3.58 | 0.49 | 23.89 | 0.96 | 0.60 | 2.88 | 10.24 | 103.6 |
| pruned, d3 | 2132 | 2.42 | 0.49 | 23.87 | 0.55 | 0.23 | 2.88 | 9.69 | 110.0 |
| pruned, adaptive (rounds at d1 / d2 / d3 / d4) | 173 / 1197 / 374 / 481 | | | | | | 1.75 / 2.40 / 3.21 / 3.67 | | 109.1 |

So at depth 2 a round is 91% verify (20.7 of 22.7 ms), 7% drafting, 2% host; the gather (0.14-0.76 ms between runs:
page cache) overlaps A (0.41 ms) and shows as gap when longer. Acceptance per position (sampled, d3): code_lru 0.80
0.78 0.77, text_trains 0.93 0.91 0.87, text_crisis 0.68-0.71 / 0.58-0.60 / 0.58-0.59, code_parser 0.87 0.79 0.83,
text_climate 0.72 0.67 0.67, math_proof 0.87 0.86 0.81; tokens per round 2.17-2.82 at d2, 2.34-3.53 at d3. Depth 3
loses to depth 2 on the median because the 4th verify row and a draft step cost 4.6 ms of round wall (+3.6 verify,
+0.8 draft, +0.2 gap) for +0.36 tokens, while a token costs ~9.1 ms: a row pays only above ~0.5 cumulative acceptance. **The verify rows, not the
drafting, are what keeps our MTP gain below Sushi's**: Sushi's step is slower, so its extra rows cost less relative
to a token.

### 11.2 Pruned draft vocabulary (default on; `NE_MTP_DRAFT_VOCAB`)

The draft readout's 4-bit coarse lm_head copy now holds only the kept rows (`set_draft_head(.., rows)`, requantised
at load from those rows; `MtpW::draft_ids` maps a compact row to its id inside `mtp_shortlist` before the exact 8-bit
re-score). Kept set = Splish's (`models/flash_next/draft_vocab.h`, copied from Splish's `DraftVocab.hpp`): ids <
98,304 (TensorFold's prefix) + the 33 special tokens + the 16,384 most frequent higher ids = 114,721 of 248,320 rows
(165 MB instead of 358). Rows ascend, so ties still go to the lowest id. Verification reads the full head: greedy text
is unchanged and sampled acceptance uses the proposal actually drawn (q over the kept ids, zero elsewhere: the residual
max(p - q, 0) then covers the rest, `accept_sampled_p` unchanged; `mtp-accept-test`'s 4-id shortlist case is this).
`NE_MTP_DRAFT_VOCAB=full` (or `0`) = the old head; `prefix` = ids < 98,304; a number N = ids < N.

Coverage on our outputs (every generated token of the runs here): **ids < 98,304: 99.60-99.70%** on the h2h prompts
but **89.4%** on the whole-file edits (code identifiers); **in the kept set: 99.86-99.89% / 99.97%**. The prefix alone
would have cost the edits a tenth of their drafts. Result: draft step 1.12 -> 0.74 ms; d2 **108.4 -> 112.2 tok/s
median (+3.5%; mean 106.7 -> 111.7, +4.7%), every prompt faster** (+2.2..+6.4%), acceptance unchanged (the same seeds
gave the same acceptance histograms on 5 of 6 prompts: the re-scored shortlist rarely changes); d3 103.6 -> 110.0.
Greedy (bench_mtp, 3 prompts x 512, d2 / d3, pruned + copy rule): first mismatches vs the no-MTP step path at 1-row
logit gaps 0.000 / 0.125 / 0.250 only (bf16 ties, section 10.2).

### 11.3 Copy rule (default on, 16 tokens; `NE_MTP_COPY_MIN_MATCH`, `NE_MTP_COPY_DEPTH`)

TensorFold's rule as in Splish (`SPLASH_M5_COPY_MIN_MATCH`): when the last 16 tokens, **t1 included** (the last
emitted token, which the round's verify row 0 carries), occurred earlier in the context, the round drafts the tokens
that followed the most recent earlier occurrence (up to the loop's max depth, 4) instead of the head's: no draft
recording runs, the drafts go straight into the verify token buffer. Greedy accept as usual; sampled accept with
point-mass proposals (`accept_sampled_p(.., q = nullptr)`: accept with p(d), correction from p without d; exact,
`mtp-accept-test`'s argmax case). Splish's first build rejected every copy because its window ended one token early
(its anchor is emitted first); here the window is S = context + [t1] and the continuation starts right after the
match, checked against a naive search in `mtp-accept-test` (12,000 random lookups, 0 mismatches, incl. runs and
overlaps).
- `mtp::copy_find` / `CopyIndex` (mtp.h): FNV-1a hashes of every 16-token window -> its most recent end, indexed
  incrementally as the history grows, matches verified token by token.
- History: the loop records every token it feeds, steps or commits (`MtpLoop::ctx`, positions [ctx0, T)); snapshots
  carry it; after a state import it knows nothing until `ne_fn_mtp_set_context` (the server passes `N.toks` when
  `ne_fn_mtp_context_len` is short: parked conversations).
- The head after a copy round: it did not run, so its pairs (the old seed rows + every committed verify row's target
  hidden) are queued as pending head rows; the next head round drains them (KV-only steps, 10.10) and seeds from the
  last one; > 136 pending rows drain in whole 8-row groups inside the copy run.
- C API: `ne_fn_mtp_set_copy(p, min_match, depth)` (A/B), `ne_fn_mtp_context_len`, `ne_fn_mtp_set_context`.

Whole-file edits (two prompts: a 2.0K-token file of this repo shown, "rename X, add docstrings / flush=True, reply
with the complete file"; 2,180 prompt tokens, 4,096 new, thinking on): d2 **125.9 -> 154.7 tok/s (+23%)**; 711 / 728
copy rounds emitted 3,186 / 3,591 of the 4,095 tokens at 4.5-4.9 tokens per round. Plain h2h prompts: 0-2 copy rounds
per 1,024 tokens, d2 112.2 vs 112.5 without, d3 110.3 vs 108.7 (noise).

### 11.4 Gated verify (opt-in `NE_MTP_QGATE=1`, `ne_fn_mtp_set_qgate`) and the depth (retuned for the cheaper drafting)

With drafting at 0.74 ms a step, drafting one deeper is cheap; verifying a row is not (+3.0-3.6 ms). So the loop now
drafts the fixed depth Dd, reads the drafts and their proposals q (sampled drafting: what each d_j was drawn from),
and verifies only the prefix k <= Dd that maximises E[tokens](k) - rate x vcost(k + 1): E(k) = 1 + sum_{j<=k}
prod_{i<=j} acc(q_i(d_i)), acc = P(d accepted | the drafts before it accepted, q(d) in a 0.1 bin) (EMAs over every
sampled round, priors 0.33-0.92), rate = tokens per ms and vcost(R) = round ms at R verify rows (EMAs). A is then
committed after the drafts are read (it used to run right behind the draft; the gather overlaps it either way). The
stopping rule looks only at the drafts, never at the target, so the accept stays exact speculative sampling (the
rounds just verify fewer drafts). Greedy and argmax drafts have no q: no gate; the server drafts greedy requests 2
deep. Adaptive depth (0) keeps its DepthPolicy (no gate), whose priors are now the measured costs (kF 13.1 / 15.9 /
19.5 / 22.5 / 25.3 ms, draft step 0.74 / 1.12).

Calibration over the h2h runs (P(accepted | q(d) bin), sampled, 28,412 draft positions): 0.36 (q < 0.1), 0.43, 0.48,
0.48, 0.55, 0.57, 0.58, 0.60, 0.68, **0.93 (q >= 0.9, 69% of the drafts)**: a low-q draft is a coin flip, so a row
behind one rarely pays. Rounds verifying 1 / 2 / 3 / 4 drafts: depth 2 + gate 16 / 84%, depth 3 + gate 17 / 18 / 65%, depth 4 + gate 24 /
16 / 11 / 49%.

In-process (6 prompts x 1,024, one run each, tok/s median / mean): pruned d2 112.5 / 112.8 (repeat of 11.2's 112.2),
gate d2 113.4 / 110.4, **gate d3 114.2 / 113.1**, gate d4 115.5 / 112.0, pruned d3 without the gate 108.7 / 106.7,
copy rule + gate d4 112.8 / 111.8. So the gate makes depth 3-4 as good as depth 2 (without it they lose 2-4%), no
better within the noise; through the server it measured 1.052 [1.030, 1.074] and 0.975 [0.951, 1.000] x master in two
sessions (11.5: session drift is +-5% today), depth 4 + gate 0.966. **Depth 2 stays the default, the gate opt-in**
(it also changes which sampled text a seed gives, since the rounds differ). The server now drafts greedy requests 2
deep whenever `--mtp` > 2 (greedy d3 103.0 vs d2 111.4 tok/s median, bench_mtp 3 x 512).

### 11.5 Same-client harness (server, `--seed-mtp`, 6 prompts x 3 seeds, 1,024 tokens; master = 04cae67's build)

| session (time) | tok/s mean +- sd | paired / master of the same pair (95% CI), wins |
|---|---|---|
| master (12:40) | 105.6 +- 9.6 | |
| **lab, d2: pruned vocabulary + copy rule (12:43, right after)** | **108.9 +- 9.5** | **1.031 [1.028, 1.035], 18 / 18** (every prompt +1.9..+3.6%) |
| lab, d3 + gate (12:46) | 111.3 +- 12.1 | 1.052 [1.030, 1.074], 16 / 18 |
| lab, d3 + gate (12:57, after tests) | 103.1 +- 10.3 | 0.975 [0.951, 1.000], 5 / 18 |
| lab, d4 + gate (13:00) | 102.4 +- 12.3 | 0.966 [0.944, 0.990], 5 / 18 |
| master (13:05) / lab d3 + gate / lab d2 / master again (13:15) | 100.9 / 107.7 / 108.3 / 105.7 | (vs the first) 1.067 / 1.074 / 1.048 |

The d2 lab session gives master's sampled texts (same seeds, same rounds: only the draft cost changed), hence the
tight interval; the gated sessions give other texts and swing with the machine (master itself moved +4.8% in 10
minutes). Against the 18 Oct baseline (98.9 +- 10.2) the lab d2 server is at 108.3-108.9; Sushi 85.8, oMLX oQ4e 105.2
on that client. Against the step path on that client (72.7): 108.9 / 72.7 = 1.50x (was 98.9 / 72.7 = 1.36x; Sushi 1.49x).

### 11.6 Correctness (all on the final code unless noted)

- `mtp-accept-test`: all pass, incl. the copy-rule lookup vs a naive search (12,000 lookups, 0 mismatches) and the
  sparse / point-mass proposal cases the pruned head and the copy rule use.
- `test_sampled_dist.py` (text_crisis, depth 3, 400 runs; with the gate, the pruned head and the copy rule on, i.e.
  the gate's rounds too): out[1] TV 0.049 (noise 95% 0.062, chi2 5.8 / 5 ids), out[2] | ' Explain' 0.043 (0.092),
  | ' Assume' 0.022 (0.030): OK.
- Greedy vs the no-MTP step path (bench_mtp d2 / d3 x 3 prompts x 512; bench_draft `--greedy`, copy rule on, code_lru /
  text_crisis / a whole-file edit x 1,024): every first mismatch at a 1-row logit gap of 0.000 / 0.125 / 0.250 (ties).
- `test_mtp_cont.py` (4-layer model, copy rule on): all pass (exact snapshot / restore / lazy / park checks).
- `make test` (with_lock): ALL PASS. `mtp-loop-check`: passes on the K5 pack; on the T6 pack (IJAH_PACK) its check (2)
  stopped with "argument with neither buffer nor bytes" (a test bug, fixed by lab-loopfix: section 10.2). Server `run_mock.sh`: all pass.

### 11.7 What is left

- The verify row (+3.0-3.6 ms, 23-27% of a step) is the MTP limit here (11.1): MoE union bytes per verify row.
- Draft readout at 2 bits over the pruned rows (~0.2 ms, acceptance to check), the head layer itself (0.33 ms).
- A better acceptance predictor for the gate than q(d) alone (position, the head's entropy, the verify's own p of
  earlier rounds), then re-test depth 3-4; greedy gating needs per-step shortlists kept (only the last is now).
- Copy rounds deeper than 4: R = 7 verify kernels are not instantiated (fn_moe_router_r7, ...); depth 5 (R = 6) gave
  the same 157.9 tok/s on the edits as 4.

### 11.8 Attribution

The copy rule is TensorFold's (github.com/ashhart/TensorFold) as built in Splish (`runtime/engine/Engine.cpp`
`fillCopyDraft`, `SPLASH_M5_COPY_MIN_MATCH`; read, not copied); the pruned draft vocabulary is Splish's kept set
(`runtime/model/flashnext/DraftVocab.hpp`, copied into `models/flash_next/draft_vocab.h`, the prefix idea
TensorFold's). Both Splish files are this project's own work.

## 12. Verify cost (lab-verify, 18 Oct)

Why: 11.1's profile put 91% of a depth-2 round in the verify, each extra verify row +3.0-3.6 ms (~25% of a step).
Tools added: `NE_PROF_ROWS=R build/fn-prof PACK 15 512 3` (the per-kernel leave-one-out profile of section
docs/kernels.md 5b at R rows; 4 loaded layers replayed as 48, random distinct rows), `NE_PROF_DDCHECK=1` (the R-row
body with the routed-expert dedupe off vs on from the same state: outputs after 1 / 4 / 48 layers), `NE_VERIFY_ROUTES=
file` (every verify's routing, u16 [48][R][10] per round, written by the MTP loop: a debug copy after each layer),
`ne_fn_mtp_set_moe_dedupe` + bench_draft `@dedupe=N`, trunk6 `T6_NAX_MIN` / `T6_P_<in>x<out>=simd|nax`.

### 12.1 Where an extra row goes (fn-prof, T 512, ms per 48-layer body, leave-one-out in context; master kernels)

| kernel kind | R 1 | 2 | 3 | 4 | 5 | per extra row (R 1 -> 5) / 4 |
|---|---|---|---|---|---|---|
| **body** | 11.81 | 14.33 | 17.75 | 20.61 | 23.50 | **2.92** |
| routed experts gate / up (`fn_moe_gateup`) | 2.24 | 3.45 | 4.92 | 6.53 | 8.09 | **1.46** |
| routed experts down + HC write (`fn_moe_down`) | 1.34 | 1.82 | 2.48 | 3.17 | 3.97 | **0.66** |
| router + shared expert gate / up (+ SwiGLU) | 0.76 | 0.71 | 0.77 | 1.34 | 1.61 | 0.21 (the jump at R 4: t6 NAX path) |
| GDN step (36) | 0.48 | 0.61 | 0.66 | 0.72 | 0.87 | 0.10 |
| GDN out_proj + HC (36) / o_proj + HC (12) | 1.03 / 0.35 | 1.05 / 0.36 | 1.12 / 0.32 | 1.30 / 0.41 | 1.40 / 0.53 | 0.09 / 0.04 |
| HC reads (4 kinds, 48 each) | 2.29 | 2.40 | 2.98 | 2.81 | 2.95 | 0.16 |
| QSA attention step (12) | 0.42 | 0.50 | 0.51 | 0.65 | 0.65 | 0.06 |
| GDN in_proj (36), attention projections (12), PLE | 3.17 | 3.07 | 3.50 | 3.24 | 3.35 | ~0.04 |

So **73% of an extra row is the routed experts** (2.12 of 2.92 ms: 10 more experts x 48 layers, 1.47 GB at K5),
7% the router finisher + shared expert, the rest per-row GDN / attention / HC work at 0.04-0.16 ms each. The head
(mixer + lm_head on NAX) is flat in R. At R >= 2 the routed kernels run at the read peak on the nominal bytes
(gateup R 4: 82 MB in 136 us), i.e. only fewer bytes help.

### 12.2 The expert union on real text (`NE_VERIFY_ROUTES`, sampled d4, the six h2h prompts x 512, 976 rounds)

| rows of a verify | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| distinct experts per layer (of 10 R) | 10 | 16.2 (81%) | 21.7 (72%) | 26.7 (67%) | 31.3 (63%) |
| expert-rows saved if each touched expert were read once | 0 | 0.38 | **0.83** | 1.33 | 1.87 |
| experts used by 1 / 2 / 3+ rows | | 77 / 23% | 71 / 20 / 9% | 68 / 20 / 13% | 65 / 20 / 15% |

Committed rows only (the accepted, consecutive tokens) give the same (21.9 of 30 at 3). Sharing grows with depth:
layer 0 shares least (1.7 slots of 30 at 3 rows), the later layers 5-13. Section 3's "39 of 40 at 4 rows, ~2.5%
recoverable" came from L00_verify4, i.e. layer 0 only: wrong for the model. The ceiling is ~0.83 rows of expert
bytes at R 3 (~1.8 ms of the ~6.6 ms that R 3 adds), 1.87 rows at R 5 (~4 ms).

### 12.3 What was tried (one indicative A/B each; in-loop = bench_draft verify GPU ms per round, same process)

| change | result | kept |
|---|---|---|
| dedupe kernels (`fn_moe_gateup_u` / `fn_moe_down_u`, existed for R 2 / 4; now 3 / 5 too, hsh aliased onto prep so R 5 fits 32 KB): each touched expert decoded once, dotted with all R rows | in-loop d2 (R 3) 21.73 -> 20.83, 20.64 -> 20.33, 20.85 -> 20.53 ms (3 runs); d3 (R 4) 25.34 -> 24.50; d4 (R 5) 28.59 -> 30.41 | **R 2-3 only (new default `NE_FN_MOE_DEDUPE` = 3)** |
| v2: dedupe with the rows compacted (dynamic row list) | d2 / d3 / d4 20.65 -> 22.27 / 26.24 -> 26.73 / 29.10 -> 30.99 ms; fn-prof R 5 +4 ms (register arrays indexed at run time) | no |
| v3: dedupe with per-row predicates in the hot loop | d2 / d3 / d4 20.45 -> 20.20 / 24.50 -> 26.22 / 27.67 -> 30.45 ms | no |
| grouped threadgroup order (row, slot pairs sorted by expert so duplicates hit the cache back to back; bit-exact, ddcheck 100%) | d2 20.64 -> 20.73, d3 24.60 -> 24.17 ms: nothing | no (removed) |
| trunk6 SIMD instead of NAX at R 4-5 for every Linear (`T6_NAX_MIN=6`) | fn-prof R 4 / 5 +0.5 / +0.7 ms; in-loop -0.5 / +0.3 | no |
| only the 2560 -> 640 Linears (shared-expert gate / up, indexer qk) SIMD up to 5 rows | fn-prof paired R 4 -0.42, R 5 -1.29 ms (both shared shapes: -0.21 / -1.02) | **yes (default)** |

Why the dedupe loses from R 4: one threadgroup per touched expert holds R prepared rows (5 KB each, threadgroup
memory 21-26 KB at R 4-5 -> fewer threadgroups resident on a read-bound kernel) and prepares / dots all R rows
(2/3 of the experts are used by one row only). Next: split the routed work by multiplicity (single-row experts on
the 1-row-prep kernel, multi-row experts on a 2-row kernel, one dispatch each, no barrier between) to get at the
0.83-1.87 expert-rows of 12.2.

### 12.4 Result (T6 pack, T 512)

| | R 2 | R 3 | R 4 | R 5 |
|---|---|---|---|---|
| verify ms (`ne_fn_mtp_prof`, sampled; 11.1 / master same day) | 15.86 / 15.85 | 19.45 / 19.54 | 22.49 / 22.57 | 25.27 / 25.37 |
| lab-verify | 15.92 | **19.26** | **22.06** | **24.78** |

(prof's verify tokens are synthetic and route nearly disjointly: the dedupe gain on real text shows in the loop:
d2 20.85 -> 20.53 ms per round, same tokens per round 2.54.) In-process tok/s (6 prompts x 1,024, sampled, one run,
median / mean): d2 113.5 / 113.0 vs 111.5 / 111.3 with the dedupe off; d3 110.7 / 109.7 vs 112.5 / 111.4. Depth 2
stays the default (d3 is level with d2 within the noise; the rows still cost ~2.8 ms each).

Same client (`session.sh ijah`, `--seed-mtp`, REPS 1, DREPS 3: 6 prompts x 3 seeds x 1,024 tokens, port 8143;
master = 29307ca's build in the main tree, lab = 8bbb98f; two master -> lab pairs back to back, 14:21-14:35):

| pair | master tok/s | lab tok/s | paired lab / master (95% CI), wins |
|---|---|---|---|
| 1 (14:21-14:28) | 108.3 +- 9.5 | 110.9 +- 9.7 | 1.024 [1.009, 1.041], 15 / 18 |
| 2 (14:29-14:35) | 108.3 +- 9.4 | 106.8 +- 9.7 | 0.986 [0.969, 1.003], 4 / 18 |
| both | | | **1.005**, 19 / 36 |

So through the server the change is within the session drift (+-2-5%): the ~0.3 ms per d2 round it saves is ~1.5% of
a round, and the dedupe's rounding differences give other sampled texts than master's (no tight pairing as in 11.5).

### 12.5 Correctness

- Dedupe vs per (row, slot) (`NE_PROF_DDCHECK`, random rows): 1 layer 41-94% of the outputs bit-identical, max
  |diff| 1-4e-6 at |h| 14-59 (f32 rounding of the same sums), growing through the layers as any two arithmetics do;
  greedy MTP vs the no-MTP step path (bench_draft `--greedy`, d2 / d3, 6 prompts x 512): every first mismatch at a 1-row
  logit gap of 0.000 / 0.125 (ties, as in 10.2 / 11.6).
- `test_sampled_dist.py --depth 2` (R 3: the dedupe path): out[1] TV 0.049 (noise 95% 0.062, chi2 5.9 / 5 ids),
  out[2] | ' Explain' 0.043 (0.092), | ' Assume' 0.022 (0.030): OK. `mtp-accept-test`: all passed.
- `make test` (with_lock, lab-verify build): ALL PASS (141 checks, the forward GPU layer goldens included).
- The dedupe default changes every R 2-3 step's MoE (the verify, the 2-row seed / flush steps, the head's 2-row seed),
  at f32 rounding level; `NE_FN_MOE_DEDUPE=0` restores master's arithmetic exactly.

### 12.6 Answer

The extra verify row is the routed experts (73%), and real text does share experts between consecutive tokens (28%
of the expert reads at 3 rows, 37% at 5), so the bytes are there to save; the existing one-threadgroup-per-expert
dedupe kernels recover only a fifth to a third of it at R 3 and lose from R 4 (threadgroup memory / per-row prep).
Kept: dedupe at R 2-3 (-0.3..-0.9 ms per d2 verify) and the 2560 -> 640 Linears on SIMD to 5 rows (-0.4 / -0.5 ms
at R 4 / 5 in prof). Depth 2 stays the default; depth 3 / the gate need the multiplicity-split kernel of 12.3 first.

### 12.7 Split by multiplicity (opt-in `NE_FN_MOE_SPLIT=1`, `ne_fn_mtp_set_moe_split`, bench_draft `@split=1`; negative)

The router's finisher also writes two lists after uq (`[ns, nm, singles: (row, slot) of experts used by one row,
multis: u of experts used by >= 2 rows]`); MoeGateUp then runs `fn_moe_gateup_s` (the (row, slot) kernel over the
singles) and `fn_moe_gateup_m` (the dedupe kernel over the multis, storing into the (row, slot) layout) concurrently,
and the down stays per (row, slot). T6 layers, R 2-5. Bit-identical to the per-pair path (`NE_PROF_DDCHECK=split`,
R 3 / 5, 1 / 4 / 48 layers: every output bit-identical). One A/B in the loop (sampled, 6 prompts x 1,024, verify
GPU ms per round, split off = the 12.4 defaults, i.e. dedupe at R 3, per pair at R 4-5):

| depth (rows) | off | split | tokens / round |
|---|---|---|---|
| d2 (R 3) | 20.46 | 21.74 (+1.28) | 2.51 / 2.54 |
| d3 (R 4) | 26.15 | 24.92 (-1.23) | 2.98 / 2.98 |
| d4 (R 5) | 29.51 | 28.84 (-0.67) | 3.34 / 3.34 |

It loses at R 3 (the d2 verify), and the R 4-5 gains are no bigger than this session's swings (the same off config
measured 24.1-26.2 ms at d3 today). Default off. Gate/up only: the down still reads every (row, slot) pair, and two
concurrent dispatches over one barrier did not hide the multi kernel's R-row prep. The router builds the lists only
when the split runs (a constant argument), so the default path does no extra work; `make test` ALL PASS after it.

## 13. DFlash2 block drafter (lab-dflash, 19 Oct; opt-in `--drafter dflash2`)

Inco's **Qwen3.8-Flash-Next-DFlash2** (huggingface incoai/Qwen3.8-Flash-Next-DFlash2 @ a22a1744e85c, Apache-2.0; 0.6B
bf16, data only, downloaded to `~/.cache/ijah/drafters/Qwen3.8-Flash-Next-DFlash2`) as an alternative drafter in the MTP loop.
**Result: same-client server (`--seed-mtp`, 6 prompts x 3 seeds x 1,024 tokens) 1.055-1.101x master's MTP d2 in three
pairs (CIs above 1, 13-17 / 18 wins; final code: 120.4 vs 112.4 / 108.9 tok/s); in-process +7.7-8.3% mean.** Only
text_crisis (low acceptance) loses (0.95-0.98). Default off.

### 13.1 What the drafter is (config + the SGLang capture patch + z-lab `dflash/model.py`, read not run)
- **Features**: for every target position, the *completed* outputs of layers **5, 15, 25, 35, 45** (0-based), i.e. the
  4-stream hyper-connection residual (pre-mixer) after the layer, **averaged over the 4 streams** (the patch's
  `hidden.unflatten(-1, (4, 2560)).mean(-2)`), concatenated -> 12,800. Checked on the engine: shifting every layer by
  +-1 drops 3 drafts' tokens / round from 2.95 to 2.86-2.87.
- **Context** row j = `hidden_norm(fc(f_j))`; per drafter layer its K / V = rope(k_norm(k_proj c_j)) / v_proj c_j (no
  input norm, no conv), at the target position j. Sliding window 2048 (non-causal: |q - k| < 2048).
- **Block**: rows `[anchor, mask (248077) x (B - 1)]` (anchor = the last emitted token, in no cache yet) at positions
  T .. T + B - 1, input = the target's embedding. 5 Qwen3 layers (32 / 8 heads x 128, rope theta 1e4), each with two
  **two-tap dynamic grouped convolutions**: `conv_s(x)[r] = sum_t x[r - t] (base[s][t] + dyn[r][s][t][c / 16])`,
  dyn = kernel_projection(norm(h)) [2 stages][2 taps][160 groups]: stage 0 before q / k / v (and gate / up), stage 1
  after o_proj (and down), then the residual. Attention over context + all block rows. Final norm, the **target's
  lm_head** on rows 1 .. B - 1, top 16 per row (unary = raw logits).
- **Selector**: from the anchor, position p scores its 16 candidates `unary_i + <pred_cb[prev] * W_sel hn_p,
  succ_cb[cand_i]>` (rank 256); greedy argmax, sampled `q = softmax(score / T)` over the 16 (no top-k / top-p): the
  draft is drawn from q and **q is the proposal** for exact acceptance (section 5.2). Trained block 8; the card serves 4.

### 13.2 Step A: acceptance before building (host-driven rounds; `tools/dflash_ref.py`, MLX reference drafter)
`ne_fn_dfl_verify` / `_commit` run real verifies (features captured) while the MLX drafter drafts; exact sampled
accept at T 1 / top-p 0.95 / top-k 20, 6 h2h prompts x 512 tokens, MTP from the loop on the same seeds:

| drafts verified / round | DFlash2 B 4 | DFlash2 B 8 | native MTP |
|---|---|---|---|
| 2 | | 2.47 tok / round | **2.50** (d2) |
| 3 | 2.95 | 2.91 | **2.98** (d3) |
| 4 / 5 | | 3.23 / 3.43 | |

So per verified draft DFlash2 is **not** better than our MTP here (the card's +0.2-0.4 tokens is against SGLang's MTP;
ours drafts from the head's shortlist with exact q, 10.3). Its edge is a free deep block: a **q-gated verify** (verify
the prefix k <= 5 that maximises E[tokens](k) / cost(k + 1), E from P(accepted | q(d) bin)) simulated exactly from
k = 5 rounds (2-fold held-out calibration): **+9-14% over MTP d2** at a 1.0-2.5 ms drafter, fixed k = 3 only +1-3%.

### 13.3 Engine implementation (`models/flash_next/dflash.{h,mm,metal}`, `mtp_loop.mm`)
- Weights: the checkpoint mapped (`Pack::open`), matrices quantised at load to 8-bit MLX affine g64 (CPU, 0.6 s;
  `NE`/server `--dflash-bits 4`: q / o / MLP at 4 bits — no faster, record_qn is ALU-bound), the codebooks read in
  place. GEMMs = `nax::record_q8` at <= 8 rows; small kernels: `dfl_rms`, `dfl_conv`, `dfl_qk_prep`, `dfl_attn` (8 KV
  heads x 16 key splits, online softmax, fixed-order `dfl_attn_reduce`), `dfl_select`. Readout = the loop's pruned
  4-bit draft head + `mtp_topk_local` / `mtp_shortlist` at K 16 (exact 8-bit re-score; the proposal is what is drawn,
  so exactness holds). **One block draft = 2.0-2.4 ms GPU** (B 8, incl. adding the last verify's rows to the context).
- Features: every verify (`route_cap` hook, `L.feat`), the 1-row / 4-row steps (`record_b` hook, `Model::dfl_f`,
  `set_dfl_capture`), chunked prefill (`get_rec` hook: each chunk's last <= 2048 rows into the ring `fs` by position;
  `prefill_set_dfl`).
- Context bookkeeping (`MtpLoop::Dfl`): rings (c bf16 [2048][2560], K / V bf16 per layer) hold positions [T - n, T);
  then `vcnt` committed rows of the last verify, then `fsn` feature rows (prefill / steps) — drained in 16-row groups
  (`dfl_drain`; 2048 rows ~15 ms) at a feed's end and a run's start. Validity like the head's (`gen`). A round uses
  the block drafter when the context reaches T with >= min(T, 512) rows (`NE_DFL_MIN_CTX`), else the MTP head (kept
  in step meanwhile like after copy rounds: its pairs queue as pending rows); the copy rule keeps priority.
- Round: draft recording (context add + block) -> **A (embedding + layer 0) committed right behind it at kmax + 1
  rows** (layer 0 is a causal GDN layer with lazy commit, so B at the gated R <= kmax + 1 reads its first R rows; this
  hid the host's gate round trip: +4% vs committing A after the gate) -> host reads drafts + q, gate picks k -> B at
  R = k + 1 -> accept (greedy, or `accept_sampled_p` with q). Greedy rounds gate on the selector's softmax confidence
  (drafts stay argmax). The gate's acceptance table starts from the priors every run and learns per run; round costs
  are fixed priors (learning them from wall time made runs not repeat).
- Checkpoints: `ne_fn_mtp_state_io` / snapshots carry header + the c ring (+10.5 MB); import rebuilds K / V from it
  (`record_rebuild`, ~10 ms); a stale / absent block restarts the context at T.
- API: `ne_fn_dfl_enable(p, dir, block, kmax, bits, gate)`, `_set(gate, kmax, active)`, `_on`, `_last`, checks
  `_ctx_reset` / `_ctx_push` / `_probe`, step A `_capture` / `_reset` / `_verify` / `_commit`. Server: `--drafter
  dflash2 [--drafter-path DIR] [--dflash-kmax 5] [--dflash-bits 8]` (the loop is created with max depth >= kmax).
  Env: `NE_DFL_EARLY_A=0`, `NE_DFL_KGREEDY`, `NE_DFL_MIN_CTX`, `NE_DFL_COST_EMA=1`.

### 13.4 Speed
In-process (`tools/bench_draft.py --dflash DIR --max-depth 5 --configs copy,copy@dfl=1`, configs alternated per
prompt — config-major order biased earlier A/Bs by the GPU's drift), sampled, 6 prompts x 1,024:

| run | MTP d2 (copy) mean / median | DFlash2 gated mean / median |
|---|---|---|
| A after the gate (first build) | 113.5 / 114.4 | 115.0 / 113.6 |
| A behind the draft, 8-bit | 109.8 / 110.2 | 117.7 / 123.7 |
| same, 2 seeds | 110.2 / 112.5 | **119.4 / 126.9** (9 / 12 wins) |
| final (per-run gate state, fixed costs) | 112.3 / 114.5 | **121.0 / 122.5** |
| fixed k = 3, no gate | | 100.4 (-12%: R 4 every round) |
| greedy (gated, 512 tokens) | 111.4 | 118.3 |

Per round (trace): drafting 2.2 ms + host gap 0.8-1.1 vs MTP 1.6 + 0.5; rounds by verified drafts d1..d5 cost 20.2 /
24.0 / 27.6 / 31.9 / 34.6 ms for 1.5 / 2.1 / 3.0 / 3.9 / 4.9 tokens (MTP d2: 22.0 ms, 2.47).
Same client (`session.sh ijah`, `--seed-mtp`, REPS 1, DREPS 3), paired by (prompt, seed):

| pair | base tok/s | DFlash2 tok/s | ratio (95% CI), wins |
|---|---|---|---|
| ABBA 1 (11:44-11:56) | 109.5 / 111.8 | 117.5 / 118.5 | 1.070 [1.038, 1.103] 14 / 18; 1.055 [1.022, 1.088] 14 / 18 |
| final code (12:20-12:35) | 112.4 / 108.9 | 120.4 | 1.067 [1.036, 1.097] 13 / 18; 1.101 [1.069, 1.132] 17 / 18 |

Per prompt (ABBA 1): code_parser +11%, math_proof +14%, text_trains +13%, code_lru +2%, text_climate +1%,
text_crisis -3%.

### 13.5 Correctness
- `tests/mtp/test_dflash.py` (engine vs the MLX reference on the same real-feature context, 3 prompts): final hidden
  rel L2 1.3-2.2% (8-bit weights), 14-16 / 16 candidates shared, logit error <= 0.21, **greedy drafts identical (7 / 7)**,
  q TV <= 0.009 (the reference's selector uses the bf16 projection): ALL PASS.
- `test_sampled_dist.py --dflash` (text_crisis, depth 2, 400 runs, first round a DFlash round): out[1] TV 0.059 (noise
  95% 0.062, P 0.06), out[2] | ' Explain' 0.063 (0.101), | ' Assume' 0.015 (0.031): OK.
- Greedy vs the no-MTP step path (`bench_draft --greedy`): every first mismatch at a 1-row logit gap 0.000 / 0.125.
- `test_mtp_cont.py --whole --dflash` (both prefill modes, d2 + adaptive): **all pass** (fresh repeats, snapshot /
  restore exact, live continuation, cut runs, state / KV park, lazy head); 4-layer model too. (The park check now
  hands the copy rule its token history back after the import, as the server does: without it a copy round of the
  reference text cannot fire after the park.)
- `mtp-accept-test`, server `run_mock.sh`: all pass; `make test` ALL PASS (141 checks).

### 13.6 Traps
- Gate state that persists across runs, or is learned from wall time, makes runs not repeat (different verify
  widths fork at bf16 ties): reset per run, fixed costs.
- The A-after-gate round trip cost ~0.5 ms per round (GPU idle while the host gates); A at kmax + 1 behind the
  draft fixed it — valid only because layer 0 is causal with lazy commit.
- A shared Prefill (lab-srvperf, base + Swift) writes the features of either model's prefill into the ring of the one
  that enabled DFlash: only acceptance can suffer (pending step rows), never output.
- R = 7 verify kernels were not instantiated (`fn_moe_router_r7`, ...), which is why kmax was capped at 5 (d5 rounds
  are the cheapest per token, 7.1 ms). They turn out to compile and link with no other change (19 Oct, +124 kernels,
  metallib 6.65 -> 6.98 MB): `kernels/fn_trunk.metal` + `kernels/fn_moe.metal` now instantiate R = 7, and the MTP GDN
  step (`mtp.metal` MTP_INST / MTP_INST_L) and QSA (`QSA_INST_R`) already had it. `fn_hc_pre_*_r7` fits the staging
  that forces r8 into 2 x 4 rows, so 7 rows stays one dispatch (`nax.mm` hc_pre check now allows R <= 7).
  **The 7-row round has never been run**: the whole-model GPU slot was held by the :8140 server all day.
  A missing pipeline at 7 rows fails loudly in `Pipelines::get`, not silently; the 6-row path proves the ple /
  dense-attn families (`fn_ple_mix_r6`, `fn_attn_core_r6` are not instantiated either) are not on the verify path.
- `--dflash DIR` is `tools/bench_draft.py`'s flag, **not** the server's: `ne_server.py` spells it `--drafter dflash2
  --drafter-path DIR`. argparse prefix-matches, so handing `--dflash` to the server is `error: ambiguous option:
  --dflash could match --dflash-kmax, --dflash-bits` and exits 2 in 0 s — before any model load, so the supervisor
  counts it as a crash and restarts it (19 Oct 13:39: 2 restarts before it was caught by hand).

### 13.7 What is left
- **Serving it as a model id (`Swift-1.5-Flash-Next-native-dflash`).** `server/models.json` maps the id to Swift-1.5's
  packs; the drafter itself is a start flag: `tools/serve.sh start -- --mtp 5 --drafter dflash2 --drafter-path DIR
  --dflash-kmax 5 --dflash-bits 8` (`--mtp` must be >= kmax, and kmax clamps to min(kmax, block - 1, the loop depth)).
  Verified CPU-only first: `--mock` with that exact flag set comes up (MTP depth 5, listening). The GPU start
  (`ne_fn_dfl_enable` on the 86 GB model) and the +6% at the server were the untested part at the time of writing;
  both have since been seen on a GPU: the drafter enabled in 1.3 s with `--mtp 5` not clamped, the served id answered
  requests, and 120.4 vs 112.4 tok/s measured through the server (13.4).
- **Serving flags and the id are one trap.** A launch script that passes a flag the server does not have exits 2 in 0 s,
  which a supervisor reads as a crash and burns its restarts on; argparse prefix-matching makes it worse, because
  `--dflash DIR` is accepted as ambiguous between `--dflash-kmax` and `--dflash-bits` long after the typo. A launcher
  should check the server it is about to start accepts the flags (`grep -q 'add_argument("--drafter"` is enough) and
  fail in the menu, not in the restart loop.
- **kmax 6-7: enabled in code, needs one measurement.** Everything above is committed; `mtp_prof`
  (`tools/bench_draft.py --prof`) now profiles a round at up to 7 rows, and the per-R verify table comes from
  `tools/bench_mtp.py --time`, which already loops R over `--max_depth + 2` (no change needed to price R = 7). To settle it on a free GPU slot:
  `tools/with_lock.sh "DFlash kmax 6-7 (~85 GB)" -- <tools/bench_draft.py --dflash DIR --max-depth 7
  --configs copy,copy@dfl=1 (kmax 5 / 6 / 7)>`, then the same client:
  `REPS=1 DREPS=3 tests/h2h/session.sh ijah --seed-mtp` with `--dflash-kmax 6` and `7`.
  Expect it to pay only on predictable text (text_crisis is already -3% at kmax 5); the gate has to be willing to
  verify 6-7 rows, and A runs at kmax + 1 = 8 rows (the 2 x 4 split path).
- a cheaper draft (2.2 ms: ~110 dispatches; fused norm + conv); per-request choice
  between the block drafter and the head on low-acceptance text (text_crisis); a better gate predictor.

### 13.8 Attribution
Drafter weights: Inco AI, DFlash 2 (inco.ai/blog/dflash2), Apache-2.0. Architecture / generation loop: z-lab/dflash
(`dflash/model.py`, @07ebd93; DFlash: Block Diffusion for Flash Speculative Decoding, Chen, Liang, Liu, ICML 2026);
read, nothing run or copied. Splash's DFlash2 port for the dense 27B (splash-m5 `runtime/model/DFlashDraft.cpp`,
`DraftCheckpoint.cpp`, `runtime/metal/kernels/decode/draft.metal`, `sampling.metal` draft_select_*; this project's own
work) fixed the reading of the convolution layout, the context K / V path and the selector; the kernels here are
written for this engine's layouts.
