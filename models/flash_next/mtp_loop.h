// Flash-Next MTP token loop (lab-loop): the MTP head drafts D tokens, one (D + 1)-row verify forward of the whole
// model checks them, the accept (greedy or exact sampled) commits KV / GDN / PLE state lazily, next round.
// Design, numbers and traps: docs/mtp.md section 10.
//
// One round (D = depth; R = D + 1 verify rows [t1, d_1 .. d_D]; t1 = the last emitted token, in no cache yet):
//   draft  (1 command buffer): head seed step over n = 1 or 2 rows (the target's pre-mixer hidden of the last round's
//          kept verify rows + the tokens after them; mlx-vlm's accept_verified_tokens) -> d_1, then D - 1 chained
//          1-row head steps -> d_2 .. d_D, copied on the GPU into the verify token buffer behind t1.
//   A_R    embedding of the R tokens + layer 0 (lazy GDN commit), committed behind the draft (no host round trip).
//   host   waits for the draft only, reads the D drafts, gathers the R n-gram rows (A_R runs meanwhile), commits B_R.
//   B_R    PLE (lazy conv commit) + layers 1..47 (GDN layers lazy, attention layers QSA at pos = T) + mixer + lm_head
//          at R rows + per-row argmax (greedy) or top-K (sampled; K = top_k <= 32).
//   host   reads R ids (greedy) or R x K (id, logit) pairs (sampled), accepts a drafts, emits d_1..d_a + the
//          correction / bonus, sets keep[0] = a + 1 (the GDN / PLE layers commit those rows of this round at the
//          start of the next one), pos = T + a + 1 (attention KV truncation), the head's KV length and inputs.
#pragma once
#include "models/flash_next/model.h"
#include "models/flash_next/mtp.h"

namespace ne::fn {

struct MtpLoop;
struct MtpLoopDeleter { void operator()(MtpLoop *p) const; };
using MtpLoopPtr = std::unique_ptr<MtpLoop, MtpLoopDeleter>;

// Head weights + draft readout (draft_bits 4 / 2 = requantised lm_head copy with an exact top-`shortlist` re-score;
// 0 = drafts read the full 8-bit head), verify buffers for 2 .. max_depth + 1 rows (recordings built on first use).
MtpLoopPtr mtp_create(Model &m, int max_depth = 4, int draft_bits = 4, int shortlist = 8);

struct MtpStats {
  double prefill_ms = 0, head_prefill_ms = 0, decode_ms = 0;  // wall
  int tokens = 0, rounds = 0;                                 // tokens emitted by rounds (after the prefill's one)
  double draft_gpu_ms = 0, verify_gpu_ms = 0;                 // sums over rounds (A + B)
  double gather_us = 0, host_us = 0;                          // sums: n-gram gather, accept + bookkeeping
  double flush_ms = 0;  // wall: the end-of-run commit of the last round's lazy rows (one 2-row verify)
  int copy_rounds = 0, copy_tokens = 0;  // lab-draft: rounds drafted by the copy rule, tokens they emitted
  int dfl_rounds = 0, dfl_tokens = 0;    // lab-dflash: rounds drafted by the DFlash2 block drafter, tokens emitted
  std::vector<int> hist;  // [(max_depth + 1) * (max_depth + 1)]: rounds at depth d with a accepted (d * (M+1) + a)
};
// Prompt (prefill()'s policy: chunked from 16 tokens; layer 47's hidden of every position feeds the head's prefill)
// then rounds until max_new tokens (the first one = the prefill's sample) or `stop`. depth 1..max_depth fixed, 0 = adaptive (mtp::DepthPolicy). The
// sampler is the model's (set_sampler): greedy when temp < 0.01 or top_k == 1, else exact speculative sampling with
// argmax drafts (point-mass proposals), the target's temperature -> top-k -> top-p.
// Leaves a clean state = prompt + out[0, k - 1) (the last emitted token is in no cache; a round cut by `stop` /
// max_new commits only up to it) with the head in step: plain steps, snapshots and mtp_generate_from continue it.
// cb (optional; lab-server's streaming / abort): called with every newly emitted run of tokens (the prefill's first
// token, then each round's accepted drafts + correction / bonus, as written to out); a nonzero return ends the run
// after that round (stop strings, a second EOS, client disconnect).
using MtpTokenCb = int (*)(void *ctx, const int *tokens, int n);
int mtp_generate(MtpLoop &L, const int *prompt, int n, int max_new, int stop, int depth, uint64_t seed, int *out,
                 MtpStats *st, MtpTokenCb cb = nullptr, void *cb_ctx = nullptr);

// ---- MTP from the model's current state (docs/mtp.md 10.9; lab-server's prefix reuse)
// The loop keeps the head's state in step with the model through its own calls: head KV rows [0, Lh) for the pairs
// (H_j, x_{j+1}) and the hidden rows of positions [Lh, T) pending (H = layer 47's pre-mixer hidden). Any plain
// step / prefill / restore / reset of the Model since (Model::gen changed) makes it stale, except T = 0 (empty).
bool mtp_head_ok(MtpLoop &L);
// toks[0, n) at the current position, no sample (prefill()'s policy, all n tokens chunked from 16), head kept in step
// ahead (lab-pfpath; may be NULL): the next feed's first tokens, whose n-gram rows are read during this feed's last
// chunk (prefill_tokens' ahead)
void mtp_feed(MtpLoop &L, const int *toks, int n, const int *ahead = nullptr, int n_ahead = 0);
// one decode step (step(m, tok, head)) with the head kept in step. Lazy head (lab-headlazy): the step only queues
// its hidden row (40 KB host copy); the head's KV catches up in KV-only 8-row steps once pend_cap rows are pending
// (default 1024, NE_MTP_PEND_CAP; <= 8 = eager), before an MTP run and before a snapshot.
StepOut mtp_step(MtpLoop &L, int tok, bool head = true);
// run the head over every whole 8-row group of pending rows now (<= 8 stay pending); returns the wall ms
double mtp_catch_up(MtpLoop &L);
void mtp_set_pend_cap(MtpLoop &L, int cap);  // drains at once if that many rows are already pending
// mtp_generate continuing from the current state: toks[0, n) (n >= 1) are prefilled at m.T (the head's pending rows
// first), then the rounds. Same result as mtp_generate over (the cached tokens + toks) up to prefill reduction order.
int mtp_generate_from(MtpLoop &L, const int *toks, int n, int max_new, int stop, int depth, uint64_t seed, int *out,
                      MtpStats *st, MtpTokenCb cb = nullptr, void *cb_ctx = nullptr);

// Check: prompt prefill, then the verify at R rows over the given continuation, keeping keeps[round % nk] rows per
// round (rollback of the rest, as a rejection would): logits f32 [nc][248320] of every kept row (row j of a round
// predicts the token after cont[i + j]). Returns the number of continuation tokens covered. Leaves the model dirty.
int mtp_verify_forced(MtpLoop &L, const int *prompt, int n, const int *cont, int nc, int R, const int *keeps, int nk,
                      float *logits);
// GPU ms (median of reps) of the R-row verify (A + B, greedy readout) at context T with P pending lazy rows, and of
// the 1-row-seed draft recording of depth R - 1. Positions faked (KV past T is whatever is there); leaves the model reset.
void mtp_time(MtpLoop &L, int R, int T, int P, int reps, double *verify_ms, double *draft_ms);
}  // namespace ne::fn

extern "C" {
typedef struct ne_fn_model ne_fn_model;
typedef struct ne_fn_mtp ne_fn_mtp;
NE_API ne_fn_mtp *ne_fn_mtp_create(ne_fn_model *m, int max_depth, int draft_bits);
NE_API void ne_fn_mtp_free(ne_fn_mtp *p);
NE_API int ne_fn_mtp_time(ne_fn_mtp *p, int R, int T, int P, int reps, double *verify_ms, double *draft_ms);
// lab-draft: GPU ms medians of a round's parts at head / target position T (each its own command buffer): [0] head step
// (1 row, the draft readout), [1] the same without readout, [2] KV-only step, [3] coarse draft-head GEMM, [4] its top-K
// stage 1, [5] shortlist (merge + exact re-score), [6] draft sample, [7] full 8-bit lm_head + argmax (1 row), [8] 2-row
// seed step, [9..12] sampled draft recordings seed 1 row depth 1..4, [13..16] seed 2 rows, [17..20] sampled verify
// (A + B with top-K) R 2..5, [21..24] greedy verify R 2..5. Returns the count (25); leaves the model reset.
// lab-draft: switch the copy rule (min_match 0 = off; depth = drafts per copy round, 0 = max_depth) and the draft
// vocabulary (NE_MTP_DRAFT_VOCAB's values; re-records the drafts, resets the model) at run time (A/B benches)
NE_API int ne_fn_mtp_set_copy(ne_fn_mtp *p, int min_match, int depth);
NE_API int ne_fn_mtp_set_draft_vocab(ne_fn_mtp *p, const char *mode);
// lab-draft: the gated verify (NE_MTP_QGATE; sampled drafts at a fixed depth: verify the drafts' best prefix), 0 / 1
NE_API int ne_fn_mtp_set_qgate(ne_fn_mtp *p, int on);
// lab-dflash step A (docs/mtp.md 13): host-driven rounds for an external drafter. capture = the DFlash2 target
// features (4-stream mean of layers 5 / 15 / 25 / 35 / 45) in every verify (re-records); reset = empty model;
// verify R rows at T: topk f32 [R][20][2] (id bits, logit), feat f32 [R][12800], gpu ms; commit c rows of it.
NE_API int ne_fn_dfl_capture(ne_fn_mtp *p, int on);
NE_API int ne_fn_dfl_reset(ne_fn_mtp *p);
NE_API int ne_fn_dfl_verify(ne_fn_mtp *p, const int *toks, int R, float *topk, float *feat, double *gpu_ms);
NE_API int ne_fn_dfl_commit(ne_fn_mtp *p, const int *toks, int c);
// lab-dflash step B (docs/mtp.md 13): the DFlash2 block drafter in the MTP loop (default off). enable: dir = the
// checkpoint (config.json + model.safetensors), block rows B (8 = the trained block; 4 = the card's serving size),
// kmax = most drafts a round verifies (<= B - 1 and <= the loop's max depth), drafter weights at bits 8 / 4, gate 1 =
// sampled rounds verify the prefix 1 .. kmax that the drafts' proposal probabilities say pays for its verify rows.
// Rounds then draft with the block drafter whenever its context reaches the current position (fresh prompts, feeds,
// steps, MTP runs keep it in step), else with the MTP head; the copy rule keeps priority. set: gate (-1 = keep),
// kmax (0 = keep), active (-1 = keep; 0 = no block drafts and no context upkeep, for A/B). last: the last run's DFlash rounds / tokens.
NE_API int ne_fn_dfl_enable(ne_fn_mtp *p, const char *dir, int block, int kmax, int bits, int gate);
NE_API int ne_fn_dfl_set(ne_fn_mtp *p, int gate, int kmax, int active);  // -1 / 0 = keep
NE_API int ne_fn_dfl_on(ne_fn_mtp *p);
NE_API int ne_fn_dfl_last(ne_fn_mtp *p, int *rounds, int *tokens);
// checks: the drafter's context from host feature rows f32 [n][12800] at positions [pos, pos + n); one block draft at
// the context's end with its internals (hn [B - 1][2560], sl / qo [B - 1][16][2], drafts [B - 1], GPU ms).
NE_API int ne_fn_dfl_ctx_reset(ne_fn_mtp *p, int T);
NE_API int ne_fn_dfl_ctx_push(ne_fn_mtp *p, int pos, const float *feat, int n);
NE_API int ne_fn_dfl_probe(ne_fn_mtp *p, int anchor, int sampled, float temp, uint32_t seed, float *hn, float *sl,
                           int *drafts, float *qo, double *ms);
// lab-verify: routed experts read once per verify for all rows routed to them (1 = at every R, N >= 2 = at R <= N)
// or per (row, slot) pair (0) = g_moe_dedupe; re-records the verifies.
NE_API int ne_fn_mtp_set_moe_dedupe(ne_fn_mtp *p, int on);
// lab-verify: g_moe_split (NE_FN_MOE_SPLIT) on / off; re-records the verifies.
NE_API int ne_fn_mtp_set_moe_split(ne_fn_mtp *p, int on);
NE_API int ne_fn_mtp_prof(ne_fn_mtp *p, int T, int reps, double *out, int nout);
NE_API int ne_fn_mtp_verify_forced(ne_fn_mtp *p, const int *prompt, int n, const int *cont, int nc, int R,
                                   const int *keeps, int nk, float *logits);
// Returns the number of tokens written to out (max_new entries) or -1. stats (may be NULL, 10 doubles): prefill ms,
// head prefill ms, decode ms, tokens (after the first), rounds, draft GPU ms / round, verify GPU ms / round, gather
// us / round, host us / round, max_depth. hist (may be NULL): (max_depth + 1)^2 ints, rounds by (depth, accepted).
NE_API int ne_fn_mtp_generate(ne_fn_mtp *p, const int *prompt, int n, int max_new, int stop, int depth,
                              uint64_t seed, int *out, double *stats, int *hist);
// Same with a token callback (mtp_generate's cb: tokens emitted since the last call; nonzero = stop after this round).
// ---- MTP from the model's current state (mtp_loop.h ne::fn::mtp_generate_from; docs/mtp.md 10.9). All return 0 /
// a count, or -1 (error text: ne_last_error) -- e.g. when the head is stale (a plain ne_fn_model_* call changed the
// state since the last MTP call): reset, or restore an ne_fn_mtp snapshot.
// 1 = the head is in step with the model (MTP can continue from it), 0 = not.
NE_API int ne_fn_mtp_head_ok(ne_fn_mtp *p);
// Lh = head KV rows, npend = pending positions (Lh + npend == T when in step); either may be NULL.
NE_API int ne_fn_mtp_head_state(ne_fn_mtp *p, int *Lh, int *npend);
// toks at the current position, no sample; the head kept in step.
NE_API int ne_fn_mtp_feed(ne_fn_mtp *p, const int *toks, int n);
// lab-pfpath: requestResidency of the loop's recordings (verify, drafts, head steps): the idle keep-warm, with
// ne_fn_gpu_ping on the model
NE_API int ne_fn_mtp_touch(ne_fn_mtp *p);
// the same, with the next feed's first tokens (lab-pfpath: their n-gram rows are read in this feed's GPU shadow)
NE_API int ne_fn_mtp_feed_ahead(ne_fn_mtp *p, const int *toks, int n, const int *ahead, int n_ahead);
// lab-headlazy: catch the lazy head up now (wall ms in *ms, may be NULL); pending-row cap (<= 8 = eager head).
NE_API int ne_fn_mtp_catch_up(ne_fn_mtp *p, double *ms);
NE_API int ne_fn_mtp_set_pend_cap(ne_fn_mtp *p, int cap);
// One decode step with the head kept in step (= ne_fn_model_step(m, tok, 1, ...) with next / logprob).
NE_API int ne_fn_mtp_step(ne_fn_mtp *p, int tok, int *next, float *logprob);
// ne_fn_mtp_generate_cb continuing from the current state over the new tokens toks[0, n), n >= 1.
NE_API int ne_fn_mtp_generate_from(ne_fn_mtp *p, const int *toks, int n, int max_new, int stop, int depth,
                                   uint64_t seed, int *out, double *stats, int *hist,
                                   int (*cb)(void *, const int *, int), void *cb_ctx);
// Snapshot = ne_fn_model_snapshot (113 MB) + the head state (Lh, pending rows: 40 KB each, <= 9: a lazy head
// catches up first). Same validity rule
// as the model's (KV rows [0, T) of the target and [0, Lh) of the head must still hold its tokens). A snapshot taken
// while the head was stale restores the model only (head stale after it).
// lab-draft, the copy rule's token history (NE_MTP_COPY_MIN_MATCH, docs/mtp.md 11): the loop records every token it
// feeds / steps / emits; after a state import (ne_fn_mtp_state_io) or anything else it did not see, it knows none of
// the state's tokens. context_len = the tokens it knows ending at T (0 = none); set_context gives it the last n tokens
// of the state (positions [T - n, T)).
NE_API int ne_fn_mtp_context_len(ne_fn_mtp *p);
NE_API int ne_fn_mtp_set_context(ne_fn_mtp *p, const int *toks, int n);
typedef struct ne_fn_mtp_snap ne_fn_mtp_snap;
NE_API ne_fn_mtp_snap *ne_fn_mtp_snapshot(ne_fn_mtp *p);
NE_API int ne_fn_mtp_restore(ne_fn_mtp *p, const ne_fn_mtp_snap *s);
NE_API int ne_fn_mtp_snap_T(const ne_fn_mtp_snap *s);
NE_API int64_t ne_fn_mtp_snap_bytes(const ne_fn_mtp_snap *s);
NE_API void ne_fn_mtp_snap_free(ne_fn_mtp_snap *s);
// lab-srvcache: the same as host buffers (the server parks conversations in RAM / on disk): the head's KV rows
// [a, b) <-> buf for `rows` rows (ne_fn_mtp_kv_bytes; ne_fn_kv_io's layout, one attention layer), and the model's
// recurrent state + the head's bookkeeping (Lh, pending rows) <-> buf of ne_fn_mtp_state_bytes() with meta int64
// [6] = T, hist0, hist1, head in step (0/1), Lh, pending rows. Import restores both (the head stays stale if it was
// stale at export). KV rows are not touched by the state calls: copy both models' rows with the kv calls. Export
// catches a lazy head up first (lab-headlazy; GPU work: export the state before copying the head's KV rows [0, Lh)).
NE_API int64_t ne_fn_mtp_kv_bytes(ne_fn_mtp *p, int rows);
NE_API int ne_fn_mtp_kv_io(ne_fn_mtp *p, int to_model, int a, int b, void *buf, int rows);
NE_API int64_t ne_fn_mtp_state_bytes(ne_fn_mtp *p);
NE_API int ne_fn_mtp_state_io(ne_fn_mtp *p, int to_model, void *buf, int64_t nbytes, int64_t *meta);
NE_API int ne_fn_mtp_generate_cb(ne_fn_mtp *p, const int *prompt, int n, int max_new, int stop, int depth,
                                 uint64_t seed, int *out, double *stats, int *hist, int (*cb)(void *, const int *, int),
                                 void *cb_ctx);
}
