// Flash-Next MTP draft head on the engine (lab-mtp). Math, design, verify / accept and rollback: docs/mtp.md.
//
// One head step for R rows (R <= 8), recorded through the core Recorder:
//   tok u32 [R], hid f32 [R][4*2560] (the target's PRE-mixer streams, or the previous draft step's hc_out)
//   -> mtp_fuse_pre: e = embed(tok) (8-bit, target's), en = rmsnorm(e) * w_e, hn = rmsnorm_10240(hid) * w_h
//   -> NAX fc_embedding(en) = pe;  NAX fc_hidden(hn as [4R][2560]) + pe (row bias) = h_in [R][4][2560]
//   -> the head's decoder layer (QSA attention + MoE K4, own HC read/writes; nax::record_layer) -> h_out
//   -> mixer (HC read without inject) -> NAX lm_head (target's, 8-bit) -> logits [R][248320] -> argmax [R]
// and pos[0] += R on the GPU (so chained steps replay in one command buffer).
#pragma once
#include <unordered_map>
#include <vector>

#include "kernels/nax/nax.h"

namespace ne::mtp {

struct MtpW {
  fn::Q8 fc_e, fc_h, embed, lm_head;
  Tensor norm_e, norm_h;  // pre_fc_norm_embedding [2560], pre_fc_norm_hidden [10240] (bf16, ones-centred: as stored)
  fn::LayerW layer;       // the head's QSA attention + MoE layer (experts K4 on the K5-a16 pack)
  fn::HcW mixer;          // its hyper_connection_mixer (inject = the layer's attn inject weight, unused)
  int vocab = 0;
  // Optional draft-only coarse head (set_draft_head): lm_head requantised to draft_bits (4 or 2, g64); drafts take
  // the argmax among its top `shortlist` ids re-scored exactly through lm_head. 0 = drafts read the full head.
  fn::Q8 draft_head;
  int draft_bits = 0, shortlist = 8;
  // lab-draft: pruned draft vocabulary (set_draft_head's `rows`): draft_head holds only these lm_head rows (ascending
  // ids, so ties still resolve to the lowest id); draft_ids u32 [draft_head.rows] maps a compact row to its token id
  // (empty = the full vocabulary, rows = ids). Verification always reads the full head.
  Tensor draft_ids;
};
MtpW load_mtp(const Pack &pack);  // language_model.mtp.* + the target's embed_tokens / lm_head
// Requantise an 8-bit affine matrix to `bits` (4 or 2) affine, group `group` (min/max per group, bf16 scale/bias;
// CPU, parallel; ~0.36 GB for the 4-bit lm_head). Layout = MLX's (32 / bits values per u32).
// rows (optional, ascending): requantise only these rows of src (row i of the result = src row rows[i]).
fn::Q8 requantize(const fn::Q8 &src, int bits, int group = 64, const std::vector<uint32_t> *rows = nullptr);
// rows: the draft vocabulary (nullptr / empty = all ids); see MtpW::draft_ids. draft_vocab_rows() parses
// NE_MTP_DRAFT_VOCAB: "full" / "0" = all ids; unset / "1" / "on" = the Splish set (models/flash_next/draft_vocab.h:
// ids < 98,304 + the specials + 16,384 frequent ids, 114,721 rows); "prefix" = ids < 98,304; a number N = ids < N.
void set_draft_head(MtpW &w, int bits, int shortlist = 8, const std::vector<uint32_t> *rows = nullptr);
std::vector<uint32_t> draft_vocab_rows(int vocab);
// The draft-head readout alone (coarse GEMM, 2-stage top-K, exact re-score) on x f32 [R][2560] -> amax[at ..].
void record_draft_readout(Recorder &r, const MtpW &w, int R, const Tensor &x, const Tensor &coarse, const Tensor &cand,
                          const Tensor &amax, const Tensor &sl, uint32_t at);

// Per head: its KV state + scratch for `rows` rows per step.
struct MtpHead {
  int rows = 0;
  fn::GdnState st;        // KV cache of the head's attention layer (pos[0] = tokens cached)
  fn::Scratch s;
  nax::LayerExtra ex;
  Tensor tok;             // u32 [R]
  Tensor hid;             // f32 [R][10240]
  Tensor en, pe, hn;      // f32 [R][2560], [R][2560], [R][10240]
  Tensor h_in, h_out;     // f32 [R][4][2560]
  Tensor h_out2;          // ping-pong for chains
  Tensor xh;              // mixer output f32 [R][2560]
  Tensor logits;          // f32 [R][vocab] (full head; with a draft head: its coarse logits, masked by the shortlist)
  Tensor amax;            // u32 [max(R, 16)] argmax per row (chains: draft k at amax[k])
  Tensor sl;              // f32 [R][32][2]: the shortlist's (id bits, exact logit) per row (draft head only)
  Tensor cand;            // f32 [R][128][32][2]: stage-1 candidates of the shortlist
};
MtpHead alloc_head(const MtpW &w, int rows, int tmax = 4096);

// One head step over `head.rows` rows: inputs head.tok (or `tok` when given: e.g. the previous step's argmax),
// head.hid (or `hid`). Writes h_out (or `h_out`), logits, amax[amax_at ...]. Barriers inside; ends with one.
void record_step(Recorder &r, const MtpW &w, MtpHead &head, const Tensor *tok = nullptr, const Tensor *hid = nullptr,
                 const Tensor *h_out = nullptr, uint32_t amax_at = 0, bool readout = true);
// The head's KV rows only (lab-headlazy; head prefill / catch-up): fuse, fc_embedding, fc_hidden, the layer up to
// lab-attn's prep dispatch (nax::record_layer kv_only), pos += rows. The cache rows written are bit-identical to
// record_step's (the same dispatches up to there); no attention core, MoE, mixer or readout (h_out not written).
void record_kv_step(Recorder &r, const MtpW &w, MtpHead &head, const Tensor &tok, const Tensor &hid);
// Head prefill over n (hidden_j, token_{j+1}) pairs already in head.hid / head.tok (n <= head.rows per call; call
// in chunks): KV only, no mixer / lm_head (readout = false) except `readout_last` on the final chunk's rows.
void record_prefill_chunk(Recorder &r, const MtpW &w, MtpHead &head, bool readout_last);
// Greedy draft chain of depth d (rows == 1): step 0 from head.tok / head.hid, step k from step k-1's argmax and
// hc_out; drafts in head.amax[0..d). One recording, no host round trip.
void record_chain(Recorder &r, const MtpW &w, MtpHead &head, int depth);

// ------------------------------------------------------------------------------------------------ verify / accept
// Host side, plain C++ (no GPU): lab-forward's token loop calls these after a verify forward over the D + 1 rows
// [t1, d_1 .. d_D] (t1 = the last committed token, not yet in any cache; row i's logits predict the token after it).
struct SampleParams {  // the target's sampler; drafts are scored under the same filter
  float temperature = 0.0f;  // < 0.01 = greedy (argmax), as Sushi's isGreedyTemperature
  int top_k = 0;             // 0 = off
  float top_p = 1.0f;
  float min_p = 0.0f;        // lab-sampler: tokens under min_p * p_max (after the temperature softmax over the top-k) are dropped, before top-p
};
bool is_greedy(const SampleParams &sp);
// Filtered, renormalised distribution of one logits row (temperature, top-k, top-p, in that order; ids sorted by
// descending probability). Dense rows are never materialised beyond the kept ids.
struct Dist {
  std::vector<int> ids;
  std::vector<float> p;
  float prob(int id) const;           // 0 if filtered out
  int sample(double u) const;         // inverse CDF, u in [0, 1)
};
Dist make_dist(const float *logits, int vocab, const SampleParams &sp);
struct Rng {  // splitmix64 -> uniform [0, 1); deterministic per seed
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed) {}
  double uniform();
};
struct VerifyResult {
  int accepted = 0;          // drafts accepted, 0..D
  int next_token = -1;       // the correction (first rejection) or the bonus (all accepted)
  std::vector<int> emitted;  // d_1 .. d_accepted, next_token  (accepted + 1 tokens)
};
// Greedy: accept d_i while d_i == argmax(row i-1); next = argmax(row accepted). target_argmax [D + 1] (a GPU argmax,
// e.g. mtp_argmax on the verify logits, so only D + 1 ids cross to the host).
VerifyResult accept_greedy(const int *drafts, int D, const int *target_argmax);
// Sampled, exact (Leviathan / Chen speculative sampling = Sushi's `exact` mode): accept d_i with probability
// min(1, p_i(d_i) / q_i(d_i)); on the first rejection draw the correction from norm(max(p_i - q_i, 0)); all accepted:
// the bonus is drawn from p_D. q_i = the draft's filtered distribution (draft_logits [D][V]) or, when the drafts were
// argmax-drafted (draft_logits == nullptr: a point mass), accept with probability p_i(d_i) and draw the correction
// from p_i with d_i removed. The emitted tokens are distributed exactly as serial sampling from the target.
VerifyResult accept_sampled(const int *drafts, int D, const float *target_logits, const float *draft_logits, int V,
                            const SampleParams &sp, Rng &rng);
// Same with the proposals given as distributions (q[i] = what d_{i+1} was drawn from, e.g. shortlist_dist of the
// draft readout's K exact logits); q == nullptr: argmax drafts (point masses).
VerifyResult accept_sampled_q(const int *drafts, int D, const float *target_logits, const Dist *q, int V,
                              const SampleParams &sp, Rng &rng);
// Same with the target distributions given (p[0 .. D]: row i's filtered distribution, e.g. shortlist_dist of the
// verify's top-K row when K >= top_k: the token loop reads only K (id, logit) pairs per row from the GPU).
VerifyResult accept_sampled_p(const int *drafts, int D, const Dist *p, const Dist *q, Rng &rng);
// The draft readout's shortlist row (MtpHead::sl: K (id bits, exact logit) pairs) as a proposal distribution under the
// target's sampler (temperature / top-k / top-p over the K ids; greedy = the best id).
Dist shortlist_dist(const float *sl_row, int K, const SampleParams &sp);

// What to keep after a verify of D + 1 rows from cache length T0 with `accepted` drafts accepted (docs/mtp.md):
struct CommitPlan {
  int keep_rows = 0;     // verify rows whose state is committed: accepted + 1 (t1 and the accepted drafts)
  int kv_len = 0;        // every attention layer's KV (and QSA block keys) truncated to T0 + keep_rows (lab-attn)
  int gdn_row = 0;       // GDN layers: the recurrent / conv state after verify row gdn_row (= keep_rows - 1)
  int head_trim = 0;     // MTP head cache: rows appended while drafting that are dropped (mlx-vlm semantics)
  int head_keep = 0;     // ... and kept (drafted with the head's own hidden, accepted: not recomputed)
  int head_rows = 0;     // head rows to run next from the verify hidden: drafts [head_keep, accepted) + the new token
};
CommitPlan plan_commit(int T0, int D, int accepted, int head_appended);

// Adaptive depth: per-position acceptance EMAs + a verify cost table (ms per round at depth d: draft steps + the
// (d + 1)-row verify); picks d maximising expected tokens per ms. observe() after every round.
struct DepthPolicy {
  int min_depth = 1, max_depth = 6;
  double ema = 0.1;
  std::vector<double> acc;   // acc[i] = P(draft i+1 accepted | drafts 1..i accepted)
  std::vector<double> cost;  // cost[d] = ms of a round at depth d (cost[0] = a plain decode step)
  explicit DepthPolicy(std::vector<double> round_cost_ms, double prior_accept = 0.8);
  double expected_tokens(int d) const;  // 1 + sum_k prod_{i<k} acc[i]
  int next_depth() const;
  void observe(int drafted, int accepted, double round_ms = -1.0);
};

// ------------------------------------------------------------------------------------------------ copy rule
// lab-draft (TensorFold's copy rule, as Splish's SPLASH_M5_COPY_MIN_MATCH; docs/mtp.md 11): S = ctx + [t1] (t1 = the
// last emitted token, not in ctx yet). If S's last N tokens occurred earlier in S (a window ending at e < |S|; the most
// recent occurrence), out = S[e, e + max) clipped to |S|: the earlier continuation, i.e. the drafts for the tokens after
// t1. Returns its length (0 = no match, or N <= 0). idx is updated incrementally (ctx may only grow between calls;
// clear() it when ctx is replaced). FNV-1a window hashes, every match verified token by token.
struct CopyIndex {
  std::unordered_map<uint64_t, int> map;  // window hash -> e (the index of the continuation's first token in S)
  int end = 0;                            // windows ending at e < end are indexed
  void clear() { map.clear(); end = 0; }
};
int copy_find(const std::vector<int> &ctx, CopyIndex &idx, int N, int t1, int max, int *out);

}  // namespace ne::mtp
