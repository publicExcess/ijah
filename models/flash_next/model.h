// Flash-Next whole model (lab-forward): embedding -> 48 layers (PLE on layer 1) -> final hyper-connection mixer ->
// 8-bit lm_head -> GPU sampler, one recording per decode step. Pipeline, buffers and numbers: docs/forward.md.
//
//   token (u32 GPU buffer; the sampler writes the next one there) -> fn_embed -> h0 [4][2560]
//   layer l: h[l % 2] -> h[(l + 1) % 2] (record_gdn_layer; layer 1 first adds PLE from the host-gathered n-gram rows)
//   mixer (fn_hc_pre + fn_hc_mix with the hyper_connection_mixer weights, no inject) -> x [2560]
//   fn_lm_head -> logits [248320] (bf16 values) -> fn_sample_p1 / p2 -> token, logprob (head skipped on prefill
//   tokens through zero indirect threadgroup counts)
// Positions: one shared pos buffer for the 12 attention layers (host-written before each replay). Rope native.
#pragma once
#include "models/flash_next/layer.h"

namespace ne::fn {

constexpr int VOCAB = 248320;
constexpr uint32_t kSampleMaxK = 64;

// lab-prefill's chunked prefill (prefill.h includes this header): owned by the Model, created on first use
struct Prefill;
struct PrefillDeleter { void operator()(Prefill *p) const; };

struct SampleParams {  // = fn_model.metal SampleParams
  float temp = 0.0f, top_p = 1.0f;
  uint32_t top_k = 1, seed = 0, step = 0, nv = VOCAB, mask_on = 0;  // mask_on: fn_sample_p1 applies Model::mask
  // lab-sampler: min_p, penalties (Model::cnt), logit_bias (Model::bias), candidate list for logprobs (Model::top)
  float min_p = 0.0f, presence = 0.0f, frequency = 0.0f, repetition = 1.0f;
  uint32_t bias_on = 0, pen_on = 0, cand_k = 0, lp_on = 0;
};

struct Model {
  // counts live models (ne_fn_set_ref_rounding refuses a mode change while any lives)
  struct Live {
    Live() { ++g_models_live; }
    ~Live() { --g_models_live; }
    Live(const Live &) = delete;
    Live &operator=(const Live &) = delete;
  } live;
  bool ref_round = g_ref_round;  // the reference-rounding mode every recording of this model uses (fixed at load)
  // keeps a C-API pack (ne_pack_open) alive while the model lives (ne_pack_close before ne_fn_model_free is fine);
  // empty for a pack the C++ caller owns. First member after the counters: destroyed after everything using it.
  std::shared_ptr<const Pack> pack_hold;
  const Pack *pack = nullptr;
  int tmax = 0, rows = 1;
  int64_t eos = 248044;
  std::vector<LayerW> L;
  std::vector<GdnState> st;
  Scratch s;
  HcW mixer;
  Q8 embed, head;
  Tensor dummy_inj, tok, h[2], pos, x, logits, cand, lse, sp, res;
  Tensor mask;  // u32 [(VOCAB + 31) / 32]: allowed-token bits for fn_sample_p1 (P.mask_on; ne_fn_model_set_mask)
  Tensor bias;  // f32 [VOCAB]: logit_bias added by fn_sample_p1 (P.bias_on; ne_fn_model_set_logit_bias)
  Tensor cnt;   // u32 [VOCAB]: bit 31 = in the prompt, low 28 bits = generated count (P.pen_on; fn_sample_p2 counts)
  Tensor top;   // u32 [SK_MAX][2]: (id, logprob bits) of the last sample's candidates (P.lp_on; ne_fn_model_top)
  int n_top = 0;
  // A step is two command buffers: A = embedding + layer 0 (needs only the token), B = PLE + layers 1..47 + mixer,
  // lm_head, sampler (the head's 5 dispatches are indirect: threadgroup counts in `ind`, 0 for prefill tokens). The
  // host gathers the n-gram rows while A runs; B is committed behind A (queue order).
  std::unique_ptr<Recorder> rec_a, rec_b;
  Tensor ind;  // u32 [5][4]: indirect threadgroup counts of the head dispatches
  std::unique_ptr<Recorder> rec_head;  // debug: mixer + head + sampler on h[1] (ne_fn_model_run_head)
  size_t n_full = 0, n_body = 0;       // dispatches per step (A + B)
  int ngram_fd = -1;                   // ngram_table.bin, pread (no read-ahead)
  uint64_t ngram_off = 0;              // file offset of row 0
  bool dirty = false;                  // generate() ran layer 0 one token ahead: reset() before reuse
  uint64_t gen = 0;                    // bumped by every state change (step, step4, prefill_tokens, reset, restore):
                                       // the MTP loop's head state is valid only for the gen it was synced at
  // 4-row prefill (consecutive prompt tokens through the r4 kernels; no head): own scratch / hidden / token
  // buffers, the same states. NE_FN_PREFILL4=0 disables it.
  Scratch s4;
  Tensor tok4, h4[2];
  std::unique_ptr<Recorder> rec_a4, rec_b4;
  size_t n_pre4 = 0;
  id<MTLResidencySet> queue_rs = nil;  // the whole text model, resident on the MTL4 queue (recorder switches)
  std::vector<std::pair<std::string, id<MTLResidencySet>>> extra_rs;  // add_queue_residency
  // chunked prefill (prefill.h), created by the first prefill() of >= kChunkedMin tokens; NE_FN_PREFILL=steps keeps
  // the 4-row / 1-row path
  std::unique_ptr<Prefill, PrefillDeleter> pf;
  ~Model();
  int T = 0;                                     // tokens in the caches
  int64_t hist[2] = {0, 0};                      // the two previous tokens (EOS at the start)
  SampleParams P;
  double last_gpu_ms = 0, last_wall_ms = 0, gather_us = 0;
  // lab-vision (image input): multimodal positions and image embeddings, set by the host before a prefill
  Tensor mrope;               // f32 [36 + 4 tmax]: rope invf + MRoPE header / table shared by the 12 attention layers
                              // (their attn::QsaCache::invf; qsa.metal mrope_pos). Zero header = text positions.
  Tensor vfeat;               // bf16 [vcap][2560]: image embeddings (NE_VISION_ROWS rows, default 16384)
  std::vector<int32_t> vmap;  // cache row -> vfeat row (-1 or past the end: the token's embedding)
  int vcap = 0;
};
// The id written into a token buffer for cache row `row`: VOCAB + the vfeat row for image rows (fn_embed reads the
// image embedding), else the token itself.
inline uint32_t embed_id(const Model &m, int row, int token) {
  return row >= 0 && row < (int)m.vmap.size() && m.vmap[row] >= 0 ? (uint32_t)(VOCAB + m.vmap[row]) : (uint32_t)token;
}
// Multimodal positions: rows [0, n) take pos3 (t, h, w) [n][3], later rows row + delta (n = 0, delta = 0: text).
void set_mrope(Model &m, const int32_t *pos3, int n, int delta);
// Image rows: vfeat[i] = feats[i] (bf16 [k][2560]) is the input embedding of cache row rows[i]; k = 0 clears.
void set_image_rows(Model &m, const int32_t *rows, int k, const uint16_t *feats);
// n-gram rows for token t at the model's history (parallel preads) -> s.ple_emb
void gather_ple(Model &m, int token);
// n-gram rows of n consecutive tokens after the two previous tokens (p2, p1) (each later row's history = the tokens
// before it): emb f32 [n][2560]; 16 n preads in one parallel batch. Model state unchanged (MTP verify rows).
void gather_ple_rows(Model &m, int64_t p2, int64_t p1, const int *tokens, int n, float *emb);
// Make every pack tensor under `prefix` resident on the MTL4 queue as well (e.g. "language_model.mtp." for the MTP
// head: recorder switches between the head and the target then cost nothing). Idempotent per prefix.
void add_queue_residency(Model &m, const std::string &prefix);

// state_from (lab-srvperf): another model of the same architecture whose state buffers this one uses (KV caches,
// GDN / PLE conv + ssm + pending rows, positions, MRoPE table, image rows): two weight sets (base + a fine-tune) over
// ONE conversation state. Only one of them holds a valid state at a time: the host restores (ne_fn_state_io /
// ne_fn_kv_io) or resets the model it switches to; never run both at once.
std::unique_ptr<Model> load_model(const Pack &pack, int tmax, const Model *state_from = nullptr);
void reset(Model &m);  // zero every state, T = 0, history = EOS
// One step for `token` at position m.T: gathers the PLE rows, runs the full (head = true) or body recording, then
// T += 1. With the head: returns the sampled next token (and its logprob, the argmax token).
struct StepOut { int token = -1, argmax = -1; float logprob = 0; };
StepOut step(Model &m, int token, bool head);
// 4 consecutive prompt tokens at positions T..T+3 (no head), T += 4.
void step4(Model &m, const int *tokens);
// prompt[0, n): n >= kChunkedMin: lab-prefill's chunked prefill (chunks of kChunk tokens as matrix work) for the
// first n - 1 tokens; shorter prompts (or NE_FN_PREFILL=steps): 4-row chunks while at least 4 tokens remain before
// the last, then 1-row steps. The last token always goes through the decode step with the head. Returns its sample.
constexpr int kChunkedMin = 16, kChunk = 4096;  // 4096: +4% at 32K over 2048 (docs/prefill.md 4b)
bool use_chunked(const Model &m, int n);  // the policy above
Prefill &chunked(Model &m);               // the model's Prefill (created on first use)
StepOut prefill(Model &m, const int *prompt, int n);
// Entry checks of a multi-token call, before any state is touched: n tokens fit (m.T + rows <= m.tmax), the model is
// not dirty, every token is in the vocabulary (a mid-call failure would leave a half-fed context).
void check_feed(const Model &m, const int *toks, int n, int rows, const char *what);
// An exception while GPU work is in flight (an n-gram pread of the next gather, ...): wait for that work (its own
// errors swallowed) and mark the model dirty (its caches hold rows T does not count: reset / restore before reuse);
// the caller rethrows.
void abort_inflight(Model &m, std::initializer_list<Recorder *> rs);

// lab-srvcache (kvio.mm): KV rows [a, b) of these QSA caches <-> a host buffer laid out for `rows` rows (the layout
// of ne_fn_kv_io); host memcpy, no GPU work may be in flight.
int64_t kv_bytes(const std::vector<attn::QsaCache *> &cs, int rows);
void kv_io(const std::vector<attn::QsaCache *> &cs, bool to_model, int a, int b, void *buf, int rows);

}  // namespace ne::fn

// The C handle (owns the Model); shared by model.mm, prefill.mm and mtp_loop.mm
struct ne_fn_model {
  std::unique_ptr<ne::fn::Model> m;
};

extern "C" {
typedef struct ne_fn_model ne_fn_model;
// Loads all 48 layers + embedding + mixer + lm_head of an open pack (no RAM copies needed on an aligned pack),
// allocates the states (attention KV for tmax positions, multiple of 32) and records the step. NULL on error.
NE_API ne_fn_model *ne_fn_model_create(ne_pack *pack, int tmax);
// The model of `pack` over state_from's state buffers (load_model's state_from; same tmax and layer kinds): one KV +
// recurrent state for two weight sets, used one at a time (the server's --share-state). NULL on error.
NE_API ne_fn_model *ne_fn_model_create_shared(ne_pack *pack, int tmax, ne_fn_model *state_from);
NE_API void ne_fn_model_free(ne_fn_model *m);
NE_API int ne_fn_model_reset(ne_fn_model *m);
// lab-pfpath: the idle keep-warm (docs/prefill.md "server path"): one trivial command buffer on the engine's queue
// (waited) + requestResidency of the model's recordings (decode steps, prefill chunks); wall ms in *ms (may be NULL)
NE_API int ne_fn_gpu_ping(ne_fn_model *m, double *ms);
// Sampler for later steps: temp <= 0 or top_k == 1 = greedy (argmax, lowest id on ties); else temperature ->
// top-k (<= 64) -> top-p -> draw (seed + step counter).
NE_API int ne_fn_model_set_sampler(ne_fn_model *m, float temp, float top_p, int top_k, uint32_t seed);
// Allowed-token mask for the sampler of the following steps / prefill samples (grammar-constrained decoding): bits =
// (VOCAB + 31) / 32 = 7760 u32 words, bit i % 32 of word i / 32 set = token i may be sampled (at least one bit must be
// set; the step's greedy / top-k / top-p / logprob are then over the allowed tokens). bits == NULL clears the mask.
// Persists across reset() and set_sampler(); the host sets it between steps (no GPU work in flight). Applied by
// step(), prefill() and the pipelined generate() (one fixed mask for the whole run); the MTP loop does NOT apply it
// and ne_fn_mtp_generate refuses to run while a mask is set. Cost: one bit read per vocabulary entry in pass 1.
NE_API int ne_fn_model_set_mask(ne_fn_model *m, const uint32_t *bits);
// --- lab-sampler (16 Oct): the OpenAI / vLLM sampling knobs, all inside the sampler's pass 1 / 2 (no new dispatch).
// Order on the logits: logit_bias, then penalties, then the mask; temperature / top-k / min_p / top-p / draw follow in
// pass 2. Per-model-instance state like the mask: persists across reset() / set_sampler(); the host sets it between steps.
// Extended parameters: min_p (0 = off; tokens under min_p * p_max after the temperature softmax are dropped, before
// top-p), presence / frequency (OpenAI: logit -= frequency * count + presence * (count > 0) over THIS request's
// generated tokens) and repetition (> 0; 1 = off; HF / vLLM: logit < 0 ? * rep : / rep over prompt + generated tokens),
// and logprob_k (0..64): keep that many candidates for ne_fn_model_top. Setting them ZEROES the generated-token counts
// and the prompt flags (a request's start); counting is done by pass 2 for every token it samples. penalty-free
// (0, 0, 1) and min_p 0 and logprob_k 0 = the sampler of before, nothing extra read or written.
NE_API int ne_fn_model_set_sampler_ex(ne_fn_model *m, float min_p, float presence, float frequency, float repetition,
                                      int logprob_k);
// Repetition penalty over the prompt: marks the n token ids as seen (call after set_sampler_ex; ids outside the vocabulary are ignored).
NE_API int ne_fn_model_set_seen(ne_fn_model *m, const int *ids, int n);
// logit_bias: ids[i] gets bias[i] added (bias <= -100: banned); n = 0 clears. Replaces the previous table.
NE_API int ne_fn_model_set_logit_bias(ne_fn_model *m, const int *ids, const float *bias, int n);
// The RNG counter of the next draw (the draw hashes seed and counter; counted per sampled token, so a request that
// sets 0 at its start gets the same tokens for the same seed whatever was fed before).
NE_API int ne_fn_model_set_rng_step(ne_fn_model *m, uint32_t step);
// The last sample's candidates under the transformed logits (needs logprob_k > 0): up to `max` (token id, log-probability
// at temperature 1 over the full vocabulary, sorted by descending logit, ties by lowest id); returns the count.
NE_API int ne_fn_model_top(ne_fn_model *m, int *ids, float *logprobs, int max);
// One step for `token` at the current position. logits (may be NULL): f32 [248320] (bf16 values) of this position.
// next / logprob / argmax may be NULL. head = 0: layers only (prefill; logits / next not produced).
NE_API int ne_fn_model_step(ne_fn_model *m, int token, int head, float *logits, int *next, float *logprob,
                            int *argmax, double *gpu_ms);
// Prompt (n tokens, prefilled as prefill() does, the head only on the last) then up to max_new sampled tokens (stops
// after `stop` if >= 0). out / out_logprob: max_new entries; returns the number generated (or -1). stats (may be
// NULL, 6 doubles): prefill wall ms, decode wall ms, decode steps, mean decode GPU ms, mean gather us, dispatches.
NE_API int ne_fn_model_generate(ne_fn_model *m, const int *prompt, int n, int max_new, int stop, int *out,
                                float *out_logprob, double *stats);
// GPU time of the full decode step at context T (positions faked: KV contents are whatever is in the caches):
// median / min over reps replays of one command buffer each, and wall time per synchronous step.
NE_API int ne_fn_model_time(ne_fn_model *m, int T, int reps, double *gpu_ms_median, double *gpu_ms_min,
                            double *wall_ms_median);
NE_API int ne_fn_model_info(ne_fn_model *m, int *n_dispatch_full, int *n_dispatch_body, int *T);
// lab-vision: see ne::fn::set_mrope / set_image_rows (host-synchronous; call between prefills / steps)
NE_API int ne_fn_model_set_mrope(ne_fn_model *m, const int32_t *pos3, int n, int delta);
NE_API int ne_fn_model_set_image_rows(ne_fn_model *m, const int32_t *rows, int k, const uint16_t *feats);
// Prompt prefill (prefill(): chunked from 16 tokens, else 4-row chunks + 1-row steps; the head on the last token): its logits / sampled next token; wall ms.
NE_API int ne_fn_model_prefill(ne_fn_model *m, const int *prompt, int n, float *logits, int *next, double *ms);
// State snapshot / restore (lab-server's prefix reuse; = its server/shim/ne_serve_state.mm): host copies of every
// recurrent state (GDN conv + ssm of the 36 GDN layers, the PLE conv) + T + the n-gram history, ~113 MB. Attention KV /
// QSA caches are not copied: restore sets pos = T (the MTP rollback's truncation), so a snapshot stays valid only while
// KV rows [0, T) still hold its tokens (the caller tracks that). Only with no GPU work in flight; snapshot of a dirty
// model (after generate() / mtp_generate) returns NULL; restore clears dirty. restore: 0, or < 0 on a shape mismatch.
typedef struct ne_fn_snap ne_fn_snap;
NE_API ne_fn_snap *ne_fn_model_snapshot(ne_fn_model *m);
NE_API int ne_fn_model_restore(ne_fn_model *m, const ne_fn_snap *s);
NE_API int ne_fn_snap_T(const ne_fn_snap *s);
NE_API int64_t ne_fn_snap_bytes(const ne_fn_snap *s);
NE_API void ne_fn_snap_free(ne_fn_snap *s);
// lab-srvcache (kvio.mm; the server's multi-conversation prefix cache): KV rows [a, b) of the 12 attention layers
// <-> buf laid out for `rows` rows (ne_fn_kv_bytes(rows) bytes); the recurrent state (= what a snapshot holds) <->
// buf of ne_fn_state_bytes() with meta int64 [3] = T, hist0, hist1 (import = restore: T / pos / history set, dirty
// cleared, Model::gen bumped). Host memcpy, no GPU work in flight. 0 or -1 (ne_last_error).
NE_API int64_t ne_fn_kv_bytes(ne_fn_model *m, int rows);
NE_API int ne_fn_kv_io(ne_fn_model *m, int to_model, int a, int b, void *buf, int rows);
NE_API int64_t ne_fn_state_bytes(ne_fn_model *m);
NE_API int ne_fn_state_io(ne_fn_model *m, int to_model, void *buf, int64_t nbytes, int64_t *meta);
// Debug: mixer + lm_head + sampler alone on a final hidden state h f32 [4*2560] (logits / next / logprob nullable).
NE_API int ne_fn_model_run_head(ne_fn_model *m, const float *h, float *logits, int *next, float *logprob);
// The 16 n-gram row ids of token t after p2, p1 (EOS = none) and, if emb != NULL, the gathered f32 [2560] row.
NE_API int ne_fn_ngram(ne_pack *pack, int64_t p2, int64_t p1, int64_t t, int64_t *ids, float *emb);
}
