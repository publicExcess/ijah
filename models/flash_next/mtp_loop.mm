// Flash-Next MTP token loop (lab-loop). Header: mtp_loop.h; design and numbers: docs/mtp.md section 10.
#include "models/flash_next/mtp_loop.h"

#include "kernels/fn_rec.h"
#include "models/flash_next/dflash.h"
#include "models/flash_next/prefill.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <tuple>

namespace ne::fn {
namespace {
MTLSize sz(int n) { return MTLSizeMake(n, 1, 1); }
using clk = std::chrono::steady_clock;
double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }
uint32_t *u32(const Tensor &t) { return (uint32_t *)t.host(); }
// verify forward F(R) ms: the adaptive policy's prior round costs. lab-draft, 18 Oct: re-measured on the T6 pack
// (ne_fn_mtp_prof, sampled verify A + B at T 512: R 2..5; 1 row = the decode step; 6..8 extrapolated at +3 ms per row);
// docs/mtp.md 11 (was section 6's projection 14.11 / 17.45 / 21.06 / 24.67 ...)
const double kF[9] = {0, 13.1, 15.9, 19.5, 22.5, 25.3, 28.3, 31.3, 34.3};
struct U4 { uint32_t x, y, z, w; };
struct DraftSP { float temp, top_p; uint32_t top_k, seed, round, K, pad0, pad1; };  // = mtp.metal MtpDraftSP
}  // namespace

struct MtpLoop {
  Model *m = nullptr;
  int maxd = 4;
  mtp::MtpW W;
  mtp::MtpHead h1, h2, h8;  // 1-row seed / chain steps, 2-row seed (all drafts accepted), 8-row head prefill
  Tensor vtok;              // u32 [16]: verify tokens [t1, d_1 .. d_D]
  Tensor keep;              // u32 [4]: keep[0] = rows of the previous verify that the next step commits
  Tensor hv[2], xv, lg, amv, cand, sl, kb, vb, ple_pend;
  Tensor qb, dsp;  // sampled drafting: the proposals [16][32] (id, prob) per draft slot; DraftSP
  Tensor xind;     // u32 [4]: indirect grid of mtp_logit_xform {ceil(vocab / 1024) or 0, rows, 1, 0} (lab-sampler)
  std::vector<nax::LayerExtra> ex;  // per layer (GDN layers: lazy-commit pending rows)
  HcW mixer;                        // the model's final mixer with the zero inject matrix
  struct Verify {
    Scratch s;
    std::unique_ptr<Recorder> a, bg, bs;  // A; B with the greedy argmax / with the sampled top-K readout
    uint32_t K = 0;                        // bs's K
  };
  std::map<int, Verify> ver;                                  // by rows
  std::map<std::tuple<int, int, bool>, std::unique_ptr<Recorder>> dr;  // drafts by (seed rows, depth, sampled)
  std::unique_ptr<Recorder> hp8, hp1;                         // head prefill (8 rows / 1 row, no readout)
  // lab-headlazy: KV-only head steps (mtp::record_kv_step: the same KV rows as hp8 / hp1, without the attention
  // core / MoE / mixer) over two host-filled staging buffers (double-buffered catch-up): kG groups of 8 rows or one
  // group, and the 1-row tail. NE_MTP_HEAD_KV=full: hp8 / hp1 instead (A/B)
  static constexpr int kG = 16;
  std::unique_ptr<Recorder> hkG[2], hk8[2], hk1;
  Tensor sh[2], stk[2];  // staging: hidden f32 [kG * 8][10240], tokens u32 [kG * 8]
  bool kv_full = false;
  // Lazy head (lab-headlazy): plain steps / feeds only queue their hidden rows; the head's KV catches up (head_drain)
  // when pend_cap rows are pending, before an MTP run and before a snapshot. pend_cap <= 8 = the eager head
  // (mtpcont: an 8-row head step as soon as 8 rows are known). NE_MTP_PEND_CAP, ne_fn_mtp_set_pend_cap.
  int pend_cap = 1024;
  HeadKvW hkw;  // lab-pfpath: the head's weights for prefill_head_kv (feed_body)
  // The head's state between calls (docs/mtp.md 10.9): head KV rows [0, Lh) cached (row j = the pair (H_j, x_{j+1}),
  // H = layer 47's pre-mixer hidden); target positions [Lh, T) pending: their hidden rows ph and the token after each
  // (pt; -1 = not fed yet, only ever the last row). Valid while head_gen == m.gen (no plain step / prefill / restore
  // since); at T = 0 it is trivially valid (empty).
  int Lh = 0;
  std::vector<float> ph;
  std::vector<int> pt;
  uint64_t head_gen = ~0ull;
  double head_ms = 0;  // wall ms of head prefill work (drains) since the last mtp_generate started
  // lab-draft: the copy rule (TensorFold's, as Splish's SPLASH_M5_COPY_MIN_MATCH): when the last copy_min tokens (incl.
  // t1) occurred earlier in the context, the round drafts the tokens that followed that occurrence (one-hot q) instead
  // of the head's. ctx = the tokens at positions [ctx0, T) as far as the loop knows them (its own feeds / steps /
  // rounds; ne_fn_mtp_set_context after a state import); cidx = window hash -> continuation index in ctx (the most
  // recent occurrence), every window ending at index < cidx_end indexed. NE_MTP_COPY_MIN_MATCH (0 = off),
  // NE_MTP_COPY_DEPTH (drafts per copy round, <= max_depth).
  std::vector<int> ctx;
  int ctx0 = 0;
  mtp::CopyIndex cidx;
  int copy_min = 16, copy_maxd = 0;
  FILE *trace = nullptr;  // NE_MTP_TRACE=path: one line per round (docs/mtp.md 11)
  FILE *rdump = nullptr;  // lab-verify: NE_VERIFY_ROUTES=path: per round R, c, then route ids u16 [48][R][10]
  Tensor rcap;            // u32 [48][8][20]: every layer's route of the last verify (rdump only)
  // lab-draft: the gated verify (NE_MTP_QGATE=1, sampled drafts): verify the first k <= drafted drafts maximising
  // E[tokens](k) - rate * vcost[k + 1], E from qacc = P(d_j accepted | d_<j accepted, q(d_j) in bin b) (10 bins of
  // 0.1, EMAs over every round), rate = tokens per ms (EMAs), vcost[R] = ms of a round's verify + host at R rows.
  bool qgate = false;
  double qacc[10], qrc = 2.4, qrm = 22.0, vcost[9];
  // lab-dflash (docs/mtp.md 13): DFlash2 target features in every verify: the 4-stream mean of layers kDflLayers'
  // outputs -> feat f32 [8][5 * 2560] (row j = verify row j)
  bool dfl_cap = false;
  Tensor feat;
  // lab-dflash: the DFlash2 block drafter (dfl_enable; docs/mtp.md 13). Its context = target positions [c.T - c.n, c.T)
  // in the rings; then (in order) vcnt rows of the last verify (feat rows 0 .. vcnt - 1, positions c.T ..) and fsn rows
  // of the feature ring fs (positions [fsT - fsn, fsT): prefill chunks write it on the GPU, steps on the host) not
  // yet added. Valid while gen == Model::gen (as the head).
  struct Dfl {
    dfl::Weights w;
    dfl::Ctx c;
    dfl::Scratch s;                        // block rows B
    std::unique_ptr<Recorder> draft;       // the vcnt verify rows -> context, then the block (drafts -> vtok)
    std::unique_ptr<Recorder> cfeat;       // the vcnt verify rows -> context alone (copy / MTP-head rounds, end of run)
    std::unique_ptr<Recorder> rebuild;     // key / value rings from the c ring (after a restore)
    std::map<int, std::unique_ptr<Recorder>> drain;  // fs rows -> context, by 16-row groups
    Tensor fs, stage;                      // f32 [2048][12800] by position % 2048; f32 [16][12800]
    int fsT = 0, fsn = 0, vcnt = 0;
    uint64_t gen = ~0ull;
    int B = 8, kmax = 5, kgreedy = 3, minctx = 512;
    bool gate = true, active = true, early_a = true;  // early_a: NE_DFL_EARLY_A (A behind the draft, see the round)  // active = 0: no block drafts and no context upkeep (A/B in one process)
    double qacc[10], qaccg[10], rc = 3.0, rm = 25.0, vcost[9], vcostg[9];  // (..g: greedy rounds)
    double qacc0[10], vcost0[9];  // the priors: every run starts from them (runs repeat exactly, as DepthPolicy's)
  };
  std::unique_ptr<Dfl> dfl;
  int last_dfl_rounds = 0, last_dfl_tokens = 0;
};

void MtpLoopDeleter::operator()(MtpLoop *p) const {
  if (p && p->trace) fclose(p->trace);
  if (p && p->rdump) fclose(p->rdump);
  delete p;
}
// ---- copy rule (lab-draft)
static void ctx_reset(MtpLoop &L, int T) {
  L.ctx.clear();
  L.ctx0 = T;
  L.cidx.clear();
}
// tokens t[0, n) at positions [T, T + n) (a history that does not end at T is dropped first: unknown tokens between)
static void ctx_add(MtpLoop &L, int T, const int *t, int n) {
  if (L.copy_min <= 0) return;
  if (L.ctx0 + (int)L.ctx.size() != T) ctx_reset(L, T);
  L.ctx.insert(L.ctx.end(), t, t + n);
}
static int qbin(double qd) { return std::clamp((int)(qd * 10.0), 0, 9); }
static int qgate_depth(MtpLoop &L, const int *drafts, const mtp::Dist *q, int Dd) {
  const double rate = L.qrc / L.qrm;
  int best = 1;
  double run = 1.0, e = 1.0, bv = -1e30;
  for (int k = 1; k <= Dd; k++) {
    run *= L.qacc[qbin(q[k - 1].prob(drafts[k - 1]))];
    e += run;
    const double v = e - rate * L.vcost[k + 1];
    if (v > bv) { bv = v; best = k; }
  }
  return best;
}
static void qgate_observe(MtpLoop &L, const int *drafts, const mtp::Dist *q, int D, int a, int R, double round_ms,
                          int c) {
  for (int j = 0; j < D && j <= a; j++) {
    double &x = L.qacc[qbin(q[j].prob(drafts[j]))];
    x = 0.97 * x + 0.03 * (j < a ? 1.0 : 0.0);
  }
  L.qrc = 0.95 * L.qrc + 0.05 * c;
  L.qrm = 0.95 * L.qrm + 0.05 * round_ms;
  if (R < 9) L.vcost[R] = 0.9 * L.vcost[R] + 0.1 * round_ms;  // (incl. the draft: the same for every k of a round)
}
static int copy_find(MtpLoop &L, int t1, int max, int *out) {
  return mtp::copy_find(L.ctx, L.cidx, L.copy_min, t1, max, out);
}
static MtpLoop::Verify &get_verify(MtpLoop &L, int R, bool sampled, uint32_t K);
static Recorder &get_draft(MtpLoop &L, int n, int D, bool sampled);

MtpLoopPtr mtp_create(Model &m, int max_depth, int draft_bits, int shortlist) {
  NE_CHECK(max_depth >= 1 && max_depth <= 7, "max_depth %d (1..7)", max_depth);
  NE_CHECK(g_attn_qsa, "the MTP loop needs the QSA attention path (NE_FN_ATTN=qsa)");
  MtpLoopPtr L(new MtpLoop);
  L->m = &m;
  L->maxd = max_depth;
  L->W = mtp::load_mtp(*m.pack);
  if (draft_bits) {
    const std::vector<uint32_t> rows = mtp::draft_vocab_rows(L->W.vocab);  // lab-draft: NE_MTP_DRAFT_VOCAB
    mtp::set_draft_head(L->W, draft_bits, shortlist, &rows);
  }
  if (const char *e = getenv("NE_MTP_COPY_MIN_MATCH")) L->copy_min = std::max(0, atoi(e));
  L->copy_maxd = max_depth;
  if (const char *e = getenv("NE_MTP_COPY_DEPTH")) L->copy_maxd = std::clamp(atoi(e), 1, max_depth);
  if (const char *e = getenv("NE_MTP_TRACE")) L->trace = fopen(e, "a");
  if (const char *e = getenv("NE_VERIFY_ROUTES")) {
    L->rdump = fopen(e, "ab");
    L->rcap = alloc(DType::U32, {48 * 8 * 20});
  }
  if (const char *e = getenv("NE_MTP_QGATE")) L->qgate = atoi(e) > 0;
  for (int b = 0; b < 10; b++) L->qacc[b] = std::clamp(0.3 + 0.65 * (b + 0.5) / 10.0, 0.3, 0.95);
  for (int R = 0; R < 9; R++) L->vcost[R] = 13.0 + 3.4 * std::max(0, R - 1) + 1.5;  // ms, measured 18 Oct (prof)
  add_queue_residency(m, "language_model.mtp.");
  L->h1 = mtp::alloc_head(L->W, 1, m.tmax);
  // lab-mtpimg: head positions after image prompts. Head row j (the pair (H_j, x_{j+1})) sits at rope position j in
  // the reference (mlx-vlm's drafter: 1-D ids = its cache length = text rope on all three MRoPE axes; "text", the
  // default). "mrope" gives it the target's multimodal position of row j (the model's MRoPE table; identical for
  // text prompts, whose table is empty).
  if (const char *e = getenv("NE_MTP_HEAD_POS")) {
    NE_CHECK(!*e || !strcmp(e, "text") || !strcmp(e, "mrope"), "NE_MTP_HEAD_POS=%s: text or mrope", e);
    if (!strcmp(e, "mrope")) L->h1.st.qc.invf = m.mrope;
  }
  L->h2 = mtp::alloc_head(L->W, 2, 64);
  L->h8 = mtp::alloc_head(L->W, 8, 64);
  L->h2.st = L->h1.st;  // one head KV cache
  L->h8.st = L->h1.st;
  L->h2.amax = L->h1.amax;  // the chain's draft buffer: d_k at amax[k]
  const int64_t RM = 8;
  L->vtok = alloc(DType::U32, {16});
  L->keep = alloc(DType::U32, {4});
  L->hv[0] = alloc(DType::F32, {RM, HC, HID});
  L->hv[1] = alloc(DType::F32, {RM, HC, HID});
  L->xv = alloc(DType::F32, {RM, HID});
  L->lg = alloc(DType::F32, {RM, VOCAB});
  L->amv = alloc(DType::U32, {16});
  L->cand = alloc(DType::F32, {RM, 128, 32, 2});
  L->sl = alloc(DType::F32, {RM, 32, 2});
  L->kb = alloc(DType::F32, {RM, HCD});
  L->vb = alloc(DType::F32, {RM, HID});
  L->ple_pend = alloc(DType::F32, {RM, HCD});
  L->qb = alloc(DType::F32, {16, 32, 2});
  L->dsp = alloc(DType::U32, {8});
  L->xind = alloc(DType::U32, {4});
  L->feat = alloc(DType::F32, {RM, 5 * HID});
  if (const char *e = getenv("NE_DFL_CAPTURE")) L->dfl_cap = atoi(e) > 0;
  for (auto &w : m.L) L->ex.push_back(w.linear ? nax::alloc_layer_extra(0, true, &L->keep) : nax::alloc_layer_extra());
  L->mixer = m.mixer;
  L->mixer.inject = m.dummy_inj;
  L->hp8 = std::make_unique<Recorder>();
  L->hp8->begin();
  mtp::record_prefill_chunk(*L->hp8, L->W, L->h8, false);
  L->hp8->end();
  L->hp1 = std::make_unique<Recorder>();
  L->hp1->begin();
  mtp::record_step(*L->hp1, L->W, L->h1, nullptr, nullptr, nullptr, 0, false);
  L->hp1->end();
  if (const char *e = getenv("NE_MTP_HEAD_KV")) L->kv_full = std::string(e) == "full";
  if (const char *e = getenv("NE_MTP_PEND_CAP")) L->pend_cap = std::max(0, atoi(e));
  for (int b = 0; b < 2; b++) {
    L->sh[b] = alloc(DType::F32, {MtpLoop::kG * 8, HCD});
    L->stk[b] = alloc(DType::U32, {MtpLoop::kG * 8});
    for (int G : {MtpLoop::kG, 1}) {
      auto &p = G == 1 ? L->hk8[b] : L->hkG[b];
      p = std::make_unique<Recorder>();
      p->begin();
      for (int g = 0; g < G; g++)
        mtp::record_kv_step(*p, L->W, L->h8, L->stk[b].view(4 * 8 * (size_t)g, DType::U32, {8}), L->sh[b].rows(8 * g, 8));
      p->end();
    }
  }
  L->hk1 = std::make_unique<Recorder>();
  L->hk1->begin();
  mtp::record_kv_step(*L->hk1, L->W, L->h1, L->h1.tok, L->h1.hid);
  L->hk1->end();
  L->ph.reserve((size_t)(std::max(L->pend_cap, 8) + 16) * HCD);
  L->hkw = HeadKvW{&L->W.fc_e, &L->W.fc_h, &L->W.embed, L->W.norm_e, L->W.norm_h, &L->W.layer, (uint32_t)L->W.vocab};
  // record every verify (greedy + sampled K = 20) and draft variant now, and replay each once: pipelines compiled,
  // residency made, first-replay costs paid here, not in the first rounds (the states are reset by mtp_generate)
  for (int R = 2; R <= max_depth + 1; R++) {
    MtpLoop::Verify &v = get_verify(*L, R, false, 0);
    get_verify(*L, R, true, 20);
    v.a->run();
    v.bg->run();
    v.bs->run();
  }
  for (int n = 1; n <= 2; n++)
    for (int D = 1; D <= max_depth; D++)
      for (int sd = 0; sd < 2; sd++)
        if (!sd || draft_bits) get_draft(*L, n, D, sd).run();
  reset(m);
  return L;
}

// lab-verify: NE_VERIFY_ROUTES debug capture of layer l's routing (s.route [R][20] is rewritten by the next layer's router)
static void route_cap(MtpLoop &L, Recorder &r, Scratch &s, int l) {
  if (L.dfl_cap && dfl_slot(l) >= 0) {  // lab-dflash: layer l's output = hv[(l + 1) % 2]
    r.dispatch(pso("dfl_hc_mean"), {L.hv[(l + 1) % 2], L.feat, Arg::value(U4{(uint32_t)s.rows, (uint32_t)dfl_slot(l), 0, 0})},
               sz(s.rows * 10), sz(256));
    r.barrier();
  }
  if (!L.rdump) return;
  const uint32_t n = (uint32_t)s.rows * 20u, off = (uint32_t)l * 8u * 20u;
  r.dispatch(pso("mtp_copy_u32", 32), {s.route, L.rcap, Arg::value(U4{0, off, n, 0})}, sz(1), sz(256));
  r.barrier();
}

static MtpLoop::Verify &get_verify(MtpLoop &L, int R, bool sampled, uint32_t K) {
  Model &m = *L.m;
  MtpLoop::Verify &v = L.ver[R];
  if (v.a && (sampled ? (v.bs && v.K == K) : (bool)v.bg)) return v;
  if (!v.a) {
    v.s = alloc_scratch(R);  // not nax::alloc_scratch_rows: its R = 3, 5..8 path keeps the PLE buffers at 4 rows
    v.a = std::make_unique<Recorder>();
    Recorder &r = *v.a;
    r.begin();
    r.dispatch(pso("fn_embed_g" + std::to_string(m.embed.group)), {L.vtok, m.embed.w, m.embed.s, m.embed.b, L.hv[0]},
               sz(R), sz(256));
    r.barrier();
    nax::record_layer(r, m.L[0], m.st[0], v.s, L.ex[0], L.hv[0], L.hv[1], true);
    route_cap(L, r, v.s, 0);
    r.end();
  }
  auto &bp = sampled ? v.bs : v.bg;
  if (sampled) v.K = K;
  bp = std::make_unique<Recorder>();
  Recorder &r = *bp;
  Scratch &s = v.s;
  const std::string rs = "_r" + std::to_string(R);
  r.begin();
  const PleW &e = m.L[1].ple;
  nax::record_ple_proj(r, R, s, e, s.ple_emb, L.kb, L.vb);  // q8: NAX GEMM; T6: K8 t6 Linears (mtp-loop-check (2))
  r.barrier();
  r.dispatch(pso("mtp_ple_mix_lazy" + rs, 1024),
             {L.hv[1], L.kb, L.vb, e.nk, e.nq, e.nc, e.conv, m.st[1].ple_conv, s.hp, L.ple_pend, L.keep}, sz(HC),
             sz(1024));
  r.barrier();
  nax::record_layer(r, m.L[1], m.st[1], s, L.ex[1], s.hp, L.hv[0], true);
  route_cap(L, r, s, 1);
  for (size_t l = 2; l < m.L.size(); l++) {
    nax::record_layer(r, m.L[l], m.st[l], s, L.ex[l], L.hv[l % 2], L.hv[(l + 1) % 2], true);
    route_cap(L, r, s, (int)l);
  }
  const Tensor &hf = L.hv[m.L.size() % 2];
  const HcW &x = L.mixer;
  if (x.down.t6) {  // lab-t6wire: bf16 mixer (T6 pack)
    t6_hc_read(r, x, R, hf, false, s, s.mix_a, s.inj_a, L.xv);
  } else if (hc_v2(R)) {  // HC read v2 (kernels/fn_rec.h)
    krec::hc_pre3(r, x, R, hf, x.inject, s.inj_a, s.hc_part);
    krec::hc_mix(r, x, R, hf, s.mix_a, s.inj_a, L.xv, s.hc_part);
  } else {
    nax::record_hc_pre(r, R, L.mixer, hf, s.mix_a, s.inj_a, s, L.ex[0]);
    r.dispatch(pso("fn_hc_mix_g" + std::to_string(x.up.group) + rs),
               {hf, x.norm, x.up.w, x.up.s, x.up.b, s.mix_a, s.inj_a, L.xv}, sz(HID / 8), sz(256));
    r.barrier();
  }
  nax::record_q8(r, R, L.xv, {&m.head}, {L.lg});
  r.barrier();
  // the sampler's logit_bias / penalties on the rows (0 threadgroups unless the request set them; lab-sampler)
  r.dispatch_indirect(pso("mtp_logit_xform", 1024), {L.lg, L.vtok, m.sp, m.bias, m.cnt}, L.xind.buf, L.xind.offset, sz(1024));
  r.barrier();
  const uint32_t V = VOCAB;
  if (!sampled) {
    r.dispatch(pso("mtp_argmax", 1024), {L.lg, L.amv, Arg::value(V), Arg::value(0u)}, sz(R), sz(1024));
  } else {
    r.dispatch(pso("mtp_topk_local"), {L.lg, L.cand, Arg::value(V), Arg::value(K)}, MTLSizeMake(128, R, 1), sz(256));
    r.barrier();
    r.dispatch(pso("mtp_topk_merge", 1024), {L.cand, L.sl, L.amv, Arg::value(K)}, sz(R), sz(1024));
  }
  r.barrier();
  r.end();
  return v;
}

static Recorder &get_draft(MtpLoop &L, int n, int D, bool sampled) {
  auto &p = L.dr[{n, D, sampled}];
  if (p) return *p;
  p = std::make_unique<Recorder>();
  Recorder &r = *p;
  r.begin();
  mtp::MtpHead &hs = n == 1 ? L.h1 : L.h2;
  // sampled drafting: replace each step's argmax by a draw from its shortlist (and keep the proposal for the accept)
  auto sample = [&](const mtp::MtpHead &h, uint32_t at) {
    if (!sampled) return;
    r.dispatch(pso("mtp_draft_sample", 32), {h.sl, L.h1.amax, L.qb, L.dsp, Arg::value(at)}, sz(h.rows), sz(32));
    r.barrier();
  };
  mtp::record_step(r, L.W, hs, nullptr, nullptr, nullptr, n == 1 ? 1u : 0u);  // d_1 -> amax[1]
  sample(hs, n == 1 ? 1u : 0u);
  Tensor prev = hs.h_out.rows(n - 1, 1);
  for (int k = 2; k <= D; k++) {
    const Tensor tk = L.h1.amax.view(4 * (size_t)(k - 1), DType::U32, {1});
    const Tensor out = prev.buf == L.h1.h_out.buf ? L.h1.h_out2 : L.h1.h_out;
    mtp::record_step(r, L.W, L.h1, &tk, &prev, &out, (uint32_t)k);
    sample(L.h1, (uint32_t)k);
    prev = out;
  }
  r.dispatch(pso("mtp_copy_u32", 32), {L.h1.amax, L.vtok, Arg::value(U4{1, 1, (uint32_t)D, 0})}, sz(1), sz(32));
  r.barrier();
  r.end();
  return r;
}


// ---- lab-dflash: the DFlash2 drafter's context (docs/mtp.md 13)
static bool dfl_on(const MtpLoop &L) { return L.dfl && L.dfl->active; }
static void dfl_reset(MtpLoop &L, int T) {
  if (!L.dfl) return;
  MtpLoop::Dfl &d = *L.dfl;
  d.c.T = T;
  d.c.n = 0;
  d.fsT = T;
  d.fsn = 0;
  d.vcnt = 0;
  d.gen = L.m->gen;
}
// n feature rows at positions [pos, pos + n) are in the fs ring (the prefill wrote the last <= 2048 on the GPU, or
// dfl_push); a row that does not continue the known ones starts the context over at pos
static void dfl_note(MtpLoop &L, int pos, int n) {
  MtpLoop::Dfl &d = *L.dfl;
  NE_CHECK(d.vcnt == 0, "dflash: verify rows pending before new feature rows");
  if (pos != d.fsT) {
    d.c.n = 0;
    d.c.T = pos;
    d.fsn = 0;
  }
  d.fsT = pos + n;
  d.fsn = std::min(dfl::W, d.fsn + n);
}
static void dfl_push(MtpLoop &L, int pos, const float *rows, int n) {
  MtpLoop::Dfl &d = *L.dfl;
  for (int i = 0; i < n; i++)
    memcpy((float *)d.fs.host() + (size_t)((pos + i) % dfl::W) * dfl::FEAT, rows + (size_t)i * dfl::FEAT,
           (size_t)dfl::FEAT * 4);
  dfl_note(L, pos, n);
}
static Recorder &dfl_drain_rec(MtpLoop &L, int G) {
  MtpLoop::Dfl &d = *L.dfl;
  auto &p = d.drain[G];
  if (p) return *p;
  p = std::make_unique<Recorder>();
  Recorder &r = *p;
  r.begin();
  for (int i = 0; i < G; i++) {
    r.dispatch(pso("dfl_fs_gather"), {d.fs, d.stage, d.s.P, Arg::value((uint32_t)(16 * i))},
               sz(16 * dfl::FEAT / 256), sz(256));
    r.barrier();
    dfl::record_ctx_rows(r, d.w, d.c, d.s, d.stage, 16, (uint32_t)(16 * i));
  }
  r.end();
  return r;
}
// the fs rows into the context (16-row groups, the group count rounded up to a power of 2; extra rows skip)
static void dfl_drain(MtpLoop &L) {
  MtpLoop::Dfl &d = *L.dfl;
  if (d.fsn == 0) return;
  NE_CHECK(d.vcnt == 0, "dflash: verify rows pending at a drain");
  if (d.fsT - d.fsn != d.c.T) {  // rows dropped from the ring (more than 2048 since the last drain)
    d.c.n = 0;
    d.c.T = d.fsT - d.fsn;
  }
  int G = 1;
  while (G * 16 < d.fsn) G *= 2;
  uint32_t *P = u32(d.s.P);
  P[1] = (uint32_t)d.fsn;
  P[2] = (uint32_t)d.c.T;
  dfl_drain_rec(L, G).run();
  d.c.T += d.fsn;
  d.c.n = std::min(dfl::W, d.c.n + d.fsn);
  d.fsn = 0;
}
// the last verify's committed rows (feat rows 0 .. vcnt - 1) into the context: its own command buffer, queued before
// the next verify (rounds that run no block draft, the end of a run)
static void dfl_flush_v(MtpLoop &L, bool wait) {
  MtpLoop::Dfl &d = *L.dfl;
  if (d.vcnt == 0) return;
  if (d.cfeat->in_flight()) d.cfeat->wait();
  uint32_t *P = u32(d.s.P);
  P[1] = (uint32_t)d.vcnt;
  P[2] = (uint32_t)d.c.T;
  d.cfeat->commit();
  if (wait) d.cfeat->wait();
  d.c.T += d.vcnt;
  d.c.n = std::min(dfl::W, d.c.n + d.vcnt);
  d.vcnt = 0;
}
// a block draft can run at T: the context reaches T (after the pending verify rows) with enough rows behind it
static bool dfl_usable(MtpLoop &L, int T) {
  if (!L.dfl) return false;
  const MtpLoop::Dfl &d = *L.dfl;
  return d.fsn == 0 && d.c.T + d.vcnt == T && d.c.n + d.vcnt >= std::min(T, d.minctx);
}
// checkpoints (state_io, snapshots): header int64 [4] = (valid, T, n, 0) + the c ring bf16 [2048][2560]
static constexpr int64_t kDflStateBytes = 32 + (int64_t)dfl::W * dfl::H * 2;
static void dfl_export(MtpLoop &L, void *buf) {
  int64_t *h = (int64_t *)buf;
  memset(h, 0, 32);
  if (!dfl_on(L) || L.dfl->gen != L.m->gen || L.m->dirty) return;  // not in step: the importer restarts the context
  MtpLoop::Dfl &d = *L.dfl;
  dfl_drain(L);
  if (d.vcnt) return;
  h[0] = 1;
  h[1] = d.c.T;
  h[2] = d.c.n;
  memcpy((char *)buf + 32, d.c.cr.host(), (size_t)dfl::W * dfl::H * 2);
}
static Recorder &dfl_rebuild_rec(MtpLoop &L) {
  MtpLoop::Dfl &d = *L.dfl;
  if (!d.rebuild) {
    d.rebuild = std::make_unique<Recorder>();
    d.rebuild->begin();
    dfl::record_rebuild(*d.rebuild, d.w, d.c, d.s);
    d.rebuild->end();
  }
  return *d.rebuild;
}
// after the model's state import (Model::gen bumped): the drafter's context from buf, or restarted at T
static void dfl_import(MtpLoop &L, const void *buf) {
  if (!L.dfl) return;
  const int64_t *h = (const int64_t *)buf;
  if (!h || h[0] != 1 || h[1] != L.m->T || h[2] < 0 || h[2] > dfl::W || h[2] > h[1]) {
    dfl_reset(L, L.m->T);
    return;
  }
  MtpLoop::Dfl &d = *L.dfl;
  dfl_reset(L, (int)h[1]);
  memcpy(d.c.cr.host(), (const char *)buf + 32, (size_t)dfl::W * dfl::H * 2);
  d.c.n = (int)h[2];
  if (d.c.n > 0) {
    uint32_t *P = u32(d.s.P);
    P[1] = (uint32_t)d.c.n;
    P[2] = (uint32_t)(d.c.T - d.c.n);
    dfl_rebuild_rec(L).run();
  }
}
static int dfl_gate_depth(const MtpLoop::Dfl &d, const float *qd, int K, bool greedy) {
  int best = 1;
  double bv = -1.0, run = 1.0, e = 1.0;
  const double *acc = greedy ? d.qaccg : d.qacc, *vc = greedy ? d.vcostg : d.vcost;
  for (int k = 1; k <= K; k++) {
    run *= acc[qbin(qd[k - 1])];
    e += run;
    const double v = e / vc[k + 1];
    if (v > bv) { bv = v; best = k; }
  }
  return best;
}

// ---- head state kept in step with the model (MTP from the current state; docs/mtp.md 10.9)
bool mtp_head_ok(MtpLoop &L) {
  Model &m = *L.m;
  if (m.dirty) return false;
  if (m.T == 0) {  // empty model (reset): empty head
    L.Lh = 0;
    L.ph.clear();
    L.pt.clear();
    L.head_gen = m.gen;
    if (L.ctx0 != 0 || !L.ctx.empty()) ctx_reset(L, 0);
  }
  return L.head_gen == m.gen;
}
// the head is out of step with the model (a stale snapshot / state imported): drop its bookkeeping too, so no old
// pending rows survive into a later export
static void head_invalidate(MtpLoop &L) {
  L.Lh = 0;
  L.ph.clear();
  L.pt.clear();
  L.head_gen = ~0ull;
  ctx_reset(L, L.m->T);
}
// a new pending row (hidden of the target position just stepped, the token after it if known)
static void head_pend(MtpLoop &L, const float *H, int next) {
  NE_CHECK(L.pt.empty() || L.pt.back() >= 0, "head: pending row without its token");
  L.ph.insert(L.ph.end(), H, H + HCD);
  L.pt.push_back(next);
}
// the token after the last pending row (= the next token the target is fed)
static void head_next(MtpLoop &L, int tok) {
  if (!L.pt.empty() && L.pt.back() < 0) L.pt.back() = tok;
}
// run the head (KV only) over the pending rows whose token is known, leaving `tail` rows pending in any case: whole
// 8-row steps (rows [Lh, Lh + 8)), then with `all` the rest one row at a time (the fresh prefill's order, so a
// prefix fed in pieces gives the head the same 8-row steps as one prompt)
static void head_drain(MtpLoop &L, bool all, int tail) {
  int avail = (int)L.pt.size() - tail;
  while (avail > 0 && L.pt[avail - 1] < 0) avail--;
  if (avail < (all ? 1 : 8)) return;
  auto t0 = clk::now();
  u32(L.h1.st.pos)[0] = (uint32_t)L.Lh;
  int i = 0;
  if (L.kv_full) {
    for (; i + 8 <= avail; i += 8) {
      memcpy(L.h8.hid.host(), L.ph.data() + (size_t)i * HCD, (size_t)8 * HCD * 4);
      for (int r = 0; r < 8; r++) u32(L.h8.tok)[r] = (uint32_t)L.pt[i + r];
      L.hp8->run();
    }
  } else {  // KV-only 8-row steps, kG groups per command buffer; the next staging buffer fills while one runs
    Recorder *busy = nullptr;
    for (int b = 0; avail - i >= 8; b ^= 1) {
      const int n = avail - i >= MtpLoop::kG * 8 ? MtpLoop::kG * 8 : 8;
      memcpy(L.sh[b].host(), L.ph.data() + (size_t)i * HCD, (size_t)n * HCD * 4);
      for (int r = 0; r < n; r++) u32(L.stk[b])[r] = (uint32_t)L.pt[i + r];
      Recorder &rc = n == 8 ? *L.hk8[b] : *L.hkG[b];
      rc.commit();
      if (busy) busy->wait();
      busy = &rc;
      i += n;
    }
    if (busy) busy->wait();
  }
  if (all)
    for (; i < avail; i++) {
      memcpy(L.h1.hid.host(), L.ph.data() + (size_t)i * HCD, (size_t)HCD * 4);
      u32(L.h1.tok)[0] = (uint32_t)L.pt[i];
      (L.kv_full ? L.hp1 : L.hk1)->run();
    }
  L.Lh += i;
  NE_CHECK((int)u32(L.h1.st.pos)[0] == L.Lh, "head: KV length %u, expected %d", u32(L.h1.st.pos)[0], L.Lh);
  L.ph.erase(L.ph.begin(), L.ph.begin() + (size_t)i * HCD);
  L.pt.erase(L.pt.begin(), L.pt.begin() + i);
  L.head_ms += ms_since(t0);
  static const bool flog = getenv("NE_FEED_LOG") != nullptr;
  if (flog) fprintf(stderr, "head_drain: %d rows to Lh %d: %.1f ms\n", i, L.Lh, ms_since(t0));
}
// plain steps / feeds: drain only once pend_cap rows are pending (the lazy head; pend_cap <= 8: eager)
static void head_lazy(MtpLoop &L) {
  if ((int)L.pt.size() >= std::max(L.pend_cap, 8)) head_drain(L, false, 0);
}
// the target over toks[0, n) at m.T (no sample: chunked when `chunk_policy`, else 4-row + 1-row body steps, as
// prefill()), layer 47's hidden of every position into the head's pending rows
static void feed_body(MtpLoop &L, const int *toks, int n, bool chunk_policy, const int *ahead = nullptr,
                      int n_ahead = 0) {
  Model &m = *L.m;
  if (n <= 0) return;
  head_next(L, toks[0]);
  ctx_add(L, m.T, toks, n);
  auto nxt = [&](int j) { return j + 1 < n ? toks[j + 1] : -1; };
  const Tensor &hlast = m.h[m.L.size() % 2], &hlast4 = m.h4[m.L.size() % 2];
  int i = 0;
  static const bool flog = getenv("NE_FEED_LOG") != nullptr;  // lab-pfpath: per-chunk wall split (no behaviour change)
  if (chunk_policy) {
    Prefill &pf = chunked(m);
    const Tensor &hc = prefill_hidden(pf, (int)m.L.size());
    // lab-pfpath: the head's KV rows of each chunk as one batched pass (prefill_head_kv) instead of queueing the
    // chunk's rows for 8-row KV-only steps; NE_MTP_HEAD_CHUNK=0: the queue (read per feed: in-process A/B)
    const char *hce = getenv("NE_MTP_HEAD_CHUNK");
    const bool hchunk = !(hce && std::string(hce) == "0") && prefill_head_kv_ok(pf, L.h1.st);
    for (; i < n; i += kChunk) {
      const int cn = std::min(kChunk, n - i);
      const auto f0 = clk::now();
      // the next chunk's n-gram rows are gathered while this one runs (within the call: prefill_tokens' own
      // overlap; the call's last chunk: the caller's `ahead`, the next call's first tokens)
      const bool last = i + cn >= n;
      if (dfl_on(L)) prefill_set_dfl(pf, &L.dfl->fs);  // (a shared Prefill may have been swapped in: lab-srvperf)
      prefill_tokens(pf, toks + i, cn, last ? ahead : toks + i + cn, last ? n_ahead : n - i - cn);
      if (dfl_on(L)) dfl_note(L, m.T - cn, cn);
      const auto f1 = clk::now();
      int r0 = 0;
      if (hchunk) {
        // rows queued before this chunk first (every one's next token is known now), so the chunk's rows follow
        // them in the head's cache; then rows [T0, T0 + known) in one pass (the call's last row waits for its token)
        head_drain(L, true, 0);
        const int T0 = m.T - cn;
        if (L.pt.empty() && L.Lh == T0) {
          const int known = last ? cn - 1 : cn;
          if (known > 0) {
            const auto h0 = clk::now();
            prefill_head_kv(pf, L.hkw, L.h1.st, toks + i + 1, known, T0);
            L.Lh += known;
            u32(L.h1.st.pos)[0] = (uint32_t)L.Lh;
            L.head_ms += ms_since(h0);
          }
          r0 = known;
        }
      }
      for (int r = r0; r < cn; r++) head_pend(L, (const float *)hc.host() + (size_t)r * HCD, nxt(i + r));
      const auto f2 = clk::now();
      head_lazy(L);
      if (flog) {
        const PfStats &s = prefill_stats(pf);
        fprintf(stderr, "feed: chunk %d at T %d: prefill %.1f ms (GPU %.1f, gather %.1f, rec %.1f), head %d rows batched "
                "+ %d queued %.1f, head_drain %.1f\n", cn, m.T - cn,
                std::chrono::duration<double, std::milli>(f1 - f0).count(), s.gpu_ms, s.gather_ms, s.rec_ms, r0, cn - r0,
                std::chrono::duration<double, std::milli>(f2 - f1).count(), ms_since(f2));
      }
    }
  }
  const auto fs = clk::now();
  const int i0 = i;
  if (m.rec_a4)
    for (; i + 4 <= n; i += 4) {
      step4(m, toks + i);
      if (dfl_on(L)) dfl_push(L, m.T - 4, (const float *)m.dfl_f.host(), 4);
      for (int r = 0; r < 4; r++) head_pend(L, (const float *)hlast4.host() + (size_t)r * HCD, nxt(i + r));
      head_lazy(L);
    }
  for (; i < n; i++) {
    step(m, toks[i], false);
    if (dfl_on(L)) dfl_push(L, m.T - 1, (const float *)m.dfl_f.host(), 1);
    head_pend(L, (const float *)hlast.host(), nxt(i));
    head_lazy(L);
  }
  if (flog && i > i0) fprintf(stderr, "feed: %d step rows at T %d: %.1f ms\n", i - i0, m.T - (i - i0), ms_since(fs));
}

void mtp_feed(MtpLoop &L, const int *toks, int n, const int *ahead, int n_ahead) {
  NE_CHECK(n >= 1, "mtp_feed: no tokens");
  check_feed(*L.m, toks, n, n, "mtp_feed");
  const int htmax = L.h1.st.qc.tmax;  // the head's KV rows [0, Lh) with Lh <= T: its cache must hold T + n too
  NE_CHECK(htmax == 0 || L.m->T + n <= htmax, "mtp_feed: the MTP head's context is full (%d + %d > %d)", L.m->T, n, htmax);
  NE_CHECK(mtp_head_ok(L), "mtp_feed: the MTP head is not in step with the model (plain step / prefill / restore "
                           "since its last MTP call): reset or restore an MTP snapshot");
  if (dfl_on(L) && L.dfl->gen != L.m->gen) dfl_reset(L, L.m->T);
  feed_body(L, toks, n, use_chunked(*L.m, n), ahead, n_ahead);
  L.head_gen = L.m->gen;
  if (dfl_on(L)) {
    dfl_drain(L);
    L.dfl->gen = L.m->gen;
  }
}

StepOut mtp_step(MtpLoop &L, int tok, bool head) {
  Model &m = *L.m;
  NE_CHECK(mtp_head_ok(L), "mtp_step: the MTP head is not in step with the model");
  head_next(L, tok);
  ctx_add(L, m.T, &tok, 1);
  if (dfl_on(L) && L.dfl->gen != m.gen) dfl_reset(L, m.T);
  StepOut o = step(m, tok, head);
  head_pend(L, (const float *)m.h[m.L.size() % 2].host(), -1);
  head_lazy(L);
  L.head_gen = m.gen;
  if (dfl_on(L)) {
    dfl_push(L, m.T - 1, (const float *)m.dfl_f.host(), 1);
    L.dfl->gen = m.gen;
  }
  return o;
}

double mtp_catch_up(MtpLoop &L) {
  NE_CHECK(mtp_head_ok(L), "mtp_catch_up: the MTP head is not in step with the model");
  const double h0 = L.head_ms;
  head_drain(L, false, 0);
  return L.head_ms - h0;
}
void mtp_set_pend_cap(MtpLoop &L, int cap) {
  L.pend_cap = std::max(0, cap);
  if (mtp_head_ok(L)) head_lazy(L);
}

static int generate_impl(MtpLoop &L, bool fresh, const int *prompt, int n, int max_new, int stop, int depth,
                         uint64_t seed, int *out, MtpStats *st, MtpTokenCb cb, void *cb_ctx);
int mtp_generate(MtpLoop &L, const int *prompt, int n, int max_new, int stop, int depth, uint64_t seed, int *out,
                 MtpStats *st, MtpTokenCb cb, void *cb_ctx) {
  return generate_impl(L, true, prompt, n, max_new, stop, depth, seed, out, st, cb, cb_ctx);
}
int mtp_generate_from(MtpLoop &L, const int *toks, int n, int max_new, int stop, int depth, uint64_t seed, int *out,
                      MtpStats *st, MtpTokenCb cb, void *cb_ctx) {
  return generate_impl(L, false, toks, n, max_new, stop, depth, seed, out, st, cb, cb_ctx);
}

static int generate_impl(MtpLoop &L, bool fresh, const int *prompt, int n, int max_new, int stop, int depth,
                         uint64_t seed, int *out, MtpStats *st, MtpTokenCb cb, void *cb_ctx) {
  Model &m = *L.m;
  NE_CHECK(n >= 1 && max_new >= 1, "bad arguments");
  NE_CHECK(!m.P.mask_on, "ne_fn_model_set_mask(NULL) first: the MTP loop does not apply the sampler's token mask");
  NE_CHECK(!m.P.lp_on, "the MTP loop gives no logprobs (lab-sampler): ne_fn_model_set_sampler_ex(.., logprob_k = 0) or use "
                       "the step path");
  const bool xform = m.P.bias_on || m.P.pen_on;   // logit_bias / penalties: applied to the verify rows (mtp_logit_xform)
  NE_CHECK(depth >= 0 && depth <= L.maxd, "depth %d (0 = adaptive, 1..%d)", depth, L.maxd);
  const bool greedy = m.P.temp < 0.01f || m.P.top_k <= 1;
  const uint32_t K = greedy ? 0 : m.P.top_k;
  NE_CHECK(greedy || K <= 32, "sampled MTP needs top_k 1..32 (got %u)", K);
  const mtp::SampleParams sp{m.P.temp, (int)m.P.top_k, m.P.top_p, m.P.min_p};
  // sampled mode: drafts drawn from the head's shortlist (default; exact accept with q = that proposal) or argmax
  // drafts (point masses: accept with p(d)); NE_MTP_DRAFT=argmax|sample
  const char *de = getenv("NE_MTP_DRAFT");
  const bool sdraft = !greedy && L.W.draft_bits && !(de && std::string(de) == "argmax");
  {
    DraftSP d{m.P.temp, m.P.top_p, m.P.top_k, (uint32_t)(seed * 2654435761u + 12345u), 0, (uint32_t)L.W.shortlist, 0, 0};
    memcpy(L.dsp.host(), &d, sizeof(d));
  }
  // context and tokens before any state is touched (a fresh run resets: T = 0 then)
  {
    const int T0 = fresh ? 0 : m.T, htmax = L.h1.st.qc.tmax;
    NE_CHECK(T0 + n + 1 <= m.tmax && (htmax == 0 || T0 + n + 1 <= htmax), "context full (%d + %d of %d)", T0, n, m.tmax);
    NE_CHECK(fresh || !m.dirty, "model state is one token ahead after generate(): reset() first");
    for (int i = 0; i < n; i++) NE_CHECK(prompt[i] >= 0 && prompt[i] < VOCAB, "token %d out of range", prompt[i]);
  }
  MtpStats S;
  const int M1 = L.maxd + 1;
  S.hist.assign((size_t)M1 * M1, 0);
  if (fresh) reset(m);
  NE_CHECK(mtp_head_ok(L), "mtp_generate_from: the MTP head is not in step with the model (plain step / prefill / "
                           "restore since its last MTP call): reset or restore an MTP snapshot");
  NE_CHECK(m.T + n + 1 <= m.tmax, "context full (%d + %d of %d)", m.T, n, m.tmax);
  u32(L.keep)[0] = 0;
  L.head_ms = 0;
  if (dfl_on(L) && L.dfl->gen != m.gen) dfl_reset(L, m.T);  // (a fresh run: T = 0; a stale context: from T on)
  if (dfl_on(L)) {  // the gate's calibration starts from the priors in every run (a run's rounds depend on it only)
    MtpLoop::Dfl &d = *L.dfl;
    memcpy(d.qacc, d.qacc0, sizeof d.qacc);
    memcpy(d.qaccg, d.qacc0, sizeof d.qaccg);
    memcpy(d.vcost, d.vcost0, sizeof d.vcost);
    memcpy(d.vcostg, d.vcost0, sizeof d.vcostg);
  }

  // ---- prefill: the target over prompt[0, n - 1) (chunked prefill from 16 tokens, else 4-row chunks + 1-row steps);
  // layer 47's pre-mixer hidden of every position feeds the head over the (H_j, x_{j+1}) pairs in 8-row steps as they
  // come (KV only). From the current state (fresh = false) the head's pending rows [Lh, T) go first.
  auto t0 = clk::now();
  const Tensor &hlast = m.h[m.L.size() % 2];
  feed_body(L, prompt, n - 1, use_chunked(m, n));
  head_next(L, prompt[n - 1]);
  ctx_add(L, m.T, prompt + n - 1, 1);
  StepOut o = step(m, prompt[n - 1], true);
  gdn_materialise(m.st);  // lab-gdnlazy: the verify's lazy-commit kernels read ssm (the last prompt step deferred)
  head_pend(L, (const float *)hlast.host(), o.token);
  if (dfl_on(L)) dfl_push(L, m.T - 1, (const float *)m.dfl_f.host(), 1);
  // lab-ttft: stream the first token before the head's catch-up (the drafts of round 0 need it, the token does not)
  out[0] = o.token;
  const bool cb_stop = cb && cb(cb_ctx, out, 1);
  head_drain(L, true, 1);  // every pair but the seed (H_{T-1}, t1) into the head's KV
  if (dfl_on(L)) dfl_drain(L);  // lab-dflash: the prompt's last <= 2048 feature rows into the drafter's context
  S.prefill_ms = ms_since(t0);
  S.head_prefill_ms = L.head_ms;
  // round 0's seed: (H_{T-1}, t1)
  int T = m.T, t1 = o.token, seed_n = 1, Lh = L.Lh;
  NE_CHECK(Lh == T - 1 && L.pt.size() == 1, "head prefill: KV length %d, T %d, pending %zu", Lh, T, L.pt.size());
  u32(L.h1.tok)[0] = (uint32_t)t1;
  memcpy(L.h1.hid.host(), L.ph.data(), (size_t)HCD * 4);
  L.ph.clear();
  L.pt.clear();
  m.dirty = true;  // pending lazy GDN / PLE rows from here until the end-of-run flush
  int k = 0;
  out[k++] = t1;
  bool done = t1 == stop || k >= max_new;
  if (cb_stop) done = true;  // streaming / abort (lab-server): t1 went out above

  std::vector<double> cost(M1);
  // draft step ms (ne_fn_mtp_prof 18 Oct: 4-bit readout 1.12 full vocabulary / 0.74 pruned; the 8-bit head 1.55)
  const double tdraft = !L.W.draft_bits ? 1.55 : L.W.draft_ids.buf ? 0.74 : 1.12;
  for (int d = 0; d < M1; d++) cost[d] = kF[d + 1] + d * tdraft + 0.15;
  mtp::DepthPolicy pol(cost, 0.75);
  pol.min_depth = 1;
  pol.max_depth = L.maxd;
  mtp::Rng rng(seed);
  if (L.trace) fprintf(L.trace, "run fresh %d n %d T %d depth %d greedy %d sdraft %d copy_min %d vocab %d\n", fresh ? 1 : 0, n,
                       T, depth, greedy ? 1 : 0, sdraft ? 1 : 0, L.copy_min, L.W.draft_head.rows);
  auto td = clk::now();
  auto th = clk::now();  // end of the previous round's B wait (host time = accept + bookkeeping)
  // lab-draft: copy rounds leave the head's pairs pending in L.ph / L.pt (seed_n = 0): a head round drains them first
  int copy_buf[16];
  double drain_ms = 0;
  while (!done) {
    int D = depth ? depth : pol.next_depth();
    // copy rule: a verbatim continuation of an earlier occurrence of the last copy_min tokens (one-hot drafts)
    int nc = L.copy_min > 0 ? copy_find(L, t1, L.copy_maxd, copy_buf) : 0;
    const bool copy = nc > 0;
    if (copy) D = nc;
    // lab-dflash: the DFlash2 block drafter drafts this round when its context reaches T (else the MTP head)
    const bool dround = !copy && dfl_on(L) && dfl_usable(L, T);
    // headroom: a round keeps up to D + 1 rows and the end-of-run flush needs 2 more (its dummy rows at T, T + 1),
    // so T + D + 1 + 2 <= tmax -- a run near the context end stops <= 2 tokens early instead of leaving a dirty state
    D = std::min(D, m.tmax - T - 3);
    if (D < 1) break;
    drain_ms = 0;
    if (!copy && !dround && seed_n == 0) {  // after copy (or DFlash) rounds: the head's KV over every pending pair but the last (the seed)
      auto t0d = clk::now();
      L.Lh = Lh;
      head_drain(L, true, 1);
      Lh = L.Lh;
      NE_CHECK(L.pt.size() == 1 && L.ph.size() == (size_t)HCD, "copy rule: %zu head rows pending after the drain", L.pt.size());
      u32(L.h1.tok)[0] = (uint32_t)L.pt[0];
      memcpy(L.h1.hid.host(), L.ph.data(), (size_t)HCD * 4);
      L.ph.clear();
      L.pt.clear();
      seed_n = 1;
      drain_ms = ms_since(t0d);
    }
    // lab-draft: the gated verify (NE_MTP_QGATE): draft Dd = D deep, then verify only the first D' <= Dd drafts whose
    // expected gain pays for their verify rows (by the drafts' own proposal probabilities, docs/mtp.md 11). Only with
    // sampled drafts (q known); A is then committed after the drafts are read instead of right behind the draft.
    const bool gate = L.qgate && sdraft && !copy && !dround && depth > 0;  // (adaptive depth: the DepthPolicy decides)
    const int Dd = copy || dround ? 0 : D;  // drafted depth (the head's chain rows)
    Recorder *drp = copy || dround ? nullptr : &get_draft(L, seed_n, Dd, sdraft);
    ((DraftSP *)L.dsp.host())->round = (uint32_t)S.rounds;
    u32(L.h1.st.pos)[0] = (uint32_t)Lh;
    u32(L.vtok)[0] = (uint32_t)t1;
    u32(m.pos)[0] = (uint32_t)T;
    const double host_pre = std::chrono::duration<double, std::micro>(clk::now() - th).count();
    S.host_us += host_pre;
    auto tr = clk::now();
    MtpLoop::Verify *vp = nullptr;
    auto arm = [&](int Dv) {  // the verify for Dv drafts: its recordings and indirect grid
      vp = &get_verify(L, Dv + 1, !greedy, K);
      u32(L.xind)[0] = xform ? (uint32_t)((VOCAB + 1023) / 1024) : 0u;
      u32(L.xind)[1] = (uint32_t)(Dv + 1);
      u32(L.xind)[2] = 1u;
    };
    // lab-dflash: rounds without a block draft add the last verify's rows to the drafter's context first (queued)
    if (dfl_on(L) && !dround) dfl_flush_v(L, false);
    double dgpu_d = 0;
    mtp::Dist dq[8];
    float dqd[8] = {};      // lab-dflash: q(d_j) of the block's drafts (greedy: the selector's softmax at the argmax)
    Recorder *ap = nullptr;  // lab-dflash: A committed behind the block draft at kmax + 1 rows (B then runs at R <= that)
    if (copy) {
      for (int j = 0; j < D; j++) u32(L.vtok)[j + 1] = (uint32_t)copy_buf[j];
    } else if (dround) {  // the last verify's rows -> context, then the block: B - 1 drafts -> vtok[1 ..], proposals
      MtpLoop::Dfl &d = *L.dfl;
      uint32_t *P = u32(d.s.P);
      const int cT = d.c.T + d.vcnt, cn = std::min(dfl::W, d.c.n + d.vcnt);
      P[0] = (uint32_t)T;
      P[1] = (uint32_t)d.vcnt;
      P[2] = (uint32_t)d.c.T;
      P[3] = (uint32_t)(T - std::min(cn, dfl::W - 1));
      const dfl::SelP sel{(uint32_t)t1, (uint32_t)(d.B - 1), greedy ? 0u : 1u, (uint32_t)(seed * 2654435761u + 977u),
                          (uint32_t)S.rounds, 0, 0, 0, greedy ? 1.0f : m.P.temp, 0, 0, 0};
      memcpy(d.s.SP.host(), &sel, sizeof sel);
      u32(d.s.tok)[0] = (uint32_t)t1;
      d.draft->commit();
      // A (embedding + layer 0: per row / causal, lazy commit) at kmax + 1 rows right behind the draft, so the GPU does
      // not wait for the host's gate; B then reads its first R rows (layer 0's pending rows: keep[0] <= R of them)
      if (d.early_a && D >= 1) {
        ap = get_verify(L, std::min(d.kmax, m.tmax - T - 3) + 1, !greedy, K).a.get();
        ap->commit();
      }
      d.draft->wait();
      dgpu_d = d.draft->gpu_ms();
      d.c.T = cT;
      d.c.n = cn;
      d.vcnt = 0;
      // the proposals q (16 (id, prob) per position) and q(d_j) for the gate
      const int Pn = d.B - 1;
      float qd[8];
      const float *qo = (const float *)d.s.qo.host();
      for (int j = 0; j < Pn; j++) {
        const int dj = (int)u32(d.s.dr)[j];
        qd[j] = 0;
        for (int c2 = 0; c2 < dfl::NC; c2++) {
          uint32_t id;
          memcpy(&id, &qo[(j * dfl::NC + c2) * 2], 4);
          const float pr = qo[(j * dfl::NC + c2) * 2 + 1];
          if (pr > 0) {
            dq[j].ids.push_back((int)id);
            dq[j].p.push_back(pr);
          }
          if ((int)id == dj) qd[j] = pr;
        }
      }
      const int kd = !d.gate ? (greedy ? std::min(d.kmax, d.kgreedy) : d.kmax) : dfl_gate_depth(d, qd, d.kmax, greedy);
      for (int j = 0; j < Pn; j++) dqd[j] = qd[j];
      D = std::max(1, std::min(kd, m.tmax - T - 3));
      arm(D);
      if (!ap) vp->a->commit();
    } else {
      drp->commit();
    }
    if (!gate && !dround) {
      arm(D);
      vp->a->commit();
    }
    if (drp) drp->wait();
    const double t_dw = ms_since(tr);
    // the proposals of sampled drafts (what each d_j was drawn from: the accept's q)
    mtp::Dist q[8];
    int drafts[8];
    for (int j = 0; j < Dd; j++) drafts[j] = (int)u32(L.vtok)[j + 1];
    if (sdraft && !copy && !dround)
      for (int j = 0; j < Dd; j++) {
        const float *qr = (const float *)L.qb.host() + (size_t)(j + 1) * 64;
        for (int c = 0; c < 32; c++) {
          uint32_t id;
          memcpy(&id, &qr[2 * c], 4);
          if (id == 0xFFFFFFFFu) break;
          q[j].ids.push_back((int)id);
          q[j].p.push_back(qr[2 * c + 1]);
        }
      }
    if (gate) {
      D = qgate_depth(L, drafts, q, Dd);
      arm(D);
      vp->a->commit();
    }
    MtpLoop::Verify &v = *vp;
    const int R = D + 1;
    int toks[8];
    toks[0] = t1;
    for (int j = 0; j < D; j++) toks[j + 1] = drafts[j] = (int)u32(L.vtok)[j + 1];
    auto tg = clk::now();
    gather_ple_rows(m, m.hist[0], m.hist[1], toks, R, (float *)v.s.ple_emb.host());
    const double gus = std::chrono::duration<double, std::micro>(clk::now() - tg).count();
    S.gather_us += gus;
    Recorder &vb = greedy ? *v.bg : *v.bs;
    vb.commit();
    const double t_bc = ms_since(tr);
    Recorder &va = ap ? *ap : *v.a;
    va.wait();
    vb.wait();
    th = clk::now();
    const double dgpu = drp ? drp->gpu_ms() : dgpu_d;
    if (dfl_on(L) && L.dfl->cfeat->in_flight()) L.dfl->cfeat->wait();
    S.draft_gpu_ms += dgpu;
    S.verify_gpu_ms += va.gpu_ms() + vb.gpu_ms();
    mtp::VerifyResult res;
    float qtr[16];
    const float *trace_q = nullptr;
    if (greedy) {
      int am[8];
      for (int j = 0; j < R; j++) am[j] = (int)u32(L.amv)[j];
      res = mtp::accept_greedy(drafts, D, am);
    } else {
      mtp::Dist p[8];
      for (int j = 0; j < R; j++) p[j] = mtp::shortlist_dist((const float *)L.sl.host() + (size_t)j * K * 2, K, sp);
      // copy drafts (and argmax drafts) are point masses: q = nullptr
      res = mtp::accept_sampled_p(drafts, D, p, dround ? dq : sdraft && !copy ? q : nullptr, rng);
      if (L.trace && sdraft && !copy && !dround) {
        for (int j = 0; j < D; j++) {
          qtr[2 * j] = q[j].prob(drafts[j]);
          qtr[2 * j + 1] = q[j].p.empty() ? 0.0f : q[j].p[0];
        }
        trace_q = qtr;
      }
      static const bool dbg = getenv("NE_MTP_DEBUG") != nullptr;
      if (dbg && S.rounds < 2) {
        fprintf(stderr, "round %d seed %llu: accepted %d next %d\n", S.rounds, (unsigned long long)seed, res.accepted,
                res.next_token);
        for (int j = 0; j < D; j++) {
          fprintf(stderr, "  d%d = %d  q:", j + 1, drafts[j]);
          for (size_t c = 0; c < q[j].ids.size(); c++) fprintf(stderr, " %d:%.3f", q[j].ids[c], q[j].p[c]);
          fprintf(stderr, "  | p:");
          for (size_t c = 0; c < std::min<size_t>(p[j].ids.size(), 6); c++) fprintf(stderr, " %d:%.3f", p[j].ids[c], p[j].p[c]);
          fprintf(stderr, "\n");
        }
      }
    }
    const int a = res.accepted;
    const int k0 = k;
    for (int tkn : res.emitted) {
      if (k >= max_new) { done = true; break; }
      out[k++] = tkn;
      if (m.P.pen_on) u32(m.cnt)[tkn] += 1u;   // the generated-token counts of the penalties (the step path counts in pass 2)
      if (tkn == stop) { done = true; break; }
    }
    if (k >= max_new) done = true;
    if (cb && k > k0 && cb(cb_ctx, out + k0, k - k0)) done = true;
    S.rounds++;
    S.hist[(size_t)D * M1 + a]++;
    // commit: c rows of this verify (t1, d_1 .. d_{c-1}) enter every cache, c = a + 1 unless the emission was cut
    // (stop token / max_new inside the round): then only t1 and the emitted tokens before the last one, so the state
    // is always prompt + out[0, k - 1) and the run can be continued from it
    const int c = k - k0;
    NE_CHECK(c >= 1, "round emitted nothing");
    for (int j = 0; j < c; j++) {
      m.hist[0] = m.hist[1];
      m.hist[1] = toks[j];
    }
    ctx_add(L, T, toks, c);
    T += c;
    m.T = T;
    u32(L.keep)[0] = (uint32_t)c;
    const float *hvr = (const float *)L.hv[m.L.size() % 2].host();
    if (dfl_on(L)) L.dfl->vcnt = c;  // lab-dflash: this verify's committed rows (feat rows 0 .. c - 1) go to the context next
    if (copy || dround) {  // (DFlash rounds: the head did not run either)
      // copy round: the head did not run. Its pending pairs = the old seed rows (if any) + every committed verify row
      // (target hidden, the token after it); a head round drains them (KV only) before its seed step.
      if (seed_n > 0) {
        const mtp::MtpHead &hs = seed_n == 1 ? L.h1 : L.h2;
        for (int j = 0; j < seed_n; j++) head_pend(L, (const float *)hs.hid.host() + (size_t)j * HCD, (int)u32(hs.tok)[j]);
        seed_n = 0;
      }
      for (int j = 0; j < c; j++) head_pend(L, hvr + (size_t)j * HCD, j < c - 1 ? drafts[j] : out[k - 1]);
      if ((int)L.pt.size() >= 136) {  // bound the host rows over long copies: whole 8-row KV-only groups now
        L.Lh = Lh;
        head_drain(L, false, 1);
        Lh = L.Lh;
      }
    } else {
      // head: keep the chain rows of accepted drafts, seed the next round from the verify hidden rows hk .. c - 1
      const int hk = std::min(c - 1, Dd - 1);  // (Dd >= D: the gated verify checks a prefix of the chain)
      Lh += seed_n + hk;
      seed_n = c - hk;
      mtp::MtpHead &hs = seed_n == 1 ? L.h1 : L.h2;
      for (int j = 0; j < seed_n; j++) {
        const int row = hk + j;
        u32(hs.tok)[j] = (uint32_t)(row < c - 1 ? drafts[row] : out[k - 1]);
        memcpy((float *)hs.hid.host() + (size_t)j * HCD, hvr + (size_t)row * HCD, (size_t)HCD * 4);
      }
    }
    t1 = res.next_token;
    const double wall = ms_since(tr);
    if (copy) {
      S.copy_rounds++, S.copy_tokens += c;
    } else if (dround) {
      S.dfl_rounds++, S.dfl_tokens += c;
      MtpLoop::Dfl &d = *L.dfl;
      for (int j = 0; j < D && j <= a; j++) {
        double &x = (greedy ? d.qaccg : d.qacc)[qbin(dqd[j])];
        x = 0.97 * x + 0.03 * (j < a ? 1.0 : 0.0);
      }
      // (round costs stay the measured priors: learning them from wall time made a run's rounds depend on timing,
      // so runs did not repeat; NE_DFL_COST_EMA=1 learns them anyway)
      static const bool cost_ema = getenv("NE_DFL_COST_EMA") && atoi(getenv("NE_DFL_COST_EMA")) > 0;
      double *vc = greedy ? d.vcostg : d.vcost;
      if (cost_ema && R < 9) vc[R] = 0.9 * vc[R] + 0.1 * (wall + host_pre / 1e3);
    } else if (!gate) {
      pol.observe(D, a, wall);
    }
    if (sdraft && !copy && !dround) qgate_observe(L, drafts, q, D, a, R, wall + host_pre / 1e3, c);
    if (L.rdump) {  // lab-verify: R, c, routes u16 [48][R][10]
      const uint32_t *rc = (const uint32_t *)L.rcap.host();
      std::vector<uint16_t> o = {(uint16_t)R, (uint16_t)c};
      for (int l = 0; l < 48; l++)
        for (int j = 0; j < R; j++)
          for (int e = 0; e < TOPK; e++) o.push_back((uint16_t)rc[l * 160 + j * 20 + e]);
      fwrite(o.data(), 2, o.size(), L.rdump);
    }
    if (L.trace)  // round D a c copy seed draft_gpu A_gpu B_gpu wall draft_wait B_commit host_pre_us gather_us drain_ms
      fprintf(L.trace, "R %d %d %d %d %d %.3f %.3f %.3f %.3f %.3f %.3f %.1f %.1f %.3f %d\n", D, a, c, copy ? 1 : 0,
              seed_n, dgpu, va.gpu_ms(), vb.gpu_ms(), wall, t_dw, t_bc, host_pre, gus, drain_ms, Dd);
    if (L.trace && trace_q) {  // + per draft: its proposal probability q(d_j) and the proposal's top probability
      fprintf(L.trace, "Q");
      for (int j = 0; j < D; j++) fprintf(L.trace, " %.4f %.4f", trace_q[2 * j], trace_q[2 * j + 1]);
      fprintf(L.trace, "\n");
    }
    if (T + 4 > m.tmax) done = true;  // no room for another round (D >= 1) plus the flush
  }
  S.decode_ms = ms_since(td);
  S.tokens = k - 1;
  if (L.trace) {
    fprintf(L.trace, "end tokens %d rounds %d decode_ms %.1f copy_rounds %d copy_tokens %d\n", S.tokens, S.rounds,
            S.decode_ms, S.copy_rounds, S.copy_tokens);
    fflush(L.trace);
  }
  u32(L.xind)[0] = 0;
  if (dfl_on(L)) dfl_flush_v(L, true);  // lab-dflash: the last round's rows into the drafter's context (before the flush)
  // ---- leave a clean state = prompt + out[0, k - 1) that plain steps, snapshots and mtp_generate_from continue:
  // the last round's kept rows are still lazy (keep[0]); a 2-row verify over dummy tokens at pos T commits them
  // (GDN ssm / conv, PLE conv) and its own rows stay pending (keep[0] = 0 drops them; KV rows >= T are past pos)
  const bool flush_fits = T + 2 <= m.tmax;  // the dummy rows' KV positions T, T + 1
  if (u32(L.keep)[0] > 0 && flush_fits) {
    auto tf = clk::now();
    MtpLoop::Verify &v = get_verify(L, 2, false, 0);
    u32(L.xind)[0] = 0u;
    u32(L.vtok)[0] = u32(L.vtok)[1] = (uint32_t)out[k - 1];
    u32(m.pos)[0] = (uint32_t)T;
    v.a->commit();
    v.bg->commit();
    v.a->wait();
    v.bg->wait();
    u32(L.keep)[0] = 0;
    S.flush_ms = ms_since(tf);
  }
  u32(m.pos)[0] = (uint32_t)T;
  // the head: KV rows [0, Lh); the seed rows [Lh, T) stay pending (the last one's token = the next one fed)
  if (seed_n > 0) {
    const mtp::MtpHead &hs = seed_n == 1 ? L.h1 : L.h2;
    L.Lh = Lh;
    L.ph.assign((const float *)hs.hid.host(), (const float *)hs.hid.host() + (size_t)seed_n * HCD);
    L.pt.resize(seed_n);
    for (int j = 0; j < seed_n; j++) L.pt[j] = j + 1 < seed_n ? (int)u32(hs.tok)[j] : -1;
    NE_CHECK(L.Lh + seed_n == T, "head bookkeeping: Lh %d + %d pending != T %d", L.Lh, seed_n, T);
  } else {  // the last round was a copy round: its pairs are pending already (lab-draft)
    L.Lh = Lh;
    NE_CHECK(!L.pt.empty() && L.Lh + (int)L.pt.size() == T, "head bookkeeping: Lh %d + %zu pending != T %d", L.Lh,
             L.pt.size(), T);
    L.pt.back() = -1;
  }
  m.gen++;
  if (u32(L.keep)[0] > 0) {  // no room for the flush (the headroom above prevents it): dirty (reset before reuse)
    m.dirty = true;
  } else {
    m.dirty = false;
    L.head_gen = m.gen;
    if (dfl_on(L)) L.dfl->gen = m.gen;
  }
  L.last_dfl_rounds = S.dfl_rounds;
  L.last_dfl_tokens = S.dfl_tokens;
  if (st) *st = S;
  return k;
}

int mtp_verify_forced(MtpLoop &L, const int *prompt, int n, const int *cont, int nc, int R, const int *keeps,
                      int nk, float *logits) {
  Model &m = *L.m;
  NE_CHECK(R >= 2 && R <= L.maxd + 1 && nk >= 1, "verify_forced: R %d", R);
  reset(m);
  ctx_reset(L, 0);
  u32(L.keep)[0] = 0;
  u32(L.xind)[0] = 0;   // raw logits out (lab-sampler: no bias / penalty transform)
  prefill(m, prompt, n);
  gdn_materialise(m.st);  // lab-gdnlazy
  MtpLoop::Verify &v = get_verify(L, R, false, 0);
  int i = 0, round = 0;
  while (i + R <= nc && m.T + R <= m.tmax) {
    for (int j = 0; j < R; j++) u32(L.vtok)[j] = (uint32_t)cont[i + j];
    u32(m.pos)[0] = (uint32_t)m.T;
    v.a->commit();
    gather_ple_rows(m, m.hist[0], m.hist[1], cont + i, R, (float *)v.s.ple_emb.host());
    v.bg->commit();
    v.a->wait();
    v.bg->wait();
    const int kp = std::max(1, std::min(R, keeps[round++ % nk]));
    memcpy(logits + (size_t)i * VOCAB, L.lg.host(), (size_t)kp * VOCAB * 4);
    for (int j = 0; j < kp; j++) {
      m.hist[0] = m.hist[1];
      m.hist[1] = cont[i + j];
    }
    m.T += kp;
    m.gen++;  // the head is not kept in step here
    u32(L.keep)[0] = (uint32_t)kp;
    i += kp;
  }
  m.dirty = true;
  return i;
}

void mtp_time(MtpLoop &L, int R, int T, int P, int reps, double *verify_ms, double *draft_ms) {
  Model &m = *L.m;
  NE_CHECK(R >= 2 && R <= L.maxd + 1 && T + 2 * R <= m.tmax && P >= 0 && P <= 8, "mtp_time: R %d T %d P %d", R, T, P);
  reset(m);
  u32(L.xind)[0] = 0;
  MtpLoop::Verify &v = get_verify(L, R, false, 0);
  Recorder &dr = get_draft(L, 1, R - 1, false);
  std::vector<double> g, d;
  for (int i = 0; i < reps + 2; i++) {
    for (int j = 0; j < R; j++) u32(L.vtok)[j] = (uint32_t)(1000 + 7 * i + j);
    u32(L.keep)[0] = (uint32_t)P;
    u32(m.pos)[0] = (uint32_t)T;
    u32(L.h1.st.pos)[0] = (uint32_t)T;
    dr.run();
    v.a->run();
    v.bg->run();
    if (i >= 2) {
      g.push_back(v.a->gpu_ms() + v.bg->gpu_ms());
      d.push_back(dr.gpu_ms());
    }
  }
  std::sort(g.begin(), g.end());
  std::sort(d.begin(), d.end());
  *verify_ms = g[g.size() / 2];
  *draft_ms = d[d.size() / 2];
  reset(m);
}

// lab-draft: GPU ms (median of reps, each piece its own recording / command buffer) of a draft round's parts at head
// position T: see mtp_loop.h ne_fn_mtp_prof for the layout. Leaves the model reset.
int mtp_prof(MtpLoop &L, int T, int reps, double *out, int nout) {
  Model &m = *L.m;
  NE_CHECK(T + 16 <= m.tmax && reps >= 1, "mtp_prof: T %d", T);
  reset(m);
  u32(L.xind)[0] = 0;
  u32(L.keep)[0] = 0;
  {
    DraftSP d{1.0f, 0.95f, 20u, 12345u, 0, (uint32_t)L.W.shortlist, 0, 0};
    memcpy(L.dsp.host(), &d, sizeof(d));
  }
  for (int j = 0; j < 16; j++) u32(L.h1.tok)[j] = u32(L.h2.tok)[j] = u32(L.vtok)[j] = (uint32_t)(1000 + 7 * j);
  auto med = [&](Recorder &r, bool verify = false, Recorder *b = nullptr) {
    std::vector<double> g;
    for (int i = 0; i < reps + 2; i++) {
      u32(L.h1.st.pos)[0] = (uint32_t)T;
      u32(m.pos)[0] = (uint32_t)T;
      r.commit();
      if (b) b->commit();
      r.wait();
      if (b) b->wait();
      if (i >= 2) g.push_back(r.gpu_ms() + (b ? b->gpu_ms() : 0.0));
    }
    std::sort(g.begin(), g.end());
    return g[g.size() / 2];
  };
  auto rec = [&](auto f) {
    auto r = std::make_unique<Recorder>();
    r->begin();
    f(*r);
    r->end();
    return r;
  };
  std::vector<double> v;
  const mtp::MtpW &W = L.W;
  v.push_back(med(*rec([&](Recorder &r) { mtp::record_step(r, W, L.h1); })));
  v.push_back(med(*rec([&](Recorder &r) { mtp::record_step(r, W, L.h1, nullptr, nullptr, nullptr, 0, false); })));
  v.push_back(med(*rec([&](Recorder &r) { mtp::record_kv_step(r, W, L.h1, L.h1.tok, L.h1.hid); })));
  const uint32_t V = (uint32_t)(W.draft_bits ? W.draft_head.rows : W.vocab), Kk = (uint32_t)W.shortlist;
  if (W.draft_bits) {
    const uint32_t mapped = W.draft_ids.buf ? 1u : 0u, at = 0;
    v.push_back(med(*rec([&](Recorder &r) { nax::record_qn(r, 1, L.h1.xh, W.draft_head, W.draft_bits, L.h1.logits); })));
    v.push_back(med(*rec([&](Recorder &r) {
      r.dispatch(pso("mtp_topk_local"), {L.h1.logits, L.h1.cand, Arg::value(V), Arg::value(Kk)}, MTLSizeMake(128, 1, 1), sz(256));
    })));
    v.push_back(med(*rec([&](Recorder &r) {
      r.dispatch(pso("mtp_shortlist"), {L.h1.cand, L.h1.xh, W.lm_head.w, W.lm_head.s, W.lm_head.b, L.h1.amax, L.h1.sl,
                                            Arg::value(V), Arg::value(at), Arg::value(Kk), mapped ? W.draft_ids : L.h1.amax,
                                            Arg::value(mapped)}, sz(1), sz(1024));
    })));
    v.push_back(med(*rec([&](Recorder &r) {
      r.dispatch(pso("mtp_draft_sample", 32), {L.h1.sl, L.h1.amax, L.qb, L.dsp, Arg::value(1u)}, sz(1), sz(32));
    })));
  } else {
    v.insert(v.end(), {0, 0, 0, 0});
  }
  v.push_back(med(*rec([&](Recorder &r) {
    nax::record_q8(r, 1, L.h1.xh, {&W.lm_head}, {L.lg});
    r.barrier();
    r.dispatch(pso("mtp_argmax", 1024), {L.lg, L.amv, Arg::value((uint32_t)VOCAB), Arg::value(0u)}, sz(1), sz(1024));
  })));
  v.push_back(med(*rec([&](Recorder &r) { mtp::record_step(r, W, L.h2); })));
  for (int n = 1; n <= 2; n++)
    for (int D = 1; D <= 4; D++)
      v.push_back(D <= L.maxd && W.draft_bits ? med(get_draft(L, n, D, true)) : 0.0);
  for (int sd = 1; sd >= 0; sd--)
    for (int R = 2; R <= 7; R++) {
      if (R > L.maxd + 1) { v.push_back(0); continue; }
      MtpLoop::Verify &vv = get_verify(L, R, sd, 20);
      v.push_back(med(*vv.a, true, sd ? vv.bs.get() : vv.bg.get()));
    }
  reset(m);
  ctx_reset(L, 0);
  const int k = std::min<int>(nout, (int)v.size());
  for (int i = 0; i < k; i++) out[i] = v[i];
  return (int)v.size();
}

}  // namespace ne::fn

// ------------------------------------------------------------------------------------------------ C entry points
using namespace ne;
using namespace ne::fn;

struct ne_fn_mtp {
  MtpLoopPtr p;
  int maxd = 0;
  ne_fn_model *model = nullptr;
};
struct ne_fn_mtp_snap {
  ne_fn_snap *ms = nullptr;
  bool head = false;
  int Lh = 0;
  std::vector<float> ph;
  std::vector<int> pt;
  std::vector<int> ctx;  // lab-draft: the copy rule's token history (positions [ctx0, T))
  int ctx0 = 0;
  std::vector<uint8_t> dfl;  // lab-dflash: the drafter's context (kDflStateBytes; empty = none)
  ~ne_fn_mtp_snap() { ne_fn_snap_free(ms); }
};
static void stats_out(const MtpStats &s, int maxd, double *stats, int *hist) {
  if (stats) {
    const double r = std::max(1, s.rounds);
    const double v[11] = {s.prefill_ms,       s.head_prefill_ms,   s.decode_ms,         (double)s.tokens,
                          (double)s.rounds,   s.draft_gpu_ms / r,  s.verify_gpu_ms / r, s.gather_us / r,
                          s.host_us / r,      (double)maxd,        s.flush_ms};
    memcpy(stats, v, 10 * sizeof(double));  // 10 doubles (the old layout); flush ms only through generate_from
  }
  if (hist) for (size_t i = 0; i < s.hist.size(); i++) hist[i] = s.hist[i];
}

extern "C" {

NE_API ne_fn_mtp *ne_fn_mtp_create(ne_fn_model *m, int max_depth, int draft_bits) {
  ne_fn_mtp *out = nullptr;
  int rc = c_call([&] {
    NE_CHECK(m && m->m, "null model");
    auto u = std::make_unique<ne_fn_mtp>();
    u->p = mtp_create(*m->m, max_depth, draft_bits);
    u->maxd = max_depth;
    u->model = m;
    out = u.release();
  });
  return rc == 0 ? out : nullptr;
}

NE_API void ne_fn_mtp_free(ne_fn_mtp *p) { delete p; }

NE_API int ne_fn_mtp_time(ne_fn_mtp *p, int R, int T, int P, int reps, double *verify_ms, double *draft_ms) {
  return c_call([&] {
    NE_CHECK(p && verify_ms && draft_ms, "null argument");
    mtp_time(*p->p, R, T, P, reps, verify_ms, draft_ms);
  });
}

NE_API int ne_fn_mtp_set_copy(ne_fn_mtp *p, int min_match, int depth) {
  return c_call([&] {
    NE_CHECK(p && min_match >= 0, "bad arguments");
    MtpLoop &L = *p->p;
    if (min_match > 0 && L.copy_min <= 0) ctx_reset(L, L.m->T);  // the history starts now
    L.copy_min = min_match;
    L.cidx.clear();
    if (min_match <= 0) ctx_reset(L, L.m->T);
    L.copy_maxd = depth > 0 ? std::min(depth, L.maxd) : L.maxd;
  });
}
NE_API int ne_fn_mtp_set_qgate(ne_fn_mtp *p, int on) {
  return c_call([&] {
    NE_CHECK(p, "null argument");
    p->p->qgate = on > 0;
  });
}
// lab-verify: routed experts deduplicated across the verify rows (g_moe_dedupe) on / off; re-records every verify
static void rerecord_verifies(MtpLoop &L) {
    for (auto &kv : L.ver) {
      kv.second.a.reset();
      kv.second.bg.reset();
      kv.second.bs.reset();
    }
    for (int R = 2; R <= L.maxd + 1; R++) {  // record + first replays now (as at creation), then reset the model
      MtpLoop::Verify &v = get_verify(L, R, false, 0);
      get_verify(L, R, true, 20);
      v.a->run();
      v.bg->run();
      v.bs->run();
    }
    reset(*L.m);
}
// ---- lab-dflash, step A (docs/mtp.md 13): a host-driven loop for the DFlash2 feasibility measurement. The drafter
// runs outside (tools/dflash_ref.py); the engine verifies R rows at the current T, returns the sampled top-K readout
// (K = 20 (id bits, logit) pairs per row, as the loop's shortlist) and the target features, and commits c rows.
NE_API int ne_fn_dfl_capture(ne_fn_mtp *p, int on) {
  return c_call([&] {
    NE_CHECK(p, "null argument");
    if (p->p->dfl_cap == (on > 0)) return;
    p->p->dfl_cap = on > 0;
    rerecord_verifies(*p->p);
  });
}
NE_API int ne_fn_dfl_reset(ne_fn_mtp *p) {
  return c_call([&] {
    NE_CHECK(p, "null argument");
    MtpLoop &L = *p->p;
    reset(*L.m);
    ctx_reset(L, 0);
    u32(L.keep)[0] = 0;
    u32(L.xind)[0] = 0;
  });
}
// topk f32 [R][20][2], feat f32 [R][12800] (either may be NULL), gpu_ms (A + B; may be NULL)
NE_API int ne_fn_dfl_verify(ne_fn_mtp *p, const int *toks, int R, float *topk, float *feat, double *gpu_ms) {
  return c_call([&] {
    NE_CHECK(p && toks, "null argument");
    MtpLoop &L = *p->p;
    Model &m = *L.m;
    NE_CHECK(L.dfl_cap, "ne_fn_dfl_capture(p, 1) first");
    NE_CHECK(R >= 2 && R <= L.maxd + 1 && m.T + R + 2 <= m.tmax, "dfl_verify: R %d at T %d", R, m.T);
    MtpLoop::Verify &v = get_verify(L, R, true, 20);
    for (int j = 0; j < R; j++) {
      NE_CHECK(toks[j] >= 0 && toks[j] < VOCAB, "token %d", toks[j]);
      u32(L.vtok)[j] = (uint32_t)toks[j];
    }
    u32(m.pos)[0] = (uint32_t)m.T;
    u32(L.xind)[0] = 0;
    u32(L.xind)[1] = (uint32_t)R;
    u32(L.xind)[2] = 1u;
    v.a->commit();
    gather_ple_rows(m, m.hist[0], m.hist[1], toks, R, (float *)v.s.ple_emb.host());
    v.bs->commit();
    v.a->wait();
    v.bs->wait();
    if (topk) memcpy(topk, L.sl.host(), (size_t)R * 20 * 2 * 4);
    if (feat) memcpy(feat, L.feat.host(), (size_t)R * 5 * HID * 4);
    if (gpu_ms) *gpu_ms = v.a->gpu_ms() + v.bs->gpu_ms();
  });
}
// the first c rows of the last verify (toks = its tokens) enter every cache (lazily, as a round's commit)
NE_API int ne_fn_dfl_commit(ne_fn_mtp *p, const int *toks, int c) {
  return c_call([&] {
    NE_CHECK(p && toks && c >= 1 && c <= 8, "bad arguments");
    MtpLoop &L = *p->p;
    Model &m = *L.m;
    for (int j = 0; j < c; j++) {
      m.hist[0] = m.hist[1];
      m.hist[1] = toks[j];
    }
    m.T += c;
    m.gen++;
    m.dirty = true;
    u32(L.keep)[0] = (uint32_t)c;
  });
}
// lab-dflash: the DFlash2 drafter on (dir = the checkpoint) with B block rows, at most kmax drafts verified (the q-gate
// picks 1 .. kmax per sampled round), drafter weights at `bits` (8 / 4). Re-records the verifies (feature capture) and
// the model's step recordings; resets the model.
static void dfl_enable(MtpLoop &L, const std::string &dir, int B, int kmax, int bits, int gate) {
  Model &m = *L.m;
  auto d = std::make_unique<MtpLoop::Dfl>();
  d->w = dfl::load(dir, bits);
  NE_CHECK(B >= 2 && B <= dfl::RMAX, "dflash block %d (2..8)", B);
  d->B = B;
  d->kmax = std::max(1, std::min({kmax, B - 1, L.maxd}));
  d->kgreedy = std::min(d->kmax, 3);
  if (const char *e = getenv("NE_DFL_KGREEDY")) d->kgreedy = std::clamp(atoi(e), 1, d->kmax);
  if (const char *e = getenv("NE_DFL_MIN_CTX")) d->minctx = std::max(0, atoi(e));
  if (const char *e = getenv("NE_DFL_EARLY_A")) d->early_a = atoi(e) > 0;
  d->gate = gate != 0;
  d->c = dfl::alloc_ctx();
  d->s = dfl::alloc_scratch(B);
  for (int i = 0; i < dfl::RMAX; i++) u32(d->s.tok)[i] = (uint32_t)d->w.mask;
  d->fs = alloc(DType::F32, {dfl::W, dfl::FEAT});
  d->stage = alloc(DType::F32, {16, dfl::FEAT});
  // acceptance by q(d) bin, priors measured on the h2h prompts (docs/mtp.md 13); round ms by verify rows
  const double pri[10] = {0.22, 0.37, 0.40, 0.48, 0.52, 0.48, 0.58, 0.57, 0.71, 0.93};
  for (int b = 0; b < 10; b++) d->qacc[b] = d->qaccg[b] = d->qacc0[b] = pri[b];
  // round ms by verify rows (draft + A + B + host; in-loop 19 Oct, docs/mtp.md 13)
  const double vc[9] = {14.0, 14.5, 20.2, 23.9, 27.3, 31.6, 34.4, 37.5, 40.5};
  for (int R = 0; R < 9; R++) d->vcost[R] = d->vcostg[R] = d->vcost0[R] = vc[R];
  d->draft = std::make_unique<Recorder>();
  d->draft->begin();
  dfl::record_ctx_rows(*d->draft, d->w, d->c, d->s, L.feat, 8, 0);
  dfl::record_block(*d->draft, d->w, d->c, d->s, L.W, m.embed, L.vtok);
  d->draft->end();
  d->cfeat = std::make_unique<Recorder>();
  d->cfeat->begin();
  dfl::record_ctx_rows(*d->cfeat, d->w, d->c, d->s, L.feat, 8, 0);
  d->cfeat->end();
  L.dfl = std::move(d);
  L.dfl_cap = true;
  set_dfl_capture(m, true);
  prefill_set_dfl(chunked(m), &L.dfl->fs);
  rerecord_verifies(L);  // (resets the model)
  {  // first replays now (pipelines, residency), an empty context
    MtpLoop::Dfl &x = *L.dfl;
    memset(x.s.P.host(), 0, 32);
    const dfl::SelP sel{0, (uint32_t)(B - 1), 1, 1, 0, 0, 0, 0, 1.0f, 0, 0, 0};
    memcpy(x.s.SP.host(), &sel, sizeof sel);
    x.draft->run();
    x.cfeat->run();
    dfl_drain_rec(L, 1).run();
  }
  u32(L.vtok)[0] = 0;
  reset(m);
  ctx_reset(L, 0);
  dfl_reset(L, 0);
}
NE_API int ne_fn_dfl_enable(ne_fn_mtp *p, const char *dir, int block, int kmax, int bits, int gate) {
  return c_call([&] {
    NE_CHECK(p && dir, "null argument");
    dfl_enable(*p->p, dir, block, kmax, bits, gate);
  });
}
// lab-dflash: the gate on / off (sampled rounds verify kmax drafts when off); kmax (1 .. B - 1, <= max_depth)
NE_API int ne_fn_dfl_set(ne_fn_mtp *p, int gate, int kmax, int active) {
  return c_call([&] {
    NE_CHECK(p && p->p->dfl, "DFlash2 drafter not enabled");
    MtpLoop::Dfl &d = *p->p->dfl;
    if (active >= 0 && d.active != (active != 0)) {
      d.active = active != 0;
      d.gen = ~0ull;  // its context restarts with the next feed / run
    }
    if (gate >= 0) d.gate = gate != 0;
    if (kmax > 0) d.kmax = std::max(1, std::min({kmax, d.B - 1, p->p->maxd}));
    d.kgreedy = std::min(d.kgreedy, d.kmax);
  });
}
NE_API int ne_fn_dfl_on(ne_fn_mtp *p) { return p && p->p->dfl ? 1 : 0; }
// lab-dflash checks (tests/mtp/test_dflash.py): the drafter's context set directly from host feature rows, and one
// block draft from it with its internals read back.
NE_API int ne_fn_dfl_ctx_reset(ne_fn_mtp *p, int T) {
  return c_call([&] {
    NE_CHECK(p && p->p->dfl, "DFlash2 drafter not enabled");
    dfl_reset(*p->p, T);
  });
}
NE_API int ne_fn_dfl_ctx_push(ne_fn_mtp *p, int pos, const float *feat, int n) {
  return c_call([&] {
    NE_CHECK(p && p->p->dfl && feat && n >= 1, "bad arguments");
    dfl_push(*p->p, pos, feat, n);
    dfl_drain(*p->p);
  });
}
// hn f32 [B - 1][2560] (final-normed rows 1 ..), sl f32 [B - 1][16][2] (candidates: id bits, exact logit), drafts
// [B - 1], qo f32 [B - 1][16][2] (proposals); any may be NULL. The block sits at the context's end (T = c.T).
NE_API int ne_fn_dfl_probe(ne_fn_mtp *p, int anchor, int sampled, float temp, uint32_t seed, float *hn, float *sl,
                           int *drafts, float *qo, double *ms) {
  return c_call([&] {
    NE_CHECK(p && p->p->dfl, "DFlash2 drafter not enabled");
    MtpLoop &L = *p->p;
    MtpLoop::Dfl &d = *L.dfl;
    NE_CHECK(d.fsn == 0 && d.vcnt == 0, "dflash probe: context rows pending");
    uint32_t *P = u32(d.s.P);
    P[0] = (uint32_t)d.c.T;
    P[1] = 0;
    P[2] = (uint32_t)d.c.T;
    P[3] = (uint32_t)(d.c.T - std::min(d.c.n, dfl::W - 1));
    const dfl::SelP sel{(uint32_t)anchor, (uint32_t)(d.B - 1), sampled ? 1u : 0u, seed, 0, 0, 0, 0, temp, 0, 0, 0};
    memcpy(d.s.SP.host(), &sel, sizeof sel);
    u32(d.s.tok)[0] = (uint32_t)anchor;
    d.draft->run();
    const int Pn = d.B - 1;
    if (hn) memcpy(hn, d.s.hn.host(), (size_t)Pn * dfl::H * 4);
    if (sl) memcpy(sl, d.s.sl.host(), (size_t)Pn * dfl::NC * 2 * 4);
    if (qo) memcpy(qo, d.s.qo.host(), (size_t)Pn * dfl::NC * 2 * 4);
    if (drafts) for (int j = 0; j < Pn; j++) drafts[j] = (int)u32(d.s.dr)[j];
    if (ms) *ms = d.draft->gpu_ms();
  });
}
// the last run's DFlash2 rounds and the tokens they emitted (MtpStats::dfl_*)
NE_API int ne_fn_dfl_last(ne_fn_mtp *p, int *rounds, int *tokens) {
  if (!p) return -1;
  if (rounds) *rounds = p->p->last_dfl_rounds;
  if (tokens) *tokens = p->p->last_dfl_tokens;
  return 0;
}
NE_API int ne_fn_mtp_set_moe_dedupe(ne_fn_mtp *p, int on) {
  return c_call([&] {
    NE_CHECK(p, "null argument");
    const int v = on == 1 ? 8 : std::max(0, on);  // 1 = every R, N >= 2 = R <= N, 0 = off
    if (g_moe_dedupe == v) return;
    g_moe_dedupe = v;
    rerecord_verifies(*p->p);
  });
}
// lab-verify: routed gate / up split by multiplicity (g_moe_split) on / off; re-records the verifies
NE_API int ne_fn_mtp_set_moe_split(ne_fn_mtp *p, int on) {
  return c_call([&] {
    NE_CHECK(p, "null argument");
    if (g_moe_split == (on > 0)) return;
    g_moe_split = on > 0;
    rerecord_verifies(*p->p);
  });
}
NE_API int ne_fn_mtp_set_draft_vocab(ne_fn_mtp *p, const char *mode) {
  return c_call([&] {
    NE_CHECK(p && mode, "null argument");
    MtpLoop &L = *p->p;
    NE_CHECK(L.W.draft_bits, "no draft readout (draft_bits 0)");
    const char *old = getenv("NE_MTP_DRAFT_VOCAB");
    const std::string keep = old ? old : "";
    setenv("NE_MTP_DRAFT_VOCAB", mode, 1);
    const std::vector<uint32_t> rows = mtp::draft_vocab_rows(L.W.vocab);
    if (old) setenv("NE_MTP_DRAFT_VOCAB", keep.c_str(), 1); else unsetenv("NE_MTP_DRAFT_VOCAB");
    mtp::set_draft_head(L.W, L.W.draft_bits, L.W.shortlist, &rows);
    L.dr.clear();  // the draft recordings bind the draft head: record and replay them again
    for (int n = 1; n <= 2; n++)
      for (int D = 1; D <= L.maxd; D++)
        for (int sd = 0; sd < 2; sd++) get_draft(L, n, D, sd).run();
    reset(*L.m);
  });
}

NE_API int ne_fn_mtp_prof(ne_fn_mtp *p, int T, int reps, double *out, int nout) {
  int n = -1;
  int rc = c_call([&] {
    NE_CHECK(p && out && nout >= 0, "null argument");
    n = mtp_prof(*p->p, T, reps, out, nout);
  });
  return rc == 0 ? n : -1;
}

NE_API int ne_fn_mtp_verify_forced(ne_fn_mtp *p, const int *prompt, int n, const int *cont, int nc, int R,
                                   const int *keeps, int nk, float *logits) {
  int count = -1;
  int rc = c_call([&] {
    NE_CHECK(p && prompt && cont && keeps && logits, "null argument");
    count = mtp_verify_forced(*p->p, prompt, n, cont, nc, R, keeps, nk, logits);
  });
  return rc == 0 ? count : -1;
}

NE_API int ne_fn_mtp_generate(ne_fn_mtp *p, const int *prompt, int n, int max_new, int stop, int depth,
                              uint64_t seed, int *out, double *stats, int *hist) {
  return ne_fn_mtp_generate_cb(p, prompt, n, max_new, stop, depth, seed, out, stats, hist, nullptr, nullptr);
}

NE_API int ne_fn_mtp_generate_cb(ne_fn_mtp *p, const int *prompt, int n, int max_new, int stop, int depth,
                                 uint64_t seed, int *out, double *stats, int *hist, int (*cb)(void *, const int *, int),
                                 void *cb_ctx) {
  int count = -1;
  int rc = c_call([&] {
    NE_CHECK(p && prompt && out, "null argument");
    MtpStats s;
    count = mtp_generate(*p->p, prompt, n, max_new, stop, depth, seed, out, &s, cb, cb_ctx);
    stats_out(s, p->maxd, stats, hist);
  });
  return rc == 0 ? count : -1;
}

NE_API int ne_fn_mtp_head_ok(ne_fn_mtp *p) {
  int ok = 0;
  int rc = c_call([&] {
    NE_CHECK(p, "null argument");
    ok = mtp_head_ok(*p->p) ? 1 : 0;
  });
  return rc == 0 ? ok : -1;
}

NE_API int ne_fn_mtp_head_state(ne_fn_mtp *p, int *Lh, int *npend) {
  return c_call([&] {
    NE_CHECK(p, "null argument");
    if (Lh) *Lh = p->p->Lh;
    if (npend) *npend = (int)p->p->pt.size();
  });
}

NE_API int ne_fn_mtp_catch_up(ne_fn_mtp *p, double *ms) {
  return c_call([&] {
    NE_CHECK(p, "null argument");
    const double t = mtp_catch_up(*p->p);
    if (ms) *ms = t;
  });
}

NE_API int ne_fn_mtp_set_pend_cap(ne_fn_mtp *p, int cap) {
  return c_call([&] {
    NE_CHECK(p, "null argument");
    mtp_set_pend_cap(*p->p, cap);
  });
}

NE_API int ne_fn_mtp_feed(ne_fn_mtp *p, const int *toks, int n) {
  return c_call([&] {
    NE_CHECK(p && toks, "null argument");
    mtp_feed(*p->p, toks, n);
  });
}
NE_API int ne_fn_mtp_touch(ne_fn_mtp *p) {
  return c_call([&] {
    NE_CHECK(p, "null argument");
    MtpLoop &L = *p->p;
    for (auto &e : L.ver)
      for (Recorder *x : {e.second.a.get(), e.second.bg.get(), e.second.bs.get()})
        if (x) x->touch();
    for (auto &e : L.dr) e.second->touch();
    for (Recorder *x : {L.hp8.get(), L.hp1.get(), L.hkG[0].get(), L.hkG[1].get(), L.hk8[0].get(), L.hk8[1].get(),
                        L.hk1.get()})
      if (x) x->touch();
  });
}
NE_API int ne_fn_mtp_feed_ahead(ne_fn_mtp *p, const int *toks, int n, const int *ahead, int n_ahead) {
  return c_call([&] {
    NE_CHECK(p && toks && (ahead || n_ahead == 0) && n_ahead >= 0, "bad arguments");
    mtp_feed(*p->p, toks, n, ahead, n_ahead);
  });
}

NE_API int ne_fn_mtp_step(ne_fn_mtp *p, int tok, int *next, float *logprob) {
  return c_call([&] {
    NE_CHECK(p, "null argument");
    StepOut o = mtp_step(*p->p, tok, true);
    if (next) *next = o.token;
    if (logprob) *logprob = o.logprob;
  });
}

NE_API int ne_fn_mtp_generate_from(ne_fn_mtp *p, const int *toks, int n, int max_new, int stop, int depth,
                                   uint64_t seed, int *out, double *stats, int *hist,
                                   int (*cb)(void *, const int *, int), void *cb_ctx) {
  int count = -1;
  int rc = c_call([&] {
    NE_CHECK(p && toks && out && n >= 1, "bad arguments");
    MtpStats s;
    count = mtp_generate_from(*p->p, toks, n, max_new, stop, depth, seed, out, &s, cb, cb_ctx);
    stats_out(s, p->maxd, stats, hist);
  });
  return rc == 0 ? count : -1;
}

NE_API int ne_fn_mtp_context_len(ne_fn_mtp *p) {
  int n = -1;
  c_call([&] {
    NE_CHECK(p, "null argument");
    MtpLoop &L = *p->p;
    n = L.ctx0 + (int)L.ctx.size() == L.m->T ? (int)L.ctx.size() : 0;
  });
  return n;
}
NE_API int ne_fn_mtp_set_context(ne_fn_mtp *p, const int *toks, int n) {
  return c_call([&] {
    NE_CHECK(p && (toks || n == 0) && n >= 0, "bad arguments");
    MtpLoop &L = *p->p;
    NE_CHECK(n <= L.m->T, "ne_fn_mtp_set_context: %d tokens for a state of %d", n, L.m->T);
    for (int i = 0; i < n; i++) NE_CHECK(toks[i] >= 0 && toks[i] < VOCAB, "token %d out of range", toks[i]);
    ctx_reset(L, L.m->T - n);
    if (L.copy_min > 0) L.ctx.assign(toks, toks + n);
  });
}

NE_API ne_fn_mtp_snap *ne_fn_mtp_snapshot(ne_fn_mtp *p) {
  ne_fn_mtp_snap *out = nullptr;
  c_call([&] {
    NE_CHECK(p && p->model, "null argument");
    MtpLoop &L = *p->p;
    const bool head = mtp_head_ok(L);
    if (head && !L.m->dirty) head_drain(L, false, 0);  // lazy head: catch up, so a snapshot holds <= 8 pending rows
    ne_fn_snap *ms = ne_fn_model_snapshot(p->model);  // NULL (error set) on a dirty model
    NE_CHECK(ms, "ne_fn_mtp_snapshot: %s", ne_last_error());
    auto s = std::make_unique<ne_fn_mtp_snap>();
    s->ms = ms;
    s->head = head;
    if (head) {
      s->Lh = L.Lh;
      s->ph = L.ph;
      s->pt = L.pt;
    }
    if (L.ctx0 + (int)L.ctx.size() == ne_fn_snap_T(ms)) {
      s->ctx = L.ctx;
      s->ctx0 = L.ctx0;
    } else {
      s->ctx0 = ne_fn_snap_T(ms);
    }
    if (L.dfl) {
      s->dfl.resize(kDflStateBytes);
      dfl_export(L, s->dfl.data());
    }
    out = s.release();
  });
  return out;
}

NE_API int ne_fn_mtp_restore(ne_fn_mtp *p, const ne_fn_mtp_snap *s) {
  return c_call([&] {
    NE_CHECK(p && p->model && s && s->ms, "null argument");
    NE_CHECK(ne_fn_model_restore(p->model, s->ms) == 0, "ne_fn_mtp_restore: %s", ne_last_error());
    MtpLoop &L = *p->p;
    if (s->head) {
      L.Lh = s->Lh;
      L.ph = s->ph;
      L.pt = s->pt;
      L.head_gen = L.m->gen;
    } else {
      head_invalidate(L);
    }
    ctx_reset(L, s->ctx0);
    L.ctx = s->ctx;
    if (L.dfl) dfl_import(L, s->dfl.empty() ? nullptr : s->dfl.data());
  });
}

NE_API int ne_fn_mtp_snap_T(const ne_fn_mtp_snap *s) { return s ? ne_fn_snap_T(s->ms) : -1; }
NE_API int64_t ne_fn_mtp_snap_bytes(const ne_fn_mtp_snap *s) {
  return s ? ne_fn_snap_bytes(s->ms) + (int64_t)(s->ph.size() * 4 + s->pt.size() * 4 + s->ctx.size() * 4 + s->dfl.size())
           : -1;
}
NE_API void ne_fn_mtp_snap_free(ne_fn_mtp_snap *s) { delete s; }

// ---- lab-srvcache: head KV rows and state <-> host buffers
static constexpr int kMtpPendCap = 16;  // pending head rows a state buffer holds (in step, after a catch-up: <= 9)

NE_API int64_t ne_fn_mtp_kv_bytes(ne_fn_mtp *p, int rows) {
  int64_t n = -1;
  c_call([&] {
    NE_CHECK(p, "null argument");
    n = kv_bytes({&p->p->h1.st.qc}, rows);
  });
  return n;
}

NE_API int ne_fn_mtp_kv_io(ne_fn_mtp *p, int to_model, int a, int b, void *buf, int rows) {
  return c_call([&] {
    NE_CHECK(p, "null argument");
    kv_io({&p->p->h1.st.qc}, to_model != 0, a, b, buf, rows);
  });
}

NE_API int64_t ne_fn_mtp_state_bytes(ne_fn_mtp *p) {
  if (!p || !p->model) return -1;
  return ne_fn_state_bytes(p->model) + (int64_t)kMtpPendCap * (HCD + 1) * 4 + (p->p->dfl ? kDflStateBytes : 0);
}

NE_API int ne_fn_mtp_state_io(ne_fn_mtp *p, int to_model, void *buf, int64_t nbytes, int64_t *meta) {
  return c_call([&] {
    NE_CHECK(p && p->model && buf && meta, "null argument");
    MtpLoop &L = *p->p;
    const int64_t mb = ne_fn_state_bytes(p->model);
    NE_CHECK(nbytes == ne_fn_mtp_state_bytes(p), "state buffer %lld B, want %lld", (long long)nbytes,
             (long long)ne_fn_mtp_state_bytes(p));
    float *ph = (float *)((char *)buf + mb);
    int32_t *pt = (int32_t *)(ph + (size_t)kMtpPendCap * HCD);
    if (to_model) {
      if (meta[3]) {  // checked before the model state is touched
        NE_CHECK(meta[5] >= 0 && meta[5] <= kMtpPendCap, "pending rows %lld", (long long)meta[5]);
        NE_CHECK(meta[4] >= 0 && meta[4] + meta[5] == meta[0], "head: KV length %lld + %lld pending != T %lld",
                 (long long)meta[4], (long long)meta[5], (long long)meta[0]);
      }
      NE_CHECK(ne_fn_state_io(p->model, 1, buf, mb, meta) == 0, "%s", ne_last_error());  // bumps Model::gen
      if (meta[3]) {
        L.Lh = (int)meta[4];
        L.ph.assign(ph, ph + (size_t)meta[5] * HCD);
        L.pt.assign(pt, pt + meta[5]);
        L.head_gen = L.m->gen;
      } else {
        head_invalidate(L);
      }
      ctx_reset(L, L.m->T);  // the tokens are unknown here (ne_fn_mtp_set_context gives them)
      if (L.dfl) dfl_import(L, (const char *)(pt + kMtpPendCap));  // lab-dflash: the drafter's context
    } else {
      const bool head = mtp_head_ok(L);
      if (head && !L.m->dirty) head_drain(L, false, 0);  // lazy head (lab-headlazy): catch up, <= 8 rows stay pending
      if (head)  // a stale head's rows are not exported (meta[3] = 0)
        NE_CHECK(L.pt.size() <= (size_t)kMtpPendCap && L.ph.size() == L.pt.size() * HCD, "head: %zu pending rows",
                 L.pt.size());
      NE_CHECK(ne_fn_state_io(p->model, 0, buf, mb, meta) == 0, "%s", ne_last_error());
      meta[3] = head ? 1 : 0;
      meta[4] = head ? L.Lh : 0;
      meta[5] = head ? (int64_t)L.pt.size() : 0;
      if (head) {
        memcpy(ph, L.ph.data(), L.ph.size() * 4);
        memcpy(pt, L.pt.data(), L.pt.size() * 4);
      }
      if (L.dfl) dfl_export(L, (char *)(pt + kMtpPendCap));  // lab-dflash
    }
  });
}

}  // extern "C"
