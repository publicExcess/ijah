// Flash-Next MTP draft head (lab-mtp): weights, state, recording. Design: docs/mtp.md; kernels: mtp.metal + the NAX
// GEMMs (kernels/nax) + lab-forward's layer kernels (through nax::record_layer).
#include "models/flash_next/mtp.h"

#include "kernels/fn_rec.h"
#include "models/flash_next/draft_vocab.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <numeric>

#include <dispatch/dispatch.h>

namespace ne::mtp {
using namespace ne::fn;

namespace {
const std::string P = "language_model.mtp.";

Tensor T(const Pack &p, const std::string &n) {
  const Tensor &t = p.at(n);
  return t.gpu_address() % 16 == 0 ? t : p.aligned(n, 16);
}
Exl3 exl3(const Pack &p, const std::string &pre, int in, int out) {
  Exl3 e;
  e.t = T(p, pre + ".trellis");
  e.suh = T(p, pre + ".suh");
  e.svh = T(p, pre + ".svh");
  NE_CHECK(e.t.dtype == DType::U16 && e.t.ndim == 4 && e.t.dim(0) == NE && e.t.dim(1) == in / 16 &&
               e.t.dim(2) == out / 16 && e.t.dim(3) % 16 == 0,
           "%s: trellis %s", pre.c_str(), e.t.desc().c_str());
  e.k = (int)e.t.dim(3) / 16;
  NE_CHECK(e.k >= 4 && e.k <= 6, "%s: K%d not instantiated", pre.c_str(), e.k);
  return e;
}
HcW hc(const Pack &p, const std::string &pre, const Tensor *inject) {
  HcW h;
  h.norm = T(p, pre + ".hc_norm.weight");
  h.inject = inject ? *inject : T(p, pre + ".block_inject_weight.weight");
  h.down = nax::load_q8(p, pre + ".input_mix_weight_down");
  h.up = nax::load_q8(p, pre + ".input_mix_weight_up");
  NE_CHECK(h.down.rows == LR && h.down.in == HCD && h.up.rows == HCD && h.up.in == LR, "%s: hc shapes", pre.c_str());
  return h;
}
// = models/flash_next/layer.mm load_layer for an attention layer at another prefix (request: load_layer_prefix)
LayerW mtp_layer(const Pack &p) {
  LayerW w;
  w.layer = -1;
  w.linear = false;
  const std::string pre = P + "layers.0.";
  w.hca = hc(p, pre + "attn_hyper_connection", nullptr);
  w.hcm = hc(p, pre + "mlp_hyper_connection", nullptr);
  const std::string sa = pre + "self_attn.";
  AttnW &a = w.attn;
  a.q = nax::load_q8(p, sa + "q_proj");
  a.k = nax::load_q8(p, sa + "k_proj");
  a.v = nax::load_q8(p, sa + "v_proj");
  a.iqk = nax::load_q8(p, sa + "indexer.index_qk_proj");
  a.o = nax::load_q8(p, sa + "o_proj");
  NE_CHECK(a.q.rows == AQD && a.k.rows == AKVD && a.v.rows == AKVD && a.iqk.rows == AIQD && a.o.rows == HID,
           "mtp attention shapes");
  a.qnorm = T(p, sa + "q_norm.weight");
  a.knorm = T(p, sa + "k_norm.weight");
  a.iqnorm = T(p, sa + "indexer.q_layernorm.weight");
  a.iknorm = T(p, sa + "indexer.k_layernorm.weight");
  w.qsa = attn::QsaW{a.qnorm, a.knorm, a.iqnorm, a.iknorm};  // lab-attn's QSA step (the default attention path)
  MoeW &m = w.moe;
  const std::string mp = pre + "mlp.";
  m.router = T(p, mp + "gate.weight");
  m.sgate = T(p, mp + "shared_expert_gate.weight");
  m.sg = nax::load_q8(p, mp + "shared_expert.gate_proj");
  m.su = nax::load_q8(p, mp + "shared_expert.up_proj");
  m.sd = nax::load_q8(p, mp + "shared_expert.down_proj");
  NE_CHECK(m.sg.group == 128 && m.su.group == 128 && m.sd.group == 128,
           "mtp shared expert groups %d %d %d: fn_moe.metal assumes g128", m.sg.group, m.su.group, m.sd.group);
  m.gate = exl3(p, mp + "switch_mlp.gate_proj", HID, INTER);
  m.up = exl3(p, mp + "switch_mlp.up_proj", HID, INTER);
  m.down = exl3(p, mp + "switch_mlp.down_proj", INTER, HID);
  return w;
}
MTLSize sz(int n) { return MTLSizeMake(n, 1, 1); }
id<MTLComputePipelineState> pso(const std::string &name) { return Pipelines::get(name, {FC::b(0, g_ref_round)}); }
}  // namespace

MtpW load_mtp(const Pack &p) {
  const Json &tc = p.text_config();
  NE_CHECK(tc.get_int("mtp_num_hidden_layers", 0) == 1, "mtp_num_hidden_layers != 1");
  const Json *ded = tc.find("mtp_use_dedicated_embeddings");
  NE_CHECK(!ded || !ded->boolean(),
           "dedicated MTP embeddings are not wired (the head uses the target's embed_tokens)");
  MtpW w;
  w.fc_e = nax::load_q8(p, P + "fc_embedding");
  w.fc_h = nax::load_q8(p, P + "fc_hidden");
  w.embed = nax::load_q8(p, "language_model.model.embed_tokens");
  w.lm_head = nax::load_q8(p, "language_model.lm_head");
  w.norm_e = T(p, P + "pre_fc_norm_embedding.weight");
  w.norm_h = T(p, P + "pre_fc_norm_hidden.weight");
  NE_CHECK(w.embed.group == 64 && w.embed.in == HID, "embed_tokens: expected 8-bit g64 [V][2560]");
  NE_CHECK(w.norm_e.numel() == HID && w.norm_h.numel() == HCD, "pre_fc norms");
  w.layer = mtp_layer(p);
  w.mixer = hc(p, P + "hyper_connection_mixer", &w.layer.hca.inject);
  w.vocab = w.lm_head.rows;
  return w;
}

Q8 requantize(const Q8 &src, int bits, int group, const std::vector<uint32_t> *sel) {
  NE_CHECK((bits == 4 || bits == 2) && src.in % group == 0 && group % 16 == 0, "requantize: %d-bit g%d", bits, group);
  if (sel && sel->empty()) sel = nullptr;
  if (sel)
    for (size_t i = 0; i < sel->size(); i++)
      NE_CHECK((int)(*sel)[i] < src.rows && (i == 0 || (*sel)[i] > (*sel)[i - 1]), "requantize: rows not ascending / in range");
  Q8 q;
  q.rows = sel ? (int)sel->size() : src.rows;
  q.in = src.in;
  q.group = group;
  const int64_t words = (int64_t)src.in * bits / 32, ng = src.in / group;
  q.w = alloc(DType::U32, {q.rows, words});
  q.s = alloc(DType::BF16, {q.rows, ng});
  q.b = alloc(DType::BF16, {q.rows, ng});
  const uint8_t *w8 = (const uint8_t *)src.w.host();
  const uint16_t *s8 = (const uint16_t *)src.s.host(), *b8 = (const uint16_t *)src.b.host();
  uint32_t *wo = (uint32_t *)q.w.host();
  uint16_t *so = (uint16_t *)q.s.host(), *bo = (uint16_t *)q.b.host();
  auto f2bf = [](float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16); };
  auto bf2f = [](uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; };
  const int g8 = src.group, ng8 = src.in / g8, maxq = (1 << bits) - 1, per = 32 / bits;
  const int64_t chunk = 1024;
  dispatch_apply((size_t)((q.rows + chunk - 1) / chunk), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t c) {
    std::vector<float> row(src.in);
    for (int64_t r = (int64_t)c * chunk; r < std::min<int64_t>(q.rows, (int64_t)(c + 1) * chunk); r++) {
      const int64_t sr = sel ? (int64_t)(*sel)[r] : r;  // source row
      for (int k = 0; k < src.in; k++)
        row[k] = bf2f(s8[sr * ng8 + k / g8]) * w8[sr * src.in + k] + bf2f(b8[sr * ng8 + k / g8]);
      for (int g = 0; g < ng; g++) {
        float mn = row[g * group], mx = mn;
        for (int k = 1; k < group; k++) { mn = std::min(mn, row[g * group + k]); mx = std::max(mx, row[g * group + k]); }
        const uint16_t sb = f2bf(std::max((mx - mn) / maxq, 1e-8f)), bb = f2bf(mn);
        so[r * ng + g] = sb;
        bo[r * ng + g] = bb;
        const float sc = bf2f(sb), bi = bf2f(bb);
        for (int k = 0; k < group; k += per) {
          uint32_t wd = 0;
          for (int j = 0; j < per; j++) {
            const int qv = std::clamp((int)std::lround((row[g * group + k + j] - bi) / sc), 0, maxq);
            wd |= (uint32_t)qv << (bits * j);
          }
          wo[r * words + (g * group + k) / per] = wd;
        }
      }
    }
  });
  return q;
}

void set_draft_head(MtpW &w, int bits, int shortlist, const std::vector<uint32_t> *rows) {
  NE_CHECK(shortlist >= 1 && shortlist <= 32, "shortlist %d (1..32)", shortlist);
  NE_CHECK(w.lm_head.group == 128 && w.lm_head.in == HID, "mtp_shortlist assumes the 8-bit g128 [V][2560] lm_head");
  if (rows && (rows->empty() || (int)rows->size() >= w.vocab)) rows = nullptr;
  NE_CHECK(!rows || bits, "a pruned draft vocabulary needs the draft readout (bits 4 or 2)");
  NE_CHECK(!rows || rows->size() >= 32 * 128, "draft vocabulary of %zu rows: too small", rows ? rows->size() : (size_t)0);
  if (bits) w.draft_head = requantize(w.lm_head, bits, 64, rows);
  w.draft_ids = Tensor();
  if (rows) {
    w.draft_ids = alloc(DType::U32, {(int64_t)rows->size()});
    memcpy(w.draft_ids.host(), rows->data(), rows->size() * 4);
  }
  w.draft_bits = bits;
  w.shortlist = shortlist;
}

std::vector<uint32_t> draft_vocab_rows(int vocab) {
  const char *e = getenv("NE_MTP_DRAFT_VOCAB");
  const std::string v = e ? e : "1";
  std::vector<uint32_t> r;
  if (v.empty() || v == "0" || v == "full") return r;
  uint32_t prefix = kDraftPrefix;
  const bool splish = v == "1" || v == "on";
  if (!splish && v != "prefix") {
    prefix = (uint32_t)strtoul(v.c_str(), nullptr, 10);
    NE_CHECK(prefix >= 4096, "NE_MTP_DRAFT_VOCAB=%s: full / 0, 1 / on, prefix, or a row count >= 4096", v.c_str());
  }
  prefix = std::min<uint32_t>(prefix, (uint32_t)vocab);
  for (uint32_t t = 0; t < prefix; t++) r.push_back(t);
  if (splish && prefix == kDraftPrefix)
    for (uint32_t t : kDraftExtra) if (t >= prefix && t < (uint32_t)vocab) r.push_back(t);
  if ((int)r.size() >= vocab) r.clear();
  return r;
}

MtpHead alloc_head(const MtpW &w, int rows, int tmax) {
  NE_CHECK(rows >= 1 && rows <= nax::kMaxRows, "mtp rows %d", rows);
  MtpHead h;
  h.rows = rows;
  const int64_t R = rows;
  h.st = nax::alloc_state_rows(w.layer, rows, tmax);
  h.s = fn::alloc_scratch(rows);  // any R
  h.ex = nax::alloc_layer_extra();
  h.tok = alloc(DType::U32, {std::max<int64_t>(R, 16)});
  h.hid = alloc(DType::F32, {R, HCD});
  h.en = alloc(DType::F32, {R, HID});
  h.pe = alloc(DType::F32, {R, HID});
  h.hn = alloc(DType::F32, {R, HCD});
  h.h_in = alloc(DType::F32, {R, HC, HID});
  h.h_out = alloc(DType::F32, {R, HC, HID});
  h.h_out2 = alloc(DType::F32, {R, HC, HID});
  h.xh = alloc(DType::F32, {R, HID});
  h.logits = alloc(DType::F32, {R, (int64_t)w.vocab});
  h.amax = alloc(DType::U32, {std::max<int64_t>(R, 16)});
  h.sl = alloc(DType::F32, {R, 32, 2});
  h.cand = alloc(DType::F32, {R, 128, 32, 2});
  return h;
}

void record_prefill_chunk(Recorder &r, const MtpW &w, MtpHead &h, bool readout_last) {
  record_step(r, w, h, nullptr, nullptr, nullptr, 0, readout_last);
}

static void record_step_impl(Recorder &r, const MtpW &w, MtpHead &h, const Tensor *tok, const Tensor *hid,
                             const Tensor *h_out, uint32_t amax_at, bool readout, bool kv_only);
void record_step(Recorder &r, const MtpW &w, MtpHead &h, const Tensor *tok, const Tensor *hid, const Tensor *h_out,
                 uint32_t amax_at, bool readout) {
  record_step_impl(r, w, h, tok, hid, h_out, amax_at, readout, false);
}
void record_kv_step(Recorder &r, const MtpW &w, MtpHead &h, const Tensor &tok, const Tensor &hid) {
  record_step_impl(r, w, h, &tok, &hid, nullptr, 0, false, true);
}
static void record_step_impl(Recorder &r, const MtpW &w, MtpHead &h, const Tensor *tok, const Tensor *hid,
                             const Tensor *h_out, uint32_t amax_at, bool readout, bool kv_only) {
  const int R = h.rows;
  const Tensor &tk = tok ? *tok : h.tok;
  const Tensor &hd = hid ? *hid : h.hid;
  const Tensor &ho = h_out ? *h_out : h.h_out;
  const uint32_t V = (uint32_t)w.vocab;
  r.dispatch(pso("mtp_fuse_pre"), {tk, w.embed.w, w.embed.s, w.embed.b, w.norm_e, hd, w.norm_h, h.en, h.hn, Arg::value(V)},
             sz(R), sz(256));
  r.barrier();
  nax::record_q8(r, R, h.en, {&w.fc_e}, {h.pe});
  r.barrier();
  // fc_hidden over the 4R (token, stream) rows (<= 16 per dispatch) + pe of each token
  for (int r0 = 0; r0 < R; r0 += 4) {
    const int n = std::min(4, R - r0);
    nax::record_q8_rowbias(r, 4 * n, h.hn.rows(r0, n).view(0, DType::F32, {4 * n, HID}), w.fc_h, h.pe.rows(r0, n),
                           h.h_in.rows(r0, n).view(0, DType::F32, {4 * n, HID}));
  }
  r.barrier();
  nax::record_layer(r, w.layer, h.st, h.s, h.ex, h.h_in, ho, true, kv_only);
  r.dispatch(pso("mtp_pos_add"), {h.st.pos, Arg::value((uint32_t)R)}, sz(1), sz(32));
  if (!readout || kv_only) { r.barrier(); return; }
  // mixer: HC read without inject (fn_hc_pre's inject gates are computed and ignored)
  const std::string rs = "_r" + std::to_string(R);
  if (fn::hc_v2(R)) {  // HC read v2 (kernels/fn_rec.h)
    fn::krec::hc_pre3(r, w.mixer, R, ho, w.mixer.inject, h.s.inj_m, h.s.hc_part);
    fn::krec::hc_mix(r, w.mixer, R, ho, h.s.mix_m, h.s.inj_m, h.xh, h.s.hc_part);
  } else {
    nax::record_hc_pre(r, R, w.mixer, ho, h.s.mix_m, h.s.inj_m, h.s, h.ex);
    r.dispatch(pso("fn_hc_mix_g" + std::to_string(w.mixer.up.group) + rs),
               {ho, w.mixer.norm, w.mixer.up.w, w.mixer.up.s, w.mixer.up.b, h.s.mix_m, h.s.inj_m, h.xh}, sz(HID / 8),
               sz(256));
    r.barrier();
  }
  if (w.draft_bits) {
    record_draft_readout(r, w, R, h.xh, h.logits, h.cand, h.amax, h.sl, amax_at);
  } else {
    nax::record_q8(r, R, h.xh, {&w.lm_head}, {h.logits});
    r.barrier();
    r.dispatch(pso("mtp_argmax"), {h.logits, h.amax, Arg::value(V), Arg::value(amax_at)}, sz(R), sz(1024));
  }
  r.barrier();
}

void record_draft_readout(Recorder &r, const MtpW &w, int R, const Tensor &x, const Tensor &coarse, const Tensor &cand,
                          const Tensor &amax, const Tensor &sl, uint32_t at) {
  // V = the draft head's rows (the pruned vocabulary's compact rows, or all ids); the shortlist maps compact rows to
  // ids through draft_ids (mapped = 1) before the exact re-score
  const uint32_t V = (uint32_t)w.draft_head.rows, K = (uint32_t)w.shortlist, mapped = w.draft_ids.buf ? 1u : 0u;
  nax::record_qn(r, R, x, w.draft_head, w.draft_bits, coarse);
  r.barrier();
  r.dispatch(pso("mtp_topk_local"), {coarse, cand, Arg::value(V), Arg::value(K)}, MTLSizeMake(128, R, 1), sz(256));
  r.barrier();
  r.dispatch(pso("mtp_shortlist"), {cand, x, w.lm_head.w, w.lm_head.s, w.lm_head.b, amax, sl, Arg::value(V),
                                    Arg::value(at), Arg::value(K), mapped ? w.draft_ids : amax, Arg::value(mapped)},
             sz(R), sz(1024));
}

void record_chain(Recorder &r, const MtpW &w, MtpHead &h, int depth) {
  NE_CHECK(h.rows == 1 && depth >= 1 && depth <= 16, "chain: rows 1, depth 1..16");
  for (int k = 0; k < depth; k++) {
    const Tensor tok_k = h.amax.view(4 * (size_t)std::max(k - 1, 0), DType::U32, {1});
    const Tensor *ho = (k & 1) ? &h.h_out2 : &h.h_out;
    const Tensor *hi = k == 0 ? &h.hid : ((k & 1) ? &h.h_out : &h.h_out2);
    record_step(r, w, h, k == 0 ? nullptr : &tok_k, hi, ho, (uint32_t)k);
  }
}

// ------------------------------------------------------------------------------------------------ verify / accept
bool is_greedy(const SampleParams &sp) { return sp.temperature < 0.01f; }

double Rng::uniform() {
  uint64_t z = (s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  z ^= z >> 31;
  return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}

float Dist::prob(int id) const {
  for (size_t i = 0; i < ids.size(); i++) if (ids[i] == id) return p[i];
  return 0.0f;
}
int Dist::sample(double u) const {
  double c = 0;
  for (size_t i = 0; i < ids.size(); i++) {
    c += p[i];
    if (u < c) return ids[i];
  }
  return ids.empty() ? -1 : ids.back();
}

Dist make_dist(const float *logits, int V, const SampleParams &sp) {
  Dist d;
  if (is_greedy(sp)) {
    d.ids.push_back((int)(std::max_element(logits, logits + V) - logits));
    d.p.push_back(1.0f);
    return d;
  }
  const double invT = 1.0 / sp.temperature;
  float mx = *std::max_element(logits, logits + V);
  std::vector<int> idx(V);
  std::iota(idx.begin(), idx.end(), 0);
  int keep = V;
  if (sp.top_k > 0 && sp.top_k < V) {
    keep = sp.top_k;
    std::partial_sort(idx.begin(), idx.begin() + keep, idx.end(), [&](int a, int b) {
      return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
    });
  } else {
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); });
  }
  std::vector<double> w(keep);
  double z = 0;
  for (int i = 0; i < keep; i++) { w[i] = std::exp(((double)logits[idx[i]] - mx) * invT); z += w[i]; }
  int nk = keep;
  if (sp.min_p > 0.0f) {   // (the list is sorted: a prefix) min_p before top-p, as fn_sample_p2
    nk = 1;
    while (nk < keep && w[nk] >= (double)sp.min_p * w[0]) nk++;
    z = 0;
    for (int i = 0; i < nk; i++) z += w[i];
  }
  int n = nk;
  if (sp.top_p < 1.0f) {
    double c = 0;
    for (int i = 0; i < nk; i++) {
      c += w[i] / z;
      if (c >= sp.top_p) { n = i + 1; break; }
    }
  }
  double zn = 0;
  for (int i = 0; i < n; i++) zn += w[i];
  d.ids.assign(idx.begin(), idx.begin() + n);
  d.p.resize(n);
  for (int i = 0; i < n; i++) d.p[i] = (float)(w[i] / zn);
  return d;
}

VerifyResult accept_greedy(const int *drafts, int D, const int *am) {
  VerifyResult r;
  while (r.accepted < D && drafts[r.accepted] == am[r.accepted]) r.emitted.push_back(drafts[r.accepted++]);
  r.next_token = am[r.accepted];
  r.emitted.push_back(r.next_token);
  return r;
}

VerifyResult accept_sampled(const int *drafts, int D, const float *tl, const float *dl, int V, const SampleParams &sp,
                            Rng &rng) {
  if (!dl || is_greedy(sp)) return accept_sampled_q(drafts, D, tl, nullptr, V, sp, rng);
  std::vector<Dist> q(D);
  for (int i = 0; i < D; i++) q[i] = make_dist(dl + (size_t)i * V, V, sp);
  return accept_sampled_q(drafts, D, tl, q.data(), V, sp, rng);
}

Dist shortlist_dist(const float *sl, int K, const SampleParams &sp) {
  std::vector<float> lg(K);
  std::vector<int> id(K);
  for (int k = 0; k < K; k++) { memcpy(&id[k], &sl[2 * k], 4); lg[k] = sl[2 * k + 1]; }
  Dist local = make_dist(lg.data(), K, sp);  // over positions 0..K-1
  for (auto &x : local.ids) x = id[x];
  return local;
}

VerifyResult accept_sampled_q(const int *drafts, int D, const float *tl, const Dist *qd, int V, const SampleParams &sp,
                              Rng &rng) {
  if (is_greedy(sp)) {
    std::vector<int> am(D + 1);
    for (int i = 0; i <= D; i++) am[i] = (int)(std::max_element(tl + (size_t)i * V, tl + (size_t)(i + 1) * V) - (tl + (size_t)i * V));
    return accept_greedy(drafts, D, am.data());
  }
  std::vector<Dist> p(D + 1);
  for (int i = 0; i <= D; i++) p[i] = make_dist(tl + (size_t)i * V, V, sp);
  return accept_sampled_p(drafts, D, p.data(), qd, rng);
}

VerifyResult accept_sampled_p(const int *drafts, int D, const Dist *pd_, const Dist *qd, Rng &rng) {
  VerifyResult r;
  for (int i = 0; i < D; i++) {
    const Dist &p = pd_[i];
    const int d = drafts[i];
    const double pd = p.prob(d);
    const Dist *q = qd ? &qd[i] : nullptr;
    const double qx = q ? q->prob(d) : 1.0;
    const double a = std::min(1.0, pd / std::max(qx, 1e-12));
    if (rng.uniform() < a) {
      r.emitted.push_back(d);
      r.accepted++;
      continue;
    }
    // residual norm(max(p - q, 0)) over p's support (q's mass outside p's support cannot be positive in p - q)
    Dist res;
    double z = 0;
    for (size_t j = 0; j < p.ids.size(); j++) {
      const double qj = q ? q->prob(p.ids[j]) : (p.ids[j] == d ? 1.0 : 0.0);
      const double v = std::max(0.0, (double)p.p[j] - qj);
      if (v > 0) { res.ids.push_back(p.ids[j]); res.p.push_back((float)v); z += v; }
    }
    if (z <= 0) res = p;  // p == q exactly: (unreachable with a < 1) keep the target
    else for (auto &v : res.p) v = (float)(v / z);
    r.next_token = res.sample(rng.uniform());
    r.emitted.push_back(r.next_token);
    return r;
  }
  r.next_token = pd_[D].sample(rng.uniform());
  r.emitted.push_back(r.next_token);
  return r;
}

CommitPlan plan_commit(int T0, int D, int accepted, int head_appended) {
  NE_CHECK(accepted >= 0 && accepted <= D, "accepted %d of %d", accepted, D);
  CommitPlan c;
  c.keep_rows = accepted + 1;
  c.kv_len = T0 + c.keep_rows;
  c.gdn_row = accepted;
  c.head_keep = std::min(accepted, head_appended);
  c.head_trim = head_appended - c.head_keep;
  c.head_rows = accepted - c.head_keep + 1;
  return c;
}

DepthPolicy::DepthPolicy(std::vector<double> round_cost_ms, double prior) : cost(std::move(round_cost_ms)) {
  NE_CHECK(cost.size() >= 2, "cost table needs depth 0 and 1");
  max_depth = std::min<int>(max_depth, (int)cost.size() - 1);
  acc.assign(16, prior);
}
double DepthPolicy::expected_tokens(int d) const {
  double e = 1, run = 1;
  for (int k = 0; k < d; k++) { run *= acc[k]; e += run; }
  return e;
}
int DepthPolicy::next_depth() const {
  int best = min_depth;
  double bv = -1;
  for (int d = min_depth; d <= max_depth && d < (int)cost.size(); d++) {
    const double v = expected_tokens(d) / cost[d];
    if (v > bv) { bv = v; best = d; }
  }
  return best;
}
void DepthPolicy::observe(int drafted, int accepted, double round_ms) {
  for (int i = 0; i < drafted && i <= accepted; i++) {
    const double hit = i < accepted ? 1.0 : 0.0;  // position i was reached; accepted iff i < accepted
    acc[i] = (1 - ema) * acc[i] + ema * hit;
  }
  // positions this depth cannot reach relax toward the deepest drafted one (lab-loop: without this, a dip in acc[d]
  // while drafting d + 1 deep makes the policy stay at depth d for good -- acc[d] is never observed again)
  if (drafted >= 1)
    for (size_t i = drafted; i < acc.size(); i++) acc[i] = (1 - ema) * acc[i] + ema * acc[drafted - 1];
  if (round_ms > 0 && drafted < (int)cost.size()) cost[drafted] = 0.9 * cost[drafted] + 0.1 * round_ms;
}

// ------------------------------------------------------------------------------------------------ copy rule
static uint64_t win_hash(const int *t, int n, int extra = -1) {  // FNV-1a over t[0, n) (+ extra if >= 0)
  uint64_t h = 1469598103934665603ull;
  for (int i = 0; i < n; i++) { h ^= (uint32_t)t[i]; h *= 1099511628211ull; }
  if (extra >= 0) { h ^= (uint32_t)extra; h *= 1099511628211ull; }
  return h;
}
int copy_find(const std::vector<int> &ctx, CopyIndex &idx, int N, int t1, int max, int *out) {
  const int n = (int)ctx.size();
  if (N <= 0 || max <= 0 || n + 1 <= N) return 0;
  const int *c = ctx.data();
  // index the windows ending at e in [max(end, N), n]: S[e - N, e) lies inside ctx; S[e] = ctx[e] or t1 (e = n)
  for (int e = std::max(idx.end, N); e <= n; e++) idx.map[win_hash(c + e - N, N)] = e;
  idx.end = n + 1;
  // the current window S[n + 1 - N, n + 1) = ctx[n + 1 - N, n) + t1 (hashed in the same order)
  const auto it = idx.map.find(win_hash(c + n + 1 - N, N - 1, t1));
  if (it == idx.map.end()) return 0;
  const int e = it->second;
  auto S = [&](int i) { return i < n ? c[i] : t1; };
  for (int i = 0; i < N; i++) if (S(e - N + i) != S(n + 1 - N + i)) return 0;  // a hash collision
  int k = 0;
  while (k < max && e + k <= n) { out[k] = S(e + k); k++; }
  return k;
}

}  // namespace ne::mtp
