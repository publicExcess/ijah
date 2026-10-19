// Flash-Next decoder layer graph: one GDN (linear_attention) layer decode step at 1 row as one recording.
// Owner: lab-layer. Kernels: kernels/fn_trunk.metal, kernels/fn_moe.metal; math and layouts: docs/kernels.md.
//
//  1 fn_hc_pre      attn HC: per-stream rms, down 10240->320 (q8) + silu(/4), block_inject gates 2 sigmoid(/4)
//  2 fn_hc_mix      attn HC: up 320->10240 (q8), sigmoid * normed, mean over streams -> GDN input [2560]
//  3 fn_gdn_inproj  qkv / z / b / a (q8 g128) + conv1d(4) + silu, beta, g
//  4 fn_gdn_step    gated delta rule (state f32 [48][128][128]) + gated rmsnorm
//  5 fn_gdn_outproj out_proj 6144->2560 (q8) + HC write h_mid = h + out * gate_s
//  6 fn_hc_pre      mlp HC on h_mid
//  7 fn_hc_mix      -> MoE input [2560]
//  8 fn_moe_router  router + top-10 + shared gate/up
//  9 fn_moe_gateup  EXL3 experts gate/up + shared down
// 10 fn_moe_down    EXL3 experts down + weighted reduce + sigmoid-gated shared + HC write h_out = h_mid + mlp * gate_s
#include "models/flash_next/layer.h"

#include "kernels/fn_rec.h"
#include "kernels/nax/nax.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace ne::fn {

// Pack tensors whose data is not 16-byte aligned (the K5 pack's EXL3 shards have odd header lengths; some trunk
// shards put U32 weights at 2 mod 16) come from Pack::aligned() (a cached aligned copy; RAM = the tensor): the
// kernels use 4..16-byte vector loads. The real fix is an aligned pack (engine-cli align-pack, docs/core.md).
size_t g_aligned_copy_bytes = 0;
static Tensor T(const Pack &p, const std::string &name) {
  const Tensor &t = p.at(name);
  if (t.gpu_address() % 16 == 0) return t;
  const Tensor &c = p.aligned(name, 16);
  g_aligned_copy_bytes = p.aligned_copy_bytes();
  return c;
}

static Q8 q8(const Pack &p, const std::string &pre) {
  Q8 q;
  const t6::Fmt f = t6::detect(p, pre);
  if (f == t6::Fmt::EXL3 || f == t6::Fmt::BF16) {  // lab-t6wire: T6 pack trunk Linear (layer_t6.mm records it)
    auto l = std::make_shared<t6::Linear>(t6::load(p, pre));
    q.rows = l->out;
    q.in = l->in;
    q.t6 = std::move(l);
    return q;
  }
  q.w = T(p, pre + ".weight");
  q.s = T(p, pre + ".scales");
  q.b = T(p, pre + ".biases");
  NE_CHECK(q.w.dtype == DType::U32 && q.s.dtype == DType::BF16 && q.b.dtype == DType::BF16 && q.w.ndim == 2,
           "%s: expected U32 weight + BF16 scales/biases, got %s %s", pre.c_str(), q.w.desc().c_str(),
           q.s.desc().c_str());
  q.rows = (int)q.w.dim(0);
  q.in = (int)q.w.dim(1) * 4;  // 8-bit: 4 per u32
  NE_CHECK(q.s.dim(0) == q.rows && q.in % q.s.dim(1) == 0, "%s: scales shape %s", pre.c_str(), q.s.desc().c_str());
  q.group = q.in / (int)q.s.dim(1);
  NE_CHECK(q.group == 64 || q.group == 128, "%s: group %d not instantiated", pre.c_str(), q.group);
  return q;
}
Q8 load_q8(const Pack &p, const std::string &prefix) { return q8(p, prefix); }
Tensor load_tensor(const Pack &p, const std::string &name) { return T(p, name); }

static Exl3 exl3(const Pack &p, const std::string &pre, int in, int out) {
  Exl3 e;
  e.t = T(p, pre + ".trellis");
  e.suh = T(p, pre + ".suh");
  e.svh = T(p, pre + ".svh");
  NE_CHECK(e.t.dtype == DType::U16 && e.t.ndim == 4 && e.t.dim(0) == NE && e.t.dim(1) == in / 16 &&
               e.t.dim(2) == out / 16 && e.t.dim(3) % 16 == 0,
           "%s: trellis %s", pre.c_str(), e.t.desc().c_str());
  NE_CHECK(e.suh.dtype == DType::F16 && e.svh.dtype == DType::F16, "%s: suh/svh must be F16", pre.c_str());
  e.k = (int)e.t.dim(3) / 16;
  NE_CHECK(e.k >= 4 && e.k <= 6, "%s: K%d not instantiated", pre.c_str(), e.k);
  return e;
}
static void expect(const Q8 &q, int rows, int in, const char *what) {
  NE_CHECK(q.rows == rows && q.in == in, "%s: [%d,%d], expected [%d,%d]", what, q.rows, q.in, rows, in);
}
static HcW hc(const Pack &p, const std::string &pre) {
  HcW h;
  h.norm = T(p, pre + ".hc_norm.weight");
  h.inject = T(p, pre + ".block_inject_weight.weight");
  h.down = q8(p, pre + ".input_mix_weight_down");
  h.up = q8(p, pre + ".input_mix_weight_up");
  expect(h.down, LR, HCD, "hc down");
  expect(h.up, HCD, LR, "hc up");
  NE_CHECK(h.down.group == h.up.group && !h.down.t6 == !h.up.t6, "%s: down/up groups / formats differ", pre.c_str());
  NE_CHECK(h.norm.dtype == DType::BF16 && h.norm.numel() == HCD, "%s: hc_norm %s", pre.c_str(), h.norm.desc().c_str());
  NE_CHECK(h.inject.dtype == DType::BF16 && h.inject.numel() == HC * HCD, "%s: inject", pre.c_str());
  return h;
}

LayerW load_layer(const Pack &p, int layer) {
  return load_layer_prefix(p, "language_model.model.layers." + std::to_string(layer) + ".", layer);
}

LayerW load_layer_prefix(const Pack &p, const std::string &pre, int layer) {
  LayerW w;
  w.layer = layer;
  w.linear = p.has(pre + "linear_attn.in_proj_qkv.weight") || p.has(pre + "linear_attn.in_proj_qkv.trellis");
  NE_CHECK(w.linear || p.has(pre + "self_attn.q_proj.weight") || p.has(pre + "self_attn.q_proj.trellis"),
           "layer %d: neither GDN nor attention", layer);
  if (p.has(pre + "ple.norm_conv.weight")) {
    PleW &e = w.ple;
    const std::string pl = pre + "ple.";
    e.on = true;
    e.key = q8(p, pl + "key_proj");
    e.value = q8(p, pl + "value_proj");
    expect(e.key, HCD, HID, "ple key_proj");
    expect(e.value, HID, HID, "ple value_proj");
    NE_CHECK((e.key.t6 && e.value.t6) || (e.key.group == 64 && e.value.group == 64),
             "ple projections must be g64 (fn_model.metal) or both t6");
    e.nk = T(p, pl + "norm_key.weight");
    e.nq = T(p, pl + "norm_query.weight");
    e.nc = T(p, pl + "norm_conv.weight");
    e.conv = T(p, pl + "conv1d.weight");
    NE_CHECK(e.nk.numel() == HCD && e.nq.numel() == HCD && e.nc.numel() == HCD && e.conv.numel() == HCD * 4 &&
                 e.conv.dtype == DType::BF16,
             "ple norms / conv shapes");
    auto i64 = [&](const std::string &n, int64_t *dst, int count) {
      const Tensor &t = p.at(pl + "ple_embedding." + n);
      NE_CHECK(t.dtype == DType::I64 && t.numel() == count, "%s: %s", n.c_str(), t.desc().c_str());
      memcpy(dst, t.host(), count * 8);
    };
    i64("layer_multipliers", e.mult, 3);
    i64("ngram_heads_vocab_sizes", e.sizes, PLE_NG);
    i64("ngram_heads_offsets", e.offsets, PLE_NG);
    const Tensor &ws = p.at(pl + "ple_embedding.ngram_embedding.weight_scale");
    NE_CHECK(ws.dtype == DType::BF16 && *(const uint16_t *)ws.host() == 0x3F80, "ple weight_scale must be 1.0");
  }
  w.hca = hc(p, pre + "attn_hyper_connection");
  w.hcm = hc(p, pre + "mlp_hyper_connection");
  if (!w.linear) {
    const std::string sa = pre + "self_attn.";
    AttnW &a = w.attn;
    a.q = q8(p, sa + "q_proj");
    a.k = q8(p, sa + "k_proj");
    a.v = q8(p, sa + "v_proj");
    a.iqk = q8(p, sa + "indexer.index_qk_proj");
    a.o = q8(p, sa + "o_proj");
    expect(a.q, AQD, HID, "q_proj");
    expect(a.k, AKVD, HID, "k_proj");
    expect(a.v, AKVD, HID, "v_proj");
    expect(a.iqk, AIQD, HID, "index_qk_proj");
    expect(a.o, HID, AOD, "o_proj");
    NE_CHECK(a.q.group == a.k.group && a.k.group == a.v.group && a.v.group == a.iqk.group,
             "layer %d: q/k/v/indexer groups differ", layer);
    a.qnorm = T(p, sa + "q_norm.weight");
    a.knorm = T(p, sa + "k_norm.weight");
    a.iqnorm = T(p, sa + "indexer.q_layernorm.weight");
    a.iknorm = T(p, sa + "indexer.k_layernorm.weight");
    NE_CHECK(a.qnorm.numel() == AHD && a.knorm.numel() == AHD && a.qnorm.dtype == DType::BF16, "q/k norm");
    if (g_attn_qsa) w.qsa = attn::QsaW{a.qnorm, a.knorm, a.iqnorm, a.iknorm};  // = load_qsa_weights, any prefix
  }
  const std::string la = pre + "linear_attn.";
  GdnW &g = w.gdn;
  if (w.linear) {
  g.qkv = q8(p, la + "in_proj_qkv");
  g.z = q8(p, la + "in_proj_z");
  g.a = q8(p, la + "in_proj_a");
  g.b = q8(p, la + "in_proj_b");
  g.out = q8(p, la + "out_proj");
  expect(g.qkv, CONVD, HID, "in_proj_qkv");
  expect(g.z, VALD, HID, "in_proj_z");
  expect(g.a, NVH, HID, "in_proj_a");
  expect(g.b, NVH, HID, "in_proj_b");
  expect(g.out, HID, VALD, "out_proj");
  NE_CHECK(g.z.group == g.a.group && g.a.group == g.b.group && (g.qkv.group == g.z.group || (g.qkv.group == 64 && g.z.group == 128)),
           "layer %d: in_proj groups (qkv %d, z %d, a %d, b %d) not instantiated", layer, g.qkv.group,
           g.z.group, g.a.group, g.b.group);
  g.conv = T(p, la + "conv1d.weight");
  g.alog = T(p, la + "A_log");
  g.dtb = T(p, la + "dt_bias");
  g.norm = T(p, la + "norm.weight");
  NE_CHECK(g.conv.dtype == DType::BF16 && g.conv.numel() == CONVD * 4, "conv1d %s", g.conv.desc().c_str());
  NE_CHECK(g.alog.dtype == DType::BF16 && g.dtb.dtype == DType::BF16 && g.norm.dtype == DType::BF16, "gdn dtypes");
  }
  MoeW &m = w.moe;
  const std::string mp = pre + "mlp.";
  m.router = T(p, mp + "gate.weight");
  m.sgate = T(p, mp + "shared_expert_gate.weight");
  NE_CHECK(m.router.dtype == DType::BF16 && m.router.numel() == (int64_t)NE * HID, "router %s", m.router.desc().c_str());
  NE_CHECK(m.sgate.dtype == DType::BF16 && m.sgate.numel() == HID, "shared_expert_gate");
  m.sg = q8(p, mp + "shared_expert.gate_proj");
  m.su = q8(p, mp + "shared_expert.up_proj");
  m.sd = q8(p, mp + "shared_expert.down_proj");
  expect(m.sg, INTER, HID, "shared gate");
  expect(m.su, INTER, HID, "shared up");
  expect(m.sd, HID, INTER, "shared down");
  NE_CHECK((m.sg.t6 && m.su.t6 && m.sd.t6) || (m.sg.group == 128 && m.su.group == 128 && m.sd.group == 128),
           "shared expert must be g128 (fn_moe.metal) or all t6");
  m.gate = exl3(p, mp + "switch_mlp.gate_proj", HID, INTER);
  m.up = exl3(p, mp + "switch_mlp.up_proj", HID, INTER);
  m.down = exl3(p, mp + "switch_mlp.down_proj", INTER, HID);
  NE_CHECK(m.gate.k == m.up.k && m.up.k == m.down.k, "per-projection K differs: instantiate per projection");
  // lab-t6wire: a T6 pack converts whole layers (every trunk Linear EXL3, docs/trunk6.md); mixed layers are not wired
  {
    std::vector<const Q8 *> lins = {&w.hca.down, &w.hca.up, &w.hcm.down, &w.hcm.up, &m.sg, &m.su, &m.sd};
    if (w.linear) lins.insert(lins.end(), {&g.qkv, &g.z, &g.a, &g.b, &g.out});
    else lins.insert(lins.end(), {&w.attn.q, &w.attn.k, &w.attn.v, &w.attn.iqk, &w.attn.o});
    if (w.ple.on) lins.insert(lins.end(), {&w.ple.key, &w.ple.value});
    int n6 = 0;
    for (const Q8 *q : lins) n6 += q->t6 ? 1 : 0;
    NE_CHECK(n6 == 0 || n6 == (int)lins.size(), "layer %d: %d of %d trunk Linears EXL3 (mixed layers not supported)",
             layer, n6, (int)lins.size());
    w.t6 = n6 > 0;
  }
  return w;
}

GdnState alloc_gdn_state() {
  GdnState s;
  s.conv = alloc(DType::F32, {3, CONVD});
  s.ssm = alloc(DType::F32, {NVH, DV, DK});
  s.lz = alloc(DType::U32, {NVH});
  s.pend = alloc(DType::F32, {NVH, 15, 2 * DK + 4});  // = fn_trunk.metal GDN_KP, GDN_PST
  memset(s.lz.host(), 0, s.lz.nbytes());
  return s;
}

bool gdn_pending(const GdnState &st) {
  if (!st.lz.buf) return false;
  const uint32_t *p = (const uint32_t *)st.lz.host();
  for (int i = 0; i < NVH; i++)
    if (p[i]) return true;
  return false;
}
void gdn_clear_pending(GdnState &st) {
  if (st.lz.buf) memset(st.lz.host(), 0, st.lz.nbytes());
}
void record_gdn_materialise(Recorder &r, GdnState &st) {
  r.dispatch(pso("fn_gdn_materialise", 1024), {st.ssm, st.lz, st.pend}, MTLSizeMake(NVH, 1, 1),
             MTLSizeMake(1024, 1, 1));
}
void gdn_materialise(std::vector<GdnState> &sts) {
  std::vector<GdnState *> todo;
  for (auto &st : sts)
    if (gdn_pending(st)) todo.push_back(&st);
  if (todo.empty()) return;
  // one-off recording (only at reader boundaries: snapshot, MTP start, chunked prefill entry, layer API reads)
  Recorder r;
  r.begin();
  for (GdnState *st : todo) record_gdn_materialise(r, *st);
  r.barrier();
  r.end();
  r.run();
}
void gdn_materialise(GdnState &st) {
  if (!gdn_pending(st)) return;
  Recorder r;
  r.begin();
  record_gdn_materialise(r, st);
  r.barrier();
  r.end();
  r.run();
}

GdnState alloc_attn_state(int tmax, int rows) {
  GdnState s;
  s.tmax = tmax;
  s.kc = alloc(DType::F32, {NKV, tmax, AHD});
  s.vc = alloc(DType::F32, {NKV, tmax, AHD});
  s.ikeys = alloc(DType::F32, {tmax, 128});
  s.pos = alloc(DType::U32, {4});
  NE_CHECK(tmax % 32 == 0, "tmax %d must be a multiple of 32", tmax);
  s.apart = alloc(DType::F32, {NKV, tmax / 32, std::max(rows, 4), NQH / NKV, AHD + 2});
  return s;
}
GdnState alloc_state(const LayerW &w, int tmax, int rows, const GdnState *share) {
  if (!w.linear && g_attn_qsa) {  // lab-attn's caches; the dense KV / partials are not used (small dummies)
    tmax = (tmax + 63) / 64 * 64;
    const bool sh = share && share->qc.tmax == tmax;
    GdnState s;
    s.tmax = tmax;
    s.kc = alloc(DType::F32, {4});
    s.vc = alloc(DType::F32, {4});
    s.apart = alloc(DType::F32, {4});
    // fn_attn_proj still writes the raw indexer keys here (f32 [tmax][128], never read on the QSA path: lab-attn
    // keeps bf16 copies in qc.ikeys): one sink shared by every attention layer of a model
    s.ikeys = sh ? share->ikeys : alloc(DType::F32, {tmax, 128});
    s.pos = alloc(DType::U32, {4});
    s.qc = attn::alloc_cache(attn::kv_fmt_from_env(attn::KvFmt::BF16), tmax, sh ? &share->qc : nullptr);
    s.qc.pos = s.pos;
    return s;
  }
  GdnState s = w.linear ? alloc_gdn_state() : alloc_attn_state(tmax, rows);
  if (w.ple.on) s.ple_conv = alloc(DType::F32, {PLE_ST, HCD});
  return s;
}

Scratch alloc_scratch(int rows) {
  // buffers for any R (kernels: 1, 2, 4 in kernels/fn_*.metal, 6 and 8 in lab-mtp's kernels/nax/rows68_*.metal)
  NE_CHECK(rows >= 1 && rows <= 16, "rows %d (1..16)", rows);
  const int64_t R = rows;
  Scratch s;
  s.rows = rows;
  s.mix_a = alloc(DType::F32, {R, LR});
  s.inj_a = alloc(DType::F32, {R, 2 * HC});
  s.xa = alloc(DType::F32, {R, HID});
  s.qkv = alloc(DType::F32, {R, CONVD});
  s.z = alloc(DType::F32, {R, VALD});
  s.gb = alloc(DType::F32, {R, 2 * NVH});
  s.o = alloc(DType::F32, {R, VALD});
  s.br_a = alloc(DType::F32, {R, HID});
  s.h2 = alloc(DType::F32, {R, HC, HID});
  s.mix_m = alloc(DType::F32, {R, LR});
  s.inj_m = alloc(DType::F32, {R, 2 * HC});
  s.xm = alloc(DType::F32, {R, HID});
  s.logits = alloc(DType::F32, {R, NE + 1});
  s.gush = alloc(DType::F32, {R, 2 * INTER});
  s.route = alloc(DType::U32, {R, 2 * TOPK});
  s.ctr1 = alloc(DType::U32, {4});
  s.guo = alloc(DType::F32, {R * TOPK, 2, INTER});
  s.sho = alloc(DType::F32, {R, HID});
  s.scr3 = alloc(DType::F32, {R, HID / 128, TOPK, 128});
  s.ctr3 = alloc(DType::U32, {R, HID / 128});
  s.br_m = alloc(DType::F32, {R, HID});
  s.hc_part = alloc(DType::F32, {LR / 8 + 1, HC, 8, R});
  s.hc_ctr = alloc(DType::U32, {64});
  s.uq = alloc(DType::U32, {1 + R * TOPK * (2 + R) + 2 + 2 * R * TOPK});  // + lab-verify's split lists
  s.guo_u = alloc(DType::F32, {R * TOPK, R, 2, INTER});
  s.aq = alloc(DType::F32, {R, AQD});
  s.akv = alloc(DType::F32, {R, 2 * AKVD});
  s.aiqk = alloc(DType::F32, {R, AIQD});
  s.ple_emb = alloc(DType::F32, {R, HID});
  s.ple_kv = alloc(DType::F32, {R, PLE_KV});
  s.hp = alloc(DType::F32, {R, HC, HID});
  return s;
}

static std::string gname(const char *base, int g, int r) {
  return std::string(base) + "_g" + std::to_string(g) + "_r" + std::to_string(r);
}
bool g_moe_split = getenv("NE_FN_MOE_SPLIT") && atoi(getenv("NE_FN_MOE_SPLIT")) > 0;
bool moe_split(int R, const LayerW &w) { return g_moe_split && R > 1 && R <= 5 && w.t6; }
int g_moe_dedupe = [] {  // lab-verify: max rows deduplicated (0 off); NE_FN_MOE_DEDUPE=1 = every R, N >= 2 = R <= N
  const char *e = getenv("NE_FN_MOE_DEDUPE");
  if (!e) return 3;  // default R 2-3 (verify d2: -0.3..-1.0 ms; slower from R 4: docs/mtp.md 12)
  const int v = atoi(e);
  return v == 1 ? 8 : std::max(0, v);
}();
bool g_nax_rows = [] {  // R >= 4: NAX trunk GEMMs (lab-mtp's request); NE_FN_NAX=0 keeps the SIMD kernels
  const char *e = getenv("NE_FN_NAX");
  return !(e && std::string(e) == "0");
}();
bool g_attn_qsa = [] {  // default: lab-attn's QSA step (the reference's attention); NE_FN_ATTN=dense = fn_attn_core
  const char *e = getenv("NE_FN_ATTN");
  NE_CHECK(!e || !*e || std::string(e) == "qsa" || std::string(e) == "dense", "NE_FN_ATTN=%s: qsa or dense", e);
  return !(e && std::string(e) == "dense");
}();
bool g_hc_v2 = [] {
  const char *e = getenv("NE_FN_HC");
  NE_CHECK(!e || !*e || std::string(e) == "v1" || std::string(e) == "v2", "NE_FN_HC=%s: v1 or v2", e);
  return !(e && std::string(e) == "v1");
}();
bool hc_v2(int R) { return g_hc_v2 && R != 8; }
int g_gdn_lazy = [] {
  const char *e = getenv("NE_GDN_LAZY");
  const int k = e && *e ? atoi(e) : 3;
  NE_CHECK(k == 0 || k == 1 || k == 2 || k == 3 || k == 4 || k == 6 || k == 8 || k == 12 || k == 16,
           "NE_GDN_LAZY=%s: 0 (fn_gdn_step) or 1, 2, 3, 4, 6, 8, 12, 16 (store every K-th step)", e);
  return k;
}();
int g_fuse = [] {
  const char *e = getenv("NE_FUSE");
  return e && *e ? (int)strtol(e, nullptr, 0) : 0;
}();
bool g_ref_round = [] {
  const char *e = getenv("NE_FN_ROUND");
  if (!e || !*e) return false;  // plain f32 unless the reference rounding is asked for
  const std::string s(e);
  NE_CHECK(s == "f32" || s == "off" || s == "0" || s == "ref" || s == "on" || s == "1",
           "NE_FN_ROUND=%s: f32 (default) or ref", e);
  return s == "ref" || s == "on" || s == "1";
}();
std::atomic<int> g_models_live{0};
id<MTLComputePipelineState> pso(const std::string &name, int min_threads) {
  id<MTLComputePipelineState> p = Pipelines::get(name, {FC::b(0, g_ref_round)});
  NE_CHECK((int)p.maxTotalThreadsPerThreadgroup >= min_threads, "%s: max %d threads per threadgroup < %d",
           name.c_str(), (int)p.maxTotalThreadsPerThreadgroup, min_threads);
  return p;
}
static MTLSize sz(int n) { return MTLSizeMake(n, 1, 1); }

const char *const kStageNames[(int)Stage::N] = {"attn_hc_pre", "attn_hc_mix", "gdn_in",     "gdn_step",
                                                "gdn_out",     "mlp_hc_pre",  "mlp_hc_mix", "moe_router",
                                                "moe_gateup",  "moe_down"};
// threadgroup counts of the HC kernels (fn_trunk.metal)
constexpr int HC_PRE_TGS = (LR / 8 + 1) * HC, HC_MIX_TGS = HID / 8;

void record_stage(Recorder &r, Stage stage, const LayerW &w, GdnState &st, Scratch &s, const Tensor &h_in,
                  const Tensor &h_out) {
  const GdnW &g = w.gdn;
  const MoeW &m = w.moe;
  const int R = s.rows;
  const std::string rs = "_r" + std::to_string(R);
  const std::string k = "_k" + std::to_string(m.gate.k) + rs;
  auto hc_pre = [&](const HcW &hw, const Tensor &h, const Tensor &mix, const Tensor &inj) {
    r.dispatch(pso(gname("fn_hc_pre", hw.down.group, R)), {h, hw.norm, hw.down.w, hw.down.s, hw.down.b, hw.inject, mix, inj, s.hc_part, s.hc_ctr},
               sz(HC_PRE_TGS), sz(256));
    r.barrier();
  };
  auto hc_mix = [&](const HcW &hw, const Tensor &h, const Tensor &mix, const Tensor &inj, const Tensor &out) {
    r.dispatch(pso(gname("fn_hc_mix", hw.up.group, R)), {h, hw.norm, hw.up.w, hw.up.s, hw.up.b, mix, inj, out},
               sz(HC_MIX_TGS), sz(256));
    r.barrier();
  };
  // HC read v2 (lab-speed, kernels/fn_rec.h): fn_hc_pre3 -> per-stream partials, fn_hc_mix2 -> mix, inj, out
  const bool v2 = hc_v2(R);
  if (w.t6 && t6_record_stage(r, stage, w, st, s, h_in, h_out)) return;  // lab-t6wire: EXL3 trunk (layer_t6.mm)
  switch (stage) {
    case Stage::AttnHCPre:
      if (v2) krec::hc_pre3(r, w.hca, R, h_in, w.hca.inject, s.inj_a, s.hc_part);
      else hc_pre(w.hca, h_in, s.mix_a, s.inj_a);
      break;
    case Stage::AttnHCMix:
      if (v2) krec::hc_mix(r, w.hca, R, h_in, s.mix_a, s.inj_a, s.xa, s.hc_part);
      else hc_mix(w.hca, h_in, s.mix_a, s.inj_a, s.xa);
      break;
    case Stage::GdnIn:
      if (R >= 4 && g_nax_rows) {  // lab-mtp's NAX trunk GEMMs (drop-ins, same buffers): faster from 4 rows
        if (w.linear) nax::record_gdn_in(r, R, w, st, s);
        else nax::record_attn_proj(r, R, w, st, s);
        r.barrier();
        break;
      }
      if (!w.linear) {
        const AttnW &a = w.attn;
        r.dispatch(pso(gname("fn_attn_proj", a.q.group, R)),
                   {s.xa, a.q.w, a.q.s, a.q.b, a.k.w, a.k.s, a.k.b, a.v.w, a.v.s, a.v.b, a.iqk.w, a.iqk.s, a.iqk.b,
                    s.aq, s.akv, s.aiqk, st.ikeys, st.pos},
                   sz((AQD + 2 * AKVD + AIQD + 7) / 8), sz(256));
        r.barrier();
        break;
      }
      r.dispatch(pso(g.qkv.group == g.z.group ? gname("fn_gdn_inproj", g.z.group, R)
                                              : "fn_gdn_inproj_q64_g128_r" + std::to_string(R)),
                 {s.xa, g.qkv.w, g.qkv.s, g.qkv.b, g.z.w, g.z.s, g.z.b, g.b.w, g.b.s, g.b.b, g.a.w, g.a.s, g.a.b,
                  g.conv, st.conv, g.alog, g.dtb, s.qkv, s.z, s.gb},
                 sz((CONVD + VALD + 2 * NVH + 7) / 8), sz(256));
      r.barrier();
      break;
    case Stage::GdnStep:
      if (!w.linear && st.qc.tmax > 0) {
        attn::record_step(r, w.qsa, st.qc, R, s.aq, s.akv, s.aiqk, s.o, false);
        r.barrier();
        break;
      }
      if (!w.linear) {
        const uint32_t tmax = (uint32_t)st.tmax;
        r.dispatch(pso("fn_attn_core" + rs, 384), {s.aq, s.akv, w.attn.qnorm, w.attn.knorm, st.kc, st.vc, st.pos,
                                                   Arg::value(tmax), st.apart},
                   sz(NKV * (st.tmax / 32)), sz(384));
        r.barrier();
        r.dispatch(pso("fn_attn_merge" + rs), {s.aq, st.apart, st.pos, Arg::value(tmax), s.o}, sz(R * NQH), sz(256));
        r.barrier();
        break;
      }
      if (g_gdn_lazy >= 1 && st.lz.buf)  // lab-gdnlazy: lazy state store (any R; defers only at R = 1)
        r.dispatch(pso("fn_gdn_step_lz" + std::to_string(g_gdn_lazy) + rs, 1024),
                   {s.qkv, s.gb, s.z, g.norm, st.ssm, s.o, st.lz, st.pend}, sz(NVH), sz(1024));
      else
        r.dispatch(pso("fn_gdn_step" + rs, 1024), {s.qkv, s.gb, s.z, g.norm, st.ssm, s.o}, sz(NVH), sz(1024));
      r.barrier();
      break;
    case Stage::GdnOut:
    {
      const Q8 &ow = w.linear ? g.out : w.attn.o;
      if (R >= 4 && g_nax_rows) {
        nax::record_out_hc(r, R, ow, s.o, h_in, s.inj_a, s.h2, s.br_a);
        r.barrier();
        break;
      }
      r.dispatch(pso(gname("fn_gdn_outproj", ow.group, R)), {s.o, ow.w, ow.s, ow.b, h_in, s.inj_a, s.h2, s.br_a},
                 sz(HID / 8), sz(256));
    }
      r.barrier();
      break;
    case Stage::MlpHCPre:
      if (v2) krec::hc_pre3(r, w.hcm, R, s.h2, w.hcm.inject, s.inj_m, s.hc_part);
      else hc_pre(w.hcm, s.h2, s.mix_m, s.inj_m);
      break;
    case Stage::MlpHCMix:
      if (v2) krec::hc_mix(r, w.hcm, R, s.h2, s.mix_m, s.inj_m, s.xm, s.hc_part);
      else hc_mix(w.hcm, s.h2, s.mix_m, s.inj_m, s.xm);
      break;
    case Stage::MoeRouter:
      r.dispatch(pso("fn_moe_router" + rs),
                 {s.xm, m.router, m.sgate, m.sg.w, m.sg.s, m.sg.b, m.su.w, m.su.s, m.su.b, s.logits, s.gush, s.route,
                  s.ctr1, s.uq, Arg::value((uint32_t)(moe_split(R, w) ? 1 : 0))},
                 sz((NE + 1 + 7) / 8 + 2 * INTER / 8), sz(256));
      r.barrier();
      break;
    case Stage::MoeGateUp:
      if (R > 1 && R <= g_moe_dedupe)
        r.dispatch(pso("fn_moe_gateup_u" + k),
                   {s.xm, m.gate.t, m.up.t, m.gate.suh, m.up.suh, s.uq, s.guo_u, s.gush, m.sd.w, m.sd.s, m.sd.b, s.sho},
                   sz(R * TOPK * 2 * (40 / 2) + HID / 8), sz(256));
      else
      r.dispatch(pso("fn_moe_gateup" + k),
                 {s.xm, m.gate.t, m.up.t, m.gate.suh, m.up.suh, s.route, s.guo, s.gush, m.sd.w, m.sd.s, m.sd.b, s.sho, s.uq},
                 sz(R * TOPK * 2 * (40 / 2) + HID / 8), sz(256));
      r.barrier();
      break;
    case Stage::MoeDown:
      if (R > 1 && R <= g_moe_dedupe && !moe_split(R, w))
        r.dispatch(pso("fn_moe_down_u" + k, 512),
                   {s.guo_u, m.down.t, m.gate.svh, m.up.svh, m.down.suh, m.down.svh, s.uq, s.logits, s.sho, s.h2,
                    s.inj_m, s.scr3, s.ctr3, h_out, s.br_m},
                   sz(R * TOPK * (HID / 128)), sz(512));
      else
      r.dispatch(pso("fn_moe_down" + k, 512),
                 {s.guo, m.down.t, m.gate.svh, m.up.svh, m.down.suh, m.down.svh, s.route, s.logits, s.sho, s.h2,
                  s.inj_m, s.scr3, s.ctr3, h_out, s.br_m},
                 sz(R * TOPK * (HID / 128)), sz(512));
      r.barrier();
      break;
    default:
      NE_CHECK(false, "bad stage %d", (int)stage);
  }
}

void record_ple_proj(Recorder &r, const PleW &e, Scratch &s) {
  const std::string rs = "_r" + std::to_string(s.rows);
  if (e.key.t6) {  // lab-t6wire: K8 key / value projections, then the bf16 stores into ple_kv [R][12800]
    T6Buf &b = t6_buf(s);
    t6_linears(r, s.rows, s, {&e.key, &e.value}, {s.ple_emb, s.ple_emb}, {b.pk, b.pv});
    t6_round(r, s.rows, b.pk, HCD, HCD, 0, s.ple_kv, PLE_KV, 0, true);
    t6_round(r, s.rows, b.pv, HID, HID, 0, s.ple_kv, PLE_KV, HCD, true);
    r.barrier();
  } else {
    r.dispatch(pso("fn_ple_proj_g64" + rs),
               {s.ple_emb, e.key.w, e.key.s, e.key.b, e.value.w, e.value.s, e.value.b, s.ple_kv}, sz(PLE_KV / 8),
               sz(256));
    r.barrier();
  }
}

void record_ple(Recorder &r, const LayerW &w, GdnState &st, Scratch &s, const Tensor &h_in, const Tensor &h_out) {
  const PleW &e = w.ple;
  const std::string rs = "_r" + std::to_string(s.rows);
  record_ple_proj(r, e, s);
  r.dispatch(pso("fn_ple_mix" + rs, 1024), {h_in, s.ple_kv, e.nk, e.nq, e.nc, e.conv, st.ple_conv, h_out}, sz(HC),
             sz(1024));
  r.barrier();
}

void record_gdn_layer(Recorder &r, const LayerW &w, GdnState &st, Scratch &s, const Tensor &h_in,
                      const Tensor &h_out) {
  if (w.ple.on) record_ple(r, w, st, s, h_in, s.hp);
  const Tensor &hi = w.ple.on ? s.hp : h_in;
  for (int i = 0; i < (int)Stage::N; i++) record_stage(r, (Stage)i, w, st, s, hi, h_out);
}

void ngram_ids(const PleW &p, int64_t p2, int64_t p1, int64_t t, int64_t eos, int64_t ids[PLE_NG]) {
  const int64_t s1 = p1, s2 = (p1 == eos) ? eos : p2;  // mlx-vlm _shift_right_ignore_eos on [p2, p1, t]
  auto mul = [](int64_t a, int64_t b) { return (int64_t)((uint64_t)a * (uint64_t)b); };  // int64 wrapping
  const int64_t m2 = mul(t, p.mult[0]) ^ mul(s1, p.mult[1]);
  const int64_t m3 = m2 ^ mul(s2, p.mult[2]);
  for (int i = 0; i < PLE_NG; i++) {
    const int64_t m = i < PLE_NG / 2 ? m2 : m3, n = p.sizes[i];
    int64_t r = m % n;
    if (r < 0) r += n;  // floor mod (numpy / MLX semantics)
    ids[i] = r + p.offsets[i];
  }
}

void ngram_gather(const Pack &pack, const int64_t ids[PLE_NG], float *emb) {
  const TensorInfo *ti = pack.info("ngram_table.weight");
  NE_CHECK(ti && ti->host, "pack has no ngram_table.weight");
  const int64_t rows = (int64_t)(ti->nbytes / (PLE_W * 2));
  const uint16_t *tab = (const uint16_t *)ti->host;
  for (int i = 0; i < PLE_NG; i++) {
    NE_CHECK(ids[i] >= 0 && ids[i] < rows, "n-gram row %lld outside the table", (long long)ids[i]);
    const uint16_t *row = tab + ids[i] * PLE_W;
    for (int j = 0; j < PLE_W; j++) {
      const uint32_t u = (uint32_t)row[j] << 16;
      memcpy(&emb[i * PLE_W + j], &u, 4);
    }
  }
}

const Tensor *scratch_tensor(Scratch &s, const std::string &name) {
  const std::map<std::string, Tensor *> m = {
      {"attn_mix", &s.mix_a}, {"attn_inj", &s.inj_a}, {"attn_in", &s.xa},     {"qkv", &s.qkv},
      {"z", &s.z},            {"gb", &s.gb},          {"core", &s.o},         {"attn_out", &s.br_a},
      {"h_mid", &s.h2},       {"mlp_mix", &s.mix_m},  {"mlp_inj", &s.inj_m},  {"mlp_in", &s.xm},
      {"logits", &s.logits},  {"route", &s.route},    {"shared_gu", &s.gush}, {"experts_gu", &s.guo},
      {"shared_down", &s.sho}, {"mlp_out", &s.br_m},   {"attn_q", &s.aq},      {"attn_kv", &s.akv},
      {"attn_iqk", &s.aiqk},  {"ple_emb", &s.ple_emb}, {"ple_kv", &s.ple_kv},   {"hc_after_ple", &s.hp}};
  auto it = m.find(name);
  return it == m.end() ? nullptr : it->second;
}

void time_layer(const LayerW &w, GdnState &st, Scratch &s, const Tensor &h_in, const Tensor &h_out, Backend be,
                int reps1, int reps2, int flush_mb, double *us_per_step, double *us_flush) {
  NE_CHECK(reps2 > reps1 && reps1 >= 1, "bad reps %d %d", reps1, reps2);
  Tensor fl, flo;
  uint32_t n4 = 0;
  if (flush_mb > 0) {
    fl = alloc(DType::U8, {(int64_t)flush_mb << 20});
    n4 = (uint32_t)(((size_t)flush_mb << 20) / 16);
    flo = alloc(DType::U32, {1 << 16});
  }
  id<MTLComputePipelineState> rd = flush_mb > 0 ? pso("ne_stream_read") : nil;
  auto chain = [&](int reps, bool layer, bool flush) {
    auto r = std::make_unique<Recorder>(be);
    r->begin();
    for (int i = 0; i < reps; i++) {
      if (layer) record_gdn_layer(*r, w, st, s, h_in, h_out);
      if (flush) {
        r->dispatch(rd, {fl, flo, Arg::value(n4)}, sz(2048), sz(256));
        r->barrier();
      }
    }
    r->end();
    return r;
  };
  auto best = [&](Recorder &r) {
    double b = 1e30;
    for (int i = 0; i < 5; i++) { r.run(); b = std::min(b, r.gpu_ms()); }
    return b;
  };
  const bool f = flush_mb > 0;
  auto a1 = chain(reps1, true, f), a2 = chain(reps2, true, f);
  const double t1 = best(*a1), t2 = best(*a2);
  double ff = 0;
  if (f) {
    auto f1 = chain(reps1, false, true), f2 = chain(reps2, false, true);
    ff = (best(*f2) - best(*f1)) * 1e3 / (reps2 - reps1);
  }
  if (us_flush) *us_flush = ff;
  if (us_per_step) *us_per_step = (t2 - t1) * 1e3 / (reps2 - reps1) - ff;
}

}  // namespace ne::fn

// ------------------------------------------------------------------------------------------------ C entry points
#ifndef NE_NO_CAPI
using namespace ne;
using namespace ne::fn;

struct ne_fn_layer {
  LayerW w;
  GdnState st;
  Scratch s;
  Tensor h_in, h_out;
  std::unique_ptr<Recorder> rec;
  std::unique_ptr<Recorder> stage[(int)Stage::N];
  std::unique_ptr<Recorder> block[kNumBlocks];
};

static const Tensor *layer_tensor(ne_fn_layer *l, const std::string &name) {
  if (name == "h_in") return &l->h_in;
  if (name == "h_out") return &l->h_out;
  if (name == "conv_state") return &l->st.conv;
  if (name == "ssm_state") return &l->st.ssm;
  if (name == "k_cache") return &l->st.kc;
  if (name == "v_cache") return &l->st.vc;
  if (name == "index_keys") return &l->st.ikeys;
  if (name == "ple_conv_state") return l->st.ple_conv.buf ? &l->st.ple_conv : nullptr;
  return scratch_tensor(l->s, name);
}

extern "C" {

NE_API ne_fn_layer *ne_fn_layer_create(ne_pack *pack, int layer) { return ne_fn_layer_create_rows(pack, layer, 1); }

NE_API ne_fn_layer *ne_fn_layer_create_rows(ne_pack *pack, int layer, int rows) {
  ne_fn_layer *l = nullptr;
  int rc = c_call([&] {
    const Pack *p = pack_from_c(pack);
    NE_CHECK(p, "null pack");
    auto u = std::make_unique<ne_fn_layer>();
    u->w = load_layer(*p, layer);
    u->st = alloc_state(u->w, 4096, rows);
    u->s = alloc_scratch(rows);
    u->h_in = alloc(DType::F32, {rows, HC, HID});
    u->h_out = alloc(DType::F32, {rows, HC, HID});
    u->rec = std::make_unique<Recorder>();
    u->rec->begin();
    record_gdn_layer(*u->rec, u->w, u->st, u->s, u->h_in, u->h_out);
    u->rec->end();
    l = u.release();
  });
  return rc == 0 ? l : nullptr;
}

NE_API void ne_fn_layer_free(ne_fn_layer *l) { delete l; }

NE_API int ne_fn_layer_set_state(ne_fn_layer *l, const float *conv, const float *ssm) {
  return c_call([&] {
    NE_CHECK(l, "null layer");
    if (conv) upload(l->st.conv, conv, l->st.conv.nbytes());
    if (ssm) {
      upload(l->st.ssm, ssm, l->st.ssm.nbytes());
      gdn_clear_pending(l->st);  // lab-gdnlazy: the written state replaces any deferred rows
    }
  });
}

NE_API int ne_fn_layer_get_state(ne_fn_layer *l, float *conv, float *ssm) {
  return c_call([&] {
    NE_CHECK(l, "null layer");
    if (conv) download(l->st.conv, conv, l->st.conv.nbytes());
    if (ssm) {
      gdn_materialise(l->st);  // lab-gdnlazy: apply deferred rows first
      download(l->st.ssm, ssm, l->st.ssm.nbytes());
    }
  });
}

NE_API int ne_fn_layer_set_kv(ne_fn_layer *l, const float *k, const float *v, const float *ikeys, int T) {
  return c_call([&] {
    NE_CHECK(l && !l->w.linear, "not an attention layer");
    NE_CHECK(T >= 0 && T + 4 <= l->st.tmax, "T %d out of range (tmax %d)", T, l->st.tmax);
    if (l->st.qc.tmax > 0) {  // QSA caches: lab-attn's import (rebuilds the block keys), sets pos = T
      attn::import_cache(l->st.qc, l->w.qsa, k, v, ikeys, T);
      if (ikeys) memcpy(l->st.ikeys.host(), ikeys, (size_t)T * 128 * 4);
      return;
    }
    for (int g = 0; g < NKV; g++) {
      if (k) memcpy((float *)l->st.kc.host() + (size_t)g * l->st.tmax * AHD, k + (size_t)g * T * AHD, (size_t)T * AHD * 4);
      if (v) memcpy((float *)l->st.vc.host() + (size_t)g * l->st.tmax * AHD, v + (size_t)g * T * AHD, (size_t)T * AHD * 4);
    }
    if (ikeys) memcpy(l->st.ikeys.host(), ikeys, (size_t)T * 128 * 4);
    ((uint32_t *)l->st.pos.host())[0] = (uint32_t)T;
  });
}

NE_API int ne_fn_layer_get_kv(ne_fn_layer *l, float *k, float *v, float *ikeys, int *n_tokens) {
  return c_call([&] {
    NE_CHECK(l && !l->w.linear, "not an attention layer");
    const int T = (int)((uint32_t *)l->st.pos.host())[0];
    if (l->st.qc.tmax > 0) {
      attn::export_cache(l->st.qc, T, k, v, ikeys, nullptr);
      if (n_tokens) *n_tokens = T;
      return;
    }
    for (int g = 0; g < NKV; g++) {
      if (k) memcpy(k + (size_t)g * T * AHD, (float *)l->st.kc.host() + (size_t)g * l->st.tmax * AHD, (size_t)T * AHD * 4);
      if (v) memcpy(v + (size_t)g * T * AHD, (float *)l->st.vc.host() + (size_t)g * l->st.tmax * AHD, (size_t)T * AHD * 4);
    }
    if (ikeys) memcpy(ikeys, l->st.ikeys.host(), (size_t)T * 128 * 4);
    if (n_tokens) *n_tokens = T;
  });
}

NE_API int ne_fn_layer_step(ne_fn_layer *l, const float *h_in, float *h_out, double *gpu_ms) {
  return c_call([&] {
    NE_CHECK(l, "null layer");
    if (h_in) upload(l->h_in, h_in, l->h_in.nbytes());
    if (!l->w.linear) {
      const uint32_t T = ((uint32_t *)l->st.pos.host())[0];
      NE_CHECK((int)T + l->s.rows <= l->st.tmax, "KV cache full (%u + %d > %d)", T, l->s.rows, l->st.tmax);
    }
    l->rec->run();
    if (!l->w.linear) ((uint32_t *)l->st.pos.host())[0] += (uint32_t)l->s.rows;
    if (h_out) download(l->h_out, h_out, l->h_out.nbytes());
    if (gpu_ms) *gpu_ms = l->rec->gpu_ms();
  });
}

// dispatches of each sub-block (the ne_fn_layer_run_stage contract): AttnHC, GdnIn, GdnStep, GdnOut, MlpHC, Moe
static const int kBlockFirst[kNumBlocks] = {0, 2, 3, 4, 5, 7}, kBlockLast[kNumBlocks] = {1, 2, 3, 4, 6, 9};

static void run_range(ne_fn_layer *l, std::unique_ptr<Recorder> &r, int first, int last) {
  if (!r) {
    r = std::make_unique<Recorder>(l->rec->backend());
    r->begin();
    for (int i = first; i <= last; i++) record_stage(*r, (Stage)i, l->w, l->st, l->s, l->h_in, l->h_out);
    r->end();
  }
  r->run();
}

NE_API int ne_fn_layer_run_stage(ne_fn_layer *l, int block) {
  return c_call([&] {
    NE_CHECK(l && block >= 0 && block < kNumBlocks, "bad sub-block %d (0..%d)", block, kNumBlocks - 1);
    run_range(l, l->block[block], kBlockFirst[block], kBlockLast[block]);
  });
}

NE_API int ne_fn_layer_run_dispatch(ne_fn_layer *l, int stage) {
  return c_call([&] {
    NE_CHECK(l && stage >= 0 && stage < (int)Stage::N, "bad dispatch %d", stage);
    run_range(l, l->stage[stage], stage, stage);
  });
}

NE_API int ne_fn_set_ref_rounding(int on) {
  return c_call([&] {
    if ((on != 0) == g_ref_round) return;
    // a model's recordings (and the ones it makes lazily: MTP head, chunked prefill, head-only) use the mode it was
    // loaded with; a change under a live model would mix the two
    NE_CHECK(g_models_live.load() == 0, "ne_fn_set_ref_rounding(%d): %d model(s) live, recorded with rounding %s; free "
             "them first", on, g_models_live.load(), g_ref_round ? "on" : "off");
    g_ref_round = on != 0;
  });
}

NE_API int ne_fn_layer_read(ne_fn_layer *l, const char *name, float *dst, int n) {
  int size = -1;
  int rc = c_call([&] {
    NE_CHECK(l && name, "null argument");
    const Tensor *t = layer_tensor(l, name);
    NE_CHECK(t, "unknown intermediate '%s'", name);
    size = (int)t->numel();
    NE_CHECK(n <= size, "%s: n %d > size %d", name, n, size);
    if (t == &l->st.ssm) gdn_materialise(l->st);  // lab-gdnlazy
    if (dst && n > 0) download(*t, dst, (size_t)n * 4);
  });
  return rc == 0 ? size : -1;
}

NE_API int ne_fn_layer_write(ne_fn_layer *l, const char *name, const float *src, int n) {
  return c_call([&] {
    NE_CHECK(l && name && src, "null argument");
    const Tensor *t = layer_tensor(l, name);
    NE_CHECK(t, "unknown intermediate '%s'", name);
    NE_CHECK(n <= (int)t->numel(), "%s: n %d > size %d", name, n, (int)t->numel());
    upload(*t, src, (size_t)n * 4);
    if (t == &l->st.ssm) gdn_clear_pending(l->st);  // lab-gdnlazy
  });
}

NE_API int ne_fn_layer_time(ne_fn_layer *l, int reps1, int reps2, int flush_mb, double *us_per_step,
                            double *us_flush) {
  return c_call([&] {
    NE_CHECK(l, "null layer");
    time_layer(l->w, l->st, l->s, l->h_in, l->h_out, l->rec->backend(), reps1, reps2, flush_mb, us_per_step,
               us_flush);
  });
}

}  // extern "C"
#endif  // NE_NO_CAPI
