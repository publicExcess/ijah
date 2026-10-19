// lab-mtp: NAX multi-row kernels vs CPU references on real pack weights, and GPU timing vs the SIMD R-row kernels.
//   build/mtp-nax-check gemm PACK [time]      in_proj (4 segments + GDN epilogue), out_proj (+HC write), lm_head
//   build/mtp-nax-check layer PACK CASE R [time]   one layer step at R rows (NAX trunk vs SIMD) vs a verify4 case
// Timing needs the machine lock (tools/with_lock.sh).
#include <chrono>
#include <functional>
#include <random>

#include "core/bench.h"
#include "kernels/nax/nax.h"
#include "tests/mtp/common.h"

using namespace ne;
using namespace ne::fn;
using namespace mtpt;

// CPU reference y[r][n] = sum_k W[n][k] x[r][k] for the listed columns (double accumulation)
static std::vector<float> ref_q8(const Q8 &q, const std::vector<float> &x, int R, const std::vector<int> &cols) {
  const uint8_t *w = (const uint8_t *)q.w.host();
  const uint16_t *s = (const uint16_t *)q.s.host(), *b = (const uint16_t *)q.b.host();
  const int ng = q.in / q.group;
  std::vector<float> y(R * cols.size());
  for (size_t ci = 0; ci < cols.size(); ci++) {
    const int n = cols[ci];
    for (int r = 0; r < R; r++) {
      double acc = 0;
      for (int k = 0; k < q.in; k++) {
        const double wv = (double)bf2f(s[(size_t)n * ng + k / q.group]) * w[(size_t)n * q.in + k] +
                          bf2f(b[(size_t)n * ng + k / q.group]);
        acc += wv * x[(size_t)r * q.in + k];
      }
      y[r * cols.size() + ci] = (float)acc;
    }
  }
  return y;
}
static std::vector<float> pick(const std::vector<float> &y, int R, int ld, const std::vector<int> &cols) {
  std::vector<float> o;
  for (int r = 0; r < R; r++) for (int c : cols) o.push_back(y[(size_t)r * ld + c]);
  return o;
}
static std::vector<int> every(int rows, int step) {
  std::vector<int> c;
  for (int i = 0; i < rows; i += step) c.push_back(i);
  if (c.back() != rows - 1) c.push_back(rows - 1);
  return c;
}

// GPU us per call: n calls each followed by a 256 MB streaming read (SLC flush), minus the flush alone
static double time_flushed(const std::function<void(Recorder &)> &rec) {
  static Tensor fl = alloc(DType::U8, {256 << 20}), flo = alloc(DType::U32, {1 << 16});
  const uint32_t n4 = (256u << 20) / 16;
  auto rd_pso = Pipelines::get("ne_stream_read");
  auto chain = [&](bool with, int n) {
    auto r = std::make_unique<Recorder>();
    r->begin();
    for (int i = 0; i < n; i++) {
      if (with) { rec(*r); r->barrier(); }
      r->dispatch(rd_pso, {fl, flo, Arg::value(n4)}, MTLSizeMake(2048, 1, 1), MTLSizeMake(256, 1, 1));
      r->barrier();
    }
    r->end();
    double b = 1e30;
    for (int i = 0; i < 5; i++) { r->run(); b = std::min(b, r->gpu_ms()); }
    return b;
  };
  const double f = (chain(false, 24) - chain(false, 8)) / 16;
  const double t = (chain(true, 24) - chain(true, 8)) / 16;
  return (t - f) * 1e3;
}

static void gemm(const Pack &pack, bool timing) {
  g_ref_round = false;  // plain f32 outputs for the CPU comparison
  LayerW w = load_layer(pack, 0);
  Q8 head = nax::load_q8(pack, "language_model.lm_head");
  printf("in_proj qkv [%d,%d] g%d; out [%d,%d] g%d; lm_head [%d,%d] g%d\n", w.gdn.qkv.rows, w.gdn.qkv.in,
         w.gdn.qkv.group, w.gdn.out.rows, w.gdn.out.in, w.gdn.out.group, head.rows, head.in, head.group);
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  const int RS[] = {1, 2, 3, 4, 6, 8, 12, 16};
  for (int R : RS) {
    std::vector<float> x(R * HID), xo(R * VALD);
    for (auto &v : x) v = nd(rng);
    for (auto &v : xo) v = nd(rng) * 0.1f;
    // plain GEMM, 4 segments (qkv z b a), no rounding effects on the check: compare f32 vs ref within bf16 + f16
    Tensor tx = alloc(DType::F32, {R, HID}), txo = alloc(DType::F32, {R, VALD});
    upload(tx, x.data(), x.size() * 4);
    upload(txo, xo.data(), xo.size() * 4);
    Tensor yq = alloc(DType::F32, {R, CONVD}), yz = alloc(DType::F32, {R, VALD}), yb = alloc(DType::F32, {R, NVH}),
           ya = alloc(DType::F32, {R, NVH}), yo = alloc(DType::F32, {R, HID}), yh = alloc(DType::F32, {R, head.rows});
    {
      Recorder r;
      r.begin();
      nax::record_q8(r, R, tx, {&w.gdn.qkv, &w.gdn.z, &w.gdn.b, &w.gdn.a}, {yq, yz, yb, ya});
      nax::record_q8(r, R, txo, {&w.gdn.out}, {yo});
      nax::record_q8(r, R, tx, {&head}, {yh});
      r.end();
      r.run();
    }
    char nm[64];
    auto chk = [&](const char *what, const Q8 &q, const Tensor &y, const std::vector<float> &xx, int step) {
      auto cols = every(q.rows, step);
      snprintf(nm, sizeof nm, "%s R=%d", what, R);
      row("gemm", nm, "cpu f64", pick(rd(y), R, q.rows, cols), ref_q8(q, xx, R, cols));
    };
    chk("qkv", w.gdn.qkv, yq, x, 7);
    chk("z", w.gdn.z, yz, x, 7);
    chk("b", w.gdn.b, yb, x, 1);
    chk("a", w.gdn.a, ya, x, 1);
    chk("out_proj", w.gdn.out, yo, xo, 3);
    chk("lm_head", head, yh, x, 997);
    if (!timing) continue;
    // timing vs the SIMD kernels (R = 1, 2, 4 instantiated): in_proj with its epilogue, out_proj + HC write
    GdnState st = alloc_gdn_state();
    Scratch s2;
    s2.rows = R;
    Tensor hin = alloc(DType::F32, {R, HC, HID}), hout = alloc(DType::F32, {R, HC, HID});
    Tensor inj = alloc(DType::F32, {R, 8}), br = alloc(DType::F32, {R, HID}), qkv = alloc(DType::F32, {R, CONVD}),
           z = alloc(DType::F32, {R, VALD}), gb = alloc(DType::F32, {R, 96});
    s2.xa = tx; s2.qkv = qkv; s2.z = z; s2.gb = gb;
    const double t_in = time_flushed([&](Recorder &r) { nax::record_gdn_in(r, R, w, st, s2); });
    const double t_out = time_flushed([&](Recorder &r) { nax::record_out_hc(r, R, w.gdn.out, txo, hin, inj, hout, br); });
    const double t_head = time_flushed([&](Recorder &r) { nax::record_q8(r, R, tx, {&head}, {yh}); });
    double s_in = 0, s_out = 0;
    if (R == 1 || R == 2 || R == 4) {
      auto gn = [&](const char *b, int g) { return std::string(b) + "_g" + std::to_string(g) + "_r" + std::to_string(R); };
      auto pin = Pipelines::get(gn("fn_gdn_inproj", w.gdn.qkv.group), {FC::b(0, g_ref_round)});
      auto pout = Pipelines::get(gn("fn_gdn_outproj", w.gdn.out.group), {FC::b(0, g_ref_round)});
      const GdnW &g = w.gdn;
      s_in = time_flushed([&](Recorder &r) {
        r.dispatch(pin, {tx, g.qkv.w, g.qkv.s, g.qkv.b, g.z.w, g.z.s, g.z.b, g.b.w, g.b.s, g.b.b, g.a.w, g.a.s, g.a.b,
                         g.conv, st.conv, g.alog, g.dtb, qkv, z, gb},
                   MTLSizeMake((CONVD + VALD + 2 * NVH + 7) / 8, 1, 1), MTLSizeMake(256, 1, 1));
      });
      s_out = time_flushed([&](Recorder &r) {
        r.dispatch(pout, {txo, g.out.w, g.out.s, g.out.b, hin, inj, hout, br}, MTLSizeMake(HID / 8, 1, 1),
                   MTLSizeMake(256, 1, 1));
      });
    }
    const double mb_in = (double)(w.gdn.qkv.w.nbytes() + w.gdn.z.w.nbytes() + 2 * w.gdn.a.w.nbytes()) * 1.0625 / 1e6;
    const double mb_out = (double)w.gdn.out.w.nbytes() * (1 + 4.0 / w.gdn.out.group) / 1e6;
    const double mb_head = (double)head.w.nbytes() * (1 + 4.0 / head.group) / 1e6;
    printf("time R=%d: in_proj NAX %6.1f us (%3.0f GB/s) SIMD %6.1f | out_proj NAX %5.1f (%3.0f GB/s) SIMD %5.1f | "
           "lm_head NAX %6.1f us (%3.0f GB/s)\n",
           R, t_in, mb_in / t_in * 1e3, s_in, t_out, mb_out / t_out * 1e3, s_out, t_head, mb_head / t_head * 1e3);
  }
}

// ---------------------------------------------------------------- layer step at R rows vs the verify4 goldens
// The verify4 case holds 4 consecutive tokens from one state; its first R rows are the R-row step (causal), the GDN
// state after R rows is gdn.ssm_state_rows[R-1]. R > 4: timing only (inputs = the 4 golden rows, repeated).
struct Lyr {
  LayerW w;
  GdnState st;
  Scratch s;
  nax::LayerExtra ex;
  Tensor h_in, h_out;
  int R = 0, T_in = 0;
};
static std::vector<float> first_rows(const std::vector<float> &v, int rows_total, int R) {
  const size_t per = v.size() / rows_total;
  std::vector<float> o;
  for (int r = 0; r < R; r++) { const int src = r % rows_total; o.insert(o.end(), v.begin() + src * per, v.begin() + (src + 1) * per); }
  return o;
}
static void set_inputs(Lyr &l, const std::string &G) {
  auto g = [&](const std::string &n) { return npy(G + "/" + n + ".npy").f; };
  if (l.w.linear) {
    wr(l.st.conv, g("gdn.conv_state_in"));
    wr(l.st.ssm, g("gdn.ssm_state_in")); gdn_clear_pending(l.st);
  } else {
    import_kv(l.st, l.w, g("attn.k_cache_in"), g("attn.v_cache_in"), g("attn.index_keys_in"), l.T_in);
  }
  wr(l.h_in, first_rows(g("hc_in"), 4, l.R));
}
static double time_layer_nax(Lyr &l, bool nax) {
  return time_flushed([&](Recorder &r) { nax::record_layer(r, l.w, l.st, l.s, l.ex, l.h_in, l.h_out, nax); });
}
static void layer(const Pack &pack, const std::string &G, int R, bool timing) {
  int layer = 0;
  {
    std::ifstream mf(G + "/manifest.json");
    std::string txt((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
    const auto k = txt.find("\"layer\":");
    if (k != std::string::npos) layer = atoi(txt.c_str() + k + 8);
  }
  Lyr l;
  l.R = R;
  l.w = load_layer(pack, layer);
  l.st = nax::alloc_state_rows(l.w, R);
  l.s = nax::alloc_scratch_rows(R);
  l.ex = nax::alloc_layer_extra();
  l.h_in = alloc(DType::F32, {R, HC, HID});
  l.h_out = alloc(DType::F32, {R, HC, HID});
  if (!l.w.linear) l.T_in = (int)(npy(G + "/attn.k_cache_in.npy").n / 512);
  auto g = [&](const std::string &n) { return npy(G + "/" + n + ".npy").f; };
  for (int nax = 1; nax >= 0 && R <= 4; nax--) {
    set_inputs(l, G);
    Recorder r;
    r.begin();
    nax::record_layer(r, l.w, l.st, l.s, l.ex, l.h_in, l.h_out, nax);
    r.end();
    r.run();
    const char *m = nax ? "NAX" : "SIMD";
    auto fr = [&](const std::string &n) { return first_rows(g(n), 4, R); };
    if (l.w.linear) {
      row(m, "qkv(conv+silu)", "gdn.conv_out", rd(l.s.qkv), fr("gdn.conv_out"));
      row(m, "z", "gdn.z", rd(l.s.z), fr("gdn.z"));
      row(m, "core(gated norm)", "gdn.gated_norm", rd(l.s.o), fr("gdn.gated_norm"));
      auto rows = g("gdn.ssm_state_rows");
      const size_t per = rows.size() / 4;
      row(m, "ssm_state", "gdn.ssm_state_rows[R-1]", (gdn_materialise(l.st), rd(l.st.ssm)),
          std::vector<float>(rows.begin() + (R - 1) * per, rows.begin() + R * per));
      row(m, "attn_out", "gdn.out", rd(l.s.br_a), fr("gdn.out"));
    } else {
      row(m, "q_proj (q|gate)", "attn.q_proj", rd(l.s.aq), fr("attn.q_proj"));
      row(m, "indexer qk", "attn.idx_qk", rd(l.s.aiqk), fr("attn.idx_qk"));
      row(m, "core * sigmoid(gate)", "attn.gated", rd(l.s.o), fr("attn.gated"));
      row(m, "attn_out", "attn.out", rd(l.s.br_a), fr("attn.out"));
    }
    row(m, "h_mid", "attn_hc.out", rd(l.s.h2), fr("attn_hc.out"));
    row(m, "mlp_in", "mlp_hc.block_in", rd(l.s.xm), fr("mlp_hc.block_in"));
    row(m, "mlp_out", "moe.out", rd(l.s.br_m), fr("moe.out"));
    row(m, "h_out", "hc_out", rd(l.h_out), fr("hc_out"));
  }
  if (!timing) return;
  set_inputs(l, G);
  const double tn = time_layer_nax(l, true);
  const bool simd_ok = true;
  const double ts = simd_ok ? time_layer_nax(l, false) : 0;
  printf("time L%02d R=%d: NAX trunk %.1f us, SIMD trunk %.1f us per layer step (256 MB flush between steps)\n", layer,
         R, tn, ts);
}

// ---------------------------------------------------------------- MTP rollback v1 on a GDN layer (verify4 case)
// For a = 0..3 accepted: run the 4-row verify step with rollback points, roll back to row a, compare the ssm state
// with gdn.ssm_state_rows[a] and the conv state with rows a+1..a+3 of [gdn.conv_state_in; bf16(gdn.qkv)].
static void rollback(const Pack &pack, const std::string &G) {
  const int R = 4;
  Lyr l;
  l.R = R;
  l.w = load_layer(pack, 0);
  l.st = nax::alloc_state_rows(l.w, R);
  l.s = nax::alloc_scratch_rows(R);
  l.ex = nax::alloc_layer_extra(R);
  l.h_in = alloc(DType::F32, {R, HC, HID});
  l.h_out = alloc(DType::F32, {R, HC, HID});
  auto g = [&](const std::string &n) { return npy(G + "/" + n + ".npy").f; };
  auto rows = g("gdn.ssm_state_rows");
  const size_t per = rows.size() / 4;
  auto cin = g("gdn.conv_state_in"), qkv = g("gdn.qkv");
  for (auto &v : qkv) { uint32_t u; memcpy(&u, &v, 4); u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u; memcpy(&v, &u, 4); }
  std::vector<float> hist(cin);
  hist.insert(hist.end(), qkv.begin(), qkv.end());
  Tensor keep = alloc(DType::U32, {4});
  for (int a = 0; a < R; a++) {
    set_inputs(l, G);
    ((uint32_t *)keep.host())[0] = (uint32_t)(a + 1);
    Recorder r;
    r.begin();
    nax::record_layer(r, l.w, l.st, l.s, l.ex, l.h_in, l.h_out, true);
    nax::record_gdn_rollback(r, l.st, l.ex, R, keep);
    r.end();
    r.run();
    char nm[64];
    snprintf(nm, sizeof nm, "ssm after row %d", a);
    row("rollbk", nm, "gdn.ssm_state_rows[a]", (gdn_materialise(l.st), rd(l.st.ssm)), std::vector<float>(rows.begin() + a * per, rows.begin() + (a + 1) * per));
    snprintf(nm, sizeof nm, "conv after row %d", a);
    row("rollbk", nm, "[conv_in; qkv] rows a+1..", rd(l.st.conv),
        std::vector<float>(hist.begin() + (size_t)(a + 1) * CONVD, hist.begin() + (size_t)(a + 4) * CONVD));
  }
  set_inputs(l, G);
  const double t_rb = time_flushed([&](Recorder &r) { nax::record_layer(r, l.w, l.st, l.s, l.ex, l.h_in, l.h_out, true); });
  nax::LayerExtra plain = nax::alloc_layer_extra();
  std::swap(plain, l.ex);
  set_inputs(l, G);
  const double t_pl = time_flushed([&](Recorder &r) { nax::record_layer(r, l.w, l.st, l.s, l.ex, l.h_in, l.h_out, true); });
  std::swap(plain, l.ex);
  const double t_back = time_flushed([&](Recorder &r) { nax::record_gdn_rollback(r, l.st, l.ex, R, keep); });
  printf("time L00 R=4: step with rollback points %.1f us, without %.1f us; rollback dispatch %.1f us\n", t_rb, t_pl, t_back);
}

// ---------------------------------------------------------------- MTP rollback v2 (lazy commit) on L00_verify4
// Two 2-row rounds from the golden state: round 1 = verify rows 0, 1 (nothing pending); the accept keeps k rows;
// round 2 = the next 2 rows after the kept ones (k = 2: rows 2, 3; k = 1: rows 1, 2 - row 1 re-verified). Round 2's
// outputs must equal the golden rows, the committed state = the state after the kept rows.
static void lazy(const Pack &pack, const std::string &G) {
  const int R = 2;
  auto g = [&](const std::string &n) { return npy(G + "/" + n + ".npy").f; };
  auto rows = g("gdn.ssm_state_rows");
  const size_t per = rows.size() / 4;
  auto cin = g("gdn.conv_state_in"), qkv = g("gdn.qkv");
  for (auto &v : qkv) { uint32_t u; memcpy(&u, &v, 4); u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u; memcpy(&v, &u, 4); }
  std::vector<float> hist(cin);
  hist.insert(hist.end(), qkv.begin(), qkv.end());
  auto pick = [&](const std::string &n, int r0) {  // rows r0, r0 + 1 of a [1, 4, ...] golden
    auto v = g(n);
    const size_t q = v.size() / 4;
    return std::vector<float>(v.begin() + r0 * q, v.begin() + (r0 + R) * q);
  };
  Lyr l;
  l.R = R;
  l.w = load_layer(pack, 0);
  l.st = nax::alloc_state_rows(l.w, R);
  l.s = nax::alloc_scratch_rows(R);
  l.h_in = alloc(DType::F32, {R, HC, HID});
  l.h_out = alloc(DType::F32, {R, HC, HID});
  for (int k = 2; k >= 1; k--) {
    l.ex = nax::alloc_layer_extra(0, true);
    wr(l.st.conv, cin);
    wr(l.st.ssm, g("gdn.ssm_state_in")); gdn_clear_pending(l.st);
    ((uint32_t *)l.ex.keep.host())[0] = 0;
    Recorder r;
    r.begin();
    nax::record_layer(r, l.w, l.st, l.s, l.ex, l.h_in, l.h_out, true);
    r.end();
    wr(l.h_in, pick("hc_in", 0));
    r.run();
    char nm[96];
    snprintf(nm, sizeof nm, "k=%d round1 h_out", k);
    row("lazy", nm, "hc_out rows 0,1", rd(l.h_out), pick("hc_out", 0));
    row("lazy", "round1 committed ssm", "gdn.ssm_state_in", (gdn_materialise(l.st), rd(l.st.ssm)), g("gdn.ssm_state_in"));
    ((uint32_t *)l.ex.keep.host())[0] = (uint32_t)k;  // the accept kept k rows
    wr(l.h_in, pick("hc_in", k));
    r.run();
    snprintf(nm, sizeof nm, "k=%d round2 h_out", k);
    char gn[64];
    snprintf(gn, sizeof gn, "hc_out rows %d,%d", k, k + 1);
    row("lazy", nm, gn, rd(l.h_out), pick("hc_out", k));
    snprintf(nm, sizeof nm, "k=%d round2 gated norm", k);
    row("lazy", nm, "gdn.gated_norm", rd(l.s.o), pick("gdn.gated_norm", k));
    snprintf(nm, sizeof nm, "k=%d committed ssm", k);
    row("lazy", nm, "gdn.ssm_state_rows[k-1]", (gdn_materialise(l.st), rd(l.st.ssm)),
        std::vector<float>(rows.begin() + (k - 1) * per, rows.begin() + k * per));
    snprintf(nm, sizeof nm, "k=%d committed conv", k);
    row("lazy", nm, "[conv_in; qkv] rows k..k+2", rd(l.st.conv),
        std::vector<float>(hist.begin() + (size_t)k * CONVD, hist.begin() + (size_t)(k + 3) * CONVD));
  }
  // cost at 4 rows: lazy (2 pending rows) vs plain
  Lyr m;
  m.R = 4;
  m.w = l.w;
  m.st = nax::alloc_state_rows(m.w, 4);
  m.s = nax::alloc_scratch_rows(4);
  m.h_in = alloc(DType::F32, {4, HC, HID});
  m.h_out = alloc(DType::F32, {4, HC, HID});
  m.ex = nax::alloc_layer_extra(0, true);
  set_inputs(m, G);
  ((uint32_t *)m.ex.keep.host())[0] = 2;
  const double tl = time_flushed([&](Recorder &r) { nax::record_layer(r, m.w, m.st, m.s, m.ex, m.h_in, m.h_out, true); });
  m.ex = nax::alloc_layer_extra();
  set_inputs(m, G);
  const double tp = time_flushed([&](Recorder &r) { nax::record_layer(r, m.w, m.st, m.s, m.ex, m.h_in, m.h_out, true); });
  printf("time L00 R=4: lazy-commit step (2 pending rows) %.1f us, plain %.1f us\n", tl, tp);
}

int main(int argc, char **argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    NE_CHECK(argc >= 3, "usage: mtp-nax-check gemm PACK [time] | layer PACK CASE_DIR ROWS [time]");
    const std::string mode = argv[1];
    auto pack = Pack::open(argv[2]);
    if (mode == "gemm") gemm(*pack, argc > 3 && std::string(argv[3]) == "time");
    else if (mode == "rollback") rollback(*pack, argv[3]);
    else if (mode == "lazy") lazy(*pack, argv[3]);
    else if (mode == "layer") {
      NE_CHECK(argc >= 5, "layer PACK CASE_DIR ROWS [time]");
      layer(*pack, argv[3], atoi(argv[4]), argc > 5 && std::string(argv[5]) == "time");
    }
    else NE_CHECK(false, "unknown mode %s", mode.c_str());
  } catch (const std::exception &e) {
    fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
