// lab-mtp: the engine's MTP head vs tools/mtp_goldens.py (tests/golden/data/MTP_*), and its GPU time.
//   build/mtp-head-check PACK GOLDEN_DATA_DIR [time]
// draft1 (1 row, from the post-prefill KV), the greedy chain of depth 3 in one recording (drafts vs MTP_draft1..3),
// rows4 (a 4-row head step). time: one head step at 1 row, a depth-3 chain (machine lock).
#include <functional>

#include "core/bench.h"
#include "models/flash_next/mtp.h"
#include "tests/mtp/common.h"

using namespace ne;
using namespace ne::fn;
using namespace mtpt;

static std::string GD;
static std::vector<float> g(const std::string &c, const std::string &n) { return npy(GD + "/" + c + "/" + n + ".npy").f; }
static std::vector<int> gi(const std::string &c, const std::string &n) { return npy(GD + "/" + c + "/" + n + ".npy").i; }

static const mtp::MtpW *gW = nullptr;
static void set_kv(mtp::MtpHead &h, const std::string &c, const char *sfx) {
  auto k = g(c, std::string("attn.k_cache_") + sfx), v = g(c, std::string("attn.v_cache_") + sfx),
       ik = g(c, std::string("attn.index_keys_") + sfx);
  import_kv(h.st, gW->layer, k, v, ik, (int)(ik.size() / 128));
}
static void set_inputs(mtp::MtpHead &h, const std::string &c) {
  auto t = gi(c, "mtp.token_ids");
  std::vector<uint32_t> tu(t.begin(), t.end());
  upload(h.tok, tu.data(), tu.size() * 4);
  wr(h.hid, g(c, "mtp.hidden_in"));
}
static void run(const std::function<void(Recorder &)> &f) {
  Recorder r;
  r.begin();
  f(r);
  r.end();
  r.run();
}
static std::vector<uint32_t> amax(const mtp::MtpHead &h, int n) {
  std::vector<uint32_t> a(16);
  download(h.amax, a.data(), 64);
  a.resize(n);
  return a;
}
static void compare_step(const char *m, const std::string &c, mtp::MtpHead &h, const Tensor &ho) {
  const int R = h.rows;
  row(m, "fc_embedding", "mtp.fc_embedding", rd(h.pe), g(c, "mtp.fc_embedding"));
  row(m, "hidden normed", "mtp.hidden_normed", rd(h.hn), g(c, "mtp.hidden_normed"));
  row(m, "h_in (fused)", "hc_in", rd(h.h_in), g(c, "hc_in"));
  row(m, "q_proj (q|gate)", "attn.q_proj", rd(h.s.aq), g(c, "attn.q_proj"));
  row(m, "core * sigmoid(gate)", "attn.gated", rd(h.s.o), g(c, "attn.gated"));
  row(m, "h_mid", "attn_hc.out", rd(h.s.h2), g(c, "attn_hc.out"));
  row(m, "mlp_out", "moe.out", rd(h.s.br_m), g(c, "moe.out"));
  row(m, "h_out (next hidden)", "hc_out", rd(ho), g(c, "hc_out"));
  row(m, "mixer out", "mixer.block_in", rd(h.xh), g(c, "mixer.block_in"));
  row(m, "logits", "head.logits", rd(h.logits), g(c, "head.logits"));
  auto ga = gi(c, "mtp.argmax");
  auto a = amax(h, R);
  int eq = 0;
  for (int r = 0; r < R; r++) eq += (int)a[r] == ga[r];
  printf("%-6s argmax: %d/%d equal (engine", m, eq, R);
  for (int r = 0; r < R; r++) printf(" %u", a[r]);
  printf(", golden");
  for (int r = 0; r < R; r++) printf(" %d", ga[r]);
  printf(")\n");
}

static double time_us(const std::function<void(Recorder &)> &rec) {
  auto chain = [&](int n) {
    auto r = std::make_unique<Recorder>();
    r->begin();
    for (int i = 0; i < n; i++) rec(*r);
    r->end();
    double b = 1e30;
    for (int i = 0; i < 5; i++) { r->run(); b = std::min(b, r->gpu_ms()); }
    return b;
  };
  return (chain(8) - chain(2)) / 6 * 1e3;
}

// Draft-head evaluation: argmax agreement of (coarse bits-bit head -> top-K -> exact re-score) with the full 8-bit
// head on every golden mixer output (the MTP cases + the target's HEAD cases), and the lm_head-part GPU time.
static double time_flushed_us(const std::function<void(Recorder &)> &rec) {
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
  return ((chain(true, 24) - chain(true, 8)) - (chain(false, 24) - chain(false, 8))) / 16 * 1e3;
}
static void draft_eval(mtp::MtpW w) {
  std::vector<float> X;  // rows of 2560
  for (const char *c : {"MTP_prefill", "MTP_draft1", "MTP_draft2", "MTP_draft3", "MTP_rows4", "HEAD_decode1", "HEAD_verify4"}) {
    auto v = g(c, "mixer.block_in");
    X.insert(X.end(), v.begin(), v.end());
  }
  const int N = (int)(X.size() / HID), V = w.vocab;
  Tensor x = alloc(DType::F32, {8, HID}), lg = alloc(DType::F32, {8, V}), am = alloc(DType::U32, {16}),
         sl = alloc(DType::F32, {8, 32, 2}), cand = alloc(DType::F32, {8, 128, 32, 2});
  auto run_rows = [&](int bits, const fn::Q8 &hd, int K, std::vector<uint32_t> &out) {
    out.assign(N, 0);
    for (int r0 = 0; r0 < N; r0 += 8) {
      const int n = std::min(8, N - r0);
      std::vector<float> xs(X.begin() + (size_t)r0 * HID, X.begin() + (size_t)(r0 + n) * HID);
      xs.resize((size_t)8 * HID, 0.0f);
      upload(x, xs.data(), xs.size() * 4);
      Recorder r;
      r.begin();
      if (bits == 8) {
        nax::record_q8(r, n, x, {&w.lm_head}, {lg});
        r.barrier();
        r.dispatch(Pipelines::get("mtp_argmax", {FC::b(0, g_ref_round)}), {lg, am, Arg::value((uint32_t)V), Arg::value(0u)},
                   MTLSizeMake(n, 1, 1), MTLSizeMake(1024, 1, 1));
      } else {
        w.draft_head = hd;
        w.draft_bits = bits;
        w.shortlist = K;
        mtp::record_draft_readout(r, w, n, x, lg, cand, am, sl, 0);
      }
      r.end();
      r.run();
      std::vector<uint32_t> a(16);
      download(am, a.data(), 64);
      for (int i = 0; i < n; i++) out[r0 + i] = a[i];
    }
  };
  std::vector<uint32_t> a8;
  run_rows(8, w.lm_head, 0, a8);
  std::vector<int> gold;
  for (const char *c : {"MTP_prefill", "MTP_draft1", "MTP_draft2", "MTP_draft3", "MTP_rows4"}) {
    auto v = gi(c, "mtp.argmax");
    gold.insert(gold.end(), v.begin(), v.end());
  }
  int eqg = 0;
  for (size_t i = 0; i < gold.size(); i++) eqg += (int)a8[i] == gold[i];
  printf("draft: %d rows; full 8-bit head argmax == reference argmax on %d/%zu MTP rows\n", N, eqg, gold.size());
  for (int bits : {4, 2}) {
    double t0 = now_us();
    fn::Q8 hd = mtp::requantize(w.lm_head, bits, 64);
    const double tq = (now_us() - t0) / 1e3;
    for (int K : {1, 4, 8, 16, 32}) {
      std::vector<uint32_t> ak;
      run_rows(bits, hd, K, ak);
      int eq = 0;
      for (int i = 0; i < N; i++) eq += ak[i] == a8[i];
      printf("draft: %d-bit g64 coarse head (requantised in %.0f ms, %.0f MB), shortlist K=%2d: argmax == full head on %d/%d rows\n",
             bits, tq, (hd.w.nbytes() + hd.s.nbytes() + hd.b.nbytes()) / 1e6, K, eq, N);
    }
    w.draft_head = hd;
    w.draft_bits = bits;
    w.shortlist = bits == 4 ? 8 : 16;
    const double tc = time_flushed_us([&](Recorder &r) { mtp::record_draft_readout(r, w, 1, x, lg, cand, am, sl, 0); });
    const double tg = time_flushed_us([&](Recorder &r) { nax::record_qn(r, 1, x, hd, bits, lg); });
    printf("time: %d-bit coarse head + shortlist K=%d (1 row, flushed): %.0f us (the coarse GEMM alone %.0f us)\n", bits,
           w.shortlist, tc, tg);
  }
  const double tf = time_flushed_us([&](Recorder &r) {
    nax::record_q8(r, 1, x, {&w.lm_head}, {lg});
    r.barrier();
    r.dispatch(Pipelines::get("mtp_argmax", {FC::b(0, g_ref_round)}), {lg, am, Arg::value((uint32_t)V), Arg::value(0u)},
               MTLSizeMake(1, 1, 1), MTLSizeMake(1024, 1, 1));
  });
  printf("time: full 8-bit head + argmax (1 row, flushed): %.0f us\n", tf);
}

int main(int argc, char **argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    if (!getenv("NE_FN_ROUND")) g_ref_round = true;  // vs the MLX goldens (the library default is plain f32)
    NE_CHECK(argc >= 3, "usage: mtp-head-check PACK GOLDEN_DATA_DIR [time|draft]");
    GD = argv[2];
    auto pack = Pack::open(argv[1]);
    double t0 = now_us();
    mtp::MtpW w = mtp::load_mtp(*pack);
    gW = &w;
    printf("MTP head loaded in %.0f ms: experts K%d, fc g%d, lm_head [%d,%d] g%d, aligned copies %.1f MB\n",
           (now_us() - t0) / 1e3, w.layer.moe.gate.k, w.fc_h.group, w.lm_head.rows, w.lm_head.in, w.lm_head.group,
           pack->aligned_copy_bytes() / 1e6);
    if (argc > 3 && std::string(argv[3]) == "draft") { draft_eval(w); return 0; }
    mtp::MtpHead h1 = mtp::alloc_head(w, 1), h4 = mtp::alloc_head(w, 4);
    // 1. draft1 alone
    set_kv(h1, "MTP_prefill", "out");
    set_inputs(h1, "MTP_draft1");
    run([&](Recorder &r) { mtp::record_step(r, w, h1); });
    compare_step("draft1", "MTP_draft1", h1, h1.h_out);
    printf("draft1 KV length after the step: %u (golden 49); attention path %s\n", ((uint32_t *)h1.st.pos.host())[0],
           h1.st.qc.tmax > 0 ? "lab-attn QSA" : "dense");
    // 2. greedy chain of depth 3 from the same state, one recording
    set_kv(h1, "MTP_prefill", "out");
    set_inputs(h1, "MTP_draft1");
    run([&](Recorder &r) { mtp::record_chain(r, w, h1, 3); });
    auto a = amax(h1, 3);
    const int g1 = gi("MTP_draft1", "mtp.argmax")[0], g2 = gi("MTP_draft2", "mtp.argmax")[0],
              g3 = gi("MTP_draft3", "mtp.argmax")[0];
    printf("chain  drafts engine %u %u %u, golden %d %d %d -> %s\n", a[0], a[1], a[2], g1, g2, g3,
           ((int)a[0] == g1 && (int)a[1] == g2 && (int)a[2] == g3) ? "EQUAL" : "DIFFER");
    row("chain", "h_out after 3 steps", "MTP_draft3 hc_out", rd(h1.h_out), g("MTP_draft3", "hc_out"));
    row("chain", "logits of step 3", "MTP_draft3 head.logits", rd(h1.logits), g("MTP_draft3", "head.logits"));
    // 3. rows4
    set_kv(h4, "MTP_prefill", "out");
    set_inputs(h4, "MTP_rows4");
    run([&](Recorder &r) { mtp::record_step(r, w, h4); });
    compare_step("rows4", "MTP_rows4", h4, h4.h_out);
    {
      auto a = rd(h4.s.o), b = g("MTP_rows4", "attn.gated");
      auto q = rd(h4.s.aq), qb = g("MTP_rows4", "attn.q_proj");
      for (int r = 0; r < 4; r++) {
        std::vector<float> x(a.begin() + r * 6144, a.begin() + (r + 1) * 6144), y(b.begin() + r * 6144, b.begin() + (r + 1) * 6144);
        std::vector<float> x2(q.begin() + r * 12288, q.begin() + (r + 1) * 12288), y2(qb.begin() + r * 12288, qb.begin() + (r + 1) * 12288);
        printf("rows4  row %d: attn.gated rel %.3e, q_proj rel %.3e\n", r, rel(x, y), rel(x2, y2));
      }
    }
    // draft2 / draft3 alone from their golden inputs (KV = the previous draft's cache out)
    for (int d = 2; d <= 3; d++) {
      const std::string c = "MTP_draft" + std::to_string(d), prev = "MTP_draft" + std::to_string(d - 1);
      set_kv(h1, prev, "out");
      set_inputs(h1, c);
      run([&](Recorder &r) { mtp::record_step(r, w, h1); });
      compare_step(("draft" + std::to_string(d)).c_str(), c, h1, h1.h_out);
      auto gk = g(c, "attn.k_cache_out"), gv = g(c, "attn.v_cache_out");
      const int T = (int)(gk.size() / 512);
      std::vector<float> kc, vc;
      export_kv(h1.st, T, kc, vc);
      printf("draft%d KV per position (rel k, rel v), positions %d..%d:", d, T - 4, T - 1);
      for (int t = T - 4; t < T; t++) {
        std::vector<float> a, b, a2, b2;
        for (int gg = 0; gg < 2; gg++) {
          a.insert(a.end(), kc.begin() + ((size_t)gg * T + t) * 256, kc.begin() + ((size_t)gg * T + t + 1) * 256);
          b.insert(b.end(), gk.begin() + ((size_t)gg * T + t) * 256, gk.begin() + ((size_t)gg * T + t + 1) * 256);
          a2.insert(a2.end(), vc.begin() + ((size_t)gg * T + t) * 256, vc.begin() + ((size_t)gg * T + t + 1) * 256);
          b2.insert(b2.end(), gv.begin() + ((size_t)gg * T + t) * 256, gv.begin() + ((size_t)gg * T + t + 1) * 256);
        }
        printf(" (%.1e, %.1e)", rel(a, b), rel(a2, b2));
      }
      printf("\n");
    }
    {  // head prefill over the 48 prompt pairs in chunks of 8 rows (KV only; readout on the last chunk)
      mtp::MtpHead h8 = mtp::alloc_head(w, 8);
      auto toks = gi("MTP_prefill", "mtp.token_ids");
      auto hid = g("MTP_prefill", "mtp.hidden_in"), gho = g("MTP_prefill", "hc_out");
      const int P = (int)toks.size();
      std::vector<float> ho;
      double gpu = 0;
      for (int c = 0; c < P; c += 8) {
        std::vector<uint32_t> tu(toks.begin() + c, toks.begin() + c + 8);
        upload(h8.tok, tu.data(), 32);
        upload(h8.hid, hid.data() + (size_t)c * HCD, (size_t)8 * HCD * 4);
        Recorder r;
        r.begin();
        mtp::record_prefill_chunk(r, w, h8, c + 8 >= P);
        r.end();
        r.run();
        gpu += r.gpu_ms();
        auto o = rd(h8.h_out);
        ho.insert(ho.end(), o.begin(), o.end());
      }
      row("prefl8", "hc_out, 48 rows", "MTP_prefill hc_out", ho, gho);
      printf("prefl8 per-row hc_out rel:");
      for (int j = 0; j < P; j++) {
        std::vector<float> x(ho.begin() + (size_t)j * HCD, ho.begin() + (size_t)(j + 1) * HCD),
            y(gho.begin() + (size_t)j * HCD, gho.begin() + (size_t)(j + 1) * HCD);
        printf(" %.0e", rel(x, y));
      }
      printf("\n");
      auto a = amax(h8, 8);
      printf("prefl8 last-row argmax engine %u, golden %d; KV length %u; %.2f ms GPU for 48 tokens (%.3f ms / token)\n",
             a[7], gi("MTP_prefill", "mtp.argmax")[47], ((uint32_t *)h8.st.pos.host())[0], gpu, gpu / P);
    }
    if (argc > 3 && std::string(argv[3]) == "time") {
      set_kv(h1, "MTP_prefill", "out");
      // each repeat appends to the head's KV (pos grows by 1 per step on the GPU): fine for timing
      const double t1 = time_us([&](Recorder &r) { mtp::record_step(r, w, h1); });
      const double t3 = time_us([&](Recorder &r) { mtp::record_chain(r, w, h1, 3); });
      printf("time: head step 1 row %.0f us; depth-3 chain %.0f us (SLC-warm, differential 2 vs 8)\n", t1, t3);
      mtp::MtpHead h2 = mtp::alloc_head(w, 2);
      set_kv(h2, "MTP_prefill", "out");
      set_kv(h4, "MTP_prefill", "out");
      const double t2r = time_us([&](Recorder &r) { mtp::record_step(r, w, h2); });
      const double t4r = time_us([&](Recorder &r) { mtp::record_step(r, w, h4); });
      const double th = time_us([&](Recorder &r) { nax::record_q8(r, 1, h1.xh, {&w.lm_head}, {h1.logits}); r.barrier(); });
      printf("time: head step 2 rows %.0f us, 4 rows %.0f us; lm_head alone (1 row) %.0f us\n", t2r, t4r, th);
    }
  } catch (const std::exception &e) {
    fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
