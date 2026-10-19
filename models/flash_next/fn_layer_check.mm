// Layer check (lab-layer): runs the Flash-Next layer-0 GDN step on the real pack and compares every sub-block
// with lab-ref's goldens (tests/golden/data/L00_decode1). Built against the core internals (no C API needed).
//   build/fn-layer-check PACK GOLDEN_DIR [chain|stage|all(=chain+stage)|time] [layer]
// chain: golden layer input + states -> whole step; stage: every sub-block alone on golden inputs; time: GPU time.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "models/flash_next/layer.h"

using namespace ne;
using namespace ne::fn;

struct Npy {
  std::vector<float> f;
  std::vector<int32_t> i;
  bool isint = false;
  size_t n = 0;
};
static Npy npy(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  NE_CHECK(in.good(), "cannot open %s", path.c_str());
  char magic[8];
  in.read(magic, 8);
  NE_CHECK(memcmp(magic, "\x93NUMPY", 6) == 0, "%s: not npy", path.c_str());
  uint32_t hl = 0;
  if (magic[6] == 1) { uint16_t h; in.read((char *)&h, 2); hl = h; }
  else in.read((char *)&hl, 4);
  std::string hdr(hl, ' ');
  in.read(hdr.data(), hl);
  Npy r;
  const bool f4 = hdr.find("'<f4'") != std::string::npos, i4 = hdr.find("'<i4'") != std::string::npos;
  NE_CHECK(f4 || i4, "%s: dtype %s", path.c_str(), hdr.c_str());
  NE_CHECK(hdr.find("'fortran_order': False") != std::string::npos, "%s: fortran order", path.c_str());
  auto a = hdr.find('('), b = hdr.find(')');
  std::stringstream ss(hdr.substr(a + 1, b - a - 1));
  size_t n = 1;
  std::string tok;
  while (std::getline(ss, tok, ',')) if (tok.find_first_not_of(" ") != std::string::npos) n *= std::stoull(tok);
  r.n = n;
  r.isint = i4;
  if (f4) { r.f.resize(n); in.read((char *)r.f.data(), n * 4); }
  else { r.i.resize(n); in.read((char *)r.i.data(), n * 4); }
  NE_CHECK(in.good(), "%s: short read", path.c_str());
  return r;
}

static std::string G;
static std::vector<float> gold(const std::string &name) { return npy(G + "/" + name + ".npy").f; }
static std::vector<float> rd(const Tensor &t, size_t n = 0, size_t off = 0) {
  if (!n) n = t.numel() - off;
  std::vector<float> v(t.numel());
  download(t, v.data(), t.numel() * 4);
  return std::vector<float>(v.begin() + off, v.begin() + off + n);
}
static void wr(const Tensor &t, const std::vector<float> &v, size_t off = 0) {
  std::vector<float> all(t.numel());
  download(t, all.data(), all.size() * 4);
  std::copy(v.begin(), v.end(), all.begin() + off);
  upload(t, all.data(), all.size() * 4);
}
static int NR = 1;    // rows of the case
static int T_in = 0;  // attention layers: cached tokens before the step
// rows r of a [NR][stride] array: elements [off, off + n) of each row, concatenated
static std::vector<float> cols(const std::vector<float> &v, size_t stride, size_t off, size_t n) {
  std::vector<float> o;
  for (int r = 0; r < NR; r++) o.insert(o.end(), v.begin() + r * stride + off, v.begin() + r * stride + off + n);
  return o;
}
static void wr_cols(const Tensor &t, const std::vector<float> &src, size_t stride, size_t off, size_t n) {
  std::vector<float> all(t.numel());
  download(t, all.data(), all.size() * 4);
  for (int r = 0; r < NR; r++) std::copy(src.begin() + r * n, src.begin() + (r + 1) * n, all.begin() + r * stride + off);
  upload(t, all.data(), all.size() * 4);
}
static double rel(const std::vector<float> &a, const std::vector<float> &b) {
  NE_CHECK(a.size() == b.size(), "size %zu vs %zu", a.size(), b.size());
  double e = 0, n = 0;
  for (size_t i = 0; i < a.size(); i++) { const double d = (double)a[i] - b[i]; e += d * d; n += (double)b[i] * b[i]; }
  return std::sqrt(e / std::max(n, 1e-30));
}
static double maxabs(const std::vector<float> &a, const std::vector<float> &b) {
  double m = 0;
  for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs((double)a[i] - b[i]));
  return m;
}
static void row(const char *mode, const char *ours, const char *golden, const std::vector<float> &a,
                const std::vector<float> &b) {
  bool fin = true;
  for (float x : a) fin &= std::isfinite(x);
  printf("%-6s %-22s %-24s n=%6zu rel_l2 %.3e  max_abs %.3e%s\n", mode, ours, golden, a.size(), rel(a, b),
         maxabs(a, b), fin ? "" : "  NONFINITE");
}

struct L {
  LayerW w;
  GdnState st;
  Scratch s;
  Tensor h_in, h_out;
};

static void compare_all(const char *mode, L &l, int stage = -1) {
  Scratch &s = l.s;
  auto want = [&](Stage st) { return stage < 0 || stage == (int)st; };
  if (want(Stage::AttnHCPre)) {
    row(mode, "attn_mix", "attn_hc.mix", rd(s.mix_a), gold("attn_hc.mix"));
    row(mode, "attn_inj[0:4]", "attn_hc.inject_w", cols(rd(s.inj_a), 8, 0, 4), gold("attn_hc.inject_w"));
  }
  if (want(Stage::AttnHCMix)) row(mode, "attn_in", "attn_hc.block_in", rd(s.xa), gold("attn_hc.block_in"));
  if (!l.w.linear) {
    const int T0 = T_in;
    if (want(Stage::GdnIn)) {
      row(mode, "q_proj (q|gate)", "attn.q_proj", rd(s.aq), gold("attn.q_proj"));
      auto kv = rd(s.akv);
      row(mode, "k_proj", "attn.k_proj", cols(kv, 1024, 0, 512), gold("attn.k_proj"));
      row(mode, "v_proj", "attn.v_proj", cols(kv, 1024, 512, 512), gold("attn.v_proj"));
      row(mode, "indexer qk", "attn.idx_qk", rd(s.aiqk), gold("attn.idx_qk"));
      auto ik = rd(l.st.ikeys, (size_t)(T0 + NR) * 128);
      row(mode, "index_keys", "attn.index_keys_out", ik, gold("attn.index_keys_out"));
    }
    if (want(Stage::GdnStep)) {
      row(mode, "core * sigmoid(gate)", "attn.gated", rd(s.o), gold("attn.gated"));
      for (int which = 0; which < 2; which++) {
        auto c = rd(which ? l.st.vc : l.st.kc);
        std::vector<float> o;
        for (int g = 0; g < 2; g++)
          o.insert(o.end(), c.begin() + (size_t)g * l.st.tmax * 256, c.begin() + ((size_t)g * l.st.tmax + T0 + NR) * 256);
        row(mode, which ? "v_cache" : "k_cache (roped)", which ? "attn.v_cache_out" : "attn.k_cache_out", o,
            gold(which ? "attn.v_cache_out" : "attn.k_cache_out"));
      }
    }
    if (want(Stage::GdnOut)) {
      row(mode, "attn_out", "attn.out", rd(s.br_a), gold("attn.out"));
      row(mode, "h_mid", "attn_hc.out", rd(s.h2), gold("attn_hc.out"));
    }
  }
  if (l.w.linear && want(Stage::GdnIn)) {
    row(mode, "qkv(conv+silu)", "gdn.conv_out", rd(s.qkv), gold("gdn.conv_out"));
    row(mode, "z", "gdn.z", rd(s.z), gold("gdn.z"));
    row(mode, "g", "gdn.g", cols(rd(s.gb), 96, 0, 48), gold("gdn.g"));
    row(mode, "beta", "gdn.beta", cols(rd(s.gb), 96, 48, 48), gold("gdn.beta"));
    row(mode, "conv_state", "gdn.conv_state_out", rd(l.st.conv), gold("gdn.conv_state_out"));
  }
  if (l.w.linear && want(Stage::GdnStep)) {
    row(mode, "core(gated norm)", "gdn.gated_norm", rd(s.o), gold("gdn.gated_norm"));
    gdn_materialise(l.st);  // lab-gdnlazy: the 1-row step deferred its store
    row(mode, "ssm_state", "gdn.ssm_state_out", rd(l.st.ssm), gold("gdn.ssm_state_out"));
  }
  if (l.w.linear && want(Stage::GdnOut)) {
    row(mode, "attn_out", "gdn.out", rd(s.br_a), gold("gdn.out"));
    row(mode, "h_mid", "attn_hc.out", rd(s.h2), gold("attn_hc.out"));
  }
  if (want(Stage::MlpHCPre)) {
    row(mode, "mlp_mix", "mlp_hc.mix", rd(s.mix_m), gold("mlp_hc.mix"));
    row(mode, "mlp_inj[0:4]", "mlp_hc.inject_w", cols(rd(s.inj_m), 8, 0, 4), gold("mlp_hc.inject_w"));
  }
  if (want(Stage::MlpHCMix)) row(mode, "mlp_in", "mlp_hc.block_in", rd(s.xm), gold("mlp_hc.block_in"));
  if (want(Stage::MoeRouter)) {
    auto lg = rd(s.logits);
    row(mode, "logits[0:512]", "moe.router_logits", cols(lg, 513, 0, 512), gold("moe.router_logits"));
    std::vector<float> sgt;
    for (int r = 0; r < NR; r++) sgt.push_back(1.0f / (1.0f + std::exp(-lg[r * 513 + 512])));
    row(mode, "sigmoid(logit 512)", "moe.shared_scale", sgt, gold("moe.shared_scale"));
    std::vector<uint32_t> route(20 * NR);
    download(s.route, route.data(), route.size() * 4);
    Npy gid = npy(G + "/moe.topk_ids.npy");
    auto gw = gold("moe.topk_weights");
    int match = 0;
    std::vector<float> ow, rw;
    for (int r = 0; r < NR; r++) {
      std::map<int, float> gm;
      for (int k = 0; k < 10; k++) gm[gid.i[r * 10 + k]] = gw[r * 10 + k];
      for (int k = 0; k < 10; k++) {
        float sc;
        memcpy(&sc, &route[r * 20 + 10 + k], 4);
        const int id = (int)route[r * 20 + k];
        if (gm.count(id)) { match++; ow.push_back(sc); rw.push_back(gm[id]); }
      }
    }
    printf("%-6s top-10 ids: %d/%d in the golden sets\n", mode, match, 10 * NR);
    if (!ow.empty()) row(mode, "scores (matched ids)", "moe.topk_weights", ow, rw);
    auto gu = rd(s.gush);
    row(mode, "shared gate", "moe.shared_gate_proj", cols(gu, 1280, 0, 640), gold("moe.shared_gate_proj"));
    row(mode, "shared up", "moe.shared_up_proj", cols(gu, 1280, 640, 640), gold("moe.shared_up_proj"));
  }
  if (want(Stage::MoeDown)) {
    auto lg = rd(s.logits);
    auto sho = rd(s.sho);
    row(mode, "shared down", "moe.shared_out", sho, gold("moe.shared_out"));
    auto br = rd(s.br_m);
    std::vector<float> routed(br.size());
    for (size_t i = 0; i < br.size(); i++) {
      const size_t r = i / 2560;
      routed[i] = br[i] - sho[i] / (1.0f + std::exp(-lg[r * 513 + 512]));
    }
    {
      auto guo = rd(s.guo);
      int bad = 0; double mx = 0;
      for (float x : guo) { if (!std::isfinite(x)) bad++; else mx = std::max(mx, (double)std::fabs(x)); }
      printf("%-6s experts_gu: %d nonfinite of %zu, max |finite| %.3e; first %g %g %g\n", mode, bad, guo.size(), mx,
             guo[0], guo[1], guo[640]);
    }
    row(mode, "routed (out - sh)", "moe.routed_out", routed, gold("moe.routed_out"));
    row(mode, "mlp_out", "moe.out", br, gold("moe.out"));
    row(mode, "h_out", "hc_out", rd(l.h_out), gold("hc_out"));
  }
}

static void run(std::unique_ptr<Recorder> &r, std::function<void(Recorder &)> rec) {
  r = std::make_unique<Recorder>();
  r->begin();
  rec(*r);
  r->end();
  r->run();
}

int main(int argc, char **argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    if (!getenv("NE_FN_ROUND")) g_ref_round = true;  // vs the MLX goldens (the library default is plain f32)
    NE_CHECK(argc >= 3, "usage: fn-layer-check PACK GOLDEN_DIR [chain|stage|all(=chain+stage)|time] [layer]");
    G = argv[2];
    const std::string mode = argc > 3 ? argv[3] : "all";
    int layer = 0;  // from the case manifest ("layer": N), or argv[4]
    {
      std::ifstream mf(G + "/manifest.json");
      std::string txt((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
      const auto k = txt.find("\"layer\":");
      if (k != std::string::npos) layer = atoi(txt.c_str() + k + 8);
    }
    if (argc > 4) layer = atoi(argv[4]);
    double t0 = now_us();
    NR = (int)(npy(G + "/hc_in.npy").n / 10240);
    auto pack = Pack::open(argv[1]);
    L l;
    l.w = load_layer(*pack, layer);
    l.st = alloc_state(l.w);
    if (!l.w.linear) T_in = (int)(npy(G + "/attn.k_cache_in.npy").n / 512);
    l.s = alloc_scratch(NR);
    l.h_in = alloc(DType::F32, {NR, HC, HID});
    l.h_out = alloc(DType::F32, {NR, HC, HID});
    printf("rows %d; pack open + layer %d load %.1f ms; trunk groups hc %d/%d gdn %d out %d; experts K%d; aligned copies %.1f MB\n", NR, layer,
           (now_us() - t0) / 1e3, l.w.hca.down.group, l.w.hca.up.group, l.w.gdn.qkv.group, l.w.gdn.out.group,
           l.w.moe.gate.k, g_aligned_copy_bytes / 1e6);
    auto set_states = [&] {
      if (l.w.linear) {
        wr(l.st.conv, gold("gdn.conv_state_in"));
        wr(l.st.ssm, gold("gdn.ssm_state_in")); gdn_clear_pending(l.st);
        return;
      }
      auto k = gold("attn.k_cache_in"), v = gold("attn.v_cache_in"), ik = gold("attn.index_keys_in");
      for (int g = 0; g < 2; g++) {
        memcpy((float *)l.st.kc.host() + (size_t)g * l.st.tmax * 256, k.data() + (size_t)g * T_in * 256, T_in * 1024);
        memcpy((float *)l.st.vc.host() + (size_t)g * l.st.tmax * 256, v.data() + (size_t)g * T_in * 256, T_in * 1024);
      }
      memcpy(l.st.ikeys.host(), ik.data(), (size_t)T_in * 512);
      ((uint32_t *)l.st.pos.host())[0] = T_in;
    };
    std::unique_ptr<Recorder> r;
    if (mode == "chain" || mode == "all") {
      set_states();
      wr(l.h_in, gold("hc_in"));
      run(r, [&](Recorder &rr) { record_gdn_layer(rr, l.w, l.st, l.s, l.h_in, l.h_out); });
      printf("chain: one step, %zu dispatches, GPU %.3f ms (cold, one command buffer)\n", r->size(), r->gpu_ms());
      compare_all("chain", l);
    }
    if (mode == "stage" || mode == "all") {
      for (int st = 0; st < (int)Stage::N; st++) {
        switch ((Stage)st) {
          case Stage::AttnHCPre: wr(l.h_in, gold("hc_in")); break;
          case Stage::AttnHCMix: wr(l.s.mix_a, gold("attn_hc.mix")); break;  // inv-rms from attn_hc_pre
          case Stage::GdnIn:
            wr(l.s.xa, gold("attn_hc.block_in"));
            set_states();
            break;
          case Stage::GdnStep:
            if (!l.w.linear) {
              set_states();
              wr(l.s.aq, gold("attn.q_proj"));
              wr_cols(l.s.akv, gold("attn.k_proj"), 1024, 0, 512);
              wr_cols(l.s.akv, gold("attn.v_proj"), 1024, 512, 512);
              break;
            }
            {
            wr(l.s.qkv, gold("gdn.conv_out"));
            wr_cols(l.s.gb, gold("gdn.g"), 96, 0, 48);
            wr_cols(l.s.gb, gold("gdn.beta"), 96, 48, 48);
            wr(l.s.z, gold("gdn.z"));
            wr(l.st.ssm, gold("gdn.ssm_state_in")); gdn_clear_pending(l.st);
            break;
          }
          case Stage::GdnOut:
            wr(l.s.o, gold(l.w.linear ? "gdn.gated_norm" : "attn.gated"));
            wr(l.h_in, gold("hc_in"));
            wr_cols(l.s.inj_a, gold("attn_hc.inject_w"), 8, 0, 4);
            break;
          case Stage::MlpHCPre: wr(l.s.h2, gold("attn_hc.out")); break;
          case Stage::MlpHCMix: wr(l.s.mix_m, gold("mlp_hc.mix")); break;
          case Stage::MoeRouter: wr(l.s.xm, gold("mlp_hc.block_in")); break;
          case Stage::MoeDown:  // experts_gu / route / shared from the two previous stages (golden MoE input)
            wr(l.s.h2, gold("attn_hc.out"));
            wr_cols(l.s.inj_m, gold("mlp_hc.inject_w"), 8, 0, 4);
            break;
          default: break;
        }
        run(r, [&](Recorder &rr) { record_stage(rr, (Stage)st, l.w, l.st, l.s, l.h_in, l.h_out); });
        compare_all("stage", l, st);
      }
    }
    if (mode == "time") {  // GPU timing: take the machine lock first
      set_states();
      wr(l.h_in, gold("hc_in"));
      double us, uf;
      time_layer(l.w, l.st, l.s, l.h_in, l.h_out, Backend::Auto, 16, 64, 0, &us, &uf);
      printf("time: backend %s; %.1f us / layer step (differential 16 vs 64 steps, same layer, SLC-warm)\n",
             backend_name(resolve_backend(Backend::Auto)), us);
      time_layer(l.w, l.st, l.s, l.h_in, l.h_out, Backend::Auto, 8, 32, 256, &us, &uf);
      printf("time: %.1f us / layer step with a 256 MB SLC flush between steps (flush %.1f us subtracted)\n", us, uf);
      // per stage, alone, with a 128 MB streaming read between repeats (its own cost subtracted): DRAM-cold weights
      Tensor fl = alloc(DType::U8, {128 << 20}), flo = alloc(DType::U32, {1 << 16});
      const uint32_t n4 = (128u << 20) / 16;
      auto rd_pso = Pipelines::get("ne_stream_read");
      auto chain = [&](int st, int n) {
        auto rr = std::make_unique<Recorder>();
        rr->begin();
        for (int i = 0; i < n; i++) {
          if (st >= 0) record_stage(*rr, (Stage)st, l.w, l.st, l.s, l.h_in, l.h_out);
          rr->dispatch(rd_pso, {fl, flo, Arg::value(n4)}, MTLSizeMake(2048, 1, 1), MTLSizeMake(256, 1, 1));
          rr->barrier();
        }
        rr->end();
        double b = 1e30;
        for (int i = 0; i < 5; i++) { rr->run(); b = std::min(b, rr->gpu_ms()); }
        return b;
      };
      const double f = (chain(-1, 32) - chain(-1, 8)) * 1e3 / 24;
      double tot = 0;
      for (int st = 0; st < (int)Stage::N; st++) {
        const double t = (chain(st, 32) - chain(st, 8)) * 1e3 / 24 - f;
        tot += t;
        printf("time: stage %d %-12s %6.1f us (alone, after a 128 MB flush)\n", st, kStageNames[st], t);
      }
      printf("time: sum of stages %.1f us (flush %.1f us subtracted per step)\n", tot, f);
    }
  } catch (const std::exception &e) {
    fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
