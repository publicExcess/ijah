// lab-prefill checks (docs/prefill.md). Every GPU mode runs under tools/with_lock.sh (LOCK_OWNER=lab-prefill); the
// binary takes no lock itself (an inner MachineLock waits forever on the lock with_lock already holds for it).
//   build/pf-check gemm PACK [time]             pf_q8 vs a CPU f64 reference on layer-0 weights (M = 1..300), timing
//   build/pf-check model PACK TOKENS N K [C]    decode-only N tokens vs chunked prefill (chunk C) of N - 1 + decode:
//                                              every state per layer, logits at the last prompt position and K
//                                              teacher-forced positions after it (KLD, top-1)  (~81 GB)
//   build/pf-check layers PACK TOKENS N K [C]   first K layers only (< 15 GB): decode layer steps vs prefill
//   build/pf-check speed PACK N [C] [profile]   prefill tok/s of N tokens (TOKENS file via NE_PF_TOKENS or synthetic),
//                                              stage breakdown with profile = 1  (~81 GB)
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "core/bench.h"
#include "engine.h"
#include "models/flash_next/prefill.h"

using namespace ne;
using namespace ne::fn;

static float bf2f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static uint16_t f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
static double rel(const float *a, const float *b, size_t n) {
  double e = 0, r = 0;
  for (size_t i = 0; i < n; i++) {
    const double d = (double)a[i] - b[i];
    e += d * d;
    r += (double)b[i] * b[i];
  }
  return r > 0 ? sqrt(e / r) : sqrt(e);
}

// ------------------------------------------------------------------------------------------------ gemm
struct PfGemmH {
  int K, N, M, ldx, ldy;
};
static void gemm_check(const Pack &pack, bool timing) {
  LayerW w = load_layer(pack, 0);
  std::mt19937 rng(3);
  std::normal_distribution<float> nd(0.f, 1.f);
  const Q8 *ws[] = {&w.gdn.qkv, &w.gdn.out, &w.gdn.b, &w.hca.down, &w.hca.up};
  const char *names[] = {"qkv", "out", "b", "hc_down", "hc_up"};
  for (int wi = 0; wi < 5; wi++) {
    const Q8 &q = *ws[wi];
    for (int M : {1, 7, 64, 100, 300}) {
      const int K = q.in, N = q.rows;
      Tensor x = alloc(DType::BF16, {M, K}), y = alloc(DType::F32, {M, N});
      std::vector<float> xf((size_t)M * K);
      uint16_t *xb = (uint16_t *)x.host();
      for (size_t i = 0; i < xf.size(); i++) {
        xb[i] = f2bf(nd(rng));
        xf[i] = bf2f(xb[i]);
      }
      PfGemmH p{K, N, M, K, N};
      const std::string fn = std::string("pf_q8_bf_g") + std::to_string(q.group) + (N % 64 ? "_u" : "");
      Recorder r;
      r.begin();
      r.dispatch(pso(fn, 128), {q.w, q.s, q.b, x, y, Arg::value(p)}, MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1),
                 MTLSizeMake(128, 1, 1));
      r.end();
      r.run();
      // CPU reference over a column sample (exact f64 weights = the decode GEMVs' f32 weights)
      const uint8_t *wb = (const uint8_t *)q.w.host();
      const uint16_t *sc = (const uint16_t *)q.s.host(), *bi = (const uint16_t *)q.b.host();
      const int ng = K / q.group;
      std::vector<float> ref, got;
      for (int n = 0; n < N; n += std::max(1, N / 97)) {
        for (int m = 0; m < M; m++) {
          double a = 0;
          for (int k = 0; k < K; k++) {
            const double wv = (double)bf2f(sc[(size_t)n * ng + k / q.group]) * wb[(size_t)n * K + k] +
                              bf2f(bi[(size_t)n * ng + k / q.group]);
            a += (double)wv * xf[(size_t)m * K + k];
          }
          ref.push_back((float)a);
          got.push_back(((float *)y.host())[(size_t)m * N + n]);
        }
      }
      printf("gemm %-8s [%5d x %5d] g%-3d M %4d: rel L2 %.2e\n", names[wi], N, K, q.group, M,
             rel(got.data(), ref.data(), ref.size()));
    }
  }
  if (!timing) return;
  for (int wi = 0; wi < 5; wi++) {
    const Q8 &q = *ws[wi];
    for (int M : {512, 2048}) {
      const int K = q.in, N = q.rows;
      Tensor x = alloc(DType::BF16, {M, K}), y = alloc(DType::BF16, {M, N});
      PfGemmH p{K, N, M, K, N};
      const std::string fn = std::string("pf_q8_bb_g") + std::to_string(q.group) + (N % 64 ? "_u" : "");
      auto rec = [&](int reps) {
        auto r = std::make_unique<Recorder>();
        r->begin();
        for (int i = 0; i < reps; i++) {
          r->dispatch(pso(fn, 128), {q.w, q.s, q.b, x, y, Arg::value(p)},
                      MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1), MTLSizeMake(128, 1, 1));
          r->barrier();
        }
        r->end();
        double b = 1e30;
        for (int i = 0; i < 4; i++) { r->run(); b = std::min(b, r->gpu_ms()); }
        return b;
      };
      const double ms = (rec(12) - rec(4)) / 8;
      printf("time %-8s [%5d x %5d] M %4d: %.3f ms  %.1f TFLOP/s\n", names[wi], N, K, M, ms,
             2.0 * M * N * K / (ms * 1e-3) / 1e12);
    }
  }
}

// ------------------------------------------------------------------------------------------------ model
struct Snap {
  std::vector<std::vector<float>> conv, ssm, ple, k, v, ik, bk;
};
static Snap snapshot(Model &m) {
  Snap s;
  const int T = m.T;
  gdn_materialise(m.st);  // lab-gdnlazy: deferred decode rows into ssm before copying
  for (size_t l = 0; l < m.L.size(); l++) {
    GdnState &st = m.st[l];
    auto cp = [](const Tensor &t) {
      std::vector<float> v(t.buf ? t.nbytes() / 4 : 0);
      if (t.buf) memcpy(v.data(), t.host(), v.size() * 4);
      return v;
    };
    s.conv.push_back(m.L[l].linear ? cp(st.conv) : std::vector<float>());
    s.ssm.push_back(m.L[l].linear ? cp(st.ssm) : std::vector<float>());
    s.ple.push_back(m.L[l].ple.on ? cp(st.ple_conv) : std::vector<float>());
    if (!m.L[l].linear) {
      std::vector<float> k((size_t)2 * T * 256), v((size_t)2 * T * 256), ik((size_t)T * 128), bk((size_t)(T / 4) * 128);
      attn::export_cache(st.qc, T, k.data(), v.data(), ik.data(), bk.data());
      s.k.push_back(k); s.v.push_back(v); s.ik.push_back(ik); s.bk.push_back(bk);
    } else {
      s.k.emplace_back(); s.v.emplace_back(); s.ik.emplace_back(); s.bk.emplace_back();
    }
  }
  return s;
}
static void kld_top1(const float *p, const float *q, int V, double *kl, bool *same) {
  double mp = -1e30, mq = -1e30;
  int ap = 0, aq = 0;
  for (int i = 0; i < V; i++) {
    if (p[i] > mp) { mp = p[i]; ap = i; }
    if (q[i] > mq) { mq = q[i]; aq = i; }
  }
  double sp = 0, sq = 0;
  for (int i = 0; i < V; i++) { sp += exp(p[i] - mp); sq += exp(q[i] - mq); }
  const double lp = log(sp) + mp, lq = log(sq) + mq;
  double k = 0;
  for (int i = 0; i < V; i++) {
    const double a = p[i] - lp;
    k += exp(a) * (a - (q[i] - lq));
  }
  *kl = k;
  *same = ap == aq;
}

static std::vector<int> read_tokens(const char *path) {
  std::ifstream f(path, std::ios::binary);
  NE_CHECK(f.good(), "cannot open %s", path);
  std::vector<int> t;
  int v;
  while (f.read((char *)&v, 4)) t.push_back(v);
  return t;
}

static void model_check(const Pack &pack, const char *tokpath, int N, int K, int C) {
  std::vector<int> tok = read_tokens(tokpath);
  NE_CHECK((int)tok.size() >= N + K, "token file has %zu < %d", tok.size(), N + K);
  auto mp = load_model(pack, ((N + K + 64) + 63) / 64 * 64);
  Model &m = *mp;
  using clk = std::chrono::steady_clock;
  // decode-only reference
  auto t0 = clk::now();
  for (int i = 0; i < N - 1; i++) step(m, tok[i], false);
  step(m, tok[N - 1], true);
  const double dec_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  Snap sd = snapshot(m);
  std::vector<std::vector<float>> ld;
  ld.emplace_back((float *)m.logits.host(), (float *)m.logits.host() + VOCAB);
  for (int i = 0; i < K; i++) {
    step(m, tok[N + i], true);
    ld.emplace_back((float *)m.logits.host(), (float *)m.logits.host() + VOCAB);
  }
  reset(m);
  // prefill
  // NE_PF_FLOOR=1: path B = lab-forward's 4-row prefill (decode kernels at R = 4) instead: the decode path's own
  // noise floor at whole-model level
  const bool floor4 = getenv("NE_PF_FLOOR") && atoi(getenv("NE_PF_FLOOR"));
  auto pf = prefill_create(m, C);
  t0 = clk::now();
  if (floor4) setenv("NE_FN_PREFILL", "steps", 1);  // prefill() itself routes through the chunked prefill now
  if (floor4) ne::fn::prefill(m, tok.data(), N);
  else prefill_prompt(*pf, tok.data(), N);
  const double pf_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  const PfStats &ps = prefill_stats(*pf);
  if (floor4) printf("FLOOR run: path B = 4-row decode kernels (lab-forward's prefill), not the chunked prefill\n");
  Snap sp = snapshot(m);
  std::vector<std::vector<float>> lp;
  lp.emplace_back((float *)m.logits.host(), (float *)m.logits.host() + VOCAB);
  for (int i = 0; i < K; i++) {
    step(m, tok[N + i], true);
    lp.emplace_back((float *)m.logits.host(), (float *)m.logits.host() + VOCAB);
  }
  printf("N %d chunk %d: decode-only %.0f ms (%.1f tok/s), prefill+last %.0f ms (prefill %d tokens: %.0f ms wall, "
         "%.0f GPU, %.1f tok/s; gather %.0f ms)\n",
         N, C, dec_ms, N / dec_ms * 1e3, pf_ms, N - 1, ps.wall_ms, ps.gpu_ms, (N - 1) / ps.wall_ms * 1e3, ps.gather_ms);
  double worst_ssm = 0, worst_conv = 0, worst_kv = 0;
  for (size_t l = 0; l < m.L.size(); l++) {
    if (m.L[l].linear) {
      const double es = rel(sp.ssm[l].data(), sd.ssm[l].data(), sd.ssm[l].size());
      const double ec = rel(sp.conv[l].data(), sd.conv[l].data(), sd.conv[l].size());
      worst_ssm = std::max(worst_ssm, es);
      worst_conv = std::max(worst_conv, ec);
      printf("  L%02zu gdn  ssm %.2e conv %.2e", l, es, ec);
      if (m.L[l].ple.on) printf("  ple conv %.2e", rel(sp.ple[l].data(), sd.ple[l].data(), sd.ple[l].size()));
      printf("\n");
    } else {
      const double ek = rel(sp.k[l].data(), sd.k[l].data(), sd.k[l].size());
      const double ev = rel(sp.v[l].data(), sd.v[l].data(), sd.v[l].size());
      const double ei = rel(sp.ik[l].data(), sd.ik[l].data(), sd.ik[l].size());
      const double eb = rel(sp.bk[l].data(), sd.bk[l].data(), sd.bk[l].size());
      worst_kv = std::max(worst_kv, std::max(ek, ev));
      printf("  L%02zu attn k %.2e v %.2e ikeys %.2e bkeys %.2e\n", l, ek, ev, ei, eb);
    }
  }
  double ksum = 0, kmax = 0;
  int same = 0;
  for (int i = 0; i <= K; i++) {
    double kl;
    bool s1;
    kld_top1(ld[i].data(), lp[i].data(), VOCAB, &kl, &s1);
    ksum += kl;
    kmax = std::max(kmax, kl);
    same += s1;
    if (i < 4 || i == K) printf("  pos %d: KLD %.3e top1 %s\n", N - 1 + i, kl, s1 ? "same" : "DIFF");
  }
  printf("%s N %d: worst ssm %.2e conv %.2e kv %.2e; logits over %d positions: KLD mean %.3e max %.3e, top-1 "
         "%d/%d\n",
         floor4 ? "FLOOR4" : "SUMMARY", N, worst_ssm, worst_conv, worst_kv, K + 1, ksum / (K + 1), kmax, same, K + 1);
}


// ------------------------------------------------------------------------------------------------ layers
// A partial model (layers 0 .. k-1 + embedding; RAM ~1.7 GB per layer) built here, the decode reference = lab-forward's
// 1-row layer recording (fn_embed + record_gdn_layer per layer, PLE rows by gather_ple) token by token; then reset and
// the chunked prefill of the same n tokens. Compares every row's hidden after layer k-1 and every state.
static std::unique_ptr<Model> partial_model(const Pack &pack, int k, int tmax) {
  auto m = std::make_unique<Model>();
  m->pack = &pack;
  m->tmax = tmax;
  m->eos = pack.text_config().get_int("eos_token_id", 248044);
  m->pos = alloc(DType::U32, {4});
  for (int l = 0; l < k; l++) {
    m->L.push_back(load_layer(pack, l));
    m->st.push_back(alloc_state(m->L.back(), tmax));
    if (!m->L.back().linear) {
      m->st.back().pos = m->pos;
      if (m->st.back().qc.tmax > 0) m->st.back().qc.pos = m->pos;
    }
  }
  m->embed = load_q8(pack, "language_model.model.embed_tokens");
  m->vcap = 1;  // fn_embed reads image rows from vfeat (lab-vision): one dummy row
  m->vfeat = alloc(DType::BF16, {1, HID});
  m->s = alloc_scratch(1);
  m->tok = alloc(DType::U32, {4});
  m->h[0] = alloc(DType::F32, {1, HC, HID});
  m->h[1] = alloc(DType::F32, {1, HC, HID});
  m->logits = alloc(DType::F32, {1, 16});
  const TensorInfo *ti = pack.info("ngram_table.weight");
  m->ngram_fd = open(ti->file.c_str(), O_RDONLY);
  fcntl(m->ngram_fd, F_RDAHEAD, 0);
  m->ngram_off = ti->file_offset;
  return m;
}
static void zero_states(Model &m) {
  for (auto &s : m.st) {
    if (s.conv.buf) memset(s.conv.host(), 0, s.conv.nbytes());
    if (s.ssm.buf) memset(s.ssm.host(), 0, s.ssm.nbytes());
    gdn_clear_pending(s);
    if (s.ple_conv.buf) memset(s.ple_conv.host(), 0, s.ple_conv.nbytes());
  }
  m.T = 0;
  m.hist[0] = m.hist[1] = m.eos;
  ((uint32_t *)m.pos.host())[0] = 0;
}
static void layers_check(const Pack &pack, const char *tokpath, int n, int k, int C) {
  std::vector<int> tok = read_tokens(tokpath);
  NE_CHECK((int)tok.size() >= n, "token file too short");
  auto mp = partial_model(pack, k, (n + 64 + 63) / 64 * 64);
  Model &m = *mp;
  zero_states(m);
  Recorder r;
  r.begin();
  r.dispatch(pso("fn_embed_g" + std::to_string(m.embed.group)), {m.tok, m.embed.w, m.embed.s, m.embed.b, m.h[0]},
             MTLSizeMake(1, 1, 1), MTLSizeMake(256, 1, 1));
  r.barrier();
  for (int l = 0; l < k; l++) record_gdn_layer(r, m.L[l], m.st[l], m.s, m.h[l % 2], m.h[(l + 1) % 2]);
  r.end();
  std::vector<float> hd((size_t)n * HCD);
  using clk = std::chrono::steady_clock;
  auto t0 = clk::now();
  for (int t = 0; t < n; t++) {
    ((uint32_t *)m.tok.host())[0] = (uint32_t)tok[t];
    if (k > 1) gather_ple(m, tok[t]);
    ((uint32_t *)m.pos.host())[0] = (uint32_t)m.T;
    r.run();
    memcpy(&hd[(size_t)t * HCD], m.h[k % 2].host(), HCD * 4);
    m.hist[0] = m.hist[1];
    m.hist[1] = tok[t];
    m.T++;
  }
  const double dms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  Snap sd = snapshot(m);
  zero_states(m);
  // noise floor: lab-forward's 4-row (MTP-verify / prefill4) layer kernels on the same tokens vs the 1-row steps
  if (n % 4 == 0) {
    Scratch s4 = alloc_scratch(4);
    Tensor t4 = alloc(DType::U32, {4}), h4[2] = {alloc(DType::F32, {4, HC, HID}), alloc(DType::F32, {4, HC, HID})};
    Recorder r4;
    r4.begin();
    r4.dispatch(pso("fn_embed_g" + std::to_string(m.embed.group)), {t4, m.embed.w, m.embed.s, m.embed.b, h4[0]},
                MTLSizeMake(4, 1, 1), MTLSizeMake(256, 1, 1));
    r4.barrier();
    for (int l = 0; l < k; l++) record_gdn_layer(r4, m.L[l], m.st[l], s4, h4[l % 2], h4[(l + 1) % 2]);
    r4.end();
    std::vector<double> e4;
    for (int t = 0; t < n; t += 4) {
      for (int i = 0; i < 4; i++) {
        ((uint32_t *)t4.host())[i] = (uint32_t)tok[t + i];
        if (k > 1) {
          int64_t ids[PLE_NG];
          ngram_ids(m.L[1].ple, m.hist[0], m.hist[1], tok[t + i], m.eos, ids);
          ngram_gather(pack, ids, (float *)s4.ple_emb.host() + (size_t)i * HID);
        }
        m.hist[0] = m.hist[1];
        m.hist[1] = tok[t + i];
      }
      ((uint32_t *)m.pos.host())[0] = (uint32_t)m.T;
      r4.run();
      for (int i = 0; i < 4; i++)
        e4.push_back(rel((const float *)h4[k % 2].host() + (size_t)i * HCD, &hd[(size_t)(t + i) * HCD], HCD));
      m.T += 4;
    }
    Snap s4n = snapshot(m);
    std::sort(e4.begin(), e4.end());
    printf("  FLOOR (4-row decode kernels vs 1-row): hidden median %.2e max %.2e;", e4[e4.size() / 2], e4.back());
    for (int l = 0; l < k; l++)
      if (m.L[l].linear) printf(" L%02d ssm %.2e", l, rel(s4n.ssm[l].data(), sd.ssm[l].data(), sd.ssm[l].size()));
      else printf(" L%02d k %.2e", l, rel(s4n.k[l].data(), sd.k[l].data(), sd.k[l].size()));
    printf("\n");
    zero_states(m);
  }
  auto pf = prefill_create(m, C);
  prefill_tokens(*pf, tok.data(), n);
  const PfStats &ps = prefill_stats(*pf);
  Snap sp = snapshot(m);
  const int last = (n - 1) % C + 1, r0 = n - last;  // rows of the last chunk
  const float *hp = (const float *)prefill_hidden(*pf, k).host();
  std::vector<double> er;
  for (int t = 0; t < last; t++) er.push_back(rel(hp + (size_t)t * HCD, &hd[(size_t)(r0 + t) * HCD], HCD));
  std::vector<double> es = er;
  std::sort(es.begin(), es.end());
  printf("layers %d, n %d, chunk %d: decode %.0f ms, prefill %.1f ms wall (%.1f GPU)\n", k, n, C, dms, ps.wall_ms,
         ps.gpu_ms);
  printf("  hidden after layer %d, rows %d..%d: rel L2 median %.2e max %.2e (last row %.2e)\n", k - 1, r0, n - 1,
         es[es.size() / 2], es.back(), er.back());
  for (int l = 0; l < k; l++) {
    if (m.L[l].linear) {
      printf("  L%02d gdn  ssm %.2e conv %.2e", l, rel(sp.ssm[l].data(), sd.ssm[l].data(), sd.ssm[l].size()),
             rel(sp.conv[l].data(), sd.conv[l].data(), sd.conv[l].size()));
      if (m.L[l].ple.on) printf("  ple conv %.2e", rel(sp.ple[l].data(), sd.ple[l].data(), sd.ple[l].size()));
      printf("\n");
    } else {
      printf("  L%02d attn k %.2e v %.2e ikeys %.2e bkeys %.2e\n", l,
             rel(sp.k[l].data(), sd.k[l].data(), sd.k[l].size()), rel(sp.v[l].data(), sd.v[l].data(), sd.v[l].size()),
             rel(sp.ik[l].data(), sd.ik[l].data(), sd.ik[l].size()),
             rel(sp.bk[l].data(), sd.bk[l].data(), sd.bk[l].size()));
    }
  }
}


// NE_PF_PROF=fine: GPU ms per kernel label (sorted), per chunk; and per-chunk GPU / attention
static void print_fine(const PfStats &s) {
  if (s.fine_ms.empty()) return;
  std::vector<std::pair<double, std::string>> v;
  for (auto &kv : s.fine_ms) v.push_back({kv.second, kv.first});
  std::sort(v.rbegin(), v.rend());
  printf("  fine labels (GPU ms per chunk, share):\n");
  for (auto &x : v) printf("    %-22s %8.2f  %5.1f%%\n", x.second.c_str(), x.first / s.chunks, 100 * x.first / s.gpu_ms);
  printf("  per chunk GPU ms / attention ms:");
  for (size_t i = 0; i < s.chunk_gpu_ms.size(); i++) printf(" %.0f/%.0f", s.chunk_gpu_ms[i], s.chunk_attn_ms[i]);
  printf("\n");
}

// prefill timing on a partial model (first k layers): per-stage GPU ms with profile = 1
static void lspeed(const Pack &pack, const char *tokpath, int n, int k, int C, bool profile) {
  std::vector<int> tok = read_tokens(tokpath);
  NE_CHECK((int)tok.size() >= n, "token file too short");
  auto mp = partial_model(pack, k, (n + 64 + 63) / 64 * 64);
  Model &m = *mp;
  zero_states(m);
  auto pf = prefill_create(m, C, profile);
  prefill_tokens(*pf, tok.data(), std::min(n, C));  // warm-up (records, wires)
  if (n % C) { zero_states(m); prefill_tokens(*pf, tok.data(), n % C); }
  zero_states(m);
  prefill_tokens(*pf, tok.data(), n);
  const PfStats &s = prefill_stats(*pf);
  printf("LSPEED layers %d n %d chunk %d%s: wall %.1f ms GPU %.1f ms gather %.1f ms, %d chunks, %d dispatches; "
         "%.2f ms per layer-chunk\n", k, n, C, profile ? " (profile)" : "", s.wall_ms, s.gpu_ms, s.gather_ms, s.chunks,
         s.dispatches, s.gpu_ms / k / s.chunks);
  if (profile)
    for (int i = 0; i < (int)PfStage::N; i++)
      printf("  %-10s %8.2f ms  %5.1f%%\n", kPfStageNames[i], s.stage_ms[i], 100 * s.stage_ms[i] / s.gpu_ms);
  print_fine(s);
}

// lab-attn's QSA step at 8 rows per dispatch on a random cache of T tokens (layer 3 weights; small)
static void attn_time(const char *packpath, int T) {
  ne_pack *pk = ne_pack_open(packpath);
  NE_CHECK(pk, "pack");
  ne_attn *a = ne_attn_create(pk, 3, 1, (T + 64 + 63) / 64 * 64);
  NE_CHECK(a, "%s", ne_last_error());
  std::mt19937 rng(5);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> k((size_t)2 * T * 256), v(k.size()), ik((size_t)T * 128);
  for (auto &x : k) x = nd(rng);
  for (auto &x : v) x = nd(rng);
  for (auto &x : ik) x = nd(rng);
  NE_CHECK(ne_attn_import(a, k.data(), v.data(), ik.data(), T) == 0, "%s", ne_last_error());
  const char *nm[] = {"step", "prep", "score", "select_local", "select", "attn", "merge"};
  for (int w = -1; w < 6; w++) {
    double us = 0;
    NE_CHECK(ne_attn_time(a, 8, w, 4, 12, 0, &us) == 0, "%s", ne_last_error());
    printf("attn T %d R 8 %-12s %8.1f us\n", T, nm[w + 1], us);
  }
  ne_attn_free(a);
  ne_pack_close(pk);
}

static void speed(const Pack &pack, int N, int C, bool profile) {
  std::vector<int> tok;
  if (const char *f = getenv("NE_PF_TOKENS")) tok = read_tokens(f);
  if ((int)tok.size() < N) {
    std::mt19937 rng(11);
    while ((int)tok.size() < N) tok.push_back((int)(rng() % 200000) + 1000);
  }
  auto mp = load_model(pack, (N + 64 + 63) / 64 * 64);
  Model &m = *mp;
  auto pf = prefill_create(m, C, profile);
  // warm-up: one small chunk (records + wires), then the timed run from position 0
  prefill_tokens(*pf, tok.data(), std::min(N, C));
  reset(m);
  prefill_tokens(*pf, tok.data(), N);
  const PfStats &s = prefill_stats(*pf);
  printf("SPEED N %d chunk %d%s: wall %.0f ms = %.1f tok/s; GPU %.0f ms; gather %.0f ms; %d chunks, %d dispatches\n",
         N, C, profile ? " (profile)" : "", s.wall_ms, N / s.wall_ms * 1e3, s.gpu_ms, s.gather_ms, s.chunks,
         s.dispatches);
  if (profile)
    for (int i = 0; i < (int)PfStage::N; i++)
      printf("  %-10s %8.1f ms  %5.1f%%  %.2f ms/chunk\n", kPfStageNames[i], s.stage_ms[i],
             100 * s.stage_ms[i] / s.gpu_ms, s.stage_ms[i] / s.chunks);
  print_fine(s);
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: pf-check gemm|model|speed PACK ...\n");
    return 2;
  }
  try {
    const std::string mode = argv[1];
    if (mode == "attn") { attn_time(argv[2], atoi(argv[3])); return 0; }
    auto pack = Pack::open(argv[2]);
    if (mode == "gemm") gemm_check(*pack, argc > 3 && std::string(argv[3]) == "time");
    else if (mode == "model") {
      NE_CHECK(argc >= 6, "model PACK TOKENS N K [C]");
      model_check(*pack, argv[3], atoi(argv[4]), atoi(argv[5]), argc > 6 ? atoi(argv[6]) : 512);
    } else if (mode == "layers") {
      NE_CHECK(argc >= 6, "layers PACK TOKENS N K [C]");
      layers_check(*pack, argv[3], atoi(argv[4]), atoi(argv[5]), argc > 6 ? atoi(argv[6]) : 512);
    } else if (mode == "lspeed") {
      NE_CHECK(argc >= 6, "lspeed PACK TOKENS N K [C] [profile]");
      lspeed(*pack, argv[3], atoi(argv[4]), atoi(argv[5]), argc > 6 ? atoi(argv[6]) : 2048, argc > 7 && atoi(argv[7]));
    } else if (mode == "speed") {
      NE_CHECK(argc >= 4, "speed PACK N [C] [profile]");
      speed(*pack, atoi(argv[3]), argc > 4 ? atoi(argv[4]) : 2048, argc > 5 && atoi(argv[5]) != 0);
    } else NE_CHECK(false, "unknown mode %s", mode.c_str());
  } catch (const std::exception &e) {
    fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
