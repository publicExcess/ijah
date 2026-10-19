// lab-loop: the MTP token loop's own kernels vs their references (layer 1's PLE weights only, < 1 GB; machine lock).
//   build/mtp-loop-check PACK
// (1) mtp_ple_mix_lazy over two verify rounds (4 rows, keep 2, then 3 rows) vs lab-forward's fn_ple_mix run serially
//     on the kept rows only (same projections fed to both): outputs and committed conv state must be bit-identical.
// (2) the loop's PLE projections at 4 rows (nax::record_ple_proj: q8 NAX GEMM, or on a T6 pack the K8 t6 Linears)
//     vs the 1-row decode's (fn::record_ple_proj) row by row (rel L2).
// (3) mtp_topk_local + mtp_topk_merge (K = 20) on random logits with ties vs a CPU sort (exact ids and values).
#include <algorithm>
#include <numeric>
#include <random>

#include "kernels/nax/nax.h"
#include "tests/mtp/common.h"

using namespace ne;
using namespace ne::fn;
using namespace mtpt;

static MTLSize sz(int n) { return MTLSizeMake(n, 1, 1); }
static float bfr(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
  memcpy(&x, &u, 4);
  return x;
}
static double max_abs(const std::vector<float> &a, const std::vector<float> &b) {
  double m = 0;
  for (size_t i = 0; i < a.size(); i++) m = std::max(m, (double)std::fabs(a[i] - b[i]));
  return m;
}

int main(int argc, char **argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  int fails = 0;
  try {
    NE_CHECK(argc >= 2, "usage: mtp-loop-check PACK");
    auto pack = Pack::open(argv[1]);
    LayerW w = load_layer(*pack, 1);
    NE_CHECK(w.ple.on, "layer 1 has no PLE");
    const PleW &e = w.ple;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto rnd = [&](size_t n, float sc) {
      std::vector<float> v(n);
      for (auto &x : v) x = bfr(nd(rng) * sc);
      return v;
    };
    constexpr int KV = HCD + HID;
    // ---- (1) lazy PLE mix
    const int RA = 4, RB = 3, KEEP = 2;
    auto s0 = rnd((size_t)PLE_ST * HCD, 0.3f);
    auto hA = rnd((size_t)RA * HCD, 1.0f), hB = rnd((size_t)RB * HCD, 1.0f);
    auto kvA = rnd((size_t)RA * KV, 1.0f), kvB = rnd((size_t)RB * KV, 1.0f);
    auto split = [&](const std::vector<float> &kv, int R, std::vector<float> &k, std::vector<float> &v) {
      k.resize((size_t)R * HCD);
      v.resize((size_t)R * HID);
      for (int r = 0; r < R; r++) {
        memcpy(&k[(size_t)r * HCD], &kv[(size_t)r * KV], HCD * 4);
        memcpy(&v[(size_t)r * HID], &kv[(size_t)r * KV + HCD], HID * 4);
      }
    };
    std::vector<float> kA, vA, kB, vB;
    split(kvA, RA, kA, vA);
    split(kvB, RB, kB, vB);
    Tensor state = alloc(DType::F32, {PLE_ST, HCD}), pend = alloc(DType::F32, {8, HCD}), keep = alloc(DType::U32, {4});
    Tensor h = alloc(DType::F32, {8, HCD}), kb = alloc(DType::F32, {8, HCD}), vb = alloc(DType::F32, {8, HID}),
           kv = alloc(DType::F32, {8, KV}), hp = alloc(DType::F32, {8, HCD});
    auto mix_ref = [&](int R, const float *hh, const float *kvv) {  // fn_ple_mix_rR on state, -> hp rows
      upload(h, hh, (size_t)R * HCD * 4);
      upload(kv, kvv, (size_t)R * KV * 4);
      Recorder r;
      r.begin();
      r.dispatch(pso("fn_ple_mix_r" + std::to_string(R), 1024), {h, kv, e.nk, e.nq, e.nc, e.conv, state, hp}, sz(HC),
                 sz(1024));
      r.end();
      r.run();
      return rd(hp, (size_t)R * HCD);
    };
    auto mix_lazy = [&](int R, const float *hh, const float *kk, const float *vv, uint32_t P) {
      upload(h, hh, (size_t)R * HCD * 4);
      upload(kb, kk, (size_t)R * HCD * 4);
      upload(vb, vv, (size_t)R * HID * 4);
      ((uint32_t *)keep.host())[0] = P;
      Recorder r;
      r.begin();
      r.dispatch(pso("mtp_ple_mix_lazy_r" + std::to_string(R), 1024),
                 {h, kb, vb, e.nk, e.nq, e.nc, e.conv, state, hp, pend, keep}, sz(HC), sz(1024));
      r.end();
      r.run();
      return rd(hp, (size_t)R * HCD);
    };
    // reference: rows A0, A1 (kept), then B0..B2, one row at a time from s0
    upload(state, s0.data(), s0.size() * 4);
    std::vector<float> refA, refB;
    for (int r = 0; r < KEEP; r++) {
      auto o = mix_ref(1, &hA[(size_t)r * HCD], &kvA[(size_t)r * KV]);
      refA.insert(refA.end(), o.begin(), o.end());
    }
    const auto s1 = rd(state, s0.size());
    for (int r = 0; r < RB; r++) {
      auto o = mix_ref(1, &hB[(size_t)r * HCD], &kvB[(size_t)r * KV]);
      refB.insert(refB.end(), o.begin(), o.end());
    }
    // lazy: round A (4 rows, nothing pending), keep 2, round B (3 rows)
    upload(state, s0.data(), s0.size() * 4);
    auto la = mix_lazy(RA, hA.data(), kA.data(), vA.data(), 0);
    const auto sa = rd(state, s0.size());
    auto lb = mix_lazy(RB, hB.data(), kB.data(), vB.data(), KEEP);
    const auto sb = rd(state, s0.size());
    la.resize(refA.size());
    const double dA = max_abs(la, refA), dB = max_abs(lb, refB), dS0 = max_abs(sa, s0), dS1 = max_abs(sb, s1);
    const bool ok1 = dA == 0 && dB == 0 && dS0 == 0 && dS1 == 0;
    printf("(1) lazy PLE mix (4 rows, keep 2, 3 rows) vs serial fn_ple_mix_r1: max |d| rows A %.3g, rows B %.3g, "
           "state after A (= unchanged) %.3g, committed state %.3g: %s\n", dA, dB, dS0, dS1, ok1 ? "OK" : "FAIL");
    fails += !ok1;
    // ---- (2) the loop's PLE projections at 4 rows (nax::record_ple_proj) vs the 1-row decode's (fn::record_ple_proj)
    // run row by row: the same quantized weights, other kernels (q8 pack: NAX GEMM vs fn_ple_proj_g64_r1; T6 pack:
    // the K8 Linears' NAX path vs their SIMD path), so only reduction order and bf16 output rounding differ.
    {
      const int R = 4;
      auto emb = rnd((size_t)R * HID, 0.05f);
      Scratch s4 = alloc_scratch(R), s1 = alloc_scratch(1);
      upload(s4.ple_emb, emb.data(), emb.size() * 4);
      Recorder r;
      r.begin();
      nax::record_ple_proj(r, R, s4, e, s4.ple_emb, kb, vb);
      r.end();
      r.run();
      auto k = rd(kb, (size_t)R * HCD), v = rd(vb, (size_t)R * HID);
      std::vector<float> ref;
      for (int t = 0; t < R; t++) {
        upload(s1.ple_emb, &emb[(size_t)t * HID], HID * 4);
        Recorder r1;
        r1.begin();
        record_ple_proj(r1, e, s1);
        r1.end();
        r1.run();
        auto o = rd(s1.ple_kv, KV);
        ref.insert(ref.end(), o.begin(), o.end());
      }
      double num = 0, den = 0;
      for (int t = 0; t < R; t++) {
        for (int i = 0; i < HCD; i++) { double d = k[t * HCD + i] - ref[t * KV + i]; num += d * d; den += ref[t * KV + i] * (double)ref[t * KV + i]; }
        for (int i = 0; i < HID; i++) { double d = v[t * HID + i] - ref[t * KV + HCD + i]; num += d * d; den += ref[t * KV + HCD + i] * (double)ref[t * KV + HCD + i]; }
      }
      const double rel = std::sqrt(num / den);
      printf("(2) PLE key/value projections (%s), loop at 4 rows vs 1-row decode x 4: rel L2 %.2e: %s\n",
             e.key.t6 ? "K8 t6 Linears, NAX vs SIMD" : "q8, NAX vs fn_ple_proj_g64_r1", rel, rel < 5e-3 ? "OK" : "FAIL");
      fails += !(rel < 5e-3);
    }
    // ---- (3) top-K merge
    {
      const int R = 3, V = 248320;
      const uint32_t K = 20;
      std::vector<float> lg((size_t)R * V);
      for (auto &x : lg) x = bfr(nd(rng) * 2.0f);  // bf16 values: many exact ties
      for (int r = 0; r < R; r++) { lg[(size_t)r * V + 1000 + r] = 30.0f; lg[(size_t)r * V + 999] = 30.0f; }  // tie at the top
      Tensor L = alloc(DType::F32, {R, V}), cand = alloc(DType::F32, {R, 128, 32, 2}), sl = alloc(DType::F32, {R, 32, 2}),
             am = alloc(DType::U32, {16});
      upload(L, lg.data(), lg.size() * 4);
      Recorder r;
      r.begin();
      const uint32_t n = V;
      r.dispatch(pso("mtp_topk_local"), {L, cand, Arg::value(n), Arg::value(K)}, MTLSizeMake(128, R, 1), sz(256));
      r.barrier();
      r.dispatch(pso("mtp_topk_merge", 1024), {cand, sl, am, Arg::value(K)}, sz(R), sz(1024));
      r.end();
      r.run();
      auto s = rd(sl, (size_t)R * K * 2);
      std::vector<uint32_t> a(16);
      download(am, a.data(), 64);
      int bad = 0;
      for (int rr = 0; rr < R; rr++) {
        std::vector<int> idx(V);
        std::iota(idx.begin(), idx.end(), 0);
        const float *l = &lg[(size_t)rr * V];
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                          [&](int x, int y) { return l[x] > l[y] || (l[x] == l[y] && x < y); });
        for (uint32_t k = 0; k < K; k++) {
          uint32_t id;
          memcpy(&id, &s[((size_t)rr * K + k) * 2], 4);
          if ((int)id != idx[k] || s[((size_t)rr * K + k) * 2 + 1] != l[idx[k]]) bad++;
        }
        if ((int)a[rr] != idx[0]) bad++;
      }
      printf("(3) top-%u merge vs CPU sort (3 rows, bf16 ties): %d mismatches: %s\n", K, bad, bad ? "FAIL" : "OK");
      fails += bad != 0;
    }
    // ---- (4) mtp_draft_sample: the stored proposal = the CPU filtered softmax, the draws follow it
    {
      struct SP { float temp, top_p; uint32_t top_k, seed, round, K, pad0, pad1; };
      const uint32_t K = 8, NR = 4096;  // NR rows x 1 draw each (slot = row, round 0)
      const float lgs[8] = {3.0f, 2.5f, 2.5f, 1.0f, 0.5f, 0.0f, -1.0f, -3.0f};
      const int ids8[8] = {50, 7, 9, 3, 11, 2, 99, 4};  // unsorted ids; ties 7 / 9 (lowest id first)
      std::vector<float> slv((size_t)NR * K * 2);
      for (uint32_t r = 0; r < NR; r++)
        for (uint32_t k = 0; k < K; k++) {
          const uint32_t id = (uint32_t)ids8[(k * 3 + r) % K];  // rows permute the pairs
          int src = 0;
          while (ids8[src] != (int)id) src++;
          memcpy(&slv[((size_t)r * K + k) * 2], &id, 4);
          slv[((size_t)r * K + k) * 2 + 1] = lgs[src];
        }
      Tensor slt = alloc(DType::F32, {(int64_t)NR, K, 2}), am = alloc(DType::U32, {(int64_t)NR}),
             qbt = alloc(DType::F32, {(int64_t)NR, 32, 2}), spt = alloc(DType::U32, {8});
      upload(slt, slv.data(), slv.size() * 4);
      SP sp{1.0f, 0.95f, 20, 777, 0, K, 0, 0};
      upload(spt, &sp, sizeof(sp));
      Recorder r;
      r.begin();
      r.dispatch(pso("mtp_draft_sample", 32), {slt, am, qbt, spt, Arg::value(0u)}, sz(NR), sz(32));
      r.end();
      r.run();
      // CPU: sorted (logit desc, id asc), softmax, top-p 0.95
      std::vector<int> ord(K);
      std::iota(ord.begin(), ord.end(), 0);
      std::sort(ord.begin(), ord.end(), [&](int x, int y) { return lgs[x] > lgs[y] || (lgs[x] == lgs[y] && ids8[x] < ids8[y]); });
      std::vector<double> w(K);
      double z = 0;
      for (uint32_t i = 0; i < K; i++) { w[i] = std::exp(lgs[ord[i]] - lgs[ord[0]]); z += w[i]; }
      uint32_t n = K;
      double c = 0;
      for (uint32_t i = 0; i < K; i++) { c += w[i] / z; if (c >= 0.95) { n = i + 1; break; } }
      double zn = 0;
      for (uint32_t i = 0; i < n; i++) zn += w[i];
      auto q = rd(qbt, (size_t)NR * 64);
      std::vector<uint32_t> a(NR);
      download(am, a.data(), NR * 4);
      double qerr = 0;
      int idbad = 0;
      std::vector<int> cnt(n, 0);
      for (uint32_t rr = 0; rr < NR; rr++) {
        for (uint32_t i = 0; i < 32; i++) {
          uint32_t id;
          memcpy(&id, &q[(size_t)rr * 64 + 2 * i], 4);
          const uint32_t want = i < n ? (uint32_t)ids8[ord[i]] : 0xFFFFFFFFu;
          idbad += id != want;
          qerr = std::max(qerr, std::fabs(q[(size_t)rr * 64 + 2 * i + 1] - (i < n ? w[i] / zn : 0.0)));
        }
        for (uint32_t i = 0; i < n; i++) if (a[rr] == (uint32_t)ids8[ord[i]]) cnt[i]++;
      }
      double zmax = 0;
      int inside = 0;
      for (uint32_t i = 0; i < n; i++) {
        const double pe = w[i] / zn, e = NR * pe;
        zmax = std::max(zmax, std::fabs(cnt[i] - e) / std::sqrt(e * (1 - pe)));
        inside += cnt[i];
      }
      const bool ok4 = idbad == 0 && qerr < 1e-5 && zmax < 4.0 && inside == (int)NR;
      printf("(4) mtp_draft_sample (%u draws, %u kept ids): proposal ids %d wrong, prob max err %.2e, draw "
             "frequencies vs q worst |z| %.2f, %d draws outside: %s\n", NR, n, idbad, qerr, zmax, (int)NR - inside,
             ok4 ? "OK" : "FAIL");
      fails += !ok4;
    }
  } catch (const std::exception &ex) {
    fprintf(stderr, "error: %s\n", ex.what());
    return 1;
  }
  printf("mtp-loop-check: %s\n", fails ? "FAILED" : "all passed");
  return fails ? 1 : 0;
}
