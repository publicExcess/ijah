// lab-mtp: CPU checks of the verify / accept rules (models/flash_next/mtp.h). No GPU, no lock.
//   build/mtp-accept-test
// 1. greedy prefix rule; 2. exact speculative sampling keeps the target distribution (first emitted token ~ p_0,
//    acceptance rate = sum_x min(p, q) = 1 - TV(p, q)), with a sampled draft and with an argmax (point-mass) draft;
// 3. top-k / top-p filter; 4. commit plan (rollback points); 5. adaptive depth picks the EV-best depth.
#include <cmath>
#include <cstdio>

#include "models/flash_next/mtp.h"

using namespace ne::mtp;
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

int main() {
  {  // greedy
    const int d[3] = {5, 7, 9}, am[4] = {5, 7, 1, 3};
    auto r = accept_greedy(d, 3, am);
    CHECK(r.accepted == 2 && r.next_token == 1 && r.emitted.size() == 3, "greedy partial: %d %d", r.accepted, r.next_token);
    const int am2[4] = {5, 7, 9, 4};
    r = accept_greedy(d, 3, am2);
    CHECK(r.accepted == 3 && r.next_token == 4 && r.emitted.size() == 4, "greedy all");
    const int am3[4] = {2, 7, 9, 4};
    r = accept_greedy(d, 3, am3);
    CHECK(r.accepted == 0 && r.next_token == 2, "greedy none");
  }
  const int V = 6;
  const float tl0[V] = {1.0f, 0.2f, -0.5f, 2.0f, 0.0f, -1.0f};   // target row 0 logits
  const float ql0[V] = {0.1f, 1.5f, 0.3f, 1.0f, -0.2f, 0.4f};    // draft row 0 logits
  SampleParams sp;
  sp.temperature = 1.0f;
  const Dist p0 = make_dist(tl0, V, sp), q0 = make_dist(ql0, V, sp);
  double tv_min = 0;
  for (int x = 0; x < V; x++) tv_min += std::min(p0.prob(x), q0.prob(x));
  for (int mode = 0; mode < 2; mode++) {  // 0: sampled draft (q), 1: argmax draft (point mass at argmax q)
    Rng rng(1234 + mode), drng(99);
    const int N = 400000, D = 2;
    std::vector<int> cnt(V, 0);
    long acc0 = 0;
    std::vector<float> tl(3 * V), ql(2 * V);
    for (int i = 0; i < 3; i++) for (int x = 0; x < V; x++) tl[i * V + x] = tl0[(x + i) % V];
    for (int i = 0; i < 2; i++) for (int x = 0; x < V; x++) ql[i * V + x] = ql0[(x + 2 * i) % V];
    for (int n = 0; n < N; n++) {
      int drafts[D];
      for (int i = 0; i < D; i++) {
        const Dist q = make_dist(&ql[i * V], V, sp);
        drafts[i] = mode == 0 ? q.sample(drng.uniform()) : q.ids[0];
      }
      auto r = accept_sampled(drafts, D, tl.data(), mode == 0 ? ql.data() : nullptr, V, sp, rng);
      cnt[r.emitted[0]]++;
      acc0 += r.accepted > 0;
    }
    double worst = 0;
    for (int x = 0; x < V; x++) {
      const double f = (double)cnt[x] / N, pe = p0.prob(x);
      const double z = std::fabs(f - pe) / std::sqrt(pe * (1 - pe) / N + 1e-12);
      worst = std::max(worst, z);
    }
    const double want = mode == 0 ? tv_min : p0.prob(q0.ids[0]);
    const double got = (double)acc0 / N;
    printf("%s draft: first emitted token vs p_0 worst |z| %.2f; P(accept d_1) %.4f (expected %.4f)\n",
           mode == 0 ? "sampled" : "argmax ", worst, got, want);
    CHECK(worst < 4.5, "emitted distribution != target (|z| %.2f)", worst);
    CHECK(std::fabs(got - want) < 0.005, "acceptance rate %.4f vs %.4f", got, want);
  }
  {  // sparse proposals (the draft readout's shortlist): q over K ids only, drafts drawn from it
    float slr[8];
    const int ids[4] = {3, 1, 0, 5};
    const float ex[4] = {1.9f, 1.4f, 0.8f, -0.2f};
    for (int k = 0; k < 4; k++) { memcpy(&slr[2 * k], &ids[k], 4); slr[2 * k + 1] = ex[k]; }
    Dist q = shortlist_dist(slr, 4, sp);
    CHECK(q.ids.size() == 4 && q.ids[0] == 3 && std::fabs(q.p[0] + q.p[1] + q.p[2] + q.p[3] - 1.0f) < 1e-5,
          "shortlist dist");
    Rng rng(77), drng(5);
    const int N = 400000;
    std::vector<int> cnt(V, 0);
    long acc = 0;
    double want = 0;
    for (int x = 0; x < V; x++) want += std::min(p0.prob(x), q.prob(x));
    for (int n = 0; n < N; n++) {
      const int d = q.sample(drng.uniform());
      auto r = accept_sampled_q(&d, 1, tl0, &q, V, sp, rng);
      cnt[r.emitted[0]]++;
      acc += r.accepted;
    }
    double worst = 0;
    for (int x = 0; x < V; x++) {
      const double f = (double)cnt[x] / N, pe = p0.prob(x);
      worst = std::max(worst, std::fabs(f - pe) / std::sqrt(pe * (1 - pe) / N + 1e-12));
    }
    printf("shortlist q (4 ids): first emitted token vs p_0 worst |z| %.2f; P(accept) %.4f (expected %.4f)\n", worst,
           (double)acc / N, want);
    CHECK(worst < 4.5 && std::fabs((double)acc / N - want) < 0.005, "sparse-q exactness");
  }
  {  // filters
    SampleParams f;
    f.temperature = 1.0f;
    f.top_k = 2;
    Dist d = make_dist(tl0, V, f);
    CHECK(d.ids.size() == 2 && d.ids[0] == 3 && d.ids[1] == 0, "top-k ids");
    CHECK(std::fabs(d.p[0] + d.p[1] - 1.0f) < 1e-6, "top-k renorm");
    f.top_k = 0;
    f.top_p = 0.5f;
    d = make_dist(tl0, V, f);
    CHECK(d.ids.size() == 1 && d.ids[0] == 3, "top-p 0.5 keeps the 2.0 logit (p = %.3f) only", p0.prob(3));
    SampleParams g;
    d = make_dist(tl0, V, g);
    CHECK(d.ids.size() == 1 && d.ids[0] == 3 && d.p[0] == 1.0f, "greedy dist");
  }
  {  // commit plan: T0 = 100, D = 3 drafts, 1 accepted; the head appended 2 rows while drafting (d_2, d_3 steps)
    auto c = plan_commit(100, 3, 1, 2);
    CHECK(c.keep_rows == 2 && c.kv_len == 102 && c.gdn_row == 1, "commit: keep %d kv %d gdn %d", c.keep_rows, c.kv_len, c.gdn_row);
    CHECK(c.head_keep == 1 && c.head_trim == 1 && c.head_rows == 1, "head: keep %d trim %d rows %d", c.head_keep, c.head_trim, c.head_rows);
    c = plan_commit(100, 3, 3, 2);
    CHECK(c.keep_rows == 4 && c.head_keep == 2 && c.head_trim == 0 && c.head_rows == 2, "commit all");
    c = plan_commit(100, 3, 0, 2);
    CHECK(c.keep_rows == 1 && c.head_keep == 0 && c.head_trim == 2 && c.head_rows == 1, "commit none");
  }
  {  // depth policy: verify cost ratio 1.0 + 0.3 per extra row, draft 0.08 per step (in decode-step units)
    std::vector<double> cost = {1.0};
    for (int d = 1; d <= 8; d++) cost.push_back(1.0 + 0.3 * d + 0.08 * d);
    DepthPolicy hi(cost, 0.9), lo(cost, 0.3);
    hi.max_depth = lo.max_depth = 8;
    printf("depth policy: a=0.9 -> d=%d (E[tok] %.2f), a=0.3 -> d=%d (E[tok] %.2f)\n", hi.next_depth(),
           hi.expected_tokens(hi.next_depth()), lo.next_depth(), lo.expected_tokens(lo.next_depth()));
    CHECK(hi.next_depth() > lo.next_depth(), "high acceptance should draft deeper");
    for (int i = 0; i < 200; i++) lo.observe(4, 4);
    CHECK(lo.acc[0] > 0.99 && lo.next_depth() == hi.next_depth() + 0 || lo.next_depth() >= hi.next_depth(),
          "observed full acceptance should raise depth (%d)", lo.next_depth());
  }
  {  // lab-draft: the copy rule's lookup (copy_find) vs a naive search; the anchor t1 is the window's last token
    std::vector<int> ctx;
    for (int i = 1; i <= 20; i++) ctx.push_back(i);
    ctx.insert(ctx.end(), {100, 101, 102});
    for (int i = 1; i <= 15; i++) ctx.push_back(i);
    CopyIndex idx;
    int out[8];
    int k = copy_find(ctx, idx, 16, 16, 4, out);  // S ends 1..16: the earlier 1..16 was followed by 17 18 19 20
    CHECK(k == 4 && out[0] == 17 && out[1] == 18 && out[2] == 19 && out[3] == 20, "copy: k %d out %d %d", k, out[0], out[1]);
    k = copy_find(ctx, idx, 16, 99, 4, out);
    CHECK(k == 0, "copy: no match expected (k %d)", k);
    ctx.push_back(16);
    k = copy_find(ctx, idx, 16, 17, 8, out);  // grown by one (incremental index): continuation 18 .. 20 100 101 102 1
    CHECK(k == 8 && out[0] == 18 && out[3] == 100 && out[6] == 1 && out[7] == 2, "copy grown: k %d out %d", k, out[0]);
    std::vector<int> run(20, 5);
    CopyIndex ri;
    k = copy_find(run, ri, 16, 5, 4, out);  // a run: the most recent window ends at |ctx|, its continuation is t1
    CHECK(k == 1 && out[0] == 5, "copy run: k %d", k);
    Rng rr(7);
    int bad = 0, hits = 0;
    for (int trial = 0; trial < 200; trial++) {
      std::vector<int> c;
      CopyIndex ci;
      const int N = 2 + trial % 4;
      for (int step = 0; step < 60; step++) {
        const int t1 = (int)(rr.uniform() * 3);
        std::vector<int> S = c;
        S.push_back(t1);
        int ref[8], nr = 0;
        const int L = (int)S.size();
        for (int e = L - 1; e >= N && nr == 0; e--) {  // most recent earlier window S[e - N, e) == S[L - N, L)
          bool eq = true;
          for (int i = 0; i < N; i++) eq = eq && S[e - N + i] == S[L - N + i];
          if (eq) for (; nr < 5 && e + nr < L; nr++) ref[nr] = S[e + nr];
        }
        const int got = copy_find(c, ci, N, t1, 5, out);
        hits += got > 0;
        bool same = got == nr;
        for (int i = 0; same && i < nr; i++) same = out[i] == ref[i];
        bad += !same;
        c.push_back(t1);
      }
    }
    printf("copy rule: naive cross-check %d mismatches of 12000 lookups (%d matches)\n", bad, hits);
    CHECK(bad == 0 && hits > 1000, "copy_find vs naive: %d mismatches", bad);
  }
  printf(fails ? "accept_test: %d FAILED\n" : "accept_test: all passed\n", fails);
  return fails ? 1 : 0;
}
