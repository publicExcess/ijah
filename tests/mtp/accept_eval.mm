// lab-mtp: MEASURED greedy MTP acceptance on the engine (target = lab-forward's whole model, drafts = the MTP head),
// and the tok/s it implies with the measured verify cost table (docs/mtp.md section 6).
//   build/mtp-accept-eval PACK GOLDEN_DATA N_NEW DEPTH [draft_bits]     (full model: ~81 GB wired, machine lock)
//   build/mtp-accept-eval PACK GOLDEN_DATA selftest DEPTH [draft_bits]  (head only, < 3 GB: mechanics check)
// The target decodes greedily from the golden prompt (tests/golden/data/L03_prefill + L03_verify4 token ids),
// keeping layer 47's pre-mixer state of every position. The head then runs serially over (hidden_j, token_{j+1}) at
// head position j (= mlx-vlm's prefill / accept path, one row at a time), and from every j drafts a greedy chain of
// DEPTH tokens; a chain's accepted length = its leading matches with the target's own tokens (greedy MTP output is
// serial's byte for byte, so this teacher-forced count is the round's acceptance). selftest: hidden = the golden
// layer-3 stand-in (acceptance meaningless) and the serial pass's first drafts are checked against MTP_prefill.
#include <functional>

#include "core/bench.h"
#include "models/flash_next/model.h"
#include "models/flash_next/mtp.h"
#include "tests/mtp/common.h"

using namespace ne;
using namespace ne::fn;
using namespace mtpt;

static std::vector<int> json_ids(const std::string &path, const char *key) {
  std::ifstream f(path);
  std::string txt((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  Json j = Json::parse(txt);
  std::vector<int> o;
  for (const Json &v : j[key].arr()) o.push_back((int)v.i64());
  return o;
}

int main(int argc, char **argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    if (!getenv("NE_FN_ROUND")) g_ref_round = true;  // vs the MLX goldens (the library default is plain f32)
    NE_CHECK(argc >= 5, "usage: mtp-accept-eval PACK GOLDEN_DATA (N_NEW|selftest) DEPTH [draft_bits]");
    const std::string GD = argv[2];
    const bool self = std::string(argv[3]) == "selftest";
    const int n_new = self ? 0 : atoi(argv[3]), D = atoi(argv[4]), bits = argc > 5 ? atoi(argv[5]) : 0;
    NE_CHECK(D >= 1 && D <= 15, "depth 1..15");
    auto pack = Pack::open(argv[1]);
    std::vector<int> toks = json_ids(GD + "/L03_prefill/manifest.json", "token_ids");
    for (int t : json_ids(GD + "/L03_verify4/manifest.json", "token_ids")) toks.push_back(t);
    const int P = (int)toks.size();
    std::vector<std::vector<float>> H;  // pre-mixer [10240] per position
    double target_ms = 0;
    int target_steps = 0;
    if (self) {
      auto a = npy(GD + "/L03_prefill/hc_out.npy").f, b = npy(GD + "/L03_verify4/hc_out.npy").f;
      a.insert(a.end(), b.begin(), b.end());
      for (int j = 0; j < P; j++) H.emplace_back(a.begin() + (size_t)j * HCD, a.begin() + (size_t)(j + 1) * HCD);
    } else {
      auto m = load_model(*pack, 4096);
      int next = -1;
      for (int i = 0; i < P; i++) {
        StepOut o = step(*m, toks[i], i == P - 1);
        H.push_back(rd(m->h[m->L.size() % 2]));
        next = o.argmax;
      }
      for (int k = 0; k < n_new; k++) {
        toks.push_back(next);
        StepOut o = step(*m, next, true);
        H.push_back(rd(m->h[m->L.size() % 2]));
        target_ms += m->last_gpu_ms;
        target_steps++;
        next = o.argmax;
      }
      toks.push_back(next);
      printf("target: %d prompt + %d greedy tokens, %.2f ms GPU per decode step\n", P, n_new, target_ms / target_steps);
    }
    mtp::MtpW w = mtp::load_mtp(*pack);
    if (bits) mtp::set_draft_head(w, bits, 8);
    mtp::MtpHead h = mtp::alloc_head(w, 1, 4096);
    const int N = (int)H.size();
    std::vector<std::vector<int>> drafts(N);
    double head_ms = 0;
    for (int j = 0; j + 1 < (int)toks.size() && j < N; j++) {
      const uint32_t t1 = (uint32_t)toks[j + 1];
      upload(h.tok, &t1, 4);
      upload(h.hid, H[j].data(), (size_t)HCD * 4);
      if (h.st.qc.tmax > 0) attn::truncate(h.st.qc, j);  // the chain of row j-1 appended past j: drop it
      else ((uint32_t *)h.st.pos.host())[0] = (uint32_t)j;
      Recorder r;
      r.begin();
      mtp::record_step(r, w, h);  // the serial (committed) head row j -> d_1
      for (int k = 1; k < D; k++) {
        const Tensor tk = h.amax.view(4 * (size_t)(k - 1), DType::U32, {1});
        const Tensor &hi = (k & 1) ? h.h_out : h.h_out2;
        const Tensor &ho = (k & 1) ? h.h_out2 : h.h_out;
        mtp::record_step(r, w, h, &tk, &hi, &ho, (uint32_t)k);
      }
      r.end();
      r.run();
      head_ms += r.gpu_ms() / D;
      std::vector<uint32_t> a(16);
      download(h.amax, a.data(), 64);
      drafts[j].assign(a.begin(), a.begin() + D);
    }
    printf("head: %.2f ms GPU per draft step (1 row%s)\n", head_ms / N, bits ? ", draft head" : "");
    if (self) {
      auto g = npy(GD + "/MTP_prefill/mtp.argmax.npy").i;
      int eq = 0;
      for (int j = 0; j < (int)g.size(); j++) eq += drafts[j][0] == g[j];
      printf("selftest: serial 1-row head drafts d_1 == the 48-row reference prefill's argmax on %d/%zu positions\n", eq,
             g.size());
    }
    // acceptance over the generated region (selftest: over the prompt, meaningless but exercised)
    const int j0 = self ? 0 : P - 1, j1 = (int)toks.size() - 1 - D;
    std::vector<int> hist(D + 1, 0);
    int rounds = 0;
    for (int j = j0; j <= j1 && j < N; j++) {
      int L = 0;
      while (L < D && drafts[j][L] == toks[j + 2 + L]) L++;
      hist[L]++;
      rounds++;
    }
    NE_CHECK(rounds > 0, "no rounds to score (N_NEW too small for DEPTH)");
    printf("acceptance over %d start positions (greedy, teacher-forced):\n", rounds);
    double reach = 1.0;
    int ge = rounds;
    for (int k = 1; k <= D; k++) {
      const int ge_k = ge - hist[k - 1];
      printf("  depth %d: P(accepted >= %d) = %.3f, conditional a_%d = %.3f\n", k, k, (double)ge_k / rounds, k,
             ge ? (double)ge_k / ge : 0.0);
      ge = ge_k;
      (void)reach;
    }
    // projected tok/s: verify cost ratio per rows (docs/mtp.md section 6), target / head step times measured here
    const double ratio[9] = {0, 1.00, 1.24, 1.50, 1.76, 2.04, 2.31, 2.48, 2.65};
    const double F1 = self ? 14.01 : target_ms / target_steps, td = head_ms / N;
    printf("projection (F(1) %.2f ms%s, draft step %.2f ms, verify ratios of docs/mtp.md):\n", F1,
           self ? " assumed" : " measured", td);
    for (int d = 1; d <= std::min(D, 7); d++) {
      double e = 1;
      for (int L = 0; L <= D; L++) e += (double)hist[L] / rounds * std::min(L, d);
      const double t = F1 * ratio[d + 1] + d * td + 0.15;
      printf("  depth %d: %.2f tokens / round, %.1f ms / round -> %.1f tok/s (serial %.1f, %.2fx)\n", d, e, t,
             1e3 * e / t, 1e3 / F1, e / t * F1);
    }
  } catch (const std::exception &e) {
    fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
