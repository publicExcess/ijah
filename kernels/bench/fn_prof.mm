// Per-kernel GPU-time profile of one Flash-Next decode step (lab-speed). Owner: lab-speed (kernels/).
//
//   build/fn-prof PACK [reps=25] [T=64] [rounds=3] [layers=0,1,2,3]    (~7 GB wired for 4 layers: take the lock)
//
// Partial model (RAM): the listed layers are loaded (GDN, PLE layer 1, attention layer 3) and replayed as the
// 48-layer pattern of the model (virtual layer v runs loaded layer list[v % n]; the type pattern must match: attention
// at v % 4 == 3; PLE only on virtual layer 1). Weights of the n loaded layers are re-read every n layers (~0.5 GB of
// trunk + experts in between: no SLC reuse), so the step costs what the 48 distinct layers cost.
//
// Method: leave-one-out differential timing in context. The body (48 layers, the PLE block, the same record_stage
// calls the model makes) is recorded once in full and once per kernel kind with every instance of that kind left out
// (its barrier too); delta / count = the in-context cost of one instance including its boundary. Rounds interleave
// the variants (drift), each variant = median of `reps` replays (one command buffer each). The head (mixer +
// lm_head + sampler) = the model's own B recording with the head on minus off. "boundary" = one extra empty
// dispatch + barrier after every kernel. Bytes = weights + states the kernel reads/writes (activations excluded);
// GB/s = bytes / in-context us; % of 594 GB/s (lead 1's read peak).
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "kernels/fn_rec.h"
#include "models/flash_next/model.h"

using namespace ne;
using namespace ne::fn;

static size_t qb(const Q8 &q) { return q.w.nbytes() + q.s.nbytes() + q.b.nbytes(); }
static size_t eb(const Exl3 &e) {  // one expert: trellis + suh + svh rows
  return e.t.nbytes() / NE + e.suh.nbytes() / NE + e.svh.nbytes() / NE;
}

// kinds: (stage, linear) + PLE
struct Kind {
  std::string name;
  int stage;    // -1 = PLE
  int linear;   // 1 GDN layers, 0 attention layers, -1 both
};

static size_t kind_bytes(const Kind &k, const LayerW &w) {
  const HcW &hc = (k.stage <= 1) ? w.hca : w.hcm;
  switch (k.stage) {
    case -1: return qb(w.ple.key) + qb(w.ple.value) + w.ple.conv.nbytes() + 2 * (size_t)PLE_ST * HCD * 4;
    case 0: case 5: return qb(hc.down) + hc.inject.nbytes() + hc.norm.nbytes();
    case 1: case 6: return qb(hc.up) + hc.norm.nbytes();
    case 2:
      if (w.linear) return qb(w.gdn.qkv) + qb(w.gdn.z) + qb(w.gdn.a) + qb(w.gdn.b) + w.gdn.conv.nbytes() + 2 * 3 * CONVD * 4;
      return qb(w.attn.q) + qb(w.attn.k) + qb(w.attn.v) + qb(w.attn.iqk);
    case 3: return w.linear ? 2 * (size_t)NVH * DV * DK * 4 : 0;
    case 4: return qb(w.linear ? w.gdn.out : w.attn.o);
    case 7: return w.moe.router.nbytes() + w.moe.sgate.nbytes() + qb(w.moe.sg) + qb(w.moe.su);
    case 8: return TOPK * (eb(w.moe.gate) + eb(w.moe.up)) + qb(w.moe.sd);
    case 9: return TOPK * eb(w.moe.down);
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s PACK [reps] [T] [rounds]\n", argv[0]);
    return 2;
  }
  const int reps = argc > 2 ? atoi(argv[2]) : 25;
  const int T = argc > 3 ? atoi(argv[3]) : 64;
  const int rounds = argc > 4 ? atoi(argv[4]) : 3;
  std::vector<int> lids;
  {
    std::string ls = argc > 5 ? argv[5] : "0,1,2,3";
    size_t a = 0;
    while (a < ls.size()) {
      size_t b = ls.find(',', a);
      if (b == std::string::npos) b = ls.size();
      lids.push_back(atoi(ls.substr(a, b - a).c_str()));
      a = b + 1;
    }
  }
  // lab-verify: NE_PROF_ROWS=R profiles the R-row step (the MTP verify shape; random distinct rows -> ~disjoint routing)
  const int PR = getenv("NE_PROF_ROWS") ? std::max(1, atoi(getenv("NE_PROF_ROWS"))) : 1;
  // lab-moegroup: NE_PROF_SPREAD=p: p % of rows 1.. R-1's routed slots moved to other experts (fn_moe_router), so the
  // synthetic rows (which route much alike) share experts like real verify rows (docs/mtp.md 12.2); 0 = as routed
  g_moe_spread = getenv("NE_PROF_SPREAD") ? std::clamp(atoi(getenv("NE_PROF_SPREAD")), 0, 100) : 0;
  try {
    auto pack = Pack::open(argv[1]);
    struct {
      std::vector<LayerW> L;
      std::vector<GdnState> st;
      Scratch s;
      Tensor h[2], pos, x, logits, cand, lse, sp, res, tok, dummy_inj;
      HcW mixer;
      Q8 head;
    } m;
    int first_attn = -1;
    m.pos = alloc(DType::U32, {4});
    m.L.reserve(lids.size());
    m.st.reserve(lids.size());
    for (int id : lids) {
      m.L.push_back(load_layer(*pack, id));
      m.st.push_back(alloc_state(m.L.back(), 4096, std::max(4, PR), first_attn >= 0 ? &m.st[first_attn] : nullptr));
      if (!m.L.back().linear) {
        if (first_attn < 0) first_attn = (int)m.L.size() - 1;
        m.st.back().pos = m.pos;
        if (m.st.back().qc.tmax > 0) m.st.back().qc.pos = m.pos;
      }
    }
    const int nl = 48, nload = (int)m.L.size();
    for (int v = 0; v < nl; v++)
      NE_CHECK(m.L[v % nload].linear == (v % 4 != 3), "layer list %d: type pattern differs from the model's", v % nload);
    const std::string mpfx = "language_model.model.hyper_connection_mixer.";
    m.mixer.norm = load_tensor(*pack, mpfx + "hc_norm.weight");
    m.mixer.down = load_q8(*pack, mpfx + "input_mix_weight_down");
    m.mixer.up = load_q8(*pack, mpfx + "input_mix_weight_up");
    m.head = load_q8(*pack, "language_model.lm_head");
    m.s = alloc_scratch(PR);
    Scratch &s = m.s;
    for (auto &h : m.h) h = alloc(DType::F32, {PR, HC, HID});
    m.dummy_inj = alloc(DType::BF16, {HC, HCD});
    m.x = alloc(DType::F32, {1, HID});
    m.logits = alloc(DType::F32, {1, VOCAB});
    m.cand = alloc(DType::F32, {(VOCAB + 1023) / 1024, kSampleMaxK, 2});
    m.lse = alloc(DType::F32, {(VOCAB + 1023) / 1024, 2});
    m.sp = alloc(DType::U32, {8});
    m.res = alloc(DType::U32, {4});
    m.tok = alloc(DType::U32, {4});
    { SampleParams P; memcpy(m.sp.host(), &P, sizeof P); }
    {  // a plausible hidden state (the model's scale): small random values
      float *h0 = (float *)m.h[0].host();
      uint32_t x = 12345;
      for (int i = 0; i < PR * HCD; i++) { x = x * 1664525u + 1013904223u; h0[i] = ((x >> 8) / 16777216.0f - 0.5f) * 0.1f; }
      // lab-moegroup: NE_PROF_CORR=a: row r > 0 = (1 - a) row 0 + a (its random row) -> consecutive-token-like expert
      // sharing (a = 1: the old independent rows); the step's distinct experts are printed below
      const char *ce = getenv("NE_PROF_CORR");
      const float a = ce ? (float)atof(ce) : 1.0f;
      for (int r = 1; r < PR; r++)
        for (int i = 0; i < HCD; i++) h0[r * HCD + i] = (1.0f - a) * h0[i] + a * h0[r * HCD + i];
    }
    ((uint32_t *)m.pos.host())[0] = (uint32_t)T;
    // the loaded tensors resident on the MTL4 queue (each recorder's own set then only adds references)
    Device &dev = Device::get();
    id<MTLResidencySet> qrs = nil;
    if (dev.has_metal4 && dev.queue4) {
      MTLResidencySetDescriptor *rsd = [MTLResidencySetDescriptor new];
      NSError *err = nil;
      qrs = [dev.dev newResidencySetWithDescriptor:rsd error:&err];
      for (const std::string &n : pack->names()) {
        bool want = n.rfind("language_model.lm_head", 0) == 0 || n.rfind(mpfx, 0) == 0;
        for (int id : lids) want = want || n.rfind("language_model.model.layers." + std::to_string(id) + ".", 0) == 0;
        const TensorInfo *ti = want ? pack->info(n) : nullptr;
        if (ti && ti->t.buf) [qrs addAllocation:ti->t.buf];
      }
      [qrs commit];
      [qrs requestResidency];
      [dev.queue4 addResidencySet:qrs];
    }
    id<MTLComputePipelineState> empty = pso("ne_empty", 32);

    std::vector<Kind> kinds;
    const char *gn[10] = {"hc_pre (attn)", "hc_mix (attn)", "gdn_inproj", "gdn_step", "gdn_outproj+HC",
                          "hc_pre (mlp)", "hc_mix (mlp)", "moe_router", "moe_gateup", "moe_down+HC"};
    const char *an[10] = {"", "", "attn_proj", "QSA step", "o_proj+HC", "", "", "", "", ""};
    kinds.push_back({"PLE (layer 1)", -1, -1});
    for (int i = 0; i < 10; i++) {
      const bool split = i >= 2 && i <= 4;
      if (split) {
        kinds.push_back({gn[i], i, 1});
        kinds.push_back({an[i], i, 0});
      } else {
        kinds.push_back({gn[i], i, -1});
      }
    }
    // variant -1 = full, -2 = full + boundary, k = without kind k
    auto skip = [&](int v, int stage, bool linear) {
      if (v < 0) return false;
      const Kind &k = kinds[v];
      return k.stage == stage && (k.linear < 0 || k.linear == (linear ? 1 : 0));
    };
    // NE_PROF_V2 = bit mask of lab-speed's v2 kernels used by the "new" recordings (1 = HC read v2, 3 = with fn_hc_pre3); the leave-one-out
    // variants use the new recordings, and the old full body is timed alongside (A/B in one process)
    const char *v2e = getenv("NE_PROF_V2");
    const int v2 = v2e ? atoi(v2e) : 0;
    int cur_mask = v2;
    auto stage_rec = [&](Recorder &r, int i, const LayerW &w, GdnState &st, const Tensor &hin, const Tensor &ho) {
      const int R = s.rows;
      if ((cur_mask & 1) && (i == 0 || i == 5)) {
        const HcW &hw = i == 0 ? w.hca : w.hcm;
        if (cur_mask & 2) krec::hc_pre3(r, hw, R, i == 0 ? hin : s.h2, hw.inject, i == 0 ? s.inj_a : s.inj_m, s.hc_part);
        else krec::hc_pre(r, hw, R, i == 0 ? hin : s.h2, hw.inject, i == 0 ? s.inj_a : s.inj_m, s.hc_part);
        return;
      }
      if ((cur_mask & 1) && (i == 1 || i == 6)) {
        const HcW &hw = i == 1 ? w.hca : w.hcm;
        if (i == 1) krec::hc_mix(r, hw, R, hin, s.mix_a, s.inj_a, s.xa, s.hc_part);
        else krec::hc_mix(r, hw, R, s.h2, s.mix_m, s.inj_m, s.xm, s.hc_part);
        return;
      }
      record_stage(r, (Stage)i, w, st, s, hin, ho);
    };
    auto record = [&](int v) {
      auto r = std::make_unique<Recorder>();
      r->begin();
      size_t nb = 0;
      for (int l = 0; l < nl; l++) {
        const LayerW &w = m.L[l % nload];
        GdnState &st = m.st[l % nload];
        const Tensor &hi = m.h[l % 2], &ho = m.h[(l + 1) % 2];
        const bool ple = w.ple.on && l == 1;
        if (ple && !skip(v, -1, w.linear)) record_ple(*r, w, st, s, hi, s.hp);
        const Tensor &hin = ple ? s.hp : hi;
        for (int i = 0; i < (int)Stage::N; i++) {
          if (skip(v, i, w.linear)) continue;
          stage_rec(*r, i, w, st, hin, ho);
          if (v == -2) {
            r->dispatch(empty, {s.ctr1}, MTLSizeMake(1, 1, 1), MTLSizeMake(32, 1, 1));
            r->barrier();
            nb++;
          }
        }
      }
      r->end();
      return std::make_pair(std::move(r), nb);
    };
    // NE_PROF_KINDS=substr1,substr2: only the kinds whose name contains one of them (quick A/B runs)
    const char *kf = getenv("NE_PROF_KINDS");
    auto wanted = [&](const std::string &n) {
      if (!kf || !*kf) return true;
      std::string f = kf;
      size_t a = 0;
      while (a <= f.size()) {
        size_t b = f.find(',', a);
        if (b == std::string::npos) b = f.size();
        if (b > a && n.find(f.substr(a, b - a)) != std::string::npos) return true;
        a = b + 1;
      }
      return false;
    };
    std::vector<int> vars = {-1};
    if (!kf || !*kf) vars.push_back(-2);
    for (int k = 0; k < (int)kinds.size(); k++) if (wanted(kinds[k].name)) vars.push_back(k);
    // NE_PROF_AB=1: A/B of lab-fuse's g_fuse (NE_FUSE for the new body, NE_PROF_AB_BASE (default 0) for the old)
    // lab-t6fuse: NE_PROF_ABENV="K=V,K=V" = A/B of env settings read at record time (t6::plan's T6_S_<in>x<out>) and
    // of FUSE2=0/1 (g_t6_fuse2), X=bits (g_t6_x): new body with them, old body (-3) without (defaults)
    const char *abenv = getenv("NE_PROF_ABENV");
    std::vector<std::pair<std::string, std::string>> abkv;
    for (std::string f = abenv ? abenv : ""; !f.empty();) {
      const size_t c = f.find(','), e = f.find('=');
      const std::string kv = f.substr(0, c);
      if (e != std::string::npos && e < kv.size()) abkv.push_back({kv.substr(0, e), kv.substr(e + 1)});
      f = c == std::string::npos ? "" : f.substr(c + 1);
    }
    const bool fuse2_def = g_t6_fuse2;
    const int x_def = g_t6_x, group_def = g_moe_group, dd_def = g_moe_dedupe;
    const bool ind_def = g_moe_group_ind;
    auto set_abenv = [&](bool on) {
      g_t6_fuse2 = fuse2_def;
      g_t6_x = x_def;
      g_moe_group = group_def;  // lab-moegroup: GROUP=N (g_moe_group), IND=0/1 (g_moe_group_ind), DEDUPE=N
      g_moe_group_ind = ind_def;
      g_moe_dedupe = dd_def;
      for (auto &kv : abkv) {
        if (kv.first == "GROUP") { if (on) g_moe_group = atoi(kv.second.c_str()); }
        else if (kv.first == "IND") { if (on) g_moe_group_ind = atoi(kv.second.c_str()) != 0; }
        else if (kv.first == "DEDUPE") { if (on) g_moe_dedupe = atoi(kv.second.c_str()); }
        else if (kv.first == "FUSE2") { if (on) g_t6_fuse2 = atoi(kv.second.c_str()) != 0; }
        else if (kv.first == "X") { if (on) g_t6_x = (int)strtol(kv.second.c_str(), nullptr, 0); }
        else if (on) setenv(kv.first.c_str(), kv.second.c_str(), 1);
        else unsetenv(kv.first.c_str());
      }
    };
    const bool ab = (getenv("NE_PROF_AB") && atoi(getenv("NE_PROF_AB")) > 0) || !abkv.empty();
    const int fuse_new = g_fuse, fuse_old = getenv("NE_PROF_AB_BASE") ? (int)strtol(getenv("NE_PROF_AB_BASE"), nullptr, 0) : 0;
    const bool lazy_ab = getenv("NE_PROF_AB_LAZY") && atoi(getenv("NE_PROF_AB_LAZY")) > 0;
    const int lazy_new = g_gdn_lazy;
    if (v2 || ab) vars.push_back(-3);  // -3 = the old full body (mask 0)
    std::map<int, std::unique_ptr<Recorder>> recs;
    std::map<int, size_t> nbound;
    std::map<int, std::vector<double>> times;
    for (int v : vars) {
      cur_mask = v == -3 ? 0 : v2;
      if (ab && abkv.empty()) g_fuse = v == -3 ? fuse_old : fuse_new;
      if (ab && lazy_ab) g_gdn_lazy = v == -3 ? 0 : lazy_new;  // lab-gdnlazy: old body stores every step
      if (!abkv.empty()) set_abenv(v != -3);
      auto pr = record(v == -3 ? -1 : v);
      nbound[v] = pr.second;
      recs[v] = std::move(pr.first);
    }
    if (getenv("NE_PROF_MICRO") && atoi(getenv("NE_PROF_MICRO")) > 0) {  // chained single-kernel timings (differential 32 vs 8 copies, SLC-warm state)
      const LayerW &w = m.L[0];
      GdnState &st = m.st[0];
      auto chain = [&](const std::string &name, int tgs, int thr, int n, std::vector<Arg> args) {
        auto r = std::make_unique<Recorder>();
        r->begin();
        for (int i = 0; i < n; i++) {
          r->dispatch(pso(name, thr), args, MTLSizeMake(tgs, 1, 1), MTLSizeMake(thr, 1, 1));
          r->barrier();
        }
        r->end();
        std::vector<double> g;
        for (int i = 0; i < 12; i++) { r->run(); if (i >= 2) g.push_back(r->gpu_ms()); }
        std::sort(g.begin(), g.end());
        return g[g.size() / 2];
      };
      for (int tgs : {8, 24, 40, 48, 96}) {
        std::vector<Arg> a = {s.qkv, s.gb, s.z, w.gdn.norm, st.ssm, s.o};
        if (tgs > 48) continue;
        const double t8 = chain("fn_gdn_step_r1", tgs, 1024, 8, a), t32 = chain("fn_gdn_step_r1", tgs, 1024, 32, a);
        printf("micro fn_gdn_step_r1 %d TGs x 1024: %.2f us each\n", tgs, (t32 - t8) * 1e3 / 24);
      }
      // NE_PROF_SRC=file.metal (runtime-compiled; `#include "ne_common.h"` is inlined) + NE_PROF_SRC_K=name:tgs:thr,...:
      // gdn_step-shaped experiments (same 6 buffers)
      if (const char *src = getenv("NE_PROF_SRC")) {
        auto slurp = [](const std::string &f) {
          FILE *fp = fopen(f.c_str(), "rb");
          NE_CHECK(fp, "cannot open %s", f.c_str());
          std::string t;
          char buf[65536];
          size_t n;
          while ((n = fread(buf, 1, sizeof buf, fp)) > 0) t.append(buf, n);
          fclose(fp);
          return t;
        };
        std::string msl = slurp(src), common = slurp("kernels/ne_common.h");
        common.replace(common.find("#pragma once"), 12, "");
        const std::string inc = "#include \"ne_common.h\"";
        if (msl.find(inc) != std::string::npos) msl.replace(msl.find(inc), inc.size(), common);
        Pipelines::add_source(msl, "fn-prof experiment");
        std::string ks = getenv("NE_PROF_SRC_K") ? getenv("NE_PROF_SRC_K") : "";
        size_t a = 0;
        while (a < ks.size()) {
          size_t b = ks.find(',', a);
          if (b == std::string::npos) b = ks.size();
          std::string e = ks.substr(a, b - a);
          a = b + 1;
          const size_t c1 = e.find(':'), c2 = e.find(':', c1 + 1);
          std::string name = e.substr(0, c1);
          const std::string gs = "_g" + std::to_string(w.hca.down.group);
          if (name == "hc_pre_old") name = "fn_hc_pre" + gs + "_r1";  // aliases for the metallib kernels
          if (name == "hc_pre2") name = "fn_hc_pre2" + gs + "_r1";
          const int tgs = atoi(e.substr(c1 + 1, c2 - c1 - 1).c_str()), thr = atoi(e.substr(c2 + 1).c_str());
          std::vector<Arg> ar = {s.qkv, s.gb, s.z, w.gdn.norm, st.ssm, s.o};
          if (e.find("hc_pre") != std::string::npos)
            ar = {m.h[0], w.hca.norm, w.hca.down.w, w.hca.down.s, w.hca.down.b, w.hca.inject, s.inj_a, s.hc_part};
          if (e.find("hc_pre_old") != std::string::npos)
            ar = {m.h[0], w.hca.norm, w.hca.down.w, w.hca.down.s, w.hca.down.b, w.hca.inject, s.mix_a, s.inj_a, s.hc_part, s.hc_ctr};
          const double t8 = chain(name, tgs, thr, 8, ar), t32 = chain(name, tgs, thr, 32, ar);
          printf("micro %s %d TGs x %d: %.2f us each\n", e.substr(0, c1).c_str(), tgs, thr, (t32 - t8) * 1e3 / 24);
        }
      }
      const double e8 = chain("ne_empty", 1, 32, 8, {s.ctr1}), e32 = chain("ne_empty", 1, 32, 32, {s.ctr1});
      printf("micro ne_empty: %.2f us each\n", (e32 - e8) * 1e3 / 24);
    }
    {  // output hash of the full body (current mask) from zero states: compare across metallibs for bit identity
      for (auto &st : m.st) {
        if (st.conv.buf) memset(st.conv.host(), 0, st.conv.nbytes());
        if (st.ssm.buf) memset(st.ssm.host(), 0, st.ssm.nbytes());
        gdn_clear_pending(st);
        if (st.ple_conv.buf) memset(st.ple_conv.host(), 0, st.ple_conv.nbytes());
      }
      std::vector<float> hsave((float *)m.h[0].host(), (float *)m.h[0].host() + HCD);
      for (int which : {-1, -3}) {
        if (!recs.count(which)) continue;
        for (auto &st : m.st) {
          if (st.conv.buf) memset(st.conv.host(), 0, st.conv.nbytes());
          if (st.ssm.buf) memset(st.ssm.host(), 0, st.ssm.nbytes());
          gdn_clear_pending(st);
          if (st.ple_conv.buf) memset(st.ple_conv.host(), 0, st.ple_conv.nbytes());
        }
        memcpy(m.h[0].host(), hsave.data(), HCD * 4);
        recs[which]->run();
        uint64_t hh = 1469598103934665603ull;
        const uint8_t *b = (const uint8_t *)m.h[0].host();
        for (int i = 0; i < HCD * 4; i++) hh = (hh ^ b[i]) * 1099511628211ull;
        printf("output hash (48 layers, v2 mask %d, %s): %016llx\n", v2, which == -1 ? "new" : "old", (unsigned long long)hh);
      }
      memcpy(m.h[0].host(), hsave.data(), HCD * 4);
    }
    if (getenv("NE_PROF_DDCHECK")) {  // lab-verify: routed-expert dedupe vs per (row, slot): outputs after n layers
      const int dd0 = g_moe_dedupe, gr0 = g_moe_group;
      const bool spl = std::string(getenv("NE_PROF_DDCHECK")) == "split", sp0 = g_moe_split;  // per pair vs split
      const std::string ddm = getenv("NE_PROF_DDCHECK");
      // lab-moegroup: group = per pair vs grouped; self / gself = per pair / grouped twice (run-to-run determinism)
      const bool grp = ddm == "group" || ddm == "self" || ddm == "gself";
      std::vector<float> h0((float *)m.h[0].host(), (float *)m.h[0].host() + (size_t)PR * HCD);
      for (int n : {1, 4, 48}) {
        std::vector<float> o[2];
        for (int k = 0; k < 2; k++) {
          g_moe_group = 0;
          if (grp) { g_moe_dedupe = 0; g_moe_split = false; g_moe_group = (ddm == "group" ? k == 1 : ddm == "gself") ? 2 : 0; }
          else if (spl) { g_moe_dedupe = 0; g_moe_split = k == 1; } else g_moe_dedupe = k == 1 ? 8 : 0;
          for (auto &st : m.st) {
            if (st.conv.buf) memset(st.conv.host(), 0, st.conv.nbytes());
            if (st.ssm.buf) memset(st.ssm.host(), 0, st.ssm.nbytes());
            gdn_clear_pending(st);
            if (st.ple_conv.buf) memset(st.ple_conv.host(), 0, st.ple_conv.nbytes());
          }
          memcpy(m.h[0].host(), h0.data(), h0.size() * 4);
          Recorder r1;
          r1.begin();
          for (int l = 0; l < n; l++) {
            const LayerW &w = m.L[l % nload];
            const bool ple = w.ple.on && l == 1;
            if (ple) record_ple(r1, w, m.st[l % nload], s, m.h[l % 2], s.hp);
            for (int i = 0; i < (int)Stage::N; i++)
              record_stage(r1, (Stage)i, w, m.st[l % nload], s, ple ? s.hp : m.h[l % 2], m.h[(l + 1) % 2]);
          }
          r1.end();
          r1.run();
          const float *hp = (const float *)m.h[n % 2].host();
          o[k].assign(hp, hp + (size_t)PR * HCD);
        }
        size_t same = 0;
        double md = 0, mx = 0;
        for (size_t i = 0; i < o[0].size(); i++) {
          same += memcmp(&o[0][i], &o[1][i], 4) == 0;
          md = std::max(md, (double)fabsf(o[0][i] - o[1][i]));
          mx = std::max(mx, (double)fabsf(o[0][i]));
        }
        printf("ddcheck rows %d, %d layers: %zu of %zu bit-identical, max |diff| %.3g (max |h| %.3g)\n", PR, n, same,
               o[0].size(), md, mx);
        if (PR > 1) {  // the last layer's routing: distinct experts, tiles (grouped run)
          const uint32_t *uq = (const uint32_t *)s.uq.host();
          const uint32_t *G = uq + 1 + PR * TOPK * (2 + PR) + 2 + 2 * PR * TOPK;
          printf("  last layer: %u distinct experts of %d; tiles of 1 / 2 / 4 rows: %u / %u / %u\n", uq[0], PR * TOPK,
                 G[0], G[1], G[2]);
          if (grp) {  // every (row, slot) pair in exactly one tile, a tile's pairs on one expert
            const uint32_t *rt = (const uint32_t *)s.route.host();
            std::vector<int> seen(PR * TOPK, 0);
            int bad = 0;
            const int Ms[3] = {1, 2, 4};
            for (int c = 0; c < 3; c++)
              for (uint32_t t = 0; t < G[c]; t++) {
                const uint32_t *te = G + 4 + c * PR * TOPK + t * Ms[c];
                for (int j = 0; j < Ms[c]; j++) {
                  if (te[j] == 0xFFFFFFFFu) { bad += c != 2 || j != 3; continue; }
                  if (te[j] >= (uint32_t)(PR * TOPK)) { bad++; continue; }
                  seen[te[j]]++;
                  bad += (rt[(te[j] / TOPK) * 20 + te[j] % TOPK] & 511) != (rt[(te[0] / TOPK) * 20 + te[0] % TOPK] & 511);
                }
              }
            for (int v : seen) bad += v != 1;
            for (int r = 0; r < PR; r++)  // a row's experts distinct (the spread keeps them so)
              for (int a = 0; a < TOPK; a++)
                for (int b2 = a + 1; b2 < TOPK; b2++) bad += (rt[r * 20 + a] & 511) == (rt[r * 20 + b2] & 511);
            printf("  tile lists: %s (%d problems)\n", bad ? "BAD" : "ok", bad);
          }
        }
      }
      g_moe_group = gr0;
      g_moe_dedupe = dd0;
      g_moe_split = sp0;
      memcpy(m.h[0].host(), h0.data(), h0.size() * 4);
    }
    if (v2) {  // old vs new body from the same states and input: the 48-layer output bit for bit
      std::vector<float> h0(HCD), outs[4];
      memcpy(h0.data(), m.h[0].host(), HCD * 4);
      {  // one stage pair on layer 0: old vs new mix / inj / x
        std::vector<float> a[3], b[3];
        for (int k = 0; k < 2; k++) {
          cur_mask = k ? v2 : 0;
          Recorder r1;
          r1.begin();
          stage_rec(r1, 0, m.L[0], m.st[0], m.h[0], m.h[1]);
          stage_rec(r1, 1, m.L[0], m.st[0], m.h[0], m.h[1]);
          r1.end();
          r1.run();
          auto &o = k ? b : a;
          o[0].assign((float *)s.mix_a.host(), (float *)s.mix_a.host() + LR);
          o[1].assign((float *)s.inj_a.host(), (float *)s.inj_a.host() + 8);
          o[2].assign((float *)s.xa.host(), (float *)s.xa.host() + HID);
        }
        cur_mask = v2;
        const char *nm[3] = {"mix", "inj", "x"};
        for (int j = 0; j < 3; j++) {
          int same = 0;
          for (size_t i = 0; i < a[j].size(); i++) same += memcmp(&a[j][i], &b[j][i], 4) == 0;
          printf("check stage pair layer 0: %s %d of %zu bit-identical\n", nm[j], same, a[j].size());
        }
      }
      auto zero_states = [&]() {
        for (auto &st : m.st) {
          if (st.conv.buf) memset(st.conv.host(), 0, st.conv.nbytes());
          if (st.ssm.buf) memset(st.ssm.host(), 0, st.ssm.nbytes());
          gdn_clear_pending(st);
          if (st.ple_conv.buf) memset(st.ple_conv.host(), 0, st.ple_conv.nbytes());
        }
      };
      for (int li = 0; li < nload; li++) {  // each loaded layer's full step alone, old vs new
        const Tensor *names[] = {&s.mix_a, &s.inj_a, &s.xa, &s.o, &s.h2, &s.mix_m, &s.inj_m, &s.xm, &m.h[1]};
        const char *nn[] = {"mix_a", "inj_a", "xa", "o", "h_mid", "mix_m", "inj_m", "xm", "h_out"};
        std::vector<float> res[2][9];
        for (int k = 0; k < 2; k++) {
          zero_states();
          memcpy(m.h[0].host(), h0.data(), HCD * 4);
          cur_mask = k ? v2 : 0;
          Recorder r1;
          r1.begin();
          for (int i = 0; i < (int)Stage::N; i++) stage_rec(r1, i, m.L[li], m.st[li], m.h[0], m.h[1]);
          r1.end();
          r1.run();
          for (int j = 0; j < 9; j++)
            res[k][j].assign((float *)names[j]->host(), (float *)names[j]->host() + names[j]->numel());
        }
        cur_mask = v2;
        printf("check layer %d:", lids[li]);
        for (int j = 0; j < 9; j++) {
          int same = 0;
          for (size_t i = 0; i < res[0][j].size(); i++) same += memcmp(&res[0][j][i], &res[1][j][i], 4) == 0;
          printf(" %s %d/%zu", nn[j], same, res[0][j].size());
        }
        printf("\n");
      }
      if (getenv("NE_PROF_BISECT")) {  // old vs new body of the first N virtual layers
        int first = -1;
        for (int N = 1; N <= nl; N++) {
          std::vector<float> o2[2];
          for (int k = 0; k < 2; k++) {
            zero_states();
            memcpy(m.h[0].host(), h0.data(), HCD * 4);
            cur_mask = k ? v2 : 0;
            Recorder r1;
            r1.begin();
            for (int l = 0; l < N; l++) {
              const LayerW &w = m.L[l % nload];
              GdnState &st = m.st[l % nload];
              const Tensor &hi = m.h[l % 2], &ho = m.h[(l + 1) % 2];
              const bool ple = w.ple.on && l == 1;
              if (ple) record_ple(r1, w, st, s, hi, s.hp);
              for (int i = 0; i < (int)Stage::N; i++) stage_rec(r1, i, w, st, ple ? s.hp : hi, ho);
            }
            r1.end();
            r1.run();
            const float *hp = (const float *)m.h[N % 2].host();
            o2[k].assign(hp, hp + HCD);
          }
          cur_mask = v2;
          int same = 0;
          for (int i = 0; i < HCD; i++) same += memcmp(&o2[0][i], &o2[1][i], 4) == 0;
          if (same != HCD && first < 0) first = N;
          if (same != HCD || N <= 6 || N % 8 == 0) printf("bisect N = %d layers: %d of %d bit-identical\n", N, same, HCD);
          if (first > 0 && N >= first + 3) break;
        }
        if (first > 0) {  // the first differing layer alone, old vs new, from the same snapshot
          const int L1 = first - 1;
          auto prefix = [&](int n) {
            zero_states();
            memcpy(m.h[0].host(), h0.data(), HCD * 4);
            cur_mask = 0;
            Recorder r1;
            r1.begin();
            for (int l = 0; l < n; l++) {
              const LayerW &w = m.L[l % nload];
              GdnState &st = m.st[l % nload];
              const bool ple = w.ple.on && l == 1;
              if (ple) record_ple(r1, w, st, s, m.h[l % 2], s.hp);
              for (int i = 0; i < (int)Stage::N; i++) stage_rec(r1, i, w, st, ple ? s.hp : m.h[l % 2], m.h[(l + 1) % 2]);
            }
            r1.end();
            r1.run();
          };
          const Tensor *names[] = {&s.mix_a, &s.inj_a, &s.xa, &s.o, &s.h2, &s.mix_m, &s.inj_m, &s.xm, &m.h[(L1 + 1) % 2]};
          const char *nn[] = {"mix_a", "inj_a", "xa", "o", "h_mid", "mix_m", "inj_m", "xm", "h_out"};
          std::vector<float> res[2][9];
          for (int k = 0; k < 2; k++) {
            prefix(L1);
            cur_mask = k ? v2 : 0;
            const LayerW &w = m.L[L1 % nload];
            Recorder r1;
            r1.begin();
            for (int i = 0; i < (int)Stage::N; i++) stage_rec(r1, i, w, m.st[L1 % nload], m.h[L1 % 2], m.h[(L1 + 1) % 2]);
            r1.end();
            r1.run();
            for (int j = 0; j < 9; j++)
              res[k][j].assign((float *)names[j]->host(), (float *)names[j]->host() + names[j]->numel());
          }
          cur_mask = v2;
          for (int j = 0; j < 9; j++) {
            int nd = 0, shown = 0;
            for (size_t i = 0; i < res[0][j].size(); i++) {
              if (memcmp(&res[0][j][i], &res[1][j][i], 4) == 0) continue;
              nd++;
              if (shown++ < 3) {
                uint32_t a, b;
                memcpy(&a, &res[0][j][i], 4);
                memcpy(&b, &res[1][j][i], 4);
                printf("  layer %d %s[%zu]: old %.9g (%08x) new %.9g (%08x)\n", L1, nn[j], i, res[0][j][i], a, res[1][j][i], b);
              }
            }
            printf("bisect layer %d %s: %d of %zu differ\n", L1, nn[j], nd, res[0][j].size());
          }
        }
      }
      for (int k = 0; k < 4; k++) {
        for (auto &st : m.st) {
          if (st.conv.buf) memset(st.conv.host(), 0, st.conv.nbytes());
          if (st.ssm.buf) memset(st.ssm.host(), 0, st.ssm.nbytes());
          gdn_clear_pending(st);
          if (st.ple_conv.buf) memset(st.ple_conv.host(), 0, st.ple_conv.nbytes());
        }
        memcpy(m.h[0].host(), h0.data(), HCD * 4);
        Recorder &rr = *recs[k % 2 == 0 ? -3 : -1];
        rr.run();
        outs[k].assign((float *)m.h[0].host(), (float *)m.h[0].host() + HCD);
      }
      int same = 0;
      double md = 0, nrm = 0;
      for (int i = 0; i < HCD; i++) {
        same += memcmp(&outs[0][i], &outs[1][i], 4) == 0;
        md = std::max(md, (double)fabsf(outs[0][i] - outs[1][i]));
        nrm = std::max(nrm, (double)fabsf(outs[0][i]));
      }
      printf("check v2=%d vs old, 48-layer output: %d of %d bit-identical, max |diff| %.3g (max |h| %.3g)\n", v2, same,
             HCD, md, nrm);
      for (int k = 0; k < 2; k++) {
        int sm = 0;
        for (int i = 0; i < HCD; i++) sm += memcmp(&outs[k][i], &outs[k + 2][i], 4) == 0;
        printf("check %s vs %s again: %d of %d bit-identical\n", k ? "new" : "old", k ? "new" : "old", sm, HCD);
      }
      memcpy(m.h[0].host(), h0.data(), HCD * 4);
    }
    auto run_med = [&](Recorder &r) {
      std::vector<double> g;
      for (int i = 0; i < reps + 2; i++) {
        r.run();
        if (i >= 2) g.push_back(r.gpu_ms());
      }
      std::sort(g.begin(), g.end());
      return g[g.size() / 2];
    };
    // head (= model.mm record_head, direct dispatches): mixer, lm_head, sampler; each part alone x 16 (in a chain)
    auto head_rec = [&](int part, int n) {
      auto r = std::make_unique<Recorder>();
      r->begin();
      const HcW &x = m.mixer;
      const Tensor &hf = m.h[0];
      auto g = [](int v) { return std::to_string(v); };
      for (int i = 0; i < n; i++) {
        if ((part < 0 || part == 0) && x.down.t6) {  // lab-t6fuse: T6 pack (bf16 final mixer)
          t6_hc_read(*r, x, 1, hf, false, s, s.mix_a, s.inj_a, m.x);
        } else if (part < 0 || part == 0) {
          r->dispatch(pso("fn_hc_pre_g" + g(x.down.group) + "_r1"),
                      {hf, x.norm, x.down.w, x.down.s, x.down.b, m.dummy_inj, s.mix_a, s.inj_a, s.hc_part, s.hc_ctr},
                      MTLSizeMake((LR / 8 + 1) * HC, 1, 1), MTLSizeMake(256, 1, 1));
          r->barrier();
          r->dispatch(pso("fn_hc_mix_g" + g(x.up.group) + "_r1"), {hf, x.norm, x.up.w, x.up.s, x.up.b, s.mix_a, s.inj_a, m.x},
                      MTLSizeMake(HID / 8, 1, 1), MTLSizeMake(256, 1, 1));
          r->barrier();
        }
        if (part < 0 || part == 1) {
          const uint32_t nv = VOCAB;
          r->dispatch(pso("fn_lm_head_g" + g(m.head.group) + "_r1"), {m.x, m.head.w, m.head.s, m.head.b, m.logits, Arg::value(nv)},
                      MTLSizeMake((VOCAB + 7) / 8, 1, 1), MTLSizeMake(256, 1, 1));
          r->barrier();
        }
        if (part < 0 || part == 2) {
          r->dispatch(pso("fn_sample_p1", 1024), {m.logits, m.sp, m.cand, m.lse}, MTLSizeMake((VOCAB + 1023) / 1024, 1, 1),
                      MTLSizeMake(1024, 1, 1));
          r->barrier();
          r->dispatch(pso("fn_sample_p2", 1024), {m.cand, m.lse, m.sp, m.res, m.tok}, MTLSizeMake(1, 1, 1), MTLSizeMake(1024, 1, 1));
          r->barrier();
        }
      }
      r->end();
      return r;
    };
    auto hfull = head_rec(-1, 1);
    std::vector<double> hon;
    for (int rd = 0; rd < rounds; rd++) {
      for (int v : vars) times[v].push_back(run_med(*recs[v]));
      hon.push_back(run_med(*hfull));
      fprintf(stderr, "round %d: full %.3f ms\n", rd, times[-1].back());
    }
    auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    const double full = med(times[-1]);
    const double head = med(hon);
    if (ab) printf("A/B: fuse 0x%x body %.3f ms, fuse 0x%x body %.3f ms (%+.3f ms)\n", fuse_old, med(times[-3]), fuse_new,
                   med(times[-1]), med(times[-1]) - med(times[-3]));
    if (v2) printf("A/B: old body %.3f ms, v2=%d body %.3f ms (%+.3f ms)\n", med(times[-3]), v2, med(times[-1]),
                   med(times[-1]) - med(times[-3]));
    printf("rows %d, T = %d, reps %d, rounds %d (medians), loaded layers %d (pattern x %d)\n", PR, T, reps, rounds, nload, nl / nload);
    printf("body (48 layers + PLE, one command buffer): %.3f ms, %zu dispatches\n", full, recs[-1]->size());
    printf("head alone (mixer + lm_head + sampler, own command buffer): %.3f ms; body + head %.3f ms\n", head,
           full + head);
    for (int part = 0; part < (kf && *kf ? 0 : 3); part++) {
      auto a = head_rec(part, 4), b = head_rec(part, 16);
      const double ta = run_med(*a), tb = run_med(*b);
      printf("  head part %s: %.1f us (chained, differential 16 vs 4)\n",
             part == 0 ? "mixer (hc_pre + hc_mix)" : part == 1 ? "lm_head (q8, 248320 rows)" : "sampler p1 + p2",
             (tb - ta) * 1e3 / 12);
    }
    if (times.count(-2))
      printf("boundary (empty dispatch + barrier): %.2f us each (%zu added)\n",
             (med(times[-2]) - full) * 1e3 / nbound[-2], nbound[-2]);
    printf("\n| kernel | count | us each (in context) | ms per step | %% of body | MB each | GB/s | %% of 594 |\n");
    printf("|---|---|---|---|---|---|---|---|\n");
    double sum = 0;
    for (int k = 0; k < (int)kinds.size(); k++) {
      if (!times.count(k)) continue;
      const Kind &kd = kinds[k];
      int cnt = 0;
      size_t bytes = 0;
      for (int l = 0; l < nl; l++) {
        const LayerW &w = m.L[l % nload];
        if (kd.stage == -1 ? !(w.ple.on && l == 1) : !(kd.linear < 0 || kd.linear == (w.linear ? 1 : 0))) continue;
        cnt++;
        bytes += kind_bytes(kd, w);
      }
      const double d = full - med(times[k]);
      sum += d;
      const double each = d * 1e3 / cnt;
      const double mb = bytes / 1e6 / cnt;
      const double gbs = each > 0 ? mb / each * 1e3 : 0;
      printf("| %s | %d | %.1f | %.3f | %.1f | %.2f | %.0f | %.0f |\n", kd.name.c_str(), cnt, each, d,
             100 * d / full, mb, gbs, 100 * gbs / 594);
    }
    printf("sum of leave-one-out deltas %.3f ms of %.3f (rest = overlap / measurement)\n", sum, full);
  } catch (const std::exception &e) {
    fprintf(stderr, "fn-prof: %s\n", e.what());
    return 1;
  }
  return 0;
}
