// lab-trunk6: t6-check -- the EXL3 / bf16 trunk kernels on real encoded matrices (docs/trunk6.md).
//
//   t6-check cases CASEDIR          run CASEDIR/cases.json (written by tests/trunk6/check_kernels.py): every case's
//                                   x through t6::record (path / rows as given), output -> CASEDIR/<case>.out.bin,
//                                   rel L2 vs the numpy reference, determinism (two runs bit-identical)
//   t6-check bench T6PACK Q8PACK PREFIX [PREFIX...]
//                                   GPU us per Linear (differential chains, weights cycled through >= 320 MB of copies
//                                   so every unit reads DRAM) at rows 1, 2, 4, 8: t6 SIMD, t6 NAX, and the q8 NAX
//                                   GEMM (kernels/nax) on the same Linear of the 8-bit pack. Env T6_BENCH_ROWS=1,4.
// GPU jobs: run under tools/with_lock.sh.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

#include "core/bench.h"
#include "core/error.h"
#include "core/json.h"
#include "core/pack.h"
#include "core/pipelines.h"
#include "core/recorder.h"
#include "kernels/nax/nax.h"
#include "kernels/trunk6/trunk6.h"

using namespace ne;

static std::vector<float> read_f32(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  NE_CHECK(f.good(), "cannot read %s", path.c_str());
  f.seekg(0, std::ios::end);
  size_t n = (size_t)f.tellg() / 4;
  f.seekg(0);
  std::vector<float> v(n);
  f.read((char *)v.data(), (std::streamsize)(n * 4));
  return v;
}
static void write_f32(const std::string &path, const std::vector<float> &v) {
  std::ofstream f(path, std::ios::binary);
  f.write((const char *)v.data(), (std::streamsize)(v.size() * 4));
}

static int cases(const std::string &dir) {
  std::ifstream jf(dir + "/cases.json");
  std::stringstream ss;
  ss << jf.rdbuf();
  Json j = Json::parse(ss.str());
  auto pack = Pack::open(j["pack"].str());
  t6::Scratch s = t6::alloc_scratch();
  int fails = 0;
  for (size_t i = 0; i < j["cases"].size(); i++) {
    const Json &c = j["cases"][i];
    const std::string name = c["name"].str(), tag = c["tag"].str(), ps = c.get_str("path", "auto");
    const int R = (int)c["R"].i64();
    t6::Linear l = t6::load(*pack, name);
    std::vector<float> x = read_f32(dir + "/" + c["x"].str()), yref = read_f32(dir + "/" + c["y"].str());
    NE_CHECK((int)x.size() == R * l.in && (int)yref.size() == R * l.out, "case %s: sizes", tag.c_str());
    Tensor xt = alloc(DType::F32, {R, l.in}), yt = alloc(DType::F32, {R, l.out});
    upload(xt, x.data(), x.size() * 4);
    t6::Plan p{};
    const t6::Plan *pp = nullptr;
    if (l.fmt == t6::Fmt::EXL3) {
      p = t6::plan(l, R, ps == "simd" ? t6::Path::Simd : ps == "nax" ? t6::Path::Nax : t6::Path::Auto,
                   (int)c.get_int("S", 0), (int)c.get_int("KS", 0));
      pp = &p;
    }
    Recorder r;
    r.begin();
    t6::record(r, l, R, xt, yt, s, pp);
    r.end();
    std::vector<float> y1(yref.size()), y2(yref.size());
    r.run();
    download(yt, y1.data(), y1.size() * 4);
    upload(yt, std::vector<float>(yref.size(), NAN).data(), yref.size() * 4);
    r.run();
    download(yt, y2.data(), y2.size() * 4);
    double e2 = 0, r2 = 0, mx = 0;
    bool same = true, finite = true;
    for (size_t k = 0; k < y1.size(); k++) {
      const double d = (double)y1[k] - yref[k];
      e2 += d * d;
      r2 += (double)yref[k] * yref[k];
      mx = std::max(mx, std::fabs(d));
      same &= memcmp(&y1[k], &y2[k], 4) == 0;
      finite &= std::isfinite(y1[k]);
    }
    const double rel = std::sqrt(e2 / std::max(r2, 1e-300));
    const double tol = c.get_num("tol", 2e-3);
    const bool ok = finite && same && rel <= tol;
    fails += !ok;
    printf("%-4s %-60s %s R%d %-4s S%-2d KS%-2d rel %.2e max %.2e %s%s\n", ok ? "ok" : "FAIL", name.c_str(),
           t6::fmt_name(l.fmt), R, l.fmt == t6::Fmt::BF16 ? "-" : (p.path == t6::Path::Nax ? "nax" : "simd"),
           p.S, p.KS, rel, mx, same ? "deterministic" : "NOT DETERMINISTIC", finite ? "" : " NONFINITE");
    write_f32(dir + "/" + tag + ".out.bin", y1);
  }
  printf("%zu cases, %d failed\n", j["cases"].size(), fails);
  return fails ? 1 : 0;
}

static int env_split(const char *k) {
  const char *e = getenv(k);
  return e ? atoi(e) : 0;
}

static int bench(const std::string &t6pack, const std::string &q8pack, const std::vector<std::string> &names) {
  auto pt = Pack::open(t6pack);
  auto pq = Pack::open(q8pack);
  std::vector<int> rows = {1, 2, 4, 8};
  if (const char *e = getenv("T6_BENCH_ROWS")) {
    rows.clear();
    std::stringstream ss(e);
    std::string tok;
    while (std::getline(ss, tok, ',')) rows.push_back(atoi(tok.c_str()));
  }
  t6::Scratch s = t6::alloc_scratch();
  // Weights cycle through NC identical copies (>= 320 MB in all), so every unit reads DRAM, not the SLC (a streaming
  // flush between units left +-5 us of noise in the differential).
  auto dup = [](const Tensor &t) {
    Tensor c = t;
    c.buf = alloc_bytes(t.nbytes());
    c.offset = 0;
    memcpy(c.host(), t.host(), t.nbytes());
    return c;
  };
  auto timeit = [&](const std::function<void(Recorder &, int)> &unit) {
    return time_per_unit([&](Recorder &r, int n) { for (int i = 0; i < n; i++) { unit(r, i); r.barrier(); } }, 16, 64, 7);
  };
  printf("%-58s %-5s %4s %8s %5s | %9s %7s | %9s %7s | %9s %7s\n", "linear", "fmt", "R", "MB", "S/KS", "t6 simd", "GB/s",
         "t6 nax", "GB/s", "q8 nax", "GB/s");
  for (const auto &nm : names) {
    t6::Linear l = t6::load(*pt, nm);
    const bool q8 = t6::detect(*pq, nm) == t6::Fmt::Q8;
    fn::Q8 qw;
    double qmb = 0;
    if (q8) {
      qw = nax::load_q8(*pq, nm);
      qmb = (qw.w.nbytes() + qw.s.nbytes() + qw.b.nbytes()) / 1e6;
    }
    const double mb = l.fmt == t6::Fmt::EXL3 ? (l.trellis.nbytes() + l.suh.nbytes() + l.svh.nbytes()) / 1e6 : l.w.nbytes() / 1e6;
    const int NC = std::min(64, std::max(2, (int)std::ceil(320.0 / std::min(mb, qmb > 0 ? qmb : mb))));
    std::vector<t6::Linear> lc(NC, l);
    std::vector<fn::Q8> qc(q8 ? NC : 0, qw);
    for (int c = 0; c < NC; c++) {
      if (l.fmt == t6::Fmt::EXL3) lc[c].trellis = dup(l.trellis), lc[c].suh = dup(l.suh), lc[c].svh = dup(l.svh);
      else lc[c].w = dup(l.w);
      if (q8) qc[c].w = dup(qw.w), qc[c].s = dup(qw.s), qc[c].b = dup(qw.b);
    }
    for (int R : rows) {
      Tensor x = alloc(DType::F32, {R, l.in}), y = alloc(DType::F32, {R, l.out});
      std::vector<float> xv((size_t)R * l.in);
      for (size_t i = 0; i < xv.size(); i++) xv[i] = (float)std::sin(0.37 * (double)i);
      upload(x, xv.data(), xv.size() * 4);
      double ts = NAN, tn = NAN, tq = NAN;
      int S = 0, KS = 0;
      if (l.fmt == t6::Fmt::EXL3) {
        const char *only = getenv("T6_BENCH_PATH");
        try {
          if (R <= 8 && (!only || !strcmp(only, "simd"))) {
            t6::Plan p = t6::plan(l, R, t6::Path::Simd, env_split("T6_SIMD_S"));
            S = p.S;
            ts = timeit([&](Recorder &r, int i) { t6::record(r, lc[i % NC], R, x, y, s, &p); });
          }
        } catch (const std::exception &e) { fprintf(stderr, "simd: %s\n", e.what()); }
        try {
          if (!only || !strcmp(only, "nax")) {
            t6::Plan pn = t6::plan(l, R, t6::Path::Nax, env_split("T6_NAX_S"), env_split("T6_NAX_KS"));
            KS = pn.S * 100 + pn.KS;
            tn = timeit([&](Recorder &r, int i) { t6::record(r, lc[i % NC], R, x, y, s, &pn); });
          }
        } catch (const std::exception &e) { fprintf(stderr, "nax: %s\n", e.what()); }
      } else if (R <= 8) {
        ts = timeit([&](Recorder &r, int i) { t6::record(r, lc[i % NC], R, x, y, s); });
      }
      if (q8 && R <= 16 && !getenv("T6_BENCH_NOQ8")) {
        Tensor yq = alloc(DType::F32, {R, qw.rows});
        tq = timeit([&](Recorder &r, int i) { nax::record_q8(r, R, x, {&qc[i % NC]}, {yq}); });
      }
      printf("%-58s %-5s %4d %8.2f %2d/%-4d| %9.1f %7.0f | %9.1f %7.0f | %9.1f %7.0f\n", nm.c_str(), t6::fmt_name(l.fmt),
             R, mb, S, KS, ts, mb * 1e3 / ts, tn, mb * 1e3 / tn, tq, qmb * 1e3 / tq);
      fflush(stdout);
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  try {
    @autoreleasepool {
      NE_CHECK(argc >= 3, "usage: t6-check cases CASEDIR | bench T6PACK Q8PACK PREFIX...");
      const std::string mode = argv[1];
      if (mode == "cases") return cases(argv[2]);
      if (mode == "bench") {
        NE_CHECK(argc >= 5, "bench T6PACK Q8PACK PREFIX...");
        return bench(argv[2], argv[3], std::vector<std::string>(argv + 4, argv + argc));
      }
      NE_CHECK(false, "unknown mode %s", mode.c_str());
    }
  } catch (const std::exception &e) {
    fprintf(stderr, "t6-check: %s\n", e.what());
    return 2;
  }
}
