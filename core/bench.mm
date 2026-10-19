// Dispatch-cost measurement. Probe kernels and method from lab lead 4 (fused/disp.mm). Owner: lab-core.
#import <Foundation/Foundation.h>

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <ctime>
#include <vector>

#include "core/bench.h"
#include "core/engine.hpp"

namespace ne {

namespace {
double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
struct Probe {
  id<MTLComputePipelineState> pso;
  Tensor cnt, src, outs[2];
  uint32_t n4 = 16384;
  bool read;
  explicit Probe(const std::string &k) : read(k == "read256") {
    NE_CHECK(k == "tiny" || k == "read256", "bench kernel must be tiny or read256");
    pso = Pipelines::get(read ? "ne_st_read256" : "ne_st_tiny");
    cnt = alloc(DType::F32, {4});
    src = alloc(DType::U32, {(int64_t)n4 * 4});
    memset(src.host(), 7, src.nbytes());
    outs[0] = alloc(DType::U32, {40 * 256 / 32});
    outs[1] = alloc(DType::U32, {40 * 256 / 32});
  }
  void record(Recorder &r, int n, bool independent) {
    r.begin();
    for (int i = 0; i < n; i++) {
      if (i && !independent) r.barrier();
      if (read)
        r.dispatch(pso, {src, outs[i & 1], outs[(i + 1) & 1], Arg::value(n4)}, MTLSizeMake(40, 1, 1),
                   MTLSizeMake(256, 1, 1));
      else
        r.dispatch(pso, {cnt}, MTLSizeMake(1, 1, 1), MTLSizeMake(32, 1, 1));
    }
    r.end();
  }
};
}  // namespace

DispatchCost bench_dispatch(Backend b, const std::string &kernel, int n1, int n2, int reps, bool independent) {
  @autoreleasepool {
    Probe pr(kernel);
    Recorder r1(b), r2(b);
    pr.record(r1, n1, independent);
    pr.record(r2, n2, independent);
    r1.run();  // warm-up (pipeline residency, clocks)
    r2.run();
    std::vector<double> t1, t2, enc;
    for (int i = 0; i < reps; i++) {
      r1.run();
      t1.push_back(r1.gpu_ms() * 1e3);
      r2.run();
      t2.push_back(r2.gpu_ms() * 1e3);
      enc.push_back(r2.encode_us());
    }
    DispatchCost c;
    double m1 = median(t1), m2 = median(t2);
    c.gpu_us = (m2 - m1) / (n2 - n1);
    c.naive_us = m2 / n2;
    c.encode_us = median(enc) / n2;
    return c;
  }
}

double time_per_unit(const std::function<void(Recorder &, int)> &record, int n1, int n2, int reps, Backend b) {
  NE_CHECK(n2 > n1 && n1 >= 0 && reps >= 1, "time_per_unit: need n2 > n1 >= 0");
  Recorder r1(b), r2(b);
  r1.begin();
  record(r1, n1);
  r1.end();
  r2.begin();
  record(r2, n2);
  r2.end();
  r1.run();
  r2.run();
  std::vector<double> t1, t2;
  for (int i = 0; i < reps; i++) {
    r1.run();
    t1.push_back(r1.gpu_ms() * 1e3);
    r2.run();
    t2.push_back(r2.gpu_ms() * 1e3);
  }
  return (median(t2) - median(t1)) / (n2 - n1);
}

// The lock script is optional (IJAH_LOCK_SCRIPT, e.g. tools/machine-lock.sh): unset = no lock.
static std::string lock_script() {
  const char *s = getenv("IJAH_LOCK_SCRIPT");
  if (!s || !*s) return "";
  if (s[0] == '~') {
    const char *h = getenv("HOME");
    return std::string(h ? h : "") + (s + 1);
  }
  return s;
}
MachineLock::MachineLock(const std::string &owner, const std::string &job, int minutes, bool wait) : owner_(owner) {
  struct stat st;
  if (getenv("NE_NO_LOCK") || lock_script().empty() || stat(lock_script().c_str(), &st) != 0) {
    held_ = true;
    return;
  }
  time_t t = time(nullptr) + minutes * 60;
  char until[8];
  strftime(until, sizeof until, "%H:%M", localtime(&t));
  std::string q = job;
  for (char &c : q)
    if (c == '\'') c = ' ';
  std::string cmd = lock_script() + (wait ? " wait " : " acquire ") + owner + " '" + q + "' " + until + " " +
                    std::to_string(getpid()) + " >/dev/null 2>&1";
  held_ = ours_ = system(cmd.c_str()) == 0;
}
MachineLock::~MachineLock() {
  if (!ours_) return;
  std::string cmd = lock_script() + " release " + owner_ + " >/dev/null 2>&1";
  system(cmd.c_str());
}

}  // namespace ne
