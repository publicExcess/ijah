// Dispatch-cost measurement (differential, as lab lead 4 found necessary). Owner: lab-core.
#pragma once
#include <functional>
#include <string>

#include "core/recorder.h"

namespace ne {

struct DispatchCost {
  double gpu_us = 0;     // per dependent dispatch, (T(n2) - T(n1)) / (n2 - n1), medians over reps
  double encode_us = 0;  // CPU encode+commit per dispatch at n2 (ICB: per replay of an already-built ICB)
  double naive_us = 0;   // T(n2) / n2 (what a one-shot measurement would report)
};

// kernel: "tiny" (1 threadgroup of 32) or "read256" (256 KiB read, 40 threadgroups of 256).
// Chains are dependent (barrier between dispatches) unless independent == true.
DispatchCost bench_dispatch(Backend b, const std::string &kernel, int n1 = 256, int n2 = 2048, int reps = 5,
                            bool independent = false);

// Generic differential timer: record(r, n) records n units (e.g. n layer steps) into r (begin/end are done for you).
// Returns GPU us per unit = (T(n2) - T(n1)) / (n2 - n1), medians over reps, after one warm-up replay of each.
double time_per_unit(const std::function<void(Recorder &, int)> &record, int n1, int n2, int reps = 5,
                     Backend b = Backend::Auto);

// The optional machine lock (the script named by IJAH_LOCK_SCRIPT, e.g. tools/machine-lock.sh) held by this process
// (pid = getpid()) for its lifetime. wait = block until free; else held() is false when someone else has it. No-op
// (held() true) when IJAH_LOCK_SCRIPT is unset, the script is absent or NE_NO_LOCK is set. Released in the destructor.
class MachineLock {
 public:
  MachineLock(const std::string &owner, const std::string &job, int minutes = 15, bool wait = false);
  ~MachineLock();
  bool held() const { return held_; }
  MachineLock(const MachineLock &) = delete;
  MachineLock &operator=(const MachineLock &) = delete;

 private:
  std::string owner_;
  bool held_ = false, ours_ = false;
};

}  // namespace ne
