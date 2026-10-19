// Errors and small helpers shared by the core. Owner: lab-core.
#pragma once
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ne {

struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline std::string strfmt(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
inline std::string strfmt(const char *fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  if (n < (int)sizeof buf) return std::string(buf, n < 0 ? 0 : n);
  std::string s(n, '\0');
  va_start(ap, fmt);
  vsnprintf(s.data(), n + 1, fmt, ap);
  va_end(ap);
  return s;
}

#define NE_CHECK(cond, ...) \
  do { if (!(cond)) throw ::ne::Error(::ne::strfmt(__VA_ARGS__)); } while (0)

// Monotonic wall clock in microseconds.
double now_us();

}  // namespace ne
