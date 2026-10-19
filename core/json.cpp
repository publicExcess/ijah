// Minimal JSON parser. Owner: lab-core. Written from scratch for native-engine.
#include "core/json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "core/error.h"

namespace ne {

struct JsonParser {
  std::string_view s;
  size_t p = 0;
  int depth = 0;

  [[noreturn]] void fail(const char *what) const { throw Error(strfmt("json: %s at byte %zu", what, p)); }
  void ws() {
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == '\r')) p++;
  }
  bool lit(const char *w) {
    size_t n = strlen(w);
    if (s.compare(p, n, w) == 0) { p += n; return true; }
    return false;
  }
  static void put_utf8(std::string &out, uint32_t c) {
    if (c < 0x80) out += char(c);
    else if (c < 0x800) { out += char(0xC0 | (c >> 6)); out += char(0x80 | (c & 63)); }
    else if (c < 0x10000) { out += char(0xE0 | (c >> 12)); out += char(0x80 | ((c >> 6) & 63)); out += char(0x80 | (c & 63)); }
    else { out += char(0xF0 | (c >> 18)); out += char(0x80 | ((c >> 12) & 63)); out += char(0x80 | ((c >> 6) & 63)); out += char(0x80 | (c & 63)); }
  }
  uint32_t hex4() {
    if (p + 4 > s.size()) fail("short \\u escape");
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
      char c = s[p++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= c - '0';
      else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
      else fail("bad \\u escape");
    }
    return v;
  }
  std::string string() {
    if (p >= s.size() || s[p] != '"') fail("expected string");
    p++;
    std::string out;
    while (true) {
      if (p >= s.size()) fail("unterminated string");
      char c = s[p++];
      if (c == '"') break;
      if ((unsigned char)c < 0x20) fail("control character in string");
      if (c != '\\') { out += c; continue; }
      if (p >= s.size()) fail("unterminated escape");
      char e = s[p++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          uint32_t c1 = hex4();
          if (c1 >= 0xD800 && c1 < 0xDC00 && p + 1 < s.size() && s[p] == '\\' && s[p + 1] == 'u') {
            p += 2;
            uint32_t c2 = hex4();
            if (c2 >= 0xDC00 && c2 < 0xE000) c1 = 0x10000 + ((c1 - 0xD800) << 10) + (c2 - 0xDC00);
            else { put_utf8(out, c1); c1 = c2; }
          }
          put_utf8(out, c1);
          break;
        }
        default: fail("bad escape");
      }
    }
    return out;
  }
  Json number() {
    size_t b = p;
    bool integer = true;
    if (p < s.size() && s[p] == '-') p++;
    if (p >= s.size() || !isdigit((unsigned char)s[p])) fail("bad number");
    while (p < s.size() && isdigit((unsigned char)s[p])) p++;
    if (p < s.size() && s[p] == '.') {
      integer = false; p++;
      if (p >= s.size() || !isdigit((unsigned char)s[p])) fail("bad fraction");
      while (p < s.size() && isdigit((unsigned char)s[p])) p++;
    }
    if (p < s.size() && (s[p] == 'e' || s[p] == 'E')) {
      integer = false; p++;
      if (p < s.size() && (s[p] == '+' || s[p] == '-')) p++;
      if (p >= s.size() || !isdigit((unsigned char)s[p])) fail("bad exponent");
      while (p < s.size() && isdigit((unsigned char)s[p])) p++;
    }
    std::string tok(s.substr(b, p - b));
    Json j;
    j.t_ = Json::Number;
    j.d_ = strtod(tok.c_str(), nullptr);
    if (integer) {
      errno = 0;
      long long v = strtoll(tok.c_str(), nullptr, 10);
      if (errno == 0) { j.is_int_ = true; j.i_ = v; }
    } else if (std::isfinite(j.d_) && j.d_ == std::floor(j.d_) && std::fabs(j.d_) < 9.0e15) {
      j.is_int_ = true; j.i_ = (int64_t)j.d_;
    }
    return j;
  }
  Json value() {
    if (++depth > 512) fail("nesting too deep");
    ws();
    if (p >= s.size()) fail("unexpected end");
    Json j;
    char c = s[p];
    if (c == '{') {
      p++;
      j.t_ = Json::Object;
      ws();
      if (p < s.size() && s[p] == '}') { p++; }
      else {
        while (true) {
          ws();
          std::string k = string();
          ws();
          if (p >= s.size() || s[p] != ':') fail("expected ':'");
          p++;
          j.o_.emplace_back(std::move(k), value());
          ws();
          if (p < s.size() && s[p] == ',') { p++; continue; }
          if (p < s.size() && s[p] == '}') { p++; break; }
          fail("expected ',' or '}'");
        }
      }
    } else if (c == '[') {
      p++;
      j.t_ = Json::Array;
      ws();
      if (p < s.size() && s[p] == ']') { p++; }
      else {
        while (true) {
          j.a_.push_back(value());
          ws();
          if (p < s.size() && s[p] == ',') { p++; continue; }
          if (p < s.size() && s[p] == ']') { p++; break; }
          fail("expected ',' or ']'");
        }
      }
    } else if (c == '"') {
      j.t_ = Json::String;
      j.s_ = string();
    } else if (lit("true")) { j.t_ = Json::Bool; j.b_ = true; }
    else if (lit("false")) { j.t_ = Json::Bool; j.b_ = false; }
    else if (lit("null")) { j.t_ = Json::Null; }
    else j = number();
    depth--;
    return j;
  }
};

Json Json::parse(std::string_view text) {
  JsonParser ps{text};
  Json j = ps.value();
  ps.ws();
  if (ps.p != text.size()) ps.fail("trailing characters");
  return j;
}

bool Json::boolean() const { if (t_ != Bool) throw Error("json: not a bool"); return b_; }
double Json::num() const { if (t_ != Number) throw Error("json: not a number"); return d_; }
int64_t Json::i64() const { if (!is_int()) throw Error("json: not an integer"); return i_; }
const std::string &Json::str() const { if (t_ != String) throw Error("json: not a string"); return s_; }
const std::vector<Json> &Json::arr() const { if (t_ != Array) throw Error("json: not an array"); return a_; }
const std::vector<std::pair<std::string, Json>> &Json::obj() const {
  if (t_ != Object) throw Error("json: not an object");
  return o_;
}
size_t Json::size() const { return t_ == Array ? a_.size() : t_ == Object ? o_.size() : 0; }
const Json *Json::find(std::string_view key) const {
  if (t_ != Object) return nullptr;
  for (auto &kv : o_) if (kv.first == key) return &kv.second;
  return nullptr;
}
const Json &Json::operator[](std::string_view key) const {
  const Json *j = find(key);
  if (!j) throw Error(strfmt("json: missing key '%.*s'", (int)key.size(), key.data()));
  return *j;
}
const Json &Json::operator[](size_t i) const {
  if (t_ != Array || i >= a_.size()) throw Error(strfmt("json: index %zu out of range", i));
  return a_[i];
}
int64_t Json::get_int(std::string_view key, int64_t dflt) const {
  const Json *j = find(key);
  return j && j->is_int() ? j->i_ : dflt;
}
double Json::get_num(std::string_view key, double dflt) const {
  const Json *j = find(key);
  return j && j->is_number() ? j->d_ : dflt;
}
std::string Json::get_str(std::string_view key, const std::string &dflt) const {
  const Json *j = find(key);
  return j && j->is_string() ? j->s_ : dflt;
}

static void dump_str(std::string &o, const std::string &s) {
  o += '"';
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += (char)c;
    }
  }
  o += '"';
}
static void dump_rec(std::string &o, const Json &j) {
  switch (j.type()) {
    case Json::Null: o += "null"; break;
    case Json::Bool: o += j.boolean() ? "true" : "false"; break;
    case Json::Number: {
      char b[40];
      if (j.is_int()) snprintf(b, sizeof b, "%lld", (long long)j.i64());
      else snprintf(b, sizeof b, "%.17g", j.num());
      o += b;
      break;
    }
    case Json::String: dump_str(o, j.str()); break;
    case Json::Array: {
      o += '[';
      bool first = true;
      for (auto &x : j.arr()) { if (!first) o += ','; first = false; dump_rec(o, x); }
      o += ']';
      break;
    }
    case Json::Object: {
      o += '{';
      bool first = true;
      for (auto &kv : j.obj()) {
        if (!first) o += ',';
        first = false;
        dump_str(o, kv.first);
        o += ':';
        dump_rec(o, kv.second);
      }
      o += '}';
      break;
    }
  }
}
std::string json_quote(std::string_view s) {
  std::string o;
  dump_str(o, std::string(s));
  return o;
}
std::string Json::dump() const { std::string o; dump_rec(o, *this); return o; }

}  // namespace ne
