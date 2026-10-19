// Minimal JSON value + parser (RFC 8259; numbers keep an exact int64 when they are integers). Owner: lab-core.
#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ne {

class Json {
 public:
  enum Type { Null, Bool, Number, String, Array, Object };

  static Json parse(std::string_view text);  // throws ne::Error with the byte offset on bad input

  Type type() const { return t_; }
  bool is_null() const { return t_ == Null; }
  bool is_bool() const { return t_ == Bool; }
  bool is_number() const { return t_ == Number; }
  bool is_int() const { return t_ == Number && is_int_; }
  bool is_string() const { return t_ == String; }
  bool is_array() const { return t_ == Array; }
  bool is_object() const { return t_ == Object; }

  bool boolean() const;           // throws if not a bool
  double num() const;             // throws if not a number
  int64_t i64() const;            // throws if not an integer-valued number
  const std::string &str() const;  // throws if not a string
  const std::vector<Json> &arr() const;
  const std::vector<std::pair<std::string, Json>> &obj() const;

  size_t size() const;                           // array/object length, else 0
  const Json *find(std::string_view key) const;  // object member or nullptr
  const Json &operator[](std::string_view key) const;  // throws if missing
  const Json &operator[](size_t i) const;              // throws if out of range
  // Convenience with defaults (path-free; use find() chains for nesting).
  int64_t get_int(std::string_view key, int64_t dflt) const;
  double get_num(std::string_view key, double dflt) const;
  std::string get_str(std::string_view key, const std::string &dflt) const;

  std::string dump() const;  // compact serialisation

 private:
  friend struct JsonParser;
  Type t_ = Null;
  bool b_ = false, is_int_ = false;
  double d_ = 0;
  int64_t i_ = 0;
  std::string s_;
  std::vector<Json> a_;
  std::vector<std::pair<std::string, Json>> o_;
};

// JSON string literal for s (quotes + escapes).
std::string json_quote(std::string_view s);

}  // namespace ne
