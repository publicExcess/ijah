// Pipeline cache: build/engine.metallib functions specialised by function constants. Owner: lab-core.
#pragma once
#import <Metal/Metal.h>

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace ne {

// One function-constant value. Build with the helpers: {FC::u32(0, 128), FC::f32(1, 0.5f), FC::b(2, true)}.
struct FC {
  enum Type : uint32_t { U32 = 0, I32 = 1, F32 = 2, BOOL = 3, U16 = 4, I16 = 5 };
  uint32_t index = 0;
  Type type = U32;
  uint32_t bits = 0;
  static FC u32(uint32_t i, uint32_t v) { return {i, U32, v}; }
  static FC i32(uint32_t i, int32_t v) { return {i, I32, (uint32_t)v}; }
  static FC f32(uint32_t i, float v) { uint32_t b; __builtin_memcpy(&b, &v, 4); return {i, F32, b}; }
  static FC b(uint32_t i, bool v) { return {i, BOOL, v ? 1u : 0u}; }
  static FC u16(uint32_t i, uint16_t v) { return {i, U16, v}; }
  static FC i16(uint32_t i, int16_t v) { return {i, I16, (uint32_t)(uint16_t)v}; }
};
using FunctionConstants = std::vector<FC>;

class Pipelines {
 public:
  // Cached by (name, constants). Thread-safe. Throws ne::Error if the function is missing or fails to compile.
  // Every pipeline supports indirect command buffers (needed by the ICB backend).
  static id<MTLComputePipelineState> get(const std::string &name, const FunctionConstants &fc = {});
  static bool has(const std::string &name);
  static std::vector<std::string> function_names();
  // The library: $NE_METALLIB, else engine.metallib next to libengine.dylib, else ./build/engine.metallib.
  static id<MTLLibrary> library();
  static std::string library_path();
  // Add functions compiled at runtime from MSL source (experiments; offline metallib is the normal path).
  // Later get() calls search these libraries after the metallib.
  static void add_source(const std::string &msl, const std::string &label = "runtime");
  static size_t cache_size();
};

}  // namespace ne
