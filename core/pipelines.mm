// Pipeline cache over build/engine.metallib (+ optional runtime-compiled sources). Owner: lab-core.
#import <Foundation/Foundation.h>

#include <dlfcn.h>
#include <sys/stat.h>

#include <map>
#include <mutex>

#include "core/engine.hpp"

namespace ne {

namespace {
struct State {
  std::mutex mu;
  id<MTLLibrary> lib = nil;
  std::string lib_path;
  std::vector<id<MTLLibrary>> extra;
  std::map<std::string, id<MTLComputePipelineState>> cache;
};
State &state() {
  static State *s = new State();
  return *s;
}
bool file_exists(const std::string &p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0;
}
std::string find_metallib() {
  if (const char *e = getenv("NE_METALLIB"); e && *e) return e;
  Dl_info info;
  if (dladdr((const void *)&find_metallib, &info) && info.dli_fname) {
    std::string so = info.dli_fname;
    std::string cand = so.substr(0, so.rfind('/') + 1) + "engine.metallib";
    if (file_exists(cand)) return cand;
  }
  return "build/engine.metallib";
}
id<MTLLibrary> load_locked(State &s) {
  if (s.lib) return s.lib;
  s.lib_path = find_metallib();
  NSError *err = nil;
  NSURL *url = [NSURL fileURLWithPath:@(s.lib_path.c_str())];
  s.lib = [Device::get().dev newLibraryWithURL:url error:&err];
  NE_CHECK(s.lib != nil, "cannot load metallib %s: %s", s.lib_path.c_str(),
           err ? err.localizedDescription.UTF8String : "?");
  return s.lib;
}
std::string key_of(const std::string &name, const FunctionConstants &fc) {
  std::string k = name;
  auto sorted = fc;
  std::sort(sorted.begin(), sorted.end(), [](const FC &a, const FC &b) { return a.index < b.index; });
  for (auto &c : sorted) k += strfmt("|%u:%u:%08x", c.index, (unsigned)c.type, c.bits);
  return k;
}
}  // namespace

id<MTLLibrary> Pipelines::library() {
  State &s = state();
  std::lock_guard<std::mutex> g(s.mu);
  return load_locked(s);
}
std::string Pipelines::library_path() {
  library();
  return state().lib_path;
}

void Pipelines::add_source(const std::string &msl, const std::string &label) {
  @autoreleasepool {
    MTLCompileOptions *o = [MTLCompileOptions new];
    o.languageVersion = MTLLanguageVersion4_0;
    NSError *err = nil;
    id<MTLLibrary> l = [Device::get().dev newLibraryWithSource:@(msl.c_str()) options:o error:&err];
    NE_CHECK(l != nil, "compile %s: %s", label.c_str(), err ? err.localizedDescription.UTF8String : "?");
    State &s = state();
    std::lock_guard<std::mutex> g(s.mu);
    s.extra.push_back(l);
  }
}

bool Pipelines::has(const std::string &name) {
  State &s = state();
  std::lock_guard<std::mutex> g(s.mu);
  NSString *n = @(name.c_str());
  if ([load_locked(s).functionNames containsObject:n]) return true;
  for (id<MTLLibrary> l : s.extra)
    if ([l.functionNames containsObject:n]) return true;
  return false;
}

std::vector<std::string> Pipelines::function_names() {
  State &s = state();
  std::lock_guard<std::mutex> g(s.mu);
  std::vector<std::string> out;
  for (NSString *n in load_locked(s).functionNames) out.push_back(n.UTF8String);
  for (id<MTLLibrary> l : s.extra)
    for (NSString *n in l.functionNames) out.push_back(n.UTF8String);
  std::sort(out.begin(), out.end());
  return out;
}

size_t Pipelines::cache_size() {
  State &s = state();
  std::lock_guard<std::mutex> g(s.mu);
  return s.cache.size();
}

id<MTLComputePipelineState> Pipelines::get(const std::string &name, const FunctionConstants &fc) {
  State &s = state();
  const std::string key = key_of(name, fc);
  std::lock_guard<std::mutex> g(s.mu);
  if (auto it = s.cache.find(key); it != s.cache.end()) return it->second;
  @autoreleasepool {
    NSString *n = @(name.c_str());
    id<MTLLibrary> lib = nil;
    if ([load_locked(s).functionNames containsObject:n]) lib = s.lib;
    else
      for (id<MTLLibrary> l : s.extra)
        if ([l.functionNames containsObject:n]) lib = l;
    NE_CHECK(lib != nil, "kernel %s not found in %s", name.c_str(), s.lib_path.c_str());
    NSError *err = nil;
    // Always specialise (even with no values): a function that declares function constants cannot be used through
    // plain newFunctionWithName (Metal validation asserts).
    MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
    for (const FC &c : fc) {
      MTLDataType t;
      uint32_t v32 = c.bits;
      uint16_t v16 = (uint16_t)c.bits;
      bool vb = c.bits != 0;
      const void *ptr = &v32;
      switch (c.type) {
        case FC::U32: t = MTLDataTypeUInt; break;
        case FC::I32: t = MTLDataTypeInt; break;
        case FC::F32: t = MTLDataTypeFloat; break;
        case FC::BOOL: t = MTLDataTypeBool; ptr = &vb; break;
        case FC::U16: t = MTLDataTypeUShort; ptr = &v16; break;
        case FC::I16: t = MTLDataTypeShort; ptr = &v16; break;
        default: throw Error(strfmt("kernel %s: bad function-constant type %u", name.c_str(), (unsigned)c.type));
      }
      [cv setConstantValue:ptr type:t atIndex:c.index];
    }
    id<MTLFunction> f = [lib newFunctionWithName:n constantValues:cv error:&err];
    NE_CHECK(f != nil, "kernel %s: %s", key.c_str(), err ? err.localizedDescription.UTF8String : "newFunction failed");
    MTLComputePipelineDescriptor *d = [MTLComputePipelineDescriptor new];
    d.computeFunction = f;
    d.supportIndirectCommandBuffers = YES;
    d.label = @(key.c_str());
    id<MTLComputePipelineState> pso = [Device::get().dev newComputePipelineStateWithDescriptor:d
                                                                                      options:MTLPipelineOptionNone
                                                                                   reflection:nil
                                                                                        error:&err];
    NE_CHECK(pso != nil, "pipeline %s: %s", key.c_str(), err ? err.localizedDescription.UTF8String : "?");
    s.cache[key] = pso;
    return pso;
  }
}

}  // namespace ne
