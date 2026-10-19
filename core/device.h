// Device singleton: MTLDevice, queues, feature flags. Owner: lab-core.
#pragma once
#import <Metal/Metal.h>

#include <cstdint>
#include <string>

namespace ne {

struct Device {
  id<MTLDevice> dev = nil;
  id<MTLCommandQueue> queue = nil;      // classic queue (classic / serial / ICB backends)
  id<MTL4CommandQueue> queue4 = nil;    // Metal 4 queue (nil if !has_metal4)
  bool has_nax = false;                 // Apple10 GPU family (M5): neural accelerators for MPP tensor ops
  bool has_metal4 = false;              // MTL4 queue + encoders usable
  uint64_t max_buffer_length = 0;       // MTLDevice.maxBufferLength (80.6 GiB on M5 Max 128 GB)
  uint64_t page_size = 16384;
  std::string name;

  static Device &get();  // created on first use; throws ne::Error if no Metal device
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;

 private:
  Device();
};

}  // namespace ne
