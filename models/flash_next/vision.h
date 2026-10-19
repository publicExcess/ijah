// Flash-Next vision tower (lab-vision): Qwen3-VL ViT (27 layers, 1152 wide, 16 heads x 72, MLP 4304) + patch
// merger -> 2560-wide image embeddings, on NAX. Kernels: kernels/vision/vision.metal; design: docs/vision.md.
//
// Input: preprocessed patches in the engine's order (server/vision.py engine_patches: per patch (t, ph, pw, c),
// 2 x 16 x 16 x 3 = 1536 f32, patches of each image in merge-block order) and each image's grid (1, gh, gw).
// Output: bf16 [sum gh * gw / 4][2560] = the reference's image features, in order.
#pragma once
#include "core/engine.hpp"

#include <array>
#include <memory>
#include <vector>

namespace ne::vis {

constexpr int HID = 1152, NH = 16, HD = 72, HDP = 96, QKVP = 3 * NH * HDP /* 4608 */, OP = NH * HDP /* 1536 */;
constexpr int MLP = 4304, MLPP = 4352, PIN = 1536, OUT = 2560, MRG = 4 * HID /* 4608 */, GRID = 48, DEPTH = 27;

struct Block {
  Tensor n1w, n1b, n2w, n2b, qkv_w, qkv_b, proj_w, proj_b, fc1_w, fc1_b, fc2_w, fc2_b;  // padded copies (bf16)
};
struct Tower {
  std::vector<Block> blk;
  Tensor pe_w, pe_b, pos_tab, mn_w, mn_b, m1_w, m1_b, m2_w, m2_b, invf;
  size_t weight_bytes = 0;
  // scratch (grown on demand)
  int cap = 0;
  Tensor xin, pos, h, ln, qkv, o, mlp, m1, out, rpos, pidx, pwt;
  int depth = DEPTH;  // NE_VIT_LAYERS (debug)
  double last_gpu_ms = 0;
};
std::unique_ptr<Tower> load_tower(const Pack &p);
// patches f32 [N][1536]; grids (t, h, w) per image (t == 1, h and w even); out bf16 [N / 4][2560]. hidden (may be
// null): bf16 [N][1152], the last block's output before the merger.
void encode(Tower &t, const float *patches, int n, const std::vector<std::array<int, 3>> &grids, uint16_t *out,
            uint16_t *hidden = nullptr);

}  // namespace ne::vis

extern "C" {
typedef struct ne_vision ne_vision;
// The pack's vision tower (vision_tower.*), copied into padded GPU buffers (~0.95 GB). NULL on error.
NE_API ne_vision *ne_vision_create(ne_pack *pack);
NE_API void ne_vision_free(ne_vision *v);
// grids int [n_img][3]; out bf16 [tokens][2560] (tokens = sum h * w / 4); hidden: see ne::vis::encode (may be NULL).
// gpu_ms (may be NULL): GPU time of the forward.
NE_API int ne_vision_encode(ne_vision *v, const float *patches, int n_patches, const int *grids, int n_img,
                            uint16_t *out, uint16_t *hidden, double *gpu_ms);
NE_API int64_t ne_vision_weight_bytes(ne_vision *v);
}
