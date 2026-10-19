// Tensor-op GEMMs for the EXL3 trunk in the chunked prefill (pfkern, 18 Oct; docs/prefill.md "kernels, Oct 18").
// Self-contained (metal_stdlib + MPP: a test can compile this file alone). Operands are MPP tensors, so the M5 tensor
// op loads them itself (pfkern microbenchmarks: per-lane fragment loads into cooperative tensors reached ~56-70% of
// the op's rate). Three kernels:
//   t6_a8_gemm   (opt-in, NE_T6_A8) int8 x int8 -> int32 per 128-block, scaled into f32:
//                y[m][n] (f32, ld ldy) = sum_b sx[b][m] * sw[b][n] * sum_{k in block b} xq[m][k] * wq[n][k]
//                xq int8 [M][K] (t6_pf_prep_a8: H_hi(x * suh), per (row, 128-block) absmax / 127), sx f32 [KB][M];
//                wq int8 [N][K] (t6_dense_a8: W' per (output row, 128-block) absmax / 127), sw f32 [KB][N].
//                int32 is exact within a block (|acc| <= 128 * 127 * 127). ~100 TOP/s vs 56-65 for f16.
//   t6_h16_gemm  f16 x f16 -> f32 yraw (the prep / post path; f32-input Linears)
//   t6_fold_gemm bf16 x f16 (folded W) -> y directly (default for bf16-input Linears, >= 64 rows)
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct T6A8P {  // = trunk6.mm T6A8P
  uint K, N, M, ldy;
};

// Operands as MPP tensors (the tensor op loads them itself: 100 TOP/s at M 4096 x K 2560 x N 10240 on the M5 Max,
// against ~70 with per-lane fragment loads into cooperative tensors and 56 TFLOP/s for the f16 GEMM it replaces);
// matmul2d BM x 64 x 128 over the 4 simdgroups, one run per 128-block, the int32 block result scaled into f32
// (element coordinates from get_multidimensional_index). Grid (N / 64, ceil(M / BM)), 128 threads.
template <int BM>
kernel void t6_a8_gemm(device const char *wq [[buffer(0)]], device const float *sw [[buffer(1)]],
                       device const char *xq [[buffer(2)]], device const float *sx [[buffer(3)]],
                       device float *y [[buffer(4)]], constant T6A8P &p [[buffer(5)]],
                       uint2 tgp [[threadgroup_position_in_grid]]) {
  constexpr int BN = 64, BK = 128, CAP = BM * BN / 128;
  const int K = p.K, N = p.N, M = p.M;
  const int row0 = tgp.y * BM, col0 = tgp.x * BN;
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> A((device int8_t *)xq, dextents<int32_t, 2>(K, M));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> B((device int8_t *)wq, dextents<int32_t, 2>(K, N));
  constexpr auto desc = matmul2d_descriptor(BM, BN, BK, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA0 = A.slice(0, row0);
  auto mB0 = B.slice(0, col0);
  auto ct = op.template get_destination_cooperative_tensor<decltype(mA0), decltype(mB0), int32_t>();
  float cf[CAP];
  for (int i = 0; i < CAP; i++) cf[i] = 0.0f;
  const int KB = (K + BK - 1) / BK;
  for (int b = 0; b < KB; b++) {
    auto mA = A.slice(b * BK, row0);
    auto mB = B.slice(b * BK, col0);
    op.run(mA, mB, ct);
    const device float *sxb = sx + (size_t)b * M, *swb = sw + (size_t)b * N;
    for (int i = 0; i < CAP; i++) {
      const auto ix = ct.get_multidimensional_index(i);
      cf[i] = fma(sxb[min(row0 + int(ix[1]), M - 1)] * swb[col0 + int(ix[0])], float(ct[i]), cf[i]);
    }
  }
  for (int i = 0; i < CAP; i++) {
    const auto ix = ct.get_multidimensional_index(i);
    const int m = row0 + int(ix[1]);
    if (m < M) y[(size_t)m * p.ldy + col0 + int(ix[0])] = cf[i];
  }
}
template [[host_name("t6_a8_gemm")]] kernel decltype(t6_a8_gemm<64>) t6_a8_gemm<64>;
template [[host_name("t6_a8_gemm_m128")]] kernel decltype(t6_a8_gemm<128>) t6_a8_gemm<128>;

// The f16 path on the same tensor ops (accuracy-neutral replacement of pf_h16_hf for >= 64 rows): yraw[m][n] (f32,
// ld N) = xh[m][:] . wd[n][:] (half), one matmul2d BM x 64 x K (dynamic K: the op runs the k loop itself) over NSG
// simdgroups. 62-66 TFLOP/s on the trunk shapes against ~56 for the 16x32x16 fragment loop (pfkern microbenchmark).
// Grid (N / 64, ceil(M / BM)), 32 NSG threads.
template <int BM, int NSG>
kernel void t6_h16_gemm(device const half *wd [[buffer(0)]], device const half *xh [[buffer(1)]],
                        device float *y [[buffer(2)]], constant T6A8P &p [[buffer(3)]],
                        uint2 tgp [[threadgroup_position_in_grid]]) {
  const int K = p.K, N = p.N, M = p.M;
  const int row0 = tgp.y * BM, col0 = tgp.x * 64;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> A((device half *)xh, dextents<int32_t, 2>(K, M));
  tensor<device half, dextents<int32_t, 2>, tensor_inline> B((device half *)wd, dextents<int32_t, 2>(K, N));
  tensor<device float, dextents<int32_t, 2>, tensor_inline> C(y, dextents<int32_t, 2>(N, M));
  constexpr auto desc = matmul2d_descriptor(BM, 64, static_cast<int>(dynamic_extent), false, true, false,
                                            matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  auto mA = A.slice(0, row0);
  auto mB = B.slice(0, col0);
  auto mC = C.slice(col0, row0);
  op.run(mA, mB, mC);
}
template [[host_name("t6_h16_gemm")]] kernel decltype(t6_h16_gemm<64, 8>) t6_h16_gemm<64, 8>;
template [[host_name("t6_h16_gemm_m128")]] kernel decltype(t6_h16_gemm<128, 4>) t6_h16_gemm<128, 4>;

// Folded-W GEMM (pfkern): y[m][n] (ld ldy; f32, or bf16: the op's conversion rounds to nearest even, = bf16r, checked
// bit-identical against a manual bf16r store) = x[m][:] . wf[n][:] for n < N (= the Linear's OUT), x bf16 [M][K]
// (contiguous rows), wf half [N rounded to 64][K] (t6_fold_k / t6_fold_n). bfloat x half runs at the f16 rate on the
// M5 tensor op (65 TFLOP/s at 4096 x 2560 x 10240); f32 x half at a quarter of it, so f32-input Linears keep the
// prep / post path. Grid (ceil(N / 64), ceil(M / 128)) x 128.
template <typename YT>
kernel void t6_fold_gemm(device const half *wf [[buffer(0)]], device const bfloat *x [[buffer(1)]],
                           device YT *y [[buffer(2)]], constant T6A8P &p [[buffer(3)]],
                           uint2 tgp [[threadgroup_position_in_grid]]) {
  constexpr int BM = 128, BN = 64;
  const int K = p.K, N = p.N, M = p.M;
  const int row0 = tgp.y * BM, col0 = tgp.x * BN;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> A((device bfloat *)x, dextents<int32_t, 2>(K, M));
  tensor<device half, dextents<int32_t, 2>, tensor_inline> B((device half *)wf, dextents<int32_t, 2>(K, (N + 63) / 64 * 64));
  tensor<device YT, dextents<int32_t, 2>, tensor_inline> C(y, dextents<int32_t, 2>(N, M), array<int32_t, 2>({1, int(p.ldy)}));
  constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false,
                                            matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA = A.slice(0, row0);
  auto mB = B.slice(0, col0);
  auto mC = C.slice(col0, row0);
  op.run(mA, mB, mC);
}
template [[host_name("t6_fold_gemm_b")]] kernel decltype(t6_fold_gemm<bfloat>) t6_fold_gemm<bfloat>;
template [[host_name("t6_fold_gemm_f")]] kernel decltype(t6_fold_gemm<float>) t6_fold_gemm<float>;
