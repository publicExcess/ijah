// Flash-Next fused MoE block at R rows (R = 1 decode, 4 verify), 3 dispatches, on the real pack tensors (port of lab lead 4's fused.metal;
// see ne_common.h for attribution):
//   D1 fn_moe_router: router GEMV (512 bf16 rows + shared_expert_gate row) + shared gate/up GEMV (8-bit affine g128);
//      the last router threadgroup to finish does top-10 + renormalised softmax (atomic counter, no extra dispatch).
//   D2 fn_moe_gateup: per (expert, gate|up, TPS output tiles): x*suh -> H128 in threadgroup memory, EXL3 trellis GEMV
//      over all k; plus the shared down GEMV (8-bit) on silu(g)*u.
//   D3 fn_moe_down: per (expert, 128 output columns, KS k-splits): finish gate/up (H128, svh), SwiGLU, * down suh,
//      H128, down trellis GEMV, H128 * svh * score; the last of the 10 experts per column block sums them in fixed
//      order + sigmoid(shared gate) * shared, and applies the hyper-connection write:
//      hout[s*HID + c] = h[s*HID + c] + branch[c] * inj[s].
// No rmsnorm here: in Flash-Next the MoE input is the HC mix (the hc_norm is the block's norm).
// R rows: router / shared expert read their weights once for all rows; the routed part runs R * TOPK (row, slot)
// pairs, each reading its expert (no union dedupe yet: rows sharing an expert read it twice).
// Pack layout (K5): trellis gate/up U16 [512][160][40][16K], down [512][40][160][16K]; suh/svh F16; router BF16.
#include "ne_common.h"

constant constexpr uint HID = 2560u, INTER = 640u, NE = 512u, TOPK = 10u, HC = 4u;
constant constexpr uint TPS = 2u;  // D2 output tiles per threadgroup
constant constexpr uint KS = 2u;   // D3 k splits
constant constexpr uint NRT = (NE + 1u + 7u) / 8u;  // router threadgroups (8 rows each over NE + 1 rows)
constant constexpr uint NST = 2u * INTER / 8u;      // shared gate/up threadgroups
constant constexpr uint OTG_ = 40u / TPS;           // (= OTG below: gate/up output tile groups per (expert, proj))

// ------------------------------------------------------------------------------------------------ D1
// grid NRT + NST threadgroups, 256 threads. x [R][HID]; logits [R][NE + 1]; gush [R][2 INTER]; route [R][20].
template <uint R>
kernel void fn_moe_router(device const float *x [[buffer(0)]], device const ushort *rw [[buffer(1)]],
                          device const ushort *sgw [[buffer(2)]],
                          device const uchar *gw [[buffer(3)]], device const ushort *gs [[buffer(4)]], device const ushort *gbi [[buffer(5)]],
                          device const uchar *uw [[buffer(6)]], device const ushort *us [[buffer(7)]], device const ushort *ubi [[buffer(8)]],
                          device atomic_float *logits [[buffer(9)]], device float *gush [[buffer(10)]],
                          device uint *route [[buffer(11)]], device atomic_uint *ctr [[buffer(12)]],
                          device uint *uq [[buffer(13)]], constant uint &split [[buffer(14)]],
                          uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float cval[80];
  threadgroup uint uids[R * TOPK], ulead[R * TOPK], ucnt[R * TOPK];
  threadgroup uint cidx[80];
  threadgroup uint last;
  if (tg >= NRT) {  // shared expert gate / up rows
    const uint row = (tg - NRT) * 8u + sg;
    const bool up = row >= INTER;
    const uint r = up ? row - INTER : row;
    float v[R];
    if (up) q8rows_dev<HID, 128u, R>(uw + (size_t)r * HID, us + (size_t)r * 20u, ubi + (size_t)r * 20u, x, lane, v);
    else q8rows_dev<HID, 128u, R>(gw + (size_t)r * HID, gs + (size_t)r * 20u, gbi + (size_t)r * 20u, x, lane, v);
    if (lane == 0u) for (uint t = 0u; t < R; t++) gush[t * 2u * INTER + row] = rr(v[t]);
    return;
  }
  const uint row = tg * 8u + sg;
  if (row <= NE) {
    const device uint4 *w4 = (const device uint4 *)(row < NE ? rw + (size_t)row * HID : sgw);
    float acc[R];
    for (uint t = 0u; t < R; t++) acc[t] = 0.0f;
    for (uint j = 0u; j < HID / 256u; j++) {
      const uint c = lane + 32u * j;
      const uint4 u = w4[c];
      for (uint t = 0u; t < R; t++) {
        const device float4 *x4 = (const device float4 *)(x + t * HID);
        acc[t] += dot8bf(u, x4[2u * c], x4[2u * c + 1u]);
      }
    }
    for (uint t = 0u; t < R; t++) {
      const float a = rr(simd_sum(acc[t]));  // reference: bf16 Linear outputs
      if (lane == 0u) atomic_store_explicit(&logits[t * (NE + 1u) + row], a, memory_order_relaxed);
    }
  }
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device);
  if (lid == 0u) last = atomic_fetch_add_explicit(&ctr[0], 1u, memory_order_relaxed) == NRT - 1u ? 1u : 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (last == 0u) return;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  for (uint t = 0u; t < R; t++) {
    device atomic_float *lg = logits + t * (NE + 1u);
    {  // per simdgroup: top-10 of its 64 logits
      const uint ia = sg * 64u + lane, ib = ia + 32u;
      float a = atomic_load_explicit(&lg[ia], memory_order_relaxed);
      float b = atomic_load_explicit(&lg[ib], memory_order_relaxed);
      for (uint k = 0u; k < TOPK; k++) {
        const float m = max(a, b);
        const uint im = (a >= b) ? ia : ib;
        const float mm = simd_max(m);
        const uint win = simd_min(m == mm ? im : 0xFFFFFFFFu);
        if (lane == 0u) { cval[sg * TOPK + k] = mm; cidx[sg * TOPK + k] = win; }
        if (ia == win) a = -INFINITY;
        if (ib == win) b = -INFINITY;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      float v[3];
      uint ix[3];
      for (uint j = 0u; j < 3u; j++) {
        const uint c = lane + 32u * j;
        v[j] = c < 80u ? cval[c] : -INFINITY;
        ix[j] = c < 80u ? cidx[c] : 0xFFFFFFFFu;
      }
      float myv = 0.0f, top = 0.0f;
      uint myi = 0u;
      for (uint k = 0u; k < TOPK; k++) {
        const float m = max(v[0], max(v[1], v[2]));
        const uint im = (v[0] == m) ? ix[0] : ((v[1] == m) ? ix[1] : ix[2]);
        const float mm = simd_max(m);
        const uint win = simd_min(m == mm ? im : 0xFFFFFFFFu);
        if (k == 0u) top = mm;
        if (lane == k) { myv = mm; myi = win; }
        for (uint j = 0u; j < 3u; j++) if (ix[j] == win) v[j] = -INFINITY;
      }
      // softmax over all 512 then top-k renormalised == softmax over the top-k logits
      const float p = lane < TOPK ? exp(myv - top) : 0.0f;
      const float s = simd_sum(p);
      // lab-moegroup (fn-prof only): split bits 8-15 = a percentage of rows 1.. R-1's slots moved to other experts (the
      // synthetic rows route alike; this sets their overlap to real text's); 0 in every engine recording
      // (a moved slot that lands on an expert the row already uses stays: a row's 10 experts stay distinct, as the
      // dedupe kernels' slot nibbles need)
      const uint spread = (split >> 8u) & 0xFFu;
      if (spread != 0u && t > 0u) {
        const bool mv = lane < TOPK && ((t * 7919u + lane * 104729u + (myi & 63u) * 31u) % 100u) < spread;
        uint f = mv ? (myi + 97u * t + 13u * lane + 1u) & (NE - 1u) : myi;
        for (uint pass = 0u; pass < 3u; pass++) {
          bool clash = false;
          for (uint j = 0u; j < TOPK; j++) {
            const uint fj = simd_shuffle(f, (ushort)j);
            clash = clash || (j != lane && fj == f);
          }
          if (mv && clash) f = myi;
        }
        myi = f;
      }
      if (lane < TOPK) { route[t * 20u + lane] = myi & (NE - 1u); route[t * 20u + TOPK + lane] = as_type<uint>(p / s); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (R > 1u) {  // unique experts over the R rows: uq = [nu, then per u: id, slot nibble per row (bits 4r..4r+3, 0xF none; R <= 8), R scores]
    threadgroup_barrier(mem_flags::mem_device);
    const uint i = lid, r = i / TOPK, k = i % TOPK;
    uint id = 0u;
    if (i < R * TOPK) { id = route[r * 20u + k]; uids[i] = id; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint lead = i, u = 0u;
    if (i < R * TOPK) {
      for (uint j = 0u; j < i; j++) if (uids[j] == id) { lead = j; break; }
      ulead[i] = lead;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i < R * TOPK) {
      for (uint j = 0u; j < lead; j++) u += (ulead[j] == j) ? 1u : 0u;  // leaders before mine = my unique index
      if (lead == i) {
        device uint *e = uq + 1u + u * (2u + R);
        e[0] = id;
        e[1] = 0xFFFFFFFFu;
      }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (i < R * TOPK) {
      device uint *e = uq + 1u + u * (2u + R);
      atomic_fetch_and_explicit((device atomic_uint *)(e + 1), (k << (4u * r)) | ~(0xFu << (4u * r)), memory_order_relaxed);
      e[2u + r] = route[r * 20u + TOPK + k];
    }
    if (i == R * TOPK - 1u) {
      uint nu = 0u;
      for (uint j = 0u; j < R * TOPK; j++) nu += (ulead[j] == j) ? 1u : 0u;
      uq[0] = nu;
    }
    threadgroup_barrier(mem_flags::mem_device);
    // rows that do not use expert u: score 0 (never read: their slot nibble is 0xF)
    // lab-verify split lists at X = uq + 1 + R TOPK (2 + R): [ns, nm, singles: es of experts used by one row [R TOPK],
    // multis: u of experts used by >= 2 rows [R TOPK]] (fn_moe_gateup_s / fn_moe_gateup_m, NE_FN_MOE_SPLIT)
    if ((split & 0xFFu) != 0u) {  // (only when the split kernels run: a constant, so the barriers stay uniform)
    if (i < R * TOPK) {
      uint c = 0u;
      for (uint j = 0u; j < R * TOPK; j++) c += (ulead[j] == lead) ? 1u : 0u;
      ucnt[i] = c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device uint *X = uq + 1u + R * TOPK * (2u + R);
    if ((split & 0xFFu) == 2u) {  // lab-moegroup: per-expert tiles of <= 4 rows (fn_moe_gateup_g / fn_moe_down_g; layout there)
      device uint *G = X + 2u + 2u * R * TOPK;
      // tiles of expert lead j with c rows: c / 4 tiles of 4, then the remainder c % 4 (1 -> class 0, 2 -> class 1,
      // 3 -> class 2 padded); classes count in leader order
      if (i < R * TOPK) {
        uint p1 = 0u, p2 = 0u, p4 = 0u, rank = 0u;
        for (uint j = 0u; j < lead; j++) {
          if (ulead[j] != j) continue;
          const uint c = ucnt[j], rm = c % 4u;
          p4 += c / 4u + (rm == 3u ? 1u : 0u);
          p2 += rm == 2u ? 1u : 0u;
          p1 += rm == 1u ? 1u : 0u;
        }
        for (uint j = lead; j < i; j++) rank += ulead[j] == lead ? 1u : 0u;
        const uint c = ucnt[i], full = c / 4u, rm = c % 4u, t = rank / 4u, pos = rank % 4u;
        const uint es = r * TOPK + k;
        if (t < full) G[4u + 2u * R * TOPK + (p4 + t) * 4u + pos] = es;
        else if (rm == 1u) G[4u + p1] = es;
        else if (rm == 2u) G[4u + R * TOPK + p2 * 2u + pos] = es;
        else {
          G[4u + 2u * R * TOPK + (p4 + full) * 4u + pos] = es;
          if (pos == 0u) G[4u + 2u * R * TOPK + (p4 + full) * 4u + 3u] = 0xFFFFFFFFu;
        }
      }
      if (i == 0u) {
        uint n1 = 0u, n2 = 0u, n4 = 0u;
        for (uint j = 0u; j < R * TOPK; j++) {
          if (ulead[j] != j) continue;
          const uint c = ucnt[j], rm = c % 4u;
          n4 += c / 4u + (rm == 3u ? 1u : 0u);
          n2 += rm == 2u ? 1u : 0u;
          n1 += rm == 1u ? 1u : 0u;
        }
        G[0] = n1; G[1] = n2; G[2] = n4;
        // the six passes' threadgroup counts (MTLDispatchThreadgroupsIndirectArguments) after the tiles: live tiles only
        device uint *A = G + 4u + 4u * R * TOPK;
        const uint nt[3] = {n1, n2, n4};
        for (uint c = 0u; c < 3u; c++) {
          A[3u * c] = max(nt[c] * 2u * OTG_, 1u); A[3u * c + 1u] = 1u; A[3u * c + 2u] = 1u;
          A[9u + 3u * c] = max(nt[c] * (HID / 128u), 1u); A[9u + 3u * c + 1u] = 1u; A[9u + 3u * c + 2u] = 1u;
        }
      }
    } else {
    if (i < R * TOPK) {
      if (ucnt[i] == 1u) {
        uint p = 0u;
        for (uint j = 0u; j < i; j++) p += (ucnt[j] == 1u) ? 1u : 0u;
        X[2u + p] = i;
      } else if (lead == i) {
        uint p = 0u;
        for (uint j = 0u; j < i; j++) p += (ulead[j] == j && ucnt[j] > 1u) ? 1u : 0u;
        X[2u + R * TOPK + p] = u;
      }
    }
    if (i == 0u) {
      uint ns = 0u, nm = 0u;
      for (uint j = 0u; j < R * TOPK; j++) { ns += (ucnt[j] == 1u) ? 1u : 0u; nm += (ulead[j] == j && ucnt[j] > 1u) ? 1u : 0u; }
      X[0] = ns;
      X[1] = nm;
    }
    }
    threadgroup_barrier(mem_flags::mem_device);
    }
  }
  if (lid == 0u) atomic_store_explicit(&ctr[0], 0u, memory_order_relaxed);
}

// ------------------------------------------------------------------------------------------------ D2
constant constexpr uint OTG = 40u / TPS;       // gate/up output tile groups per (expert, proj)
// grid R * TOPK * 2 * OTG expert threadgroups + HID/8 shared-down threadgroups, 256 threads. guo [R*TOPK][2][INTER],
// sho [R][HID].
template <int K, uint R, bool SP = false>
kernel void fn_moe_gateup(device const float *x [[buffer(0)]], device const uint *gt [[buffer(1)]],
                          device const uint *ut [[buffer(2)]], device const half *gsu [[buffer(3)]],
                          device const half *usu [[buffer(4)]], device const uint *route [[buffer(5)]],
                          device float *guo [[buffer(6)]], device const float *gush [[buffer(7)]],
                          device const uchar *sdw [[buffer(8)]], device const ushort *sds [[buffer(9)]],
                          device const ushort *sdb [[buffer(10)]], device float *sho [[buffer(11)]],
                          device const uint *uq [[buffer(12)]],
                          uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint NW = 8u * uint(K);
  constexpr uint NXT = R * TOPK * 2u * OTG;
  threadgroup half prep[HID];
  threadgroup float part[8u * TPS * 16u];
  threadgroup float hsh[R * INTER];
  if (tg >= NXT) {  // shared down rows (8-bit g128) on silu(gate) * up
    for (uint i = lid; i < R * INTER; i += 256u) {
      const uint t = i / INTER, j = i % INTER;
      hsh[i] = rr(silu(gush[t * 2u * INTER + j]) * gush[t * 2u * INTER + INTER + j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row = (tg - NXT) * 8u + sg;
    const device uint4 *w4 = (const device uint4 *)(sdw + (size_t)row * INTER);
    float acc[R];
    for (uint t = 0u; t < R; t++) acc[t] = 0.0f;
    for (uint c = lane; c < INTER / 16u; c += 32u) {
      const uint g = c >> 3u;
      const float sc = bf(sds[row * 5u + g]), bi = bf(sdb[row * 5u + g]);
      const uint4 u = w4[c];
      for (uint t = 0u; t < R; t++) {
        const threadgroup float4 *h4 = (const threadgroup float4 *)(hsh + t * INTER);
        const float2 d = dot16q(u, h4[4u * c], h4[4u * c + 1u], h4[4u * c + 2u], h4[4u * c + 3u]);
        acc[t] += sc * d.x + bi * d.y;
      }
    }
    for (uint t = 0u; t < R; t++) {
      const float a = rr(simd_sum(acc[t]));
      if (lane == 0u) sho[t * HID + row] = a;
    }
    return;
  }
  const uint slot = tg / OTG, otg = tg % OTG;
  uint es = slot >> 1u;  // es = row * TOPK + k
  const uint proj = slot & 1u;
  if (SP) {  // lab-verify split: the single-row experts' (row, slot) pairs only
    const device uint *X = uq + 1u + R * TOPK * (2u + R);
    if (es >= X[0]) return;
    es = X[2u + es];
  }
  const uint xr = es / TOPK;
  const uint eid = route[xr * 20u + es % TOPK] & (NE - 1u);
  x += xr * HID;
  const device uint *wm = (proj ? ut : gt) + (size_t)eid * (160u * 40u) * NW;
  uint3 cur[TPS];
  for (uint t = 0u; t < TPS; t++) cur[t] = lane_load<K>(wm + (size_t)(sg * 40u + otg * TPS + t) * NW, lane);
  const device half *su = (proj ? usu : gsu) + (size_t)eid * HID;
  for (uint b = sg; b < HID / 128u; b += 8u) {
    const uint o = b * 128u + lane;
    float4 v = float4(x[o] * float(su[o]), x[o + 32u] * float(su[o + 32u]), x[o + 64u] * float(su[o + 64u]),
                      x[o + 96u] * float(su[o + 96u]));
    v = fwht128(v, lane);
    prep[kp(o)] = half(v.x); prep[kp(o + 32u)] = half(v.y); prep[kp(o + 64u)] = half(v.z); prep[kp(o + 96u)] = half(v.w);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup const half4 *p4 = (threadgroup const half4 *)prep;
  const uint q = lane & 3u, col = lane >> 2u;
  float acc[TPS][2];
  for (uint t = 0u; t < TPS; t++) { acc[t][0] = 0.0f; acc[t][1] = 0.0f; }
  for (uint tk = sg; tk < 160u; tk += 8u) {
    uint3 nxt[TPS];
    const uint tn = min(tk + 8u, 159u);
    for (uint t = 0u; t < TPS; t++) nxt[t] = lane_load<K>(wm + (size_t)(tn * 40u + otg * TPS + t) * NW, lane);
    const float4 x4 = float4(p4[tk * 4u + q]);
    for (uint t = 0u; t < TPS; t++) {
      float4 w0, w1;
      dec8<K>(cur[t], lane, w0, w1);
      acc[t][0] += dot(x4, w0);
      acc[t][1] += dot(x4, w1);
      cur[t] = nxt[t];
    }
  }
  for (uint t = 0u; t < TPS; t++)
    for (uint j = 0u; j < 2u; j++) {
      float v = acc[t][j];
      v += simd_shuffle_xor(v, 1u);
      v += simd_shuffle_xor(v, 2u);
      if (q == 0u) part[(sg * TPS + t) * 16u + j * 8u + col] = v;
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < TPS * 16u) {
    float s = 0.0f;
    for (uint g = 0u; g < 8u; g++) s += part[g * TPS * 16u + lid];
    guo[((size_t)es * 2u + proj) * INTER + otg * TPS * 16u + lid] = s;
  }
}

// ------------------------------------------------------------------------------------------------ D3
// grid R * TOPK * (HID/128) threadgroups, 256 * KS threads (8 simdgroups = the 8 output tiles of a 128-column block,
// times KS k-splits). scr [R][HID/128][TOPK][128], ctr [R][HID/128]; h / hout [R][HC][HID], inj [R][8], branch [R][HID].
template <int K, uint R>
kernel void fn_moe_down(device const float *guo [[buffer(0)]], device const uint *dtr [[buffer(1)]],
                        device const half *gsv [[buffer(2)]], device const half *usv [[buffer(3)]],
                        device const half *dsu [[buffer(4)]], device const half *dsv [[buffer(5)]],
                        device const uint *route [[buffer(6)]], device const float *logits [[buffer(7)]],
                        device const float *sho [[buffer(8)]], device const float *h [[buffer(9)]],
                        device const float *inj [[buffer(10)]], device atomic_float *scr [[buffer(11)]],
                        device atomic_uint *ctr [[buffer(12)]], device float *hout [[buffer(13)]],
                        device float *branch [[buffer(14)]],
                        uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint NW = 8u * uint(K);
  constexpr uint KTS = 40u / KS;  // k tiles per simdgroup
  threadgroup half prep[INTER];
  threadgroup float o128[128u * KS];
  const uint es = tg % (R * TOPK), blk = tg / (R * TOPK);  // es = row * TOPK + k
  const uint xr = es / TOPK, k = es % TOPK;
  const uint eid = route[xr * 20u + k] & (NE - 1u);
  const device uint *wm = dtr + (size_t)eid * (40u * 160u) * NW;
  const uint ot = blk * 8u + (sg & 7u);
  const uint k0 = (sg >> 3u) * KTS;
  uint3 cur = lane_load<K>(wm + (size_t)(k0 * 160u + ot) * NW, lane);
  if (sg < INTER / 128u) {
    const uint o = sg * 128u + lane;
    const device float *g = guo + (size_t)es * 2u * INTER;  // es = row * TOPK + k
    const device float *u = g + INTER;
    const device half *sg_ = gsv + (size_t)eid * INTER;
    const device half *su_ = usv + (size_t)eid * INTER;
    const device half *sd = dsu + (size_t)eid * INTER;
    float4 gv = fwht128(float4(g[o], g[o + 32u], g[o + 64u], g[o + 96u]), lane) *
                float4(float(sg_[o]), float(sg_[o + 32u]), float(sg_[o + 64u]), float(sg_[o + 96u]));
    float4 uv = fwht128(float4(u[o], u[o + 32u], u[o + 64u], u[o + 96u]), lane) *
                float4(float(su_[o]), float(su_[o + 32u]), float(su_[o + 64u]), float(su_[o + 96u]));
    float4 hh = gv / (1.0f + exp(-gv)) * uv * float4(float(sd[o]), float(sd[o + 32u]), float(sd[o + 64u]), float(sd[o + 96u]));
    hh = fwht128(hh, lane);
    prep[kp(o)] = half(hh.x); prep[kp(o + 32u)] = half(hh.y); prep[kp(o + 64u)] = half(hh.z); prep[kp(o + 96u)] = half(hh.w);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup const half4 *p4 = (threadgroup const half4 *)prep;
  const uint q = lane & 3u, col = lane >> 2u;
  float a0 = 0.0f, a1 = 0.0f;
  for (uint tk = k0; tk < k0 + KTS; tk++) {
    const uint tn = min(tk + 1u, 39u);
    const uint3 nxt = lane_load<K>(wm + (size_t)(tn * 160u + ot) * NW, lane);
    const float4 x4 = float4(p4[tk * 4u + q]);
    float4 w0, w1;
    dec8<K>(cur, lane, w0, w1);
    a0 += dot(x4, w0);
    a1 += dot(x4, w1);
    cur = nxt;
  }
  a0 += simd_shuffle_xor(a0, 1u); a0 += simd_shuffle_xor(a0, 2u);
  a1 += simd_shuffle_xor(a1, 1u); a1 += simd_shuffle_xor(a1, 2u);
  if (q == 0u) { o128[sg * 16u + col] = a0; o128[sg * 16u + col + 8u] = a1; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0u) return;
  for (uint j = 0u; j < 4u; j++) {
    const uint c = lane + 32u * j;
    float t = o128[c];
    for (uint s2 = 1u; s2 < KS; s2++) t += o128[s2 * 128u + c];
    o128[c] = t;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  const uint c0 = blk * 128u + lane;
  const float sc = as_type<float>(route[xr * 20u + TOPK + k]);
  const device half *sv = dsv + (size_t)eid * HID;
  float4 v = fwht128(float4(o128[lane], o128[lane + 32u], o128[lane + 64u], o128[lane + 96u]), lane) *
             float4(float(sv[c0]), float(sv[c0 + 32u]), float(sv[c0 + 64u]), float(sv[c0 + 96u])) * sc;
  const uint rb = xr * (HID / 128u) + blk;  // (row, column block)
  // the finisher's inputs, loaded before the stores + fence (lab-speed: a load issued after them waits a round trip
  // on the block's critical path); every threadgroup loads them, only the last of the 10 uses them
  const float sgt = sigm(logits[xr * (NE + 1u) + NE]);
  const device float *ip = inj + xr * 2u * HC;
  const float4 ij = float4(ip[0], ip[1], ip[2], ip[3]);
  const device float *hr = h + xr * HC * HID;
  float hv[4][HC], shv[4];
  for (uint j = 0u; j < 4u; j++) {
    const uint c = blk * 128u + lane + 32u * j;
    shv[j] = sho[xr * HID + c];
    for (uint st = 0u; st < HC; st++) hv[j][st] = hr[st * HID + c];
  }
  device atomic_float *sp = scr + ((size_t)rb * TOPK + k) * 128u;
  atomic_store_explicit(&sp[lane], v.x, memory_order_relaxed);
  atomic_store_explicit(&sp[lane + 32u], v.y, memory_order_relaxed);
  atomic_store_explicit(&sp[lane + 64u], v.z, memory_order_relaxed);
  atomic_store_explicit(&sp[lane + 96u], v.w, memory_order_relaxed);
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  uint old = 0u;
  if (lane == 0u) old = atomic_fetch_add_explicit(&ctr[rb], 1u, memory_order_relaxed);
  old = simd_broadcast(old, 0);
  if (old != TOPK - 1u) return;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  device atomic_float *sb = scr + (size_t)rb * TOPK * 128u;
  device float *hor = hout + xr * HC * HID;
  for (uint j = 0u; j < 4u; j++) {
    const uint c = blk * 128u + lane + 32u * j;
    float s = 0.0f;
    for (uint e = 0u; e < TOPK; e++) s += atomic_load_explicit(&sb[e * 128u + lane + 32u * j], memory_order_relaxed);
    // reference: routed out, the gated shared term, the branch, the injection and the sum are bf16
    const float br = rr(rr(s) + rr(rr(sgt) * shv[j]));
    branch[xr * HID + c] = br;
    for (uint st = 0u; st < HC; st++) hor[st * HID + c] = rr(hv[j][st] + rr(br * ij[st]));
  }
  if (lane == 0u) atomic_store_explicit(&ctr[rb], 0u, memory_order_relaxed);
}

// ------------------------------------------------------------------------------------------------ D2, D3 deduped
// R > 1: the routed part runs over the nu unique experts of the R rows (uq, built by the router's finisher); each
// threadgroup decodes its expert's weights once and dots them with every row that routes to it.
// D2u grid R*TOPK*2*OTG + HID/8 threadgroups (u >= nu exit), 256 threads; guo [R*TOPK][R][2][INTER].
template <int K, uint R, bool SP = false>
kernel void fn_moe_gateup_u(device const float *x [[buffer(0)]], device const uint *gt [[buffer(1)]],
                            device const uint *ut [[buffer(2)]], device const half *gsu [[buffer(3)]],
                            device const half *usu [[buffer(4)]], device const uint *uq [[buffer(5)]],
                            device float *guo [[buffer(6)]], device const float *gush [[buffer(7)]],
                            device const uchar *sdw [[buffer(8)]], device const ushort *sds [[buffer(9)]],
                            device const ushort *sdb [[buffer(10)]], device float *sho [[buffer(11)]],
                            uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint NW = 8u * uint(K);
  constexpr uint NXT = R * TOPK * 2u * OTG;
  // lab-verify: the expert threadgroups' prep and the shared-down threadgroups' hsh share one buffer (R = 5: 25.6 +
  // 12.8 KB would exceed 32 KB)
  threadgroup half prepb[R * HID];
  threadgroup float part[8u * TPS * 16u];
  threadgroup float *hsh = (threadgroup float *)prepb;
  if (tg >= NXT) {  // shared down rows (8-bit g128) on silu(gate) * up, all rows
    for (uint i = lid; i < R * INTER; i += 256u) {
      const uint t = i / INTER, j = i % INTER;
      hsh[i] = rr(silu(gush[t * 2u * INTER + j]) * gush[t * 2u * INTER + INTER + j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row = (tg - NXT) * 8u + sg;
    const device uint4 *w4 = (const device uint4 *)(sdw + (size_t)row * INTER);
    float acc[R];
    for (uint t = 0u; t < R; t++) acc[t] = 0.0f;
    for (uint c = lane; c < INTER / 16u; c += 32u) {
      const uint g = c >> 3u;
      const float sc = bf(sds[row * 5u + g]), bi = bf(sdb[row * 5u + g]);
      const uint4 u = w4[c];
      for (uint t = 0u; t < R; t++) {
        const threadgroup float4 *h4 = (const threadgroup float4 *)(hsh + t * INTER);
        const float2 d = dot16q(u, h4[4u * c], h4[4u * c + 1u], h4[4u * c + 2u], h4[4u * c + 3u]);
        acc[t] += sc * d.x + bi * d.y;
      }
    }
    for (uint t = 0u; t < R; t++) {
      const float a = rr(simd_sum(acc[t]));
      if (lane == 0u) sho[t * HID + row] = a;
    }
    return;
  }
  const uint slot = tg / OTG, otg = tg % OTG;
  uint u = slot >> 1u;
  const uint proj = slot & 1u;
  if (SP) {  // lab-verify split: the multi-row experts only; results go to the (row, slot) layout of fn_moe_down
    const device uint *X = uq + 1u + R * TOPK * (2u + R);
    if (u >= X[1]) return;
    u = X[2u + R * TOPK + u];
  } else if (u >= uq[0]) return;
  const device uint *e = uq + 1u + u * (2u + R);
  const uint eid = e[0] & (NE - 1u);
  const uint kmask = e[1];
  const device uint *wm = (proj ? ut : gt) + (size_t)eid * (160u * 40u) * NW;
  uint3 cur[TPS];
  for (uint t = 0u; t < TPS; t++) cur[t] = lane_load<K>(wm + (size_t)(sg * 40u + otg * TPS + t) * NW, lane);
  const device half *su = (proj ? usu : gsu) + (size_t)eid * HID;
  for (uint r = 0u; r < R; r++) {
    if (((kmask >> (4u * r)) & 0xFu) == 0xFu) continue;
    const device float *xr = x + r * HID;
    for (uint b = sg; b < HID / 128u; b += 8u) {
      const uint o = b * 128u + lane;
      float4 v = float4(xr[o] * float(su[o]), xr[o + 32u] * float(su[o + 32u]), xr[o + 64u] * float(su[o + 64u]),
                        xr[o + 96u] * float(su[o + 96u]));
      v = fwht128(v, lane);
      threadgroup half *pr = prepb + r * HID;
      pr[kp(o)] = half(v.x); pr[kp(o + 32u)] = half(v.y); pr[kp(o + 64u)] = half(v.z); pr[kp(o + 96u)] = half(v.w);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint q = lane & 3u, col = lane >> 2u;
  float acc[TPS][R][2];
  for (uint t = 0u; t < TPS; t++) for (uint r = 0u; r < R; r++) { acc[t][r][0] = 0.0f; acc[t][r][1] = 0.0f; }
  for (uint tk = sg; tk < 160u; tk += 8u) {
    uint3 nxt[TPS];
    const uint tn = min(tk + 8u, 159u);
    for (uint t = 0u; t < TPS; t++) nxt[t] = lane_load<K>(wm + (size_t)(tn * 40u + otg * TPS + t) * NW, lane);
    float4 x4[R];
    for (uint r = 0u; r < R; r++) x4[r] = float4(((threadgroup const half4 *)(prepb + r * HID))[tk * 4u + q]);
    for (uint t = 0u; t < TPS; t++) {
      float4 w0, w1;
      dec8<K>(cur[t], lane, w0, w1);
      for (uint r = 0u; r < R; r++) { acc[t][r][0] += dot(x4[r], w0); acc[t][r][1] += dot(x4[r], w1); }
      cur[t] = nxt[t];
    }
  }
  for (uint r = 0u; r < R; r++) {
    if (((kmask >> (4u * r)) & 0xFu) == 0xFu) continue;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0u; t < TPS; t++)
      for (uint j = 0u; j < 2u; j++) {
        float v = acc[t][r][j];
        v += simd_shuffle_xor(v, 1u);
        v += simd_shuffle_xor(v, 2u);
        if (q == 0u) part[(sg * TPS + t) * 16u + j * 8u + col] = v;
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < TPS * 16u) {
      float sum = 0.0f;
      for (uint g = 0u; g < 8u; g++) sum += part[g * TPS * 16u + lid];
      const size_t gi = SP ? ((size_t)(r * TOPK + ((kmask >> (4u * r)) & 0xFu)) * 2u + proj) : (((size_t)u * R + r) * 2u + proj);
      guo[gi * INTER + otg * TPS * 16u + lid] = sum;
    }
  }
}

// D3u grid R*TOPK * (HID/128) threadgroups (u >= nu exit), 256 * KS threads. Per row r of expert u: the result goes
// to scr[(r, blk)][slot k_r(u)]; the last of the 10 experts of row r and block blk sums them in slot order
// (deterministic) + sigmoid(shared gate) * shared, and writes the HC update of row r.
template <int K, uint R>
kernel void fn_moe_down_u(device const float *guo [[buffer(0)]], device const uint *dtr [[buffer(1)]],
                          device const half *gsv [[buffer(2)]], device const half *usv [[buffer(3)]],
                          device const half *dsu [[buffer(4)]], device const half *dsv [[buffer(5)]],
                          device const uint *uq [[buffer(6)]], device const float *logits [[buffer(7)]],
                          device const float *sho [[buffer(8)]], device const float *h [[buffer(9)]],
                          device const float *inj [[buffer(10)]], device atomic_float *scr [[buffer(11)]],
                          device atomic_uint *ctr [[buffer(12)]], device float *hout [[buffer(13)]],
                          device float *branch [[buffer(14)]],
                          uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint NW = 8u * uint(K);
  constexpr uint KTS = 40u / KS;
  threadgroup half prep[R][INTER];
  threadgroup float o128[R][128u * KS];
  const uint u = tg % (R * TOPK), blk = tg / (R * TOPK);
  if (u >= uq[0]) return;
  const device uint *e = uq + 1u + u * (2u + R);
  const uint eid = e[0] & (NE - 1u);
  const uint kmask = e[1];
  const device uint *wm = dtr + (size_t)eid * (40u * 160u) * NW;
  const uint ot = blk * 8u + (sg & 7u);
  const uint k0 = (sg >> 3u) * KTS;
  uint3 cur = lane_load<K>(wm + (size_t)(k0 * 160u + ot) * NW, lane);
  if (sg < INTER / 128u) {
    const uint o = sg * 128u + lane;
    const device half *sg_ = gsv + (size_t)eid * INTER;
    const device half *su_ = usv + (size_t)eid * INTER;
    const device half *sd = dsu + (size_t)eid * INTER;
    for (uint r = 0u; r < R; r++) {
      if (((kmask >> (4u * r)) & 0xFu) == 0xFu) continue;
      const device float *g = guo + ((size_t)u * R + r) * 2u * INTER;
      const device float *uu = g + INTER;
      float4 gv = fwht128(float4(g[o], g[o + 32u], g[o + 64u], g[o + 96u]), lane) *
                  float4(float(sg_[o]), float(sg_[o + 32u]), float(sg_[o + 64u]), float(sg_[o + 96u]));
      float4 uv = fwht128(float4(uu[o], uu[o + 32u], uu[o + 64u], uu[o + 96u]), lane) *
                  float4(float(su_[o]), float(su_[o + 32u]), float(su_[o + 64u]), float(su_[o + 96u]));
      float4 hh = gv / (1.0f + exp(-gv)) * uv * float4(float(sd[o]), float(sd[o + 32u]), float(sd[o + 64u]), float(sd[o + 96u]));
      hh = fwht128(hh, lane);
      prep[r][kp(o)] = half(hh.x); prep[r][kp(o + 32u)] = half(hh.y); prep[r][kp(o + 64u)] = half(hh.z); prep[r][kp(o + 96u)] = half(hh.w);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint q = lane & 3u, col = lane >> 2u;
  float a0[R], a1[R];
  for (uint r = 0u; r < R; r++) { a0[r] = 0.0f; a1[r] = 0.0f; }
  for (uint tk = k0; tk < k0 + KTS; tk++) {
    const uint tn = min(tk + 1u, 39u);
    const uint3 nxt = lane_load<K>(wm + (size_t)(tn * 160u + ot) * NW, lane);
    float4 w0, w1;
    dec8<K>(cur, lane, w0, w1);
    for (uint r = 0u; r < R; r++) {
      const float4 x4 = float4(((threadgroup const half4 *)prep[r])[tk * 4u + q]);
      a0[r] += dot(x4, w0);
      a1[r] += dot(x4, w1);
    }
    cur = nxt;
  }
  for (uint r = 0u; r < R; r++) {
    float b0 = a0[r], b1 = a1[r];
    b0 += simd_shuffle_xor(b0, 1u); b0 += simd_shuffle_xor(b0, 2u);
    b1 += simd_shuffle_xor(b1, 1u); b1 += simd_shuffle_xor(b1, 2u);
    if (q == 0u) { o128[r][sg * 16u + col] = b0; o128[r][sg * 16u + col + 8u] = b1; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0u) return;
  const device half *sv = dsv + (size_t)eid * HID;
  const uint c0 = blk * 128u + lane;
  for (uint r = 0u; r < R; r++) {
    const uint k = (kmask >> (4u * r)) & 0xFu;
    if (k == 0xFu) continue;
    float4 ov;
    for (uint j = 0u; j < 4u; j++) {
      float t = o128[r][lane + 32u * j];
      for (uint s2 = 1u; s2 < KS; s2++) t += o128[r][s2 * 128u + lane + 32u * j];
      ov[j] = t;
    }
    const float sc = as_type<float>(e[2u + r]);
    float4 v = fwht128(ov, lane) * float4(float(sv[c0]), float(sv[c0 + 32u]), float(sv[c0 + 64u]), float(sv[c0 + 96u])) * sc;
    const uint rb = r * (HID / 128u) + blk;
    device atomic_float *sp = scr + ((size_t)rb * TOPK + k) * 128u;
    atomic_store_explicit(&sp[lane], v.x, memory_order_relaxed);
    atomic_store_explicit(&sp[lane + 32u], v.y, memory_order_relaxed);
    atomic_store_explicit(&sp[lane + 64u], v.z, memory_order_relaxed);
    atomic_store_explicit(&sp[lane + 96u], v.w, memory_order_relaxed);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    uint old = 0u;
    if (lane == 0u) old = atomic_fetch_add_explicit(&ctr[rb], 1u, memory_order_relaxed);
    old = simd_broadcast(old, 0);
    if (old != TOPK - 1u) continue;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    const float sgt = sigm(logits[r * (NE + 1u) + NE]);
    device atomic_float *sb = scr + (size_t)rb * TOPK * 128u;
    const device float *ip = inj + r * 2u * HC;
    const float4 ij = float4(ip[0], ip[1], ip[2], ip[3]);
    const device float *hr = h + r * HC * HID;
    device float *hor = hout + r * HC * HID;
    for (uint j = 0u; j < 4u; j++) {
      const uint c = blk * 128u + lane + 32u * j;
      float s = 0.0f;
      for (uint ee = 0u; ee < TOPK; ee++) s += atomic_load_explicit(&sb[ee * 128u + lane + 32u * j], memory_order_relaxed);
      const float br = rr(rr(s) + rr(rr(sgt) * sho[r * HID + c]));
      branch[r * HID + c] = br;
      for (uint st = 0u; st < HC; st++) hor[st * HID + c] = rr(hr[st * HID + c] + rr(br * ij[st]));
    }
    if (lane == 0u) atomic_store_explicit(&ctr[rb], 0u, memory_order_relaxed);
  }
}


// ------------------------------------------------------------------------------------------------ D2, D3 grouped
// lab-moegroup: the routed experts of an R-row step grouped by expert, so each touched expert's weights are read once
// per tile of <= M rows (M = 1, 2, 4) instead of once per (row, slot) pair. The router's finisher (split == 2) writes
// the tiles at G = uq + 1 + R TOPK (2 + R) + 2 + 2 R TOPK: G[0..2] = tile counts of M = 1 / 2 / 4, then the tiles of
// class c (c = 0 / 1 / 2) at G + 4 + c R TOPK, M entries es = row * TOPK + slot each (0xFFFFFFFF = padding, M = 4
// with 3 rows). An expert used by c rows: c / 4 tiles of 4 (+ one tile of the remainder: 1 -> M 1, 2 -> M 2, 3 ->
// M 4 padded). Splash 1.4 (incoai/splash, Apache-2.0, PRs #454-#456) groups its MoE rows into per-expert tiles the
// same way and runs small tiles as column GEMVs; this is our GEMV kernels' version of that idea.
// Per row the arithmetic is fn_moe_gateup's / fn_moe_down's (the same prep, each simdgroup the same k tiles in the
// same order, the same reductions and finisher), so the outputs are bit-identical to the per-(row, slot) path.
template <uint M>
static inline constexpr uint gcls() { return M == 1u ? 0u : (M == 2u ? 1u : 2u); }
// D2g grid ntiles_max(M) * 2 * OTG threadgroups (ti >= G[c] exit), 256 threads; guo [R*TOPK][2][INTER] (per pair).
// PH > 1: the k range in PH phases (prep of 20 / PH 128-blocks per row, then their k tiles; each simdgroup still runs
// its k tiles tk = sg, sg + 8, ... in order): threadgroup memory M * 5 KB / PH.
template <int K, uint M, uint PH = 1u>
kernel void fn_moe_gateup_g(device const float *x [[buffer(0)]], device const uint *gt [[buffer(1)]],
                            device const uint *ut [[buffer(2)]], device const half *gsu [[buffer(3)]],
                            device const half *usu [[buffer(4)]], device const uint *route [[buffer(5)]],
                            device float *guo [[buffer(6)]], device const uint *G [[buffer(7)]],
                            constant uint &R [[buffer(8)]],
                            uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint NW = 8u * uint(K);
  constexpr uint PART = 8u * TPS * 16u;
  constexpr uint KB = HID / PH, BP = HID / 128u / PH, TP = 160u / PH;  // per phase: halves, 128-blocks, k tiles
  constexpr bool AL = M > 1u && M * PART * 2u <= M * KB;  // M > 1: the M rows' partial sums alias prep after the k loop
  threadgroup half prep[M * KB];
  threadgroup float part1[AL ? 1u : M * PART];
  threadgroup float *part = AL ? (threadgroup float *)prep : part1;
  const uint slot = tg / OTG, otg = tg % OTG;
  const uint ti = slot >> 1u, proj = slot & 1u;
  if (ti >= G[gcls<M>()]) return;
  const device uint *te = G + 4u + gcls<M>() * R * TOPK + ti * M;
  uint es[M];
  for (uint j = 0u; j < M; j++) es[j] = te[j];
  const uint eid = route[(es[0] / TOPK) * 20u + es[0] % TOPK] & (NE - 1u);
  const device uint *wm = (proj ? ut : gt) + (size_t)eid * (160u * 40u) * NW;
  uint3 cur[TPS];
  for (uint t = 0u; t < TPS; t++) cur[t] = lane_load<K>(wm + (size_t)(sg * 40u + otg * TPS + t) * NW, lane);
  const device half *su = (proj ? usu : gsu) + (size_t)eid * HID;
  const uint q = lane & 3u, col = lane >> 2u;
  float acc[TPS][M][2];
  for (uint t = 0u; t < TPS; t++) for (uint j = 0u; j < M; j++) { acc[t][j][0] = 0.0f; acc[t][j][1] = 0.0f; }
  for (uint ph = 0u; ph < PH; ph++) {
    if (ph > 0u) threadgroup_barrier(mem_flags::mem_threadgroup);  // the last phase's prep read by every simdgroup
    for (uint j = 0u; j < M; j++) {
      if (es[j] == 0xFFFFFFFFu) continue;
      const device float *xr = x + (es[j] / TOPK) * HID;
      threadgroup half *pr = prep + j * KB - ph * KB;
      for (uint b = ph * BP + sg; b < (ph + 1u) * BP; b += 8u) {
        const uint o = b * 128u + lane;
        float4 v = float4(xr[o] * float(su[o]), xr[o + 32u] * float(su[o + 32u]), xr[o + 64u] * float(su[o + 64u]),
                          xr[o + 96u] * float(su[o + 96u]));
        v = fwht128(v, lane);
        pr[kp(o)] = half(v.x); pr[kp(o + 32u)] = half(v.y); pr[kp(o + 64u)] = half(v.z); pr[kp(o + 96u)] = half(v.w);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint tk = ph * TP + sg; tk < (ph + 1u) * TP; tk += 8u) {
      uint3 nxt[TPS];
      const uint tn = min(tk + 8u, 159u);
      for (uint t = 0u; t < TPS; t++) nxt[t] = lane_load<K>(wm + (size_t)(tn * 40u + otg * TPS + t) * NW, lane);
      float4 x4[M];
      for (uint j = 0u; j < M; j++) x4[j] = float4(((threadgroup const half4 *)(prep + j * KB))[(tk - ph * TP) * 4u + q]);
      for (uint t = 0u; t < TPS; t++) {
        float4 w0, w1;
        dec8<K>(cur[t], lane, w0, w1);
        for (uint j = 0u; j < M; j++) { acc[t][j][0] += dot(x4[j], w0); acc[t][j][1] += dot(x4[j], w1); }
        cur[t] = nxt[t];
      }
    }
  }
  if (AL) threadgroup_barrier(mem_flags::mem_threadgroup);  // every simdgroup done with prep (part aliases it)
  for (uint j = 0u; j < M; j++)
    for (uint t = 0u; t < TPS; t++)
      for (uint i = 0u; i < 2u; i++) {
        float v = acc[t][j][i];
        v += simd_shuffle_xor(v, 1u);
        v += simd_shuffle_xor(v, 2u);
        if (q == 0u) part[j * PART + (sg * TPS + t) * 16u + i * 8u + col] = v;
      }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < M * TPS * 16u) {
    const uint j = lid / (TPS * 16u), c = lid % (TPS * 16u);
    if (es[j] != 0xFFFFFFFFu) {
      float s = 0.0f;
      for (uint g = 0u; g < 8u; g++) s += part[j * PART + g * TPS * 16u + c];
      guo[((size_t)es[j] * 2u + proj) * INTER + otg * TPS * 16u + c] = s;
    }
  }
}

// D3g grid ntiles_max(M) * (HID/128) threadgroups (tile-major; ti >= G[c] exit), 256 * KS threads. Row j of the tile
// is finished by simdgroup j: its result goes to scr[(row, blk)][slot], and the last of the row's 10 experts for the
// block sums them in slot order + the gated shared term and writes the row's HC update (= fn_moe_down's finisher).
template <int K, uint M>
kernel void fn_moe_down_g(device const float *guo [[buffer(0)]], device const uint *dtr [[buffer(1)]],
                          device const half *gsv [[buffer(2)]], device const half *usv [[buffer(3)]],
                          device const half *dsu [[buffer(4)]], device const half *dsv [[buffer(5)]],
                          device const uint *route [[buffer(6)]], device const float *logits [[buffer(7)]],
                          device const float *sho [[buffer(8)]], device const float *h [[buffer(9)]],
                          device const float *inj [[buffer(10)]], device atomic_float *scr [[buffer(11)]],
                          device atomic_uint *ctr [[buffer(12)]], device float *hout [[buffer(13)]],
                          device float *branch [[buffer(14)]], device const uint *G [[buffer(15)]],
                          constant uint &R [[buffer(16)]],
                          uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint NW = 8u * uint(K);
  constexpr uint KTS = 40u / KS;
  threadgroup half prep[M][INTER];
  threadgroup float o128[M][128u * KS];
  const uint ti = tg / (HID / 128u), blk = tg % (HID / 128u);
  if (ti >= G[gcls<M>()]) return;
  const device uint *te = G + 4u + gcls<M>() * R * TOPK + ti * M;
  uint es[M];
  for (uint j = 0u; j < M; j++) es[j] = te[j];
  const uint eid = route[(es[0] / TOPK) * 20u + es[0] % TOPK] & (NE - 1u);
  const device uint *wm = dtr + (size_t)eid * (40u * 160u) * NW;
  const uint ot = blk * 8u + (sg & 7u);
  const uint k0 = (sg >> 3u) * KTS;
  uint3 cur = lane_load<K>(wm + (size_t)(k0 * 160u + ot) * NW, lane);
  for (uint task = sg; task < M * (INTER / 128u); task += 2u * 8u) {  // (row j, 128-block) prep tasks
    const uint j = task / (INTER / 128u), b = task % (INTER / 128u);
    if (es[j] == 0xFFFFFFFFu) continue;
    const uint o = b * 128u + lane;
    const device float *g = guo + (size_t)es[j] * 2u * INTER;
    const device float *u = g + INTER;
    const device half *sg_ = gsv + (size_t)eid * INTER;
    const device half *su_ = usv + (size_t)eid * INTER;
    const device half *sd = dsu + (size_t)eid * INTER;
    float4 gv = fwht128(float4(g[o], g[o + 32u], g[o + 64u], g[o + 96u]), lane) *
                float4(float(sg_[o]), float(sg_[o + 32u]), float(sg_[o + 64u]), float(sg_[o + 96u]));
    float4 uv = fwht128(float4(u[o], u[o + 32u], u[o + 64u], u[o + 96u]), lane) *
                float4(float(su_[o]), float(su_[o + 32u]), float(su_[o + 64u]), float(su_[o + 96u]));
    float4 hh = gv / (1.0f + exp(-gv)) * uv * float4(float(sd[o]), float(sd[o + 32u]), float(sd[o + 64u]), float(sd[o + 96u]));
    hh = fwht128(hh, lane);
    prep[j][kp(o)] = half(hh.x); prep[j][kp(o + 32u)] = half(hh.y); prep[j][kp(o + 64u)] = half(hh.z); prep[j][kp(o + 96u)] = half(hh.w);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint q = lane & 3u, col = lane >> 2u;
  float a0[M], a1[M];
  for (uint j = 0u; j < M; j++) { a0[j] = 0.0f; a1[j] = 0.0f; }
  for (uint tk = k0; tk < k0 + KTS; tk++) {
    const uint tn = min(tk + 1u, 39u);
    const uint3 nxt = lane_load<K>(wm + (size_t)(tn * 160u + ot) * NW, lane);
    float4 x4[M];
    for (uint j = 0u; j < M; j++) x4[j] = float4(((threadgroup const half4 *)prep[j])[tk * 4u + q]);
    float4 w0, w1;
    dec8<K>(cur, lane, w0, w1);
    for (uint j = 0u; j < M; j++) { a0[j] += dot(x4[j], w0); a1[j] += dot(x4[j], w1); }
    cur = nxt;
  }
  for (uint j = 0u; j < M; j++) {
    float b0 = a0[j], b1 = a1[j];
    b0 += simd_shuffle_xor(b0, 1u); b0 += simd_shuffle_xor(b0, 2u);
    b1 += simd_shuffle_xor(b1, 1u); b1 += simd_shuffle_xor(b1, 2u);
    if (q == 0u) { o128[j][sg * 16u + col] = b0; o128[j][sg * 16u + col + 8u] = b1; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg >= M) return;
  const uint j = sg;
  if (es[j] == 0xFFFFFFFFu) return;
  const uint xr = es[j] / TOPK, k = es[j] % TOPK;
  float4 ov;
  for (uint i = 0u; i < 4u; i++) {
    float t = o128[j][lane + 32u * i];
    for (uint s2 = 1u; s2 < KS; s2++) t += o128[j][s2 * 128u + lane + 32u * i];
    ov[i] = t;
  }
  const uint c0 = blk * 128u + lane;
  const float sc = as_type<float>(route[xr * 20u + TOPK + k]);
  const device half *sv = dsv + (size_t)eid * HID;
  float4 v = fwht128(ov, lane) * float4(float(sv[c0]), float(sv[c0 + 32u]), float(sv[c0 + 64u]), float(sv[c0 + 96u])) * sc;
  const uint rb = xr * (HID / 128u) + blk;
  const float sgt = sigm(logits[xr * (NE + 1u) + NE]);
  const device float *ip = inj + xr * 2u * HC;
  const float4 ij = float4(ip[0], ip[1], ip[2], ip[3]);
  const device float *hr = h + xr * HC * HID;
  float hv[4][HC], shv[4];
  for (uint i = 0u; i < 4u; i++) {
    const uint c = blk * 128u + lane + 32u * i;
    shv[i] = sho[xr * HID + c];
    for (uint st = 0u; st < HC; st++) hv[i][st] = hr[st * HID + c];
  }
  device atomic_float *sp = scr + ((size_t)rb * TOPK + k) * 128u;
  atomic_store_explicit(&sp[lane], v.x, memory_order_relaxed);
  atomic_store_explicit(&sp[lane + 32u], v.y, memory_order_relaxed);
  atomic_store_explicit(&sp[lane + 64u], v.z, memory_order_relaxed);
  atomic_store_explicit(&sp[lane + 96u], v.w, memory_order_relaxed);
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  uint old = 0u;
  if (lane == 0u) old = atomic_fetch_add_explicit(&ctr[rb], 1u, memory_order_relaxed);
  old = simd_broadcast(old, 0);
  if (old != TOPK - 1u) return;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  device atomic_float *sb = scr + (size_t)rb * TOPK * 128u;
  device float *hor = hout + xr * HC * HID;
  for (uint i = 0u; i < 4u; i++) {
    const uint c = blk * 128u + lane + 32u * i;
    float s = 0.0f;
    for (uint e = 0u; e < TOPK; e++) s += atomic_load_explicit(&sb[e * 128u + lane + 32u * i], memory_order_relaxed);
    const float br = rr(rr(s) + rr(rr(sgt) * shv[i]));
    branch[xr * HID + c] = br;
    for (uint st = 0u; st < HC; st++) hor[st * HID + c] = rr(hv[i][st] + rr(br * ij[st]));
  }
  if (lane == 0u) atomic_store_explicit(&ctr[rb], 0u, memory_order_relaxed);
}

#define NE_INST(name, func, ...) template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;
#define NE_INST_MOEU(R)                                     \
  NE_INST("fn_moe_gateup_u_k4_r" #R, fn_moe_gateup_u, 4, R)  \
  NE_INST("fn_moe_gateup_u_k5_r" #R, fn_moe_gateup_u, 5, R)  \
  NE_INST("fn_moe_gateup_u_k6_r" #R, fn_moe_gateup_u, 6, R)  \
  NE_INST("fn_moe_down_u_k4_r" #R, fn_moe_down_u, 4, R)      \
  NE_INST("fn_moe_down_u_k5_r" #R, fn_moe_down_u, 5, R)      \
  NE_INST("fn_moe_down_u_k6_r" #R, fn_moe_down_u, 6, R)
NE_INST_MOEU(2)
NE_INST_MOEU(3)
NE_INST_MOEU(4)
NE_INST_MOEU(5)
#define NE_INST_MOEG(M)                                     \
  NE_INST("fn_moe_gateup_g_k4_m" #M, fn_moe_gateup_g, 4, M)  \
  NE_INST("fn_moe_gateup_g_k5_m" #M, fn_moe_gateup_g, 5, M)  \
  NE_INST("fn_moe_gateup_g_k6_m" #M, fn_moe_gateup_g, 6, M)  \
  NE_INST("fn_moe_down_g_k4_m" #M, fn_moe_down_g, 4, M)      \
  NE_INST("fn_moe_down_g_k5_m" #M, fn_moe_down_g, 5, M)      \
  NE_INST("fn_moe_down_g_k6_m" #M, fn_moe_down_g, 6, M)
NE_INST_MOEG(1)
NE_INST_MOEG(2)
NE_INST_MOEG(4)
#define NE_INST_MOEGP(M, P)                                                \
  NE_INST("fn_moe_gateup_g_k4_m" #M "_p" #P, fn_moe_gateup_g, 4, M, P)  \
  NE_INST("fn_moe_gateup_g_k5_m" #M "_p" #P, fn_moe_gateup_g, 5, M, P)  \
  NE_INST("fn_moe_gateup_g_k6_m" #M "_p" #P, fn_moe_gateup_g, 6, M, P)
NE_INST_MOEGP(2, 2)
NE_INST_MOEGP(4, 2)
NE_INST_MOEGP(4, 4)
#define NE_INST_MOES(R)                                               \
  NE_INST("fn_moe_gateup_s_k4_r" #R, fn_moe_gateup, 4, R, true)       \
  NE_INST("fn_moe_gateup_s_k5_r" #R, fn_moe_gateup, 5, R, true)       \
  NE_INST("fn_moe_gateup_s_k6_r" #R, fn_moe_gateup, 6, R, true)       \
  NE_INST("fn_moe_gateup_m_k4_r" #R, fn_moe_gateup_u, 4, R, true)     \
  NE_INST("fn_moe_gateup_m_k5_r" #R, fn_moe_gateup_u, 5, R, true)     \
  NE_INST("fn_moe_gateup_m_k6_r" #R, fn_moe_gateup_u, 6, R, true)
NE_INST_MOES(2)
NE_INST_MOES(3)
NE_INST_MOES(4)
NE_INST_MOES(5)
#define NE_INST_MOE(R)                                     \
  NE_INST("fn_moe_router_r" #R, fn_moe_router, R)          \
  NE_INST("fn_moe_gateup_k4_r" #R, fn_moe_gateup, 4, R)    \
  NE_INST("fn_moe_gateup_k5_r" #R, fn_moe_gateup, 5, R)    \
  NE_INST("fn_moe_gateup_k6_r" #R, fn_moe_gateup, 6, R)    \
  NE_INST("fn_moe_down_k4_r" #R, fn_moe_down, 4, R)        \
  NE_INST("fn_moe_down_k5_r" #R, fn_moe_down, 5, R)        \
  NE_INST("fn_moe_down_k6_r" #R, fn_moe_down, 6, R)
NE_INST_MOE(1)
NE_INST_MOE(2)
NE_INST_MOE(3)
NE_INST_MOE(4)
NE_INST_MOE(5)
NE_INST_MOE(6)
NE_INST_MOE(8)
