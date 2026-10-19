"""EXL3 trunk encode / decode for one Linear (lab-trunk6). Used by tools/convert_trunk.py and tests/trunk6/.

Encode = lead 5's recipe (the lab's trunk/enc.py, itself after exllamav3's
quantize_exl3, MIT, turboderp): random signs, output / input channel scales (input scales floored at 1e-3 x median for
dead channels), Hadamard rotations (128, else the largest power of two <= 64 dividing the dimension: 64 for the HC 320,
16 for the GDN a/b 48), global-scale search, block LDL of the rotated damped FULL Hessian, LDLQ over 16-row blocks,
tail-biting Viterbi per 16x16 tile (MCG window 15, Metal kernel through MLX), 2-round scale refit in the H metric,
suh / svh rounded to f16. Unlike lab enc.py it keeps the codewords and packs them (Sushi / exllamav3 bit layout,
exl3/convert/exl3_codec.pack_codewords).

Format of one encoded Linear (y = x @ W, W [in, out] = HF weight^T):
  trellis U16 [in/16, out/16, 16 K]  tile (tk, tn) = W'[16 tk : 16 tk + 16, 16 tn : 16 tn + 16] (tensor-core slot order)
  suh     F16 [in]                   input scales (signs, channel scales, global scale folded in)
  svh     F16 [out]                  output scales
  W = diag(suh) . H_hi . W' . H_ho . diag(svh), H_n = block-diagonal normalised Sylvester Hadamard of size n,
  hi = had_size(in), ho = had_size(out).
The reference modules are imported read-only (sys.path, IJAH_REF_DIR); MIT notices in docs/trunk6.md and
THIRD_PARTY_NOTICES.md.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ijah_paths  # noqa: E402

# The EXL3 reference modules are not part of this repository: IJAH_REF_DIR/convert (exl3_codec, encode) and
# IJAH_REF_DIR/trunk (viterbi_lut_metal, encode only).
LAB = ijah_paths.ref_dir() or Path("/nonexistent-IJAH_REF_DIR")
for p in (str(LAB / "convert"), str(LAB / "trunk")):
    if p not in sys.path:
        sys.path.append(p)

import numpy as np  # noqa: E402

try:
    import exl3_codec as C  # noqa: E402  (reference: codec, tile permutation, bit pack)
    from encode import block_ldl, block_rms, refit_scales, sample_scale_tiles  # noqa: E402  (reference convert/)
except ImportError as e:  # noqa: F841
    raise ImportError("tools/convert_trunk_enc.py needs the EXL3 reference modules (exl3_codec, encode) under "
                      "IJAH_REF_DIR/convert; they are not part of this repository") from e

WINDOW = 15
_HAD = {}


def had(n):
    if n not in _HAD:
        i = np.arange(n)
        x = i[:, None] & i[None, :]
        pc = np.zeros((n, n), np.int64)
        while x.any():
            pc += x & 1
            x = x >> 1
        _HAD[n] = np.where(pc % 2 == 0, 1.0, -1.0) / np.sqrt(n)
    return _HAD[n]


def had_size(d):
    for n in (128, 64, 32, 16):
        if d % n == 0:
            return n
    raise ValueError(d)


def had_l(w, n):
    k, m = w.shape
    return np.einsum("ij,bjm->bim", had(n), w.reshape(k // n, n, m)).reshape(k, m)


def had_r(w, n):
    k, m = w.shape
    return (w.reshape(k, m // n, n) @ had(n)).reshape(k, m)


_TABLE = None


def mcg_table():
    """f32 [2^15]: decoded MCG value of every window-15 codeword (f16-rounded, as the decoder produces)."""
    global _TABLE
    if _TABLE is None:
        _TABLE = np.ascontiguousarray(C.lut("mcg", WINDOW)[: 1 << WINDOW].astype(np.float32))
    return _TABLE


def viterbi(tiles, K):
    """tiles f32 [T, 256] slot order -> (codewords u16 [T, 256], values f32 [T, 256]). GPU (MLX Metal kernel)."""
    if os.environ.get("T6_VITERBI", "metal") == "numpy":
        return C.viterbi_tiles(tiles.astype(np.float32), K, "mcg", WINDOW)
    from viterbi_lut_metal import viterbi_lut_metal  # lab engine/trunk (port of exllamav3's kernel, MIT)
    return viterbi_lut_metal(np.ascontiguousarray(tiles, np.float32), mcg_table(), K, WINDOW)


def g_scale_search(w, K):
    tiles = sample_scale_tiles(w)

    def mse(s, t):
        return float(np.mean((viterbi(t * s, K)[1] / s - t) ** 2))

    coarse = [0.1 + 0.2 * i for i in range(10)]
    m1 = [mse(s, tiles[::3]) for s in coarse]
    c = coarse[int(np.argmin(m1))]
    step = 0.075
    fine = [c + step * (i - 2) for i in range(5)]
    m2 = [mse(s, tiles) for s in fine]
    b = int(np.argmin(m2))
    off = 0.0
    if 0 < b < 4:
        y0, y1, y2 = m2[b - 1], m2[b], m2[b + 1]
        den = y0 - 2 * y1 + y2
        off = max(-0.5, min(0.5, 0.5 * (y0 - y2) / den)) if den > 0 else 0.0
    return max(fine[b] + off * step, 0.01)


def quant_rows(rows, K):
    """rows [16, out] -> (quantized rows [16, out] f64, codewords [out/16, 256] u16 slot order)."""
    tiles = C.tiles_from_inner(rows.astype(np.float32))[0]
    cw, v = viterbi(tiles, K)
    t = np.empty_like(v)
    t[:, C.PERM] = v
    return t.reshape(-1, 16, 16).transpose(1, 0, 2).reshape(16, -1).astype(np.float64), cw


def reconstruct(q, su, sv):
    hk, hm = had_size(q.shape[0]), had_size(q.shape[1])
    w = had_l(q.astype(np.float64), hk) * su.astype(np.float64)[:, None]
    return had_r(w, hm) * sv.astype(np.float64)[None, :]


def encode(W, H, K, seed=0, sigma_reg=0.025):
    """W [in, out] (f64), H [in, in] full Hessian (X^T X / rows). Returns dict(trellis u16 [in/16, out/16, 16K],
    suh f16 [in], svh f16 [out], g_scale, W_hat f64 [in, out] = the exact decoded weight)."""
    k, m = W.shape
    hk, hm = had_size(k), had_size(m)
    rng = np.random.default_rng(seed)
    su = np.where(rng.standard_normal(k) >= 0, 1.0, -1.0)
    sv = np.where(rng.standard_normal(m) >= 0, 1.0, -1.0)
    w = W.astype(np.float64).copy()
    ocs = block_rms(w, 0).ravel()
    ocs /= ocs.mean()
    ocs[ocs < 1e-30] = 0.1
    sv = sv * ocs + 1e-10
    w /= sv[None, :]
    w = had_r(w, hm)
    ics = block_rms(w, 1).ravel()
    ics[ics < 1e-30] = 0.1
    ics = np.maximum(ics, 1e-3 * np.median(ics))   # lead 5: dead input channels (GDN out_proj L1/L2)
    su = su * ics / (-C.CODEBOOK_SCALE) + 1e-10
    w /= su[:, None]
    w = had_l(w, hk)
    g = g_scale_search(w.astype(np.float32), K)
    w *= g
    su /= g
    Hd = H.astype(np.float64).copy()
    Hd[np.diag_indices(k)] += sigma_reg * np.mean(np.diag(Hd))
    Hr = had_l(had_r(Hd * su[None, :] * su[:, None], hk), hk)
    L = block_ldl(Hr)
    del Hr
    q = np.zeros_like(w)
    cws = np.zeros((k // 16, m // 16, 256), np.uint16)
    for bj in range(k, 0, -16):
        bi = bj - 16
        rows = w[bi:bj].copy()
        if bj < k:
            rows += L[bj:, bi:bj].T @ (w[bj:] - q[bj:])
        q[bi:bj], cws[bi // 16] = quant_rows(rows, K)
    del L
    Wq = reconstruct(q, su, sv)
    Wq, su, sv = refit_scales(W.astype(np.float64), Wq, Hd, su, sv)
    suh, svh = su.astype(np.float16), sv.astype(np.float16)
    trellis = C.pack_codewords(cws, 16 * K)
    return dict(trellis=trellis, suh=suh, svh=svh, g_scale=float(g), W_hat=reconstruct(q, suh, svh))


def decode(trellis, suh, svh):
    """Independent decode of a packed Linear -> W_hat [in, out] f64 (the kernels' reference)."""
    n = trellis.shape[-1]
    inner = C.decode_inner(trellis, n, "mcg", WINDOW)
    return reconstruct(inner, suh, svh)


def proxy_err(W, Q, H):
    E = (Q - W).astype(np.float64)
    return float((E * (H @ E)).sum() / (W * (H @ W)).sum())
