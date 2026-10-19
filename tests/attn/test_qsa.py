#!/usr/bin/env python3
"""lab-attn: the QSA attention step (kernels/attn/qsa.*) vs lab-ref's goldens and numpy references.

    $PY tests/attn/test_qsa.py [--lib build-attn/libengine.dylib] [--fmt bf16,kv8,f32] [--only golden,rows,rollback,kv8]

Checks (fed the goldens' own projections q_proj / k_proj / v_proj / idx_qk and caches *_in):
  golden    L03_decode1, L03_verify4, L03_long_decode1, L03_long_verify4: indexer queries, block keys, scores,
            selected set (tie-aware), K / V cache rows, gated attention output (attn.gated)
  rows      R = 2, 3 (prefix rows of L03_long_verify4 vs its goldens) and R = 5..8 (synthetic rows vs the numpy
            reference built from tools/opref.py)
  rollback  verify4 -> truncate -> decode1 equals a fresh decode1 bit for bit; truncate inside a completed block and
            re-append different tokens (numpy reference)
  kv8       the kv8 cache vs a numpy port of MLX affine_quantize, and the kv8 attention vs numpy on the dequantized
            cache (exact path check), plus its distance to the bf16 goldens
Light GPU work (no model layers): no machine lock needed. Exit status 1 on any failure.
"""
import argparse
import ctypes as C
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import refenv  # noqa: E402
import golden  # noqa: E402
import compare  # noqa: E402
import opref as O  # noqa: E402
import engine as E  # noqa: E402
import numpy as np  # noqa: E402

FMTS = {"f32": 0, "bf16": 1, "kv8": 2}
PACK = Path(refenv.ijah_paths.golden_pack())
LAYER = 3
f32p = C.POINTER(C.c_float)


def ptr(a):
    return a.ctypes.data_as(f32p) if a is not None else None


class Attn:
    def __init__(self, eng, pack, fmt, tmax=4096, layer=LAYER):
        L = eng.lib
        self.eng = eng
        sig = {
            "ne_attn_create": (C.c_void_p, [C.c_void_p, C.c_int, C.c_int, C.c_int]),
            "ne_attn_free": (None, [C.c_void_p]),
            "ne_attn_import": (C.c_int, [C.c_void_p, f32p, f32p, f32p, C.c_int]),
            "ne_attn_export": (C.c_int, [C.c_void_p, C.c_int, f32p, f32p, f32p, f32p]),
            "ne_attn_truncate": (C.c_int, [C.c_void_p, C.c_int]),
            "ne_attn_pos": (C.c_int, [C.c_void_p]),
            "ne_attn_step": (C.c_int, [C.c_void_p, C.c_int, f32p, f32p, f32p, f32p, C.POINTER(C.c_double)]),
            "ne_attn_read": (C.c_int, [C.c_void_p, C.c_char_p, C.c_void_p, C.c_int]),
            "ne_attn_time": (C.c_int, [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
                                       C.POINTER(C.c_double)]),
        }
        for n, (r, a) in sig.items():
            f = getattr(L, n)
            f.restype, f.argtypes = r, a
        self.L = L
        self.tmax = tmax
        self.h = L.ne_attn_create(pack.ptr, layer, FMTS[fmt], tmax)
        if not self.h:
            raise E.EngineError(eng.last_error())

    def ok(self, rc, what):
        if rc != 0:
            raise E.EngineError(f"{what}: {self.eng.last_error()}")

    def import_(self, k, v, ik, T):
        k = np.ascontiguousarray(k, np.float32) if k is not None else None
        v = np.ascontiguousarray(v, np.float32) if v is not None else None
        ik = np.ascontiguousarray(ik, np.float32) if ik is not None else None
        self.ok(self.L.ne_attn_import(self.h, ptr(k), ptr(v), ptr(ik), T), "import")

    def export(self, n):
        k = np.zeros((2, n, 256), np.float32)
        v = np.zeros((2, n, 256), np.float32)
        ik = np.zeros((n, 128), np.float32)
        bk = np.zeros((n // 4, 128), np.float32)
        self.ok(self.L.ne_attn_export(self.h, n, ptr(k), ptr(v), ptr(ik), ptr(bk)), "export")
        return k, v, ik, bk

    def step(self, q, kv, iqk):
        R = q.shape[0]
        out = np.zeros((R, 6144), np.float32)
        ms = C.c_double()
        q, kv, iqk = (np.ascontiguousarray(x, np.float32) for x in (q, kv, iqk))
        self.ok(self.L.ne_attn_step(self.h, R, ptr(q), ptr(kv), ptr(iqk), ptr(out), C.byref(ms)), "step")
        return out

    def read(self, name, dtype=np.float32):
        n = self.L.ne_attn_read(self.h, name.encode(), None, 0)
        a = np.zeros(n, dtype)
        self.ok(0 if self.L.ne_attn_read(self.h, name.encode(), a.ctypes.data, n) == n else -1, "read " + name)
        return a

    def truncate(self, T):
        self.ok(self.L.ne_attn_truncate(self.h, T), "truncate")

    def pos(self):
        return self.L.ne_attn_pos(self.h)

    def time(self, R, which=-1, reps1=8, reps2=40, flush_mb=0):
        us = C.c_double()
        self.ok(self.L.ne_attn_time(self.h, R, which, reps1, reps2, flush_mb, C.byref(us)), "time")
        return us.value

    def free(self):
        if self.h:
            self.L.ne_attn_free(self.h)
            self.h = None


# ------------------------------------------------------------------ numpy reference (tools/opref.py)
# rope inverse frequencies exactly as MLX computes them on the GPU (kernels/attn/qsa.mm kInvFreqBits); numpy's f32
# table differs by one ulp in 11 of 32 entries, which moves angles by ~1e-2 rad at 128K positions.
MLX_INV_FREQ = np.array([int(x, 16) for x in "0x3f800000,0x3f1ab32b,0x3ebaf81a,0x3e61f835,0x3e088d77,0x3da50956,0x3d47763e,0x3cf11177,0x3c91ad39,0x3c301051,0x3bd4ca14,0x3b80967d,0x3b1b690d,0x3abbd3ed,0x3a6301e0,0x3a092e02,0x39a5cb60,0x394860c1,0x38f22ce2,0x3892587e,0x3830df51,0x37d5c441,0x37812dab,0x371c1fc3,0x36bcb0c1,0x36640cc6,0x3609cf4a,0x35a68e4b,0x35494c57,0x34f3499b,0x3493048d,0x3431af43".split(",")], np.uint32).view(np.float32)


class Ref:
    def __init__(self, layer=LAYER):
        O.inv_freq = lambda rot_dim=64, theta=1e7: MLX_INV_FREQ  # noqa: E731 (opref.rope reads it)
        self.P = O.PackNP()
        a = f"language_model.model.layers.{layer}.self_attn."
        self.qn = self.P.norm(a + "q_norm.weight")
        self.kn = self.P.norm(a + "k_norm.weight")
        self.iqn = self.P.norm(a + "indexer.q_layernorm.weight")
        self.ikn = self.P.norm(a + "indexer.k_layernorm.weight")

    def select(self, IK, T, iqk):
        """Indexer only: (iq [R,4,128], block keys, scores [R, nb], selected sets)."""
        R = iqk.shape[0]
        pos = np.arange(T, T + R)
        g = O.bf16(iqk).reshape(R, 5, 128)
        iq = O.rope(O.rmsnorm(g[:, :4], self.iqn).transpose(1, 0, 2), pos)
        bk = O.qsa_block_keys(np.concatenate([IK[:T], g[:, 4]], 0), self.ikn)
        sc = O.qsa_scores(iq, bk)
        sets, _ = O.qsa_select(sc, T)
        return iq, bk, sc, sets

    def step(self, K, V, IK, T, q, kv, iqk, quant=None):
        """K, V [2, T, 256] (roped keys), IK [T, 128] raw keys; rows q [R,12288], kv [R,1024], iqk [R,640].
        quant: a function applied to the full K / V before attention (kv8 emulation). Returns dict."""
        R = q.shape[0]
        pos = np.arange(T, T + R)
        g = O.bf16(iqk).reshape(R, 5, 128)
        iq = O.rope(O.rmsnorm(g[:, :4], self.iqn).transpose(1, 0, 2), pos)          # [4, R, 128]
        raw = np.concatenate([IK[:T], g[:, 4]], 0)
        bk = O.qsa_block_keys(raw, self.ikn)
        sc = O.qsa_scores(iq, bk)
        sets, allowed = O.qsa_select(sc, T)
        qp = O.bf16(q).reshape(R, 24, 512)
        qr = O.rope(O.rmsnorm(qp[..., :256], self.qn).transpose(1, 0, 2), pos)        # [24, R, 256]
        kk = O.rope(O.rmsnorm(O.bf16(kv[:, :512]).reshape(R, 2, 256), self.kn).transpose(1, 0, 2), pos)
        vv = O.bf16(kv[:, 512:]).reshape(R, 2, 256).transpose(1, 0, 2)
        K2 = np.concatenate([K[:, :T], kk], 1)
        V2 = np.concatenate([V[:, :T], vv], 1)
        Ka, Va = (quant(K2), quant(V2)) if quant else (K2, V2)
        core = O.attention(qr, Ka, Va, pos, allowed)
        gated = O.bf16(core * O.sigmoid_bf(qp[..., 256:].reshape(R, -1)))
        return dict(iq=iq.transpose(1, 0, 2), bk=bk, scores=sc, sets=sets, qr=qr.transpose(1, 0, 2), K=K2, V=V2,
                    gated=gated)


def _rnd(x):  # metal round(): half away from zero
    return np.sign(x) * np.floor(np.abs(x) + np.float32(0.5))


def mlx_q8(x):
    """numpy port of MLX affine_quantize (bits 8, group 64) + dequantize with bf16 scales / biases. x [..., 256]."""
    s = x.shape
    w = np.asarray(x, np.float32).reshape(-1, 64)
    wmin = w.min(1)
    wmax = np.maximum(w.max(1), np.float32(0))
    scale = np.maximum((wmax - wmin) / np.float32(255), np.float32(1e-7)).astype(np.float32)
    side = np.abs(wmin) > np.abs(wmax)
    scale = np.where(side, scale, -scale).astype(np.float32)
    edge = np.where(side, wmin, wmax).astype(np.float32)
    q0 = _rnd(edge / scale).astype(np.float32)
    at0 = q0 == 0
    scale = np.where(at0, scale, edge / np.where(at0, np.float32(1), q0)).astype(np.float32)
    bias = np.where(at0, np.float32(0), edge).astype(np.float32)
    c = np.clip(_rnd((w - bias[:, None]) / scale[:, None]), 0, 255).astype(np.float32)
    return (c * O.bf16(scale)[:, None] + O.bf16(bias)[:, None]).astype(np.float32).reshape(s)


# ------------------------------------------------------------------ checks
RES = []


def rec(name, ok, msg):
    RES.append((name, bool(ok), msg))
    print(f"  {'ok ' if ok else 'FAIL'} {name:46s} {msg}")


def cmp(name, got, ref, tol):
    rel, mx, cos = compare.metrics(np.asarray(got, np.float32), np.asarray(ref, np.float32))
    rec(name, rel <= tol, f"rel {rel:.2e} max {mx:.2e} cos {cos:.7f} (tol {tol:g})")
    return rel


def sets_check(name, ours, ref_sets, scores, T):
    """ours: list of id arrays; ref_sets: list of python sets; tie-aware via the scores [R, nb]."""
    bad = ties = 0
    for t, (a, b) in enumerate(zip(ours, ref_sets)):
        a = set(map(int, a))
        if a == b:
            continue
        comp = (T + t + 1) // 4
        kth = np.sort(scores[t, :comp])[::-1][min(511, comp - 1)]
        if all(scores[t, x] == kth for x in a ^ b):
            ties += 1
        else:
            bad += 1
    rec(name, bad == 0, f"{len(ours)} rows: {bad} differ, {ties} only in exact ties")


def our_sets(a, R):
    sel = a.read("sel", np.uint32).reshape(-1, 512)[:R]
    n = a.read("nsel", np.uint32)[:R]
    return [sel[t, :n[t]] for t in range(R)], n


def golden_inputs(c, rows=None):
    S = c.m["rows"] if rows is None else rows
    q = c["attn.q_proj"][0, :S]
    kv = np.concatenate([c["attn.k_proj"][0, :S], c["attn.v_proj"][0, :S]], 1)
    iqk = c["attn.idx_qk"][0, :S]
    return q, kv, iqk


def run_golden(eng, pack, fmt, cname, rows=None, tol_out=None):
    c = golden.load(cname)
    T = c.m["cache_len_in"]
    S = c.m["rows"] if rows is None else rows
    tag = f"{fmt} {cname}" + (f"[:{S}]" if rows else "")
    a = Attn(eng, pack, fmt, tmax=max(4096, (T + 64 + 63) // 64 * 64))
    a.import_(c["attn.k_cache_in"][0], c["attn.v_cache_in"][0], c["attn.index_keys_in"][0], T)
    out = a.step(*golden_inputs(c, S))
    iq = a.read("iq").reshape(-1, 4, 128)[:S]
    cmp(f"{tag} idx_q", iq, c["attn.idx_q"][0].transpose(1, 0, 2)[:S], 5e-3)
    qr = a.read("qr").reshape(-1, 24, 256)[:S] * 16
    cmp(f"{tag} q_rope", qr, c["attn.q_rope"][0].transpose(1, 0, 2)[:S], 5e-3)
    n = T + S
    k, v, ik, bk = a.export(n)
    nb = n // 4
    nbg = c["attn.index_block_keys_out"].shape[2]
    if nbg >= nb:
        cmp(f"{tag} block keys ({nb})", bk[:nb], c["attn.index_block_keys_out"][0, 0, :nb], 5e-3)
    else:  # the reference skips block keys at decode with <= 512 complete blocks: compare what it has
        cmp(f"{tag} block keys ({nbg} of {nb})", bk[:nbg], c["attn.index_block_keys_out"][0, 0], 5e-3)
    cmp(f"{tag} raw index keys", ik, c["attn.index_keys_out"][0, :n], 5e-3)
    sc = a.read("scores").reshape(-1, a.tmax // 4)[:S, :nb]
    gs = c["attn.idx_scores"][0, :S, :nb]
    cmp(f"{tag} block scores", sc[:, :gs.shape[1]], gs, 1e-4)
    sets, nsel = our_sets(a, S)
    if "attn.selected_blocks" in c:
        gsel = c["attn.selected_blocks"][0, :S]
        rs = [set(map(int, x)) - {-1} for x in gsel]
        sets_check(f"{tag} selected blocks", sets, rs, gs, T)
    else:  # dense decode in the reference: every complete block must be listed
        ok = all(list(sets[t]) == list(range((T + t + 1) // 4)) for t in range(S))
        rec(f"{tag} block list = causal", ok, f"nsel {list(nsel)}")
    tk = 5e-3 if fmt != "kv8" else 1e-2
    cmp(f"{tag} k cache new rows", k[:, T:n], c["attn.k_cache_out"][0, :, T:n], tk)
    cmp(f"{tag} v cache new rows", v[:, T:n], c["attn.v_cache_out"][0, :, T:n], tk)
    if fmt != "kv8":
        cmp(f"{tag} k / v cache old rows (import)", np.concatenate([k[:, :T], v[:, :T]]),
            np.concatenate([c["attn.k_cache_in"][0], c["attn.v_cache_in"][0]]), 0)
    tol = tol_out or (5e-3 if fmt != "kv8" else 2e-2)
    r = cmp(f"{tag} attn.gated", out, c["attn.gated"][0, :S], tol)
    a.free()
    return r


def run_rows(eng, pack, ref, fmt):
    c = golden.load("L03_long_verify4")
    for R in (2, 3):
        run_golden(eng, pack, fmt, "L03_long_verify4", rows=R)
    # synthetic R = 5..8: verify4 rows reused as the extra rows' projections (positions T+4 ..)
    T = c.m["cache_len_in"]
    K, V, IK = c["attn.k_cache_in"][0], c["attn.v_cache_in"][0], c["attn.index_keys_in"][0]
    q4, kv4, iqk4 = golden_inputs(c)
    rng = np.random.default_rng(0)
    for R in (5, 8):
        idx = [t % 4 for t in range(R)]
        q, kv, iqk = q4[idx], kv4[idx], iqk4[idx]
        if R > 4:  # perturb the repeated rows so they are not copies
            q = q.copy(); q[4:] *= (1 + 0.05 * rng.standard_normal(q[4:].shape)).astype(np.float32)
            iqk = iqk.copy(); iqk[4:] *= (1 + 0.05 * rng.standard_normal(iqk[4:].shape)).astype(np.float32)
        a = Attn(eng, pack, fmt, tmax=4096)
        a.import_(K, V, IK, T)
        out = a.step(q, kv, iqk)
        x = ref.step(K, V, IK, T, q, kv, iqk, quant=mlx_q8 if fmt == "kv8" else None)
        tag = f"{fmt} synthetic R={R} @ {T}"
        cmp(f"{tag} idx_q", a.read("iq").reshape(-1, 4, 128)[:R], x["iq"], 5e-3)
        n = T + R
        k, v, ik, bk = a.export(n)
        cmp(f"{tag} block keys", bk[:n // 4], x["bk"][:n // 4], 5e-3)
        sc = a.read("scores").reshape(-1, a.tmax // 4)[:R, :n // 4]
        cmp(f"{tag} scores", sc, x["scores"], 1e-4)
        sets, _ = our_sets(a, R)
        sets_check(f"{tag} selected blocks", sets, x["sets"], x["scores"], T)
        cmp(f"{tag} attn.gated", out, x["gated"], 1e-3 if fmt != "kv8" else 2e-3)
        a.free()


def run_long(eng, pack, ref, fmt, ctxs=(9001, 20000)):
    """Synthetic caches past 2048 complete blocks (the two-stage select: per-slice top 512, then the merge)."""
    rng = np.random.default_rng(7)
    c = golden.load("L03_long_verify4")
    q4, kv4, iqk4 = golden_inputs(c)
    for T in ctxs:
        K = O.bf16(rng.standard_normal((2, T, 256), dtype=np.float32))
        V = O.bf16(rng.standard_normal((2, T, 256), dtype=np.float32))
        IK = O.bf16(rng.standard_normal((T, 128), dtype=np.float32) * 2)
        for R in (1, 4):
            a = Attn(eng, pack, fmt, tmax=(T + 64 + 63) // 64 * 64)
            a.import_(K, V, IK, T)
            q, kv, iqk = q4[:R], kv4[:R], iqk4[:R]
            out = a.step(q, kv, iqk)
            x = ref.step(K, V, IK, T, q, kv, iqk, quant=mlx_q8 if fmt == "kv8" else None)
            tag = f"{fmt} synthetic T={T} R={R}"
            n = T + R
            sc = a.read("scores").reshape(-1, a.tmax // 4)[:R, :n // 4]
            cmp(f"{tag} scores", sc, x["scores"], 1e-4)
            sets, _ = our_sets(a, R)
            sets_check(f"{tag} selected blocks", sets, x["sets"], x["scores"], T)
            cmp(f"{tag} attn.gated", out, x["gated"], 1e-3 if fmt != "kv8" else 2e-3)  # kv8: half staging
            a.free()


def run_select_huge(eng, pack, ref, T=140001):
    """Past 128K tokens the merge stage has > 8192 candidates (not register-cached): selection only."""
    rng = np.random.default_rng(9)
    c = golden.load("L03_long_verify4")
    q4, kv4, iqk4 = golden_inputs(c)
    IK = O.bf16(rng.standard_normal((T, 128), dtype=np.float32) * 2)
    a = Attn(eng, pack, "kv8", tmax=(T + 64 + 63) // 64 * 64)
    a.import_(np.zeros((2, T, 256), np.float32), np.zeros((2, T, 256), np.float32), IK, T)
    a.step(q4[:2], kv4[:2], iqk4[:2])
    _, _, sc_ref, sets_ref = ref.select(IK, T, iqk4[:2])
    sc = a.read("scores").reshape(-1, a.tmax // 4)[:2, :(T + 2) // 4]
    cmp(f"synthetic T={T} R=2 scores", sc, sc_ref, 1e-4)
    sets, _ = our_sets(a, 2)
    sets_check(f"synthetic T={T} R=2 selected blocks (uncached merge)", sets, sets_ref, sc_ref, T)
    a.free()


def run_rollback(eng, pack, ref, fmt):
    cv = golden.load("L03_long_verify4")
    cd = golden.load("L03_long_decode1")
    T = cv.m["cache_len_in"]
    K, V, IK = cv["attn.k_cache_in"][0], cv["attn.v_cache_in"][0], cv["attn.index_keys_in"][0]
    a = Attn(eng, pack, fmt, tmax=4096)
    a.import_(K, V, IK, T)
    a.step(*golden_inputs(cv))
    rec(f"{fmt} rollback: pos after verify4", a.pos() == T + 4, f"pos {a.pos()}")
    a.truncate(T)
    out1 = a.step(*golden_inputs(cd))
    b = Attn(eng, pack, fmt, tmax=4096)
    b.import_(K, V, IK, T)
    out2 = b.step(*golden_inputs(cd))
    rec(f"{fmt} rollback: verify4 -> truncate -> decode1", np.array_equal(out1, out2),
        "bit-identical to a fresh decode1" if np.array_equal(out1, out2) else f"max diff {np.abs(out1 - out2).max():.3e}")
    cmp(f"{fmt} rollback: decode1 vs golden", out1, cd["attn.gated"][0], 5e-3 if fmt != "kv8" else 2e-2)
    # accept 2 of 4 (pos 2306: block 576 = tokens 2304..2307 is now incomplete), then re-append 4 other rows
    a.truncate(T)
    a.step(*golden_inputs(cv))
    a.truncate(T + 2)
    q4, kv4, iqk4 = golden_inputs(cv)
    q, kv, iqk = q4[::-1].copy(), kv4[::-1].copy(), iqk4[::-1].copy()
    out = a.step(q, kv, iqk)
    k, v, ik, bk = a.export(T + 6)
    # reference: cache = original + verify rows 0, 1 (as appended by the first verify), then the reversed rows
    x0 = ref.step(K, V, IK, T, q4[:2], kv4[:2], iqk4[:2])
    IK2 = np.concatenate([IK, O.bf16(iqk4[:2, 512:])], 0)
    x = ref.step(x0["K"], x0["V"], IK2, T + 2, q, kv, iqk, quant=mlx_q8 if fmt == "kv8" else None)
    cmp(f"{fmt} rollback: block keys after re-append", bk[:(T + 6) // 4], x["bk"][:(T + 6) // 4], 5e-3)
    sets, _ = our_sets(a, 4)
    sets_check(f"{fmt} rollback: selection after re-append", sets, x["sets"], x["scores"], T + 2)
    cmp(f"{fmt} rollback: attn.gated after re-append", out, x["gated"], 1e-3 if fmt != "kv8" else 2e-3)
    a.free()
    b.free()


def run_kv8(eng, pack, ref):
    c = golden.load("L03_long_verify4")
    T = c.m["cache_len_in"]
    K, V, IK = c["attn.k_cache_in"][0], c["attn.v_cache_in"][0], c["attn.index_keys_in"][0]
    a = Attn(eng, pack, "kv8", tmax=4096)
    a.import_(K, V, IK, T)
    q, kv, iqk = golden_inputs(c)
    out = a.step(q, kv, iqk)
    n = T + 4
    k, v, _, _ = a.export(n)
    x = ref.step(K, V, IK, T, q, kv, iqk)
    kq = mlx_q8(x["K"])
    vq = mlx_q8(x["V"])
    dk = np.abs(k - kq) / (np.abs(kq).max() + 1e-30)
    rec("kv8 cache vs numpy MLX quantize (K)", np.mean(k == kq) > 0.999,
        f"{np.mean(k == kq) * 100:.4f}% equal, max |d| / max {dk.max():.2e}")
    rec("kv8 cache vs numpy MLX quantize (V)", np.mean(v == vq) > 0.999, f"{np.mean(v == vq) * 100:.4f}% equal")
    cmp("kv8 quantization error K (vs bf16 cache)", k, x["K"], 1e-2)
    cmp("kv8 quantization error V (vs bf16 cache)", v, x["V"], 1e-2)
    # exact path: numpy attention over OUR dequantized cache with the reference selection
    pos = np.arange(T, T + 4)
    _, allowed = O.qsa_select(x["scores"], T)
    core = O.attention(x["qr"].transpose(1, 0, 2), k, v, pos, allowed)
    gated = O.bf16(core * O.sigmoid_bf(O.bf16(q).reshape(4, 24, 512)[..., 256:].reshape(4, -1)))
    cmp("kv8 attention vs numpy on the same cache", out, gated, 2e-3)  # MMA path: K, V, P staged as half
    cmp("kv8 attn.gated vs bf16 golden (quantization)", out, c["attn.gated"][0], 2e-2)
    a.free()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=str(ROOT / "build-attn/libengine.dylib"))
    ap.add_argument("--pack", default=str(PACK))
    ap.add_argument("--fmt", default="bf16,kv8,f32")
    ap.add_argument("--only", default="golden,rows,rollback,kv8,huge")
    args = ap.parse_args()
    eng = E.Engine(args.lib)
    eng.set_ref_rounding(True)  # goldens: the reference rounding points (the library default is plain f32)
    pack = eng.pack_open(args.pack)
    only = args.only.split(",")
    ref = Ref() if set(only) & {"rows", "rollback", "kv8", "huge"} else None
    for fmt in args.fmt.split(","):
        if "golden" in only:
            print(f"== goldens, {fmt} cache")
            for cn in ("L03_decode1", "L03_verify4", "L03_long_decode1", "L03_long_verify4"):
                run_golden(eng, pack, fmt, cn)
        if "rows" in only:
            print(f"== rows 2, 3, 5, 8 and long synthetic contexts, {fmt} cache")
            run_rows(eng, pack, ref, fmt)
            run_long(eng, pack, ref, fmt)
        if "rollback" in only:
            print(f"== rollback, {fmt} cache")
            run_rollback(eng, pack, ref, fmt)
    if "kv8" in only:
        print("== kv8 path")
        run_kv8(eng, pack, ref)
    if "huge" in only:
        print("== selection past 128K")
        run_select_huge(eng, pack, ref)
    bad = [r for r in RES if not r[1]]
    print(f"\n{len(RES) - len(bad)} of {len(RES)} checks pass")
    for r in bad:
        print("  FAIL", r[0], r[2])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
