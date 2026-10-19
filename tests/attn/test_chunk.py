#!/usr/bin/env python3
"""lab-attn: the prefill-chunk attention core (kernels/attn/qsa_chunk.*) vs the per-slice step path.

    $PY tests/attn/test_chunk.py [--lib build-attn2/libengine.dylib] [--ctx 1024,8192,30720] [--time]

Synthetic caches of T0 tokens (random bf16 K / V / indexer keys) and n chunk rows (random q / kv, indexer queries
as a random walk so that neighbouring rows select overlapping blocks). Reference = record_chunk mode 0 without the
dense path: every row through lab-attn's step (prep / score / select / qsa_attn / qsa_merge, 8 rows per slice), which
tests/attn/test_qsa.py checks against the goldens and numpy. Checked: the caches after the chunk (K / V rows, raw and block indexer keys) and the selections are bit-identical, the gated
output of the multi-row core (mode 1, with and without the dense NAX rows) within 2e-3 rel L2 per row (the long
goldens' tolerance; bf16 P . V on the NAX vs f32 in the SIMT kernel). Light GPU work: no model, no lock needed
except for --time.
"""
import argparse
import ctypes as C
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests/attn"))
import refenv  # noqa: E402,F401
import engine as E  # noqa: E402
import opref as O  # noqa: E402
import numpy as np  # noqa: E402
from test_qsa import Attn, PACK, ptr  # noqa: E402

RES = []
u32p = C.POINTER(C.c_uint32)


def rec(name, ok, msg):
    RES.append((name, ok, msg))
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {msg}")


def bind(L):
    sig = {
        "ne_attn_chunk": (C.c_int, [C.c_void_p, C.c_int, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int,
                                    C.c_int, C.POINTER(C.c_double)]),
        "ne_attn_chunk_sel": (C.c_int, [C.c_void_p, C.c_int, u32p, u32p]),
        "ne_attn_chunk_time": (C.c_int, [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_double)]),
    }
    for n, (r, a) in sig.items():
        f = getattr(L, n)
        f.restype, f.argtypes = r, a


def chunk(a, q, kv, iqk, mode, dense):
    n = q.shape[0]
    out = np.zeros((n, 6144), np.float32)
    ms = C.c_double()
    a.ok(a.L.ne_attn_chunk(a.h, n, q.ctypes.data, kv.ctypes.data, iqk.ctypes.data, out.ctypes.data, mode, dense,
                           C.byref(ms)), "chunk")
    sel = np.zeros((n, 512), np.uint32)
    nsel = np.zeros(n, np.uint32)
    a.ok(a.L.ne_attn_chunk_sel(a.h, n, sel.ctypes.data_as(u32p), nsel.ctypes.data_as(u32p)), "chunk_sel")
    return out, sel, nsel, ms.value


def ctime(a, n, mode, part, reps=7):
    ms = C.c_double()
    a.ok(a.L.ne_attn_chunk_time(a.h, n, mode, part, reps, C.byref(ms)), "chunk_time")
    return ms.value


def inputs(rng, n, step=0.15):
    q = rng.standard_normal((n, 12288), dtype=np.float32)
    kv = rng.standard_normal((n, 1024), dtype=np.float32)
    iqk = rng.standard_normal((n, 640), dtype=np.float32)
    walk = np.cumsum(rng.standard_normal((n, 512), dtype=np.float32) * step, axis=0)
    iqk[:, :512] = rng.standard_normal(512).astype(np.float32) + walk  # smooth indexer queries
    return q, kv, iqk


def run(eng, pack, T0, n, do_time, seed, end=False):
    rng = np.random.default_rng(seed)
    tmax = (T0 + n + 64 + 63) // 64 * 64
    if end:  # the chunk fills the cache to its last row (n % 8 != 0: a short last slice at the context end)
        tmax = (T0 + n + 63) // 64 * 64
        T0 = tmax - n
    a = Attn(eng, pack, "bf16", tmax=tmax)
    bind(a.L)
    if T0 > 0:
        K = O.bf16(rng.standard_normal((2, T0, 256), dtype=np.float32))
        V = O.bf16(rng.standard_normal((2, T0, 256), dtype=np.float32))
        IK = O.bf16(rng.standard_normal((T0, 128), dtype=np.float32) * 2)
        a.import_(K, V, IK, T0)
    q, kv, iqk = inputs(rng, n)
    tag = f"T0={T0} n={n}{' (to tmax)' if end else ''}"
    ref, sref, nref, ms0 = chunk(a, q, kv, iqk, 0, 0)
    cache_ref = a.export(T0 + n)
    for mode, dense in ((1, 0), (1, 1)):
        a.truncate(T0)
        out, sel, nsel, ms1 = chunk(a, q, kv, iqk, mode, dense)
        lab = f"{tag} core{' + dense rows' if dense else ''}"
        if not dense:
            cache = a.export(T0 + n)
            eq = all(np.array_equal(x, y) for x, y in zip(cache, cache_ref))
            rec(f"{lab} caches (k, v, raw / block indexer keys)", eq, "bit-identical" if eq else "differ")
            same = bool((nsel == nref).all() and all((sel[i, :nsel[i]] == sref[i, :nref[i]]).all() for i in range(n)))
            rec(f"{lab} selections", same, "bit-identical" if same else "differ")
        num = np.linalg.norm(out - ref, axis=1)
        den = np.linalg.norm(ref, axis=1) + 1e-30
        rel = num / den
        tot = float(np.linalg.norm(out - ref) / np.linalg.norm(ref))
        ok = bool(np.isfinite(out).all() and rel.max() < 2e-3)
        rec(f"{lab} attn.gated", ok, f"rel L2 {tot:.2e}, worst row {rel.max():.2e} (row {int(rel.argmax())})")
    if do_time:
        a.truncate(T0)
        chunk(a, q, kv, iqk, 1, 0)
        a.truncate(T0)
        t0 = ctime(a, n, 0, 0)
        t1 = ctime(a, n, 1, 0)
        chunk(a, q, kv, iqk, 1, 0)  # selection of this pos in the scratch for part 1
        a.truncate(T0)
        ta = ctime(a, n, 1, 1)
        u = [len(np.unique(sref[r:r + 8][sref[r:r + 8] != 0xFFFFFFFF])) for r in range(0, n - 7, 8)]
        print(f"  TIME {tag}: step path {t0:.2f} ms, chunk core {t1:.2f} ms (attention kernel {ta:.2f} ms); "
              f"8-row union {np.mean(u) / 512:.2f}x of 512")
    a.free()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=str(ROOT / "build-attn2/libengine.dylib"))
    ap.add_argument("--pack", default=str(PACK))
    ap.add_argument("--ctx", default="1024,8192,30720")
    ap.add_argument("--n", default="2048,1001")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--end", default="1001,9", help="chunk sizes also run so the chunk ends at tmax ('' = none)")
    args = ap.parse_args()
    eng = E.Engine(args.lib)
    pack = eng.pack_open(args.pack)
    for i, T0 in enumerate(int(x) for x in args.ctx.split(",")):
        for n in (int(x) for x in args.n.split(",")):
            print(f"== T0 {T0}, {n} rows")
            run(eng, pack, T0, n, args.time and n == 2048, 11 + i)
        for n in (int(x) for x in args.end.split(",") if x):
            print(f"== T0 {T0}, {n} rows ending at tmax")
            run(eng, pack, T0, n, False, 31 + i, end=True)
    bad = [r for r in RES if not r[1]]
    print(f"\n{len(RES) - len(bad)} of {len(RES)} checks pass")
    for r in bad:
        print("  FAIL", r[0], r[2])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
