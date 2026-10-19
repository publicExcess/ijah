"""lab-headlazy: the lazy MTP head (docs/mtp.md 10.10) -- step-path decode, catch-up cost, feed cost, MTP speed.

    NEED_GB=85 LOCK_OWNER=... tools/with_lock.sh "headlazy bench (~83 GB)" HH:MM -- $PY tools/bench_headlazy.py
        [--n 512] [--catch 50,500,5000]

Head variants (two ne_fn_mtp objects on one model): "old" = NE_MTP_HEAD_KV=full + pending cap 0 (the eager head with
full-layer head steps = lab-mtpcont), "kv" = KV-only head steps (cap 0: eager, cap 1024: lazy = the default).
  step    decode tok/s through RealBackend.step-equivalent ctypes calls (what the server's step path runs per token):
          plain ne_fn_model_step (head stale) vs ne_fn_mtp_step old / kv eager / kv lazy; sampled T 1 / top-p 0.95 /
          top-k 20 from the same fed prompt
  catch   N lazy mtp_steps, then ne_fn_mtp_catch_up: wall ms (kv; old at the smallest N's for scale)
  feed    plain ne_fn_model_prefill vs ne_fn_mtp_feed (old / kv) from an empty state, 2048 / 8192 tokens
  mtp     fresh ne_fn_mtp_generate d2 greedy (old vs kv): tokens identical, decode tok/s, head prefill ms
"""
import argparse
import ctypes as C
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("NE_FN_ROUND", "f32")
import refenv  # noqa: E402
import ijah_paths  # noqa: E402

refenv.add_paths()
import numpy as np  # noqa: E402
import engine as E  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

vp, ip = C.c_void_p, C.c_int


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=ijah_paths.pack())
    ap.add_argument("--n", type=int, default=512, help="decode tokens per step-path variant")
    ap.add_argument("--catch", default="50,500,5000")
    ap.add_argument("--feed", default="2048,8192")
    ap.add_argument("--ctx", type=int, default=16384)
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(Path(a.pack) / "tokenizer.json"))
    text = "\n\n".join((ROOT / "docs" / f).read_text() for f in ("mtp.md", "prefill.md", "forward.md", "server.md", "attn.md"))
    ids = tok.encode(text, add_special_tokens=False).ids
    print(f"{len(ids)} text tokens", flush=True)
    eng = E.Engine()
    L = eng.lib
    for k, (r, args) in {
        "ne_fn_mtp_create": (vp, [vp, ip, ip]),
        "ne_fn_mtp_feed": (ip, [vp, vp, ip]),
        "ne_fn_mtp_step": (ip, [vp, ip, C.POINTER(ip), vp]),
        "ne_fn_mtp_catch_up": (ip, [vp, C.POINTER(C.c_double)]),
        "ne_fn_mtp_set_pend_cap": (ip, [vp, ip]),
        "ne_fn_mtp_head_state": (ip, [vp, C.POINTER(ip), C.POINTER(ip)]),
        "ne_fn_mtp_generate": (ip, [vp, vp, ip, ip, ip, ip, C.c_uint64, vp, vp, vp]),
    }.items():
        f = getattr(L, k)
        f.restype, f.argtypes = r, args
    pack = eng.pack_open(a.pack)
    t0 = time.time()
    m = eng.fn_model(pack, a.ctx)
    print(f"load {time.time() - t0:.1f} s", flush=True)
    kv = L.ne_fn_mtp_create(C.c_void_p(m.ptr), 4, 4)
    os.environ["NE_MTP_HEAD_KV"] = "full"
    old = L.ne_fn_mtp_create(C.c_void_p(m.ptr), 4, 4)
    del os.environ["NE_MTP_HEAD_KV"]
    assert kv and old, eng.last_error()
    nx = C.c_int(-1)

    def ok(rc, what):
        if rc != 0:
            raise E.EngineError(f"{what}: {eng.last_error()}")

    def feed(h, toks):
        p = np.ascontiguousarray(toks, np.int32)
        ok(L.ne_fn_mtp_feed(h, p.ctypes.data, len(p)), "feed")

    def mstep(h, t):
        ok(L.ne_fn_mtp_step(h, int(t), C.byref(nx), None), "mtp_step")
        return nx.value

    def pstep(t):
        ok(L.ne_fn_model_step(m.ptr, int(t), 1, None, C.byref(nx), None, None, None), "step")
        return nx.value

    def pend(h):
        x, y = C.c_int(0), C.c_int(0)
        L.ne_fn_mtp_head_state(h, C.byref(x), C.byref(y))
        return y.value

    P = ids[:1024]
    res = {}
    # ---- step path decode
    m.set_sampler(1.0, 0.95, 20, 7)
    variants = [("plain", None, 0), ("old eager", old, 0), ("kv lazy", kv, 1024), ("kv eager", kv, 0),
                ("plain", None, 0), ("kv lazy", kv, 1024), ("old eager", old, 0)]
    for name, h, cap in variants:
        m.reset()
        if h is not None:
            ok(L.ne_fn_mtp_set_pend_cap(h, cap), "cap")
            feed(h, P[:-1])
            y = mstep(h, P[-1])
        else:
            y, _, _ = m.prefill(P)
        t0 = time.perf_counter()
        for _ in range(a.n):
            y = mstep(h, y) if h is not None else pstep(y)
        dt = time.perf_counter() - t0
        tps = a.n / dt
        res.setdefault(name, []).append(tps)
        print(f"step {name:10s}: {tps:.2f} tok/s ({dt * 1e3 / a.n:.2f} ms/token), pending after "
              f"{pend(h) if h is not None else '-'}", flush=True)
    print("step path tok/s:", {k: [round(x, 2) for x in v] for k, v in res.items()}, flush=True)
    # ---- catch-up cost
    m.set_sampler(0.0, 1.0, 1, 0)
    for N in [int(x) for x in a.catch.split(",")]:
        for name, h in (("kv", kv), ("old", old)):
            if name == "old" and N > 500:
                continue
            ok(L.ne_fn_mtp_set_pend_cap(h, 1 << 20), "cap")
            m.reset()
            feed(h, P[:-1])
            y = mstep(h, P[-1])
            ms = C.c_double(0)
            ok(L.ne_fn_mtp_catch_up(h, C.byref(ms)), "catch_up")
            p0 = pend(h)
            for _ in range(N):
                y = mstep(h, y)
            p1 = pend(h)
            ok(L.ne_fn_mtp_catch_up(h, C.byref(ms)), "catch_up")
            print(f"catch-up {name} N {N}: pending {p1} -> {pend(h)} (before the steps {p0}): {ms.value:.1f} ms "
                  f"= {ms.value * 1e3 / max(p1 - pend(h), 1):.1f} us / row", flush=True)
    ok(L.ne_fn_mtp_set_pend_cap(kv, 1024), "cap")
    ok(L.ne_fn_mtp_set_pend_cap(old, 0), "cap")
    # ---- feed cost (eager / at-cap drain inside the feed: the cost of keeping the head in step during a prefill)
    for n in [int(x) for x in a.feed.split(",")]:
        Pn = (ids * 4)[:n]
        for rep in range(2):
            row = {}
            m.reset()
            _, _, ms = m.prefill(Pn + [Pn[0]])
            row["plain"] = ms
            for name, h in (("old", old), ("kv", kv)):
                m.reset()
                t0 = time.perf_counter()
                feed(h, Pn)
                ms = C.c_double(0)
                ok(L.ne_fn_mtp_catch_up(h, C.byref(ms)), "catch_up")
                mstep(h, Pn[0])
                row[name] = (time.perf_counter() - t0) * 1e3
            print(f"feed {n} (rep {rep}): plain prefill {row['plain']:.0f} ms, mtp_feed + catch-up + step: old "
                  f"{row['old']:.0f} ms ({(row['old'] / row['plain'] - 1) * 100:+.0f}%), kv {row['kv']:.0f} ms "
                  f"({(row['kv'] / row['plain'] - 1) * 100:+.0f}%)", flush=True)
    # ---- MTP speed (fresh generate, greedy d2)
    Q = ids[2000:2512]
    outs = {}
    for name, h in (("old", old), ("kv", kv), ("old", old), ("kv", kv)):
        p = np.ascontiguousarray(Q, np.int32)
        out = np.empty(a.n, np.int32)
        st = np.zeros(10, np.float64)
        k = L.ne_fn_mtp_generate(h, p.ctypes.data, len(p), a.n, -1, 2, 0, out.ctypes.data, st.ctypes.data, None)
        assert k > 0, eng.last_error()
        tps = st[3] / st[2] * 1e3
        same = outs.setdefault("ref", out[:k].tolist()) == out[:k].tolist()
        print(f"mtp {name}: {k} tokens, {tps:.1f} tok/s, prefill {st[0]:.0f} ms (head {st[1]:.1f} ms), rounds "
              f"{st[4]:.0f}, tokens identical to the first run {same}", flush=True)


if __name__ == "__main__":
    main()
