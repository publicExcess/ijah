"""lab-pfpath: the MTP head's KV rows after a chunked feed, batched (prefill_head_kv, NE_MTP_HEAD_CHUNK=1) vs the
8-row KV-only steps (=0), and the MTP loop's acceptance / output after each.

    NEED_GB=85 LOCK_OWNER=pfpath LOCK_WAIT=30 tools/with_lock.sh "check_headkv (~81 GB)" HH:MM -- \\
        $PY tools/check_headkv.py [--n 8197] [--gen 256]

Per arm: reset, the server's feed (4096-token pieces through ne_fn_mtp_feed) of n new-text tokens, the head caught
up; the head's KV rows [0, Lh) exported (ne_fn_mtp_kv_io: bf16 K / V / indexer keys / block keys) and the target's KV
rows (must be identical: the target path is unchanged). Then from the same feed, the MTP loop (greedy, depth 2) over
one more token for --gen tokens: tokens per round (acceptance) and the output.
"""
import argparse
import ctypes as C
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "tools"))
import refenv  # noqa: E402
import ijah_paths  # noqa: E402

refenv.add_paths()
import numpy as np  # noqa: E402
import backend as B  # noqa: E402


def bf(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=ijah_paths.pack())
    ap.add_argument("--n", type=int, default=8197)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=32768)
    a = ap.parse_args()
    ids = json.load(open(os.environ.get("IJAH_NEW_STREAM") or sys.exit("set IJAH_NEW_STREAM: a JSON list of token ids of text the model has not seen (not shipped)")))
    toks = ids[300000:300000 + a.n + 1]
    be = B.RealBackend(ROOT, a.pack, a.ctx, 2, ROOT / "build/libengine.dylib", None, print)
    L = be.eng.lib
    vp, ip = C.c_void_p, C.c_int
    L.ne_fn_mtp_head_state.argtypes = [vp, C.POINTER(ip), C.POINTER(ip)]
    L.ne_fn_mtp_catch_up.argtypes = [vp, C.POINTER(C.c_double)]
    mp = vp(be.mtp_ptr)
    res = {}
    for arm in ("0", "1", "0", "1"):
        os.environ["NE_MTP_HEAD_CHUNK"] = arm
        be.reset()
        be.set_sampler(0.0, 1.0, 1, 0)
        t = time.perf_counter()
        i = 0
        for p in B.pieces(a.n, table=B.MTP_PIECES):
            be.feed(toks[i:i + p], sample=False)
            i += p
        ms = C.c_double(0)
        assert L.ne_fn_mtp_catch_up(mp, C.byref(ms)) == 0, be.eng.last_error()
        wall = (time.perf_counter() - t) * 1e3
        lh, npend = ip(0), ip(0)
        L.ne_fn_mtp_head_state(mp, C.byref(lh), C.byref(npend))
        Lh = lh.value
        hb, cap, hcap = be.kv_save(0, a.n, 0, a.n // 8 * 8 - 8)  # the same head rows in both arms
        nm = int(L.ne_fn_kv_bytes(vp(be.m.ptr), cap))
        model_kv, head_kv = hb[:nm].copy(), hb[nm:].copy()
        out, st = be.mtp_from([toks[a.n]], a.gen, 2, 0, lambda t: None, lambda: False)
        acc = st["tokens"] / max(1, st["rounds"])
        print(f"arm HEAD_CHUNK={arm}: feed {a.n} + catch-up {wall:.0f} ms (final catch-up {ms.value:.1f}), Lh {Lh}, "
              f"pending {npend.value}; MTP greedy d2: {st['tokens']:.0f} tokens in {st['rounds']:.0f} rounds = "
              f"{acc:.3f} per round, decode {st['decode_ms']:.0f} ms", flush=True)
        res.setdefault(arm, []).append((model_kv, head_kv, out, acc, Lh))
    (m0, h0, o0, a0, l0), (m1, h1, o1, a1, l1) = res["0"][0], res["1"][0]
    print(f"target KV rows identical across arms: {np.array_equal(m0, m1)}; repeat arm 0 head KV identical: "
          f"{np.array_equal(h0, res['0'][1][1])}, arm 1: {np.array_equal(h1, res['1'][1][1])}")
    x, y = bf(h0.view(np.uint16)), bf(h1.view(np.uint16))
    ok = np.isfinite(x) & np.isfinite(y)
    rel = np.linalg.norm((x - y)[ok]) / max(1e-30, np.linalg.norm(x[ok]))
    print(f"head KV bytes {len(h0)} vs {len(h1)} (Lh {l0} vs {l1}): rel L2 (as bf16) {rel:.3e}, "
          f"max abs {np.abs(x - y)[ok].max():.3e}, nonfinite {int((~ok).sum())}, identical bytes "
          f"{np.mean(h0 == h1):.3f}")
    k = 16  # by sixteenths of the buffer (sections: K, V, indexer keys, block keys in kv_io's layout)
    seg = len(x) // k
    print("  per sixteenth rel L2: " + " ".join(
        f"{np.linalg.norm((x - y)[j * seg:(j + 1) * seg]) / max(1e-30, np.linalg.norm(x[j * seg:(j + 1) * seg])):.1e}"
        for j in range(k)))
    same = sum(1 for p, q in zip(o0, o1) if p == q)
    first = next((k for k, (p, q) in enumerate(zip(o0, o1)) if p != q), None)
    print(f"MTP greedy output: {same}/{min(len(o0), len(o1))} equal tokens, first difference at {first}; "
          f"tokens per round {a0:.3f} (8-row steps) vs {a1:.3f} (batched)")


if __name__ == "__main__":
    main()
