#!/usr/bin/env python3
"""lab-attn: GPU time of the QSA attention step (one full-attention layer) at long context, synthetic cache content.

    tools/with_lock.sh "attn bench" HH:MM -- $PY tools/attn_bench.py [--ctx 2048,8192,32768,131072] [--fmt bf16,kv8]
        [--rows 1,4] [--flush 256] [--dispatches]

Per (format, context T, rows R): the whole step (5 dispatches) and optionally each dispatch alone, differential
(SLC-warm: 16 vs 80 steps in one command buffer; flushed: 8 vs 40 steps with a flush_mb streaming read between steps,
SLC evicted, its own cost subtracted; best of 5, best of 2 runs). Per dispatch: SLC-warm, 64 vs 320 in one command
buffer (short chains of small dispatches run at low clocks and read 2-5x slower). The cache stays at T (the steps rewrite rows T..T+R-1), so every step sees the same context.
Bytes = what the step must read: block keys (T/4 x 256 B), the selected K/V rows (min(T+R, 2048 + tail) per row and
kv head), q / partials ignored; GB/s = that / time.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests/attn"))
import refenv  # noqa: E402,F401
import engine as E  # noqa: E402
import test_qsa as Q  # noqa: E402
import numpy as np  # noqa: E402

ROWB = {"f32": 1024, "bf16": 512, "kv8": 256 + 16}  # bytes per cached row of one kv head (K or V)
NAMES = ["prep", "score", "sel_local", "select", "attn", "merge"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=str(ROOT / "build-attn/libengine.dylib"))
    ap.add_argument("--ctx", default="2048,8192,32768,131072")
    ap.add_argument("--fmt", default="bf16,kv8")
    ap.add_argument("--rows", default="1,4")
    ap.add_argument("--flush", type=int, default=256)
    ap.add_argument("--dispatches", action="store_true")
    args = ap.parse_args()
    eng = E.Engine(args.lib)
    pack = eng.pack_open(str(Q.PACK))
    rng = np.random.default_rng(1)
    rows = [int(r) for r in args.rows.split(",")]
    print(f"{'fmt':5s} {'T':>7s} {'R':>2s} {'warm us':>8s} {'flushed':>8s} {'MB read':>8s} {'GB/s fl':>7s}   "
          "per dispatch us (alone, SLC-warm, 64 vs 320 in one command buffer)")
    for fmt in args.fmt.split(","):
        for T in [int(x) for x in args.ctx.split(",")]:
            tmax = (T + 64 + 63) // 64 * 64
            a = Q.Attn(eng, pack, fmt, tmax=tmax)
            K = rng.standard_normal((2, T, 256), dtype=np.float32)
            V = rng.standard_normal((2, T, 256), dtype=np.float32)
            IK = rng.standard_normal((T, 128), dtype=np.float32)
            a.import_(K, V, IK, T)
            del K, V, IK
            for R in rows:
                q = rng.standard_normal((R, 12288), dtype=np.float32)
                kv = rng.standard_normal((R, 1024), dtype=np.float32)
                iqk = rng.standard_normal((R, 640), dtype=np.float32)
                a.step(q, kv, iqk)  # loads the inputs (and records); pos -> T + R
                a.truncate(T)
                warm = min(a.time(R, -1, 16, 80, 0) for _ in range(2))
                us = min(a.time(R, -1, 8, 40, args.flush) for _ in range(2))
                keys = sum(min(T + t + 1, 2048 + 4) for t in range(R))
                mb = (T // 4 * 256 + keys * 2 * 2 * ROWB[fmt]) / 1e6
                per = ""
                if args.dispatches:
                    per = "  ".join(f"{n} {a.time(R, i, 64, 320, 0):5.1f}" for i, n in enumerate(NAMES))
                print(f"{fmt:5s} {T:7d} {R:2d} {warm:8.1f} {us:8.1f} {mb:8.2f} {mb / us * 1e3:7.0f}   {per}", flush=True)
            a.free()
    return 0


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    print(f"({time.time() - t0:.0f} s)")
    sys.exit(rc)
