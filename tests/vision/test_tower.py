"""lab-vision: engine vision tower vs the MLX reference goldens of tests/vision/vision_ref.py (same pack, same image).

  $PY -I -B tests/vision/test_tower.py GOLDEN_DIR [--pack DIR]     (GPU job: tools/with_lock.sh; ~2 GB)
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import refenv  # noqa: E402
import ijah_paths  # noqa: E402

refenv.add_paths()
import numpy as np  # noqa: E402

import engine as E  # noqa: E402


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("--pack", default=None)
    ap.add_argument("--limit", type=float, default=5e-2)  # the tower is chaotic: docs/vision.md 6
    a = ap.parse_args()
    g = Path(a.golden)
    pack = a.pack or ijah_paths.pack()
    eng = E.Engine()
    p = eng.pack_open(pack)
    t0 = time.time()
    v = eng.vision(p)
    print(f"tower loaded in {time.time() - t0:.1f} s")
    x = np.load(g / "patches.npy")
    grid = tuple(int(i) for i in np.load(g / "grid.npy"))
    out = v.encode(x, [grid])
    ms = []
    for _ in range(2):
        t1 = time.time()
        out = v.encode(x, [grid])
        ms.append((v.gpu_ms, (time.time() - t1) * 1e3))
    e = E.bf16_to_f32(out)
    rb, rf = np.load(g / "ref_bf16.npy"), np.load(g / "ref_f32.npy")
    print(f"{x.shape[0]} patches -> {e.shape}; GPU {ms[-1][0]:.1f} ms, wall {ms[-1][1]:.1f} ms")
    rb_, rf_ = rel(e, rb), rel(e, rf)
    cos = float((e * rb).sum() / np.linalg.norm(e) / np.linalg.norm(rb))
    print(f"engine vs MLX bf16 rel L2 {rb_:.3e} (cos {cos:.6f}); vs MLX f32 {rf_:.3e}; MLX bf16 vs f32 {rel(rb, rf):.3e}")
    tc = (e * rb).sum(1) / (np.linalg.norm(e, axis=1) * np.linalg.norm(rb, axis=1) + 1e-30)
    gw = grid[2] // 2
    bad = np.nonzero(tc < 0.99)[0]
    print(f"per-token cosine min {tc.min():.4f} mean {tc.mean():.5f}; {len(bad)} tokens < 0.99"
          + (f": (row, col) {[(int(i) // gw, int(i) % gw) for i in bad[:12]]}" if len(bad) else ""))
    print("PASS" if rb_ <= a.limit else "FAIL", f"(limit {a.limit:g})")
    return 0 if rb_ <= a.limit else 1


if __name__ == "__main__":
    sys.exit(main())
