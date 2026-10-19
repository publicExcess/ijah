"""lab-vision: the tower's sensitivity floor. The bf16 tower is chaotic: the MLX reference run from the ENGINE's patch
embedding (bit-identical to the reference's in all but ~0.003% of its elements, each 1 bf16 ulp) ends as far from
the reference as the whole engine does. Prints that distance next to the engine's.

  $PY -I -B tests/vision/tower_floor.py GOLDEN_DIR        (GPU, ~3 GB; GOLDEN_DIR from vision_ref.py)
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests/vision"))
import refenv  # noqa: E402
import ijah_paths  # noqa: E402

refenv.add_paths()
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import engine as E  # noqa: E402
from vision_ref import load_tower  # noqa: E402


def main():
    g = Path(sys.argv[1])
    pack = Path(ijah_paths.pack())
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))  # noqa: E731
    ref = np.load(g / "ref_bf16.npy")
    x = np.load(g / "patches.npy")
    grid = tuple(int(i) for i in np.load(g / "grid.npy"))
    pv = x.reshape(-1, 2, 16, 16, 3).transpose(0, 4, 1, 2, 3).reshape(x.shape[0], -1)
    m = load_tower(pack, False)
    eng = E.Engine()
    p = eng.pack_open(str(pack))
    os.environ["NE_VIT_LAYERS"] = "0"
    v0 = eng.vision(p)
    _, h0 = v0.encode(x, [grid], hidden=True)
    v0.free()
    del os.environ["NE_VIT_LAYERS"]
    gt = mx.array([grid])
    h_ref0 = m.patch_embed(mx.array(pv).astype(mx.bfloat16)) + m.fast_pos_embed_interpolate(gt)
    he = mx.array(h0).view(mx.bfloat16)
    diff = float((np.array(he.astype(mx.float32)) != np.array(h_ref0.astype(mx.float32))).mean())
    rot = m.rot_pos_emb(gt)
    cu = mx.array([0, x.shape[0]], dtype=mx.int32)
    h = he
    for blk in m.blocks:
        h = blk(h, cu_seqlens=cu, rotary_pos_emb=rot)
    o = m.merger(h)
    mx.eval(o)
    alt = np.array(o.astype(mx.float32))
    v = eng.vision(p)
    e = E.bf16_to_f32(v.encode(x, [grid]))
    print(f"patch embedding: engine vs reference differ in {diff * 100:.4f}% of elements")
    print(f"reference tower from the engine's patch embedding vs reference: rel L2 {rel(alt, ref):.3e}  <- floor")
    print(f"engine vs reference: rel L2 {rel(e, ref):.3e}")


if __name__ == "__main__":
    main()
