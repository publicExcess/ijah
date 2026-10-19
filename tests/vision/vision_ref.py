"""lab-vision reference checks and goldens (MLX / mlx-vlm on the pack's vision tower only, ~1 GB; no language model).

  PY=.venv/bin/python
  $PY -I -B tests/vision/vision_ref.py IMAGE OUT_DIR [--max-pixels N] [--pack DIR]

1. server/vision.py preprocessing == mlx-vlm's Qwen3VLImageProcessor (same min / max pixels): bit-exact pixel values
2. server/vision.py positions() == Qwen3_5LanguageModel.get_rope_index on prompts with 1 and 2 images
3. MLX VisionModel (mlx_vlm qwen4_exp = qwen3_vl tower) on the pack's weights, bf16 as served: OUT_DIR/ref_bf16.npy
   (f32 values of the bf16 embeddings [tokens][2560]); the same tower with f32 weights / activations: ref_f32.npy;
   engine-order patches: patches.npy (f32 [N][1536]); grid.npy
"""
import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "server"))
import refenv  # noqa: E402

refenv.add_paths()
import numpy as np  # noqa: E402

import vision as V  # noqa: E402


def check_positions():
    from mlx_vlm.models.qwen3_5.language import LanguageModel
    import mlx.core as mx

    cfg = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2), image_token_id=V.IMAGE_TOKEN,
                          video_token_id=V.VIDEO_TOKEN, vision_start_token_id=V.VISION_START)
    me = SimpleNamespace(config=cfg)
    cases = [[(1, 8, 12)], [(1, 6, 4), (1, 10, 14)], [(1, 4, 4)]]
    for grids in cases:
        toks = [11, 22, 33]
        for g in grids:
            toks += [V.VISION_START] + [V.IMAGE_TOKEN] * V.n_tokens(g) + [V.VISION_END] + [44, 55]
        toks += [66] * 5
        ref, rd = LanguageModel.get_rope_index(me, mx.array([toks]), mx.array(grids), None, None)
        ref = np.array(ref)[:, 0, :].T.astype(np.int64)
        mine, d = V.positions(toks, grids)
        assert np.array_equal(ref, mine), (grids, np.argwhere(ref != mine)[:5])
        assert int(np.array(rd).ravel()[0]) == d, (int(np.array(rd).ravel()[0]), d)
    print(f"positions == get_rope_index on {len(cases)} prompts")


def load_tower(pack, f32):
    import json
    import mlx.core as mx
    from mlx_vlm.models.qwen4_exp import VisionModel, VisionConfig

    cfg = json.loads((pack / "config.json").read_text())
    vc = VisionConfig.from_dict(cfg["vision_config"]) if hasattr(VisionConfig, "from_dict") else VisionConfig(
        **cfg["vision_config"])
    m = VisionModel(vc)
    idx = json.loads((pack / "model.safetensors.index.json").read_text())["weight_map"]
    files = sorted({f for k, f in idx.items() if k.startswith("vision_tower.")})
    w = {}
    for f in files:
        w.update({k[len("vision_tower."):]: v for k, v in mx.load(str(pack / f)).items() if k.startswith("vision_tower.")})
    if f32:
        w = {k: v.astype(mx.float32) for k, v in w.items()}
    m.load_weights(list(w.items()))
    mx.eval(m.parameters())
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--max-pixels", type=int, default=V.MAX_PIXELS)
    ap.add_argument("--pack", default=refenv.ijah_paths.pack())
    a = ap.parse_args()
    pack = Path(a.pack)
    if pack.is_file():
        pack = Path(pack.read_text().strip())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import mlx.core as mx
    from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor

    img, _ = V.load_image(a.image)
    t0 = time.time()
    pv, grid = V.preprocess(img, V.MIN_PIXELS, a.max_pixels)
    t_pre = time.time() - t0
    proc = Qwen3VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2, min_pixels=V.MIN_PIXELS,
                                 max_pixels=a.max_pixels)
    r = proc([img])
    assert tuple(r["image_grid_thw"][0]) == grid, (r["image_grid_thw"], grid)
    assert np.array_equal(r["pixel_values"], pv), np.abs(r["pixel_values"] - pv).max()
    print(f"preprocess == mlx-vlm Qwen3VLImageProcessor: grid {grid}, {pv.shape[0]} patches, {V.n_tokens(grid)} tokens "
          f"(bit-exact; {t_pre * 1e3:.0f} ms)")
    check_positions()
    np.save(out / "patches.npy", V.engine_patches(pv))
    np.save(out / "grid.npy", np.array(grid, np.int64))
    for f32 in (False, True):
        m = load_tower(pack, f32)
        # bf16 as served: get_input_embeddings casts pixel_values to the patch embed's dtype (an f32 input would run
        # the whole tower with f32 activations)
        x = mx.array(pv).astype(mx.float32 if f32 else mx.bfloat16)
        t0 = time.time()
        h, _ = m(x, mx.array([grid]))
        mx.eval(h)
        dt = time.time() - t0
        h2, _ = m(x, mx.array([grid]))  # warm timing
        t0 = time.time()
        mx.eval(h2)
        dt2 = time.time() - t0
        e = np.array(h.astype(mx.float32))
        name = "ref_f32.npy" if f32 else "ref_bf16.npy"
        np.save(out / name, e)
        print(f"MLX tower {'f32' if f32 else 'bf16'}: {e.shape}, {dt * 1e3:.0f} ms cold / {dt2 * 1e3:.0f} ms warm -> {name}")
        del m
    b, f = np.load(out / "ref_bf16.npy"), np.load(out / "ref_f32.npy")
    print(f"MLX bf16 vs f32 tower: rel L2 {np.linalg.norm(b - f) / np.linalg.norm(f):.3e}")


if __name__ == "__main__":
    main()
