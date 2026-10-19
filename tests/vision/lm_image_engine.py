"""lab-vision whole-model check on an image prompt: the engine (chunked prefill with image embeddings + MRoPE positions,
then teacher-forced decode steps on the reference's greedy tokens) vs tests/vision/lm_image_ref.py's logits.

  tools/with_lock.sh "vision LM engine (~81 GB)" HH:MM -- $PY -I -B tests/vision/lm_image_engine.py REF_DIR [--pack K5]

Two arms: the reference's image features (isolates the language-model side: scatter + positions) and the engine's
own vision tower (end to end). Per position: KLD(ref || engine) and top-1 agreement.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "server"))
import refenv  # noqa: E402

refenv.add_paths()
import ctypes as C  # noqa: E402

import numpy as np  # noqa: E402

import engine as E  # noqa: E402
import vision as V  # noqa: E402


def logp(x):
    x = np.asarray(x, np.float64)
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def kld(ref, eng):
    p, q = logp(ref), logp(eng)
    return float((np.exp(p) * (p - q)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("--pack", default=refenv.ijah_paths.golden_pack())
    ap.add_argument("--tmax", type=int, default=8192)
    a = ap.parse_args()
    r = Path(a.ref)
    meta = json.loads((r / "meta.json").read_text())
    toks = np.load(r / "tokens.npy").tolist()
    grid = tuple(int(x) for x in np.load(r / "grid.npy"))
    ref_lg = E.bf16_to_f32(np.load(r / "logits.npy"))
    greedy = np.load(r / "greedy.npy").tolist()
    ref_feats = np.load(r / "feats.npy")
    pos, delta = V.positions(toks, [grid])
    rpos = np.load(r / "positions.npy")
    print(f"positions == reference get_rope_index: {np.array_equal(pos.T, rpos)} (delta {delta}, ref {meta['rope_delta']})")
    if (r / "logits_textpos.npy").exists():
        ctl = E.bf16_to_f32(np.load(r / "logits_textpos.npy"))
        kc = [kld(ref_lg[i], ctl[i]) for i in range(len(ctl))]
        print(f"reference itself, 1-D text positions vs MRoPE: first-token KLD {kc[0]:.2e}, mean {np.mean(kc):.2e} "
              f"(the scale of a position error)")
    eng = E.Engine()
    eng.set_ref_rounding(True)  # vs the MLX reference (as before the f32 library default)
    pack = eng.pack_open(a.pack)
    t0 = time.time()
    m = eng.fn_model(pack, a.tmax)
    print(f"model loaded in {time.time() - t0:.0f} s", flush=True)
    sm = eng.fn("ne_fn_model_set_mrope", C.c_int, [C.c_void_p, C.c_void_p, C.c_int, C.c_int])
    si = eng.fn("ne_fn_model_set_image_rows", C.c_int, [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p])
    idx = np.ascontiguousarray(np.nonzero(np.array(toks) == V.IMAGE_TOKEN)[0], np.int32)
    vis = eng.vision(pack)
    from PIL import Image  # noqa: F401
    im = V.VisionImage(V.load_image(meta["image"])[0])
    assert im.grid == grid
    own = vis.encode(V.engine_patches(im.pixel_values), [grid])
    fr, fo = E.bf16_to_f32(ref_feats), E.bf16_to_f32(own)
    print(f"engine tower vs reference features: rel L2 {np.linalg.norm(fo - fr) / np.linalg.norm(fr):.3e} "
          f"({vis.gpu_ms:.0f} ms GPU)")
    vis.free()
    p32 = np.ascontiguousarray(pos, np.int32)
    res = {}
    for arm, feats in (("ref_features", ref_feats), ("engine_tower", own), ("text_positions", ref_feats)):
        m.reset()
        if arm == "text_positions":  # control: image embeddings but 1-D positions (shows the positions matter)
            E_ = sm(C.c_void_p(m.ptr), None, 0, 0)
        else:
            E_ = sm(C.c_void_p(m.ptr), p32.ctypes.data, len(toks), delta)
        assert E_ == 0, eng.last_error()
        f = np.ascontiguousarray(feats, np.uint16)
        assert si(C.c_void_p(m.ptr), idx.ctypes.data, len(idx), f.ctypes.data) == 0, eng.last_error()
        t0 = time.time()
        nx, lg, _ = m.prefill(toks, logits=True)
        t_pre = time.time() - t0
        klds, top = [kld(ref_lg[0], lg)], [int(np.argmax(lg)) == int(np.argmax(ref_lg[0]))]
        for i, t in enumerate(greedy[:-1]):
            _, _, _, lg = m.step(t, logits=True)
            klds.append(kld(ref_lg[i + 1], lg))
            top.append(int(np.argmax(lg)) == int(np.argmax(ref_lg[i + 1])))
        res[arm] = {"first_kld": klds[0], "mean_kld": float(np.mean(klds)), "max_kld": float(np.max(klds)),
                    "top1": float(np.mean(top)), "prefill_s": t_pre}
        print(f"{arm}: first-token KLD {klds[0]:.2e}, mean over {len(klds)} positions {np.mean(klds):.2e} "
              f"(max {np.max(klds):.2e}), top-1 {np.mean(top) * 100:.0f}%; prefill {len(toks)} tokens {t_pre:.2f} s",
              flush=True)
    # greedy generation with the engine's own features (sanity)
    m.reset()
    sm(C.c_void_p(m.ptr), p32.ctypes.data, len(toks), delta)
    si(C.c_void_p(m.ptr), idx.ctypes.data, len(idx), np.ascontiguousarray(own, np.uint16).ctypes.data)
    m.set_sampler(0.0, 1.0, 1, 0)
    nx, _, _ = m.prefill(toks)
    gen = [nx]
    for _ in range(len(greedy) - 1):
        nx, _, _, _ = m.step(gen[-1])
        gen.append(nx)
    print(f"engine greedy == reference greedy for {sum(1 for x, y in zip(gen, greedy) if x == y)}/{len(greedy)} tokens")
    res["engine_greedy"] = gen
    (r / "engine.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
