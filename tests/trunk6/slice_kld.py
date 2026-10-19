"""lab-trunk6: slice KLD vs the bf16 teacher with layers 0-3's trunk taken from a T6 pack (the 4-layer proof).

Same method as lab lead 5's engine/trunk/score.py (read-only imports): the 4-layer slice of the K5 pack through
fn_exl3, every 8-bit trunk Linear of layers 0-3 swapped for a dense f32 module whose weight is DECODED FROM THE T6
PACK (trellis / suh / svh -> W_hat, tools/convert_trunk_enc.decode), the final hyper-connection mixer from the pack's
bf16 tensors; scored by exl3/pack/kld_core.score_model against RUN/teacher/ref-{A,B,C}-L04; paired with lead 5's
base (Sushi 8-bit trunk) and its K6/HCu8/PLE8+fm row (dequantized lab encodes) via kld_core.report.
GPU job (MLX, ~15 GB): run it under tools/with_lock.sh.
  $PY -I -B tests/trunk6/slice_kld.py --pack ~/.cache/ijah/packs/Qwen3.8-Flash-Next-EXL3-Ijah --sets A,B,C
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import refenv  # noqa: E402
import convert_trunk_enc as E  # noqa: E402  (paths: IJAH_REF_DIR convert/, trunk/)

sys.path.insert(0, str(E.LAB / "trunk"))
import labcommon as LC  # noqa: E402  (lab-trunk paths: SLICE, RUN, BASE; read-only)

sys.path.insert(0, str(LC.EXL3 / "pack"))
import kld_core as K  # noqa: E402
import numpy as np  # noqa: E402

PFX = "language_model.model.layers."
MIX = "language_model.model.hyper_connection_mixer."
RES = ROOT / "tests/trunk6/results"


def st_get(path, key):
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    v = h[key]
    a, b = v["data_offsets"]
    dt = {"U16": np.uint16, "F16": np.float16, "BF16": np.uint16}[v["dtype"]]
    return np.array(np.memmap(path, dtype=dt, mode="r", offset=8 + n + a, shape=tuple(v["shape"])))


def set_child(root, path, new):
    parts = path.split(".")
    obj = root
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    setattr(obj, parts[-1], new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=refenv.ijah_paths.pack())
    ap.add_argument("--sets", default="A,B,C")
    ap.add_argument("--name", default="t6")
    a = ap.parse_args()
    import mlx.core as mx
    import mlx.nn as nn
    import fn_exl3 as F
    mx.set_cache_limit(1 << 30)
    pack = Path(a.pack).expanduser()
    wm = json.loads((pack / "model.safetensors.index.json").read_text())["weight_map"]

    class DenseF32(nn.Module):
        def __init__(self, w):
            super().__init__()
            self.weight = w

        def __call__(self, x):
            return (x.astype(mx.float32) @ self.weight.T).astype(x.dtype)

    t0 = time.time()
    model, _ = F.build_model(F.Pack(LC.SLICE), num_layers=4, verbose=False)
    model.eval()
    n = 0
    for li in range(4):
        for path, m in list(model.model.layers[li].named_modules()):
            if type(m) is not nn.QuantizedLinear:
                continue
            nm = f"{PFX}{li}.{path}"
            assert nm + ".trellis" in wm, f"{nm} not in the T6 pack"
            f = pack / wm[nm + ".trellis"]
            W = E.decode(st_get(f, nm + ".trellis"), st_get(f, nm + ".suh"), st_get(f, nm + ".svh"))  # [in, out]
            set_child(model.model.layers[li], path, DenseF32(mx.array(np.ascontiguousarray(W.T, np.float32))))
            n += 1
    for path, m in list(model.model.hyper_connection_mixer.named_modules()):
        if type(m) is nn.QuantizedLinear:
            key = f"{MIX}{path}.weight"
            w = (st_get(pack / wm[key], key).astype(np.uint32) << 16).view(np.float32)
            set_child(model.model.hyper_connection_mixer, path, DenseF32(mx.array(w)))
            n += 1
    mx.eval(model.parameters())
    print(f"slice built, {n} Linears from {pack.name} in {time.time() - t0:.0f} s", flush=True)
    head = K.load_head(LC.RUN)
    fwd = lambda ids, cache: model(ids, cache=cache).logits
    res = {"name": f"trunk-{a.name}", "pack": str(pack), "layers": 4, "variant": a.name, "sets": {}}
    for S in a.sets.split(","):
        t1 = time.time()
        r = K.score_model(fwd, model.make_cache, LC.RUN, S, 4, head)
        r["score_s"] = time.time() - t1
        res["sets"][S] = r
        print(f"{a.name} {S}: KLD {r['kld_mean']:.6f}, top-1 {r['top1_mean']:.4f}, rerun {r['rerun_max_abs']:.1e} "
              f"({r['score_s']:.0f} s)", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    out = RES / f"kld-L04-trunk-{a.name}.json"
    out.write_text(json.dumps(res, indent=1))
    lab = LC.OUT / "kld"
    K.report([str(lab / "kld-L04-trunk-base.json"), str(lab / "kld-L04-trunk-mix-6-hcu8-ple8+fm.json"), str(out)])


if __name__ == "__main__":
    main()
