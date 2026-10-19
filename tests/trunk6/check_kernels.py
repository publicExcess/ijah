"""lab-trunk6: kernel exactness on the real encoded matrices of a T6 pack (CPU part; the GPU part is build/t6-check).

  PY=.venv/bin/python
  $PY -I -B tests/trunk6/check_kernels.py make --pack ~/.cache/ijah/packs/Qwen3.8-Flash-Next-EXL3-Ijah --layers 0-3 --dir D
  LOCK_OWNER=lab-trunk6 tools/with_lock.sh "t6-check cases" HH:MM -- build/t6-check cases D
  $PY -I -B tests/trunk6/check_kernels.py report --dir D

For every EXL3 Linear of the layers (and the bf16 final mixer): W_hat decoded independently in numpy
(tools/convert_trunk_enc.decode: exl3_codec bit unpack, MCG table, Hadamards), x = random N(0,1) rows with a few
outlier channels; two references:
  y_emul  the kernel arithmetic: x * suh and H_hi in f32 (same butterfly order), rounded to f16 (the prepared input),
          @ W' (exact f16 codebook values), H_ho * svh  -> the strict check (tol 5e-5: f32 accumulation order only)
  y_exact x @ W_hat in f64                              -> what the f16 input preparation costs (reported)
Cases: rows 1 / 2 / 4 / 8 on the SIMD path, 4 / 8 / 16 on the NAX path, one bf16 case per mixer matrix at 1 / 4 rows.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import refenv  # noqa: E402
import convert_trunk_enc as E  # noqa: E402
import exl3_codec as C  # noqa: E402
import numpy as np  # noqa: E402

PFX = "language_model.model.layers."
MIX = "language_model.model.hyper_connection_mixer."


def st_header(path):
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def st_get(path, key):
    h, start = st_header(path)
    v = h[key]
    a, b = v["data_offsets"]
    dt = {"U16": np.uint16, "F16": np.float16, "BF16": np.uint16, "F32": np.float32}[v["dtype"]]
    return np.array(np.memmap(path, dtype=dt, mode="r", offset=start + a, shape=tuple(v["shape"])))


def fwht_f32(x, n):
    """The kernel's input Hadamard in f32, same butterfly order (index bits 0, 1, ... within blocks of n), then the
    1/sqrt(n) scale: rows of x [R, k] (H symmetric, so x H = H x per row block)."""
    R, k = x.shape
    v = x.astype(np.float32).reshape(R, k // n, n).copy()
    b = 1
    while b < n:
        v = v.reshape(R, k // n, n // (2 * b), 2, b)
        lo, hi = v[:, :, :, 0, :].copy(), v[:, :, :, 1, :].copy()
        v[:, :, :, 0, :] = lo + hi
        v[:, :, :, 1, :] = lo - hi
        v = v.reshape(R, k // n, n)
        b *= 2
    return (v * np.float32(1.0 / np.sqrt(n))).reshape(R, k)


def linears(pack, layers):
    wm = json.loads((pack / "model.safetensors.index.json").read_text())["weight_map"]
    out = []
    for L in layers:
        pre = f"{PFX}{L}."
        out += sorted(k[:-len(".trellis")] for k in wm if k.startswith(pre) and k.endswith(".trellis")
                      and ".switch_mlp." not in k)
    return out, wm


def make(a):
    pack, d = Path(a.pack).expanduser(), Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in a.layers.split(",")] if "," in a.layers else list(
        range(int(a.layers.split("-")[0]), int(a.layers.split("-")[-1]) + 1))
    names, wm = linears(pack, layers)
    rng = np.random.default_rng(7)
    cases = []
    for nm in names:
        f = pack / wm[nm + ".trellis"]
        tr, suh, svh = st_get(f, nm + ".trellis"), st_get(f, nm + ".suh"), st_get(f, nm + ".svh")
        k, m = tr.shape[0] * 16, tr.shape[1] * 16
        hk, hm = E.had_size(k), E.had_size(m)
        inner = C.decode_inner(tr, tr.shape[-1], "mcg", E.WINDOW).astype(np.float64)   # W' [in, out]
        W = E.reconstruct(inner, suh, svh)
        tag0 = nm[len(PFX):].replace(".", "_")
        for path, rows in (("simd", (1, 2, 4, 8)), ("nax", (4, 8, 16))):
            for R in rows:
                if a.quick and R not in (1, 4, 8):
                    continue
                x = rng.standard_normal((R, k))
                x[:, rng.integers(0, k, 4)] *= 20.0                    # a few outlier channels
                x = x.astype(np.float32)
                xs = (x * suh.astype(np.float32)).astype(np.float32)     # f32 product, as the kernel
                xh = fwht_f32(xs, hk).astype(np.float16).astype(np.float64)
                y_emul = E.had_r(xh @ inner, hm) * svh.astype(np.float64)[None, :]
                y_exact = x.astype(np.float64) @ W
                tag = f"{tag0}_{path}_R{R}"
                x.tofile(d / f"{tag}.x.bin")
                y_emul.astype(np.float32).tofile(d / f"{tag}.y.bin")
                y_exact.astype(np.float32).tofile(d / f"{tag}.yx.bin")
                cases.append(dict(name=nm, tag=tag, R=R, path=path, x=f"{tag}.x.bin", y=f"{tag}.y.bin",
                                  yx=f"{tag}.yx.bin", tol=5e-5, K=int(tr.shape[-1] // 16), shape=[k, m]))
        print(f"{nm}: [{k}, {m}] K{tr.shape[-1] // 16} H{hk}/{hm}", flush=True)
    for nm in ("input_mix_weight_down", "input_mix_weight_up"):
        key = f"{MIX}{nm}.weight"
        if key not in wm:
            continue
        w = (st_get(pack / wm[key], key).astype(np.uint32) << 16).view(np.float32).astype(np.float64)  # [out, in]
        for R in (1, 4):
            x = rng.standard_normal((R, w.shape[1])).astype(np.float32)
            tag = f"mixer_{nm}_R{R}"
            x.tofile(d / f"{tag}.x.bin")
            y = x.astype(np.float64) @ w.T
            y.astype(np.float32).tofile(d / f"{tag}.y.bin")
            y.astype(np.float32).tofile(d / f"{tag}.yx.bin")
            cases.append(dict(name=f"{MIX}{nm}", tag=tag, R=R, path="auto", x=f"{tag}.x.bin", y=f"{tag}.y.bin",
                              yx=f"{tag}.yx.bin", tol=1e-5, K=16, shape=[w.shape[1], w.shape[0]]))
    (d / "cases.json").write_text(json.dumps(dict(pack=str(pack), cases=cases), indent=1))
    print(f"{len(cases)} cases -> {d / 'cases.json'}")


def report(a):
    d = Path(a.dir)
    cs = json.loads((d / "cases.json").read_text())["cases"]
    worst = {}
    for c in cs:
        o = d / f"{c['tag']}.out.bin"
        if not o.exists():
            continue
        y = np.fromfile(o, np.float32).astype(np.float64)
        ye = np.fromfile(d / c["y"], np.float32).astype(np.float64)
        yx = np.fromfile(d / c["yx"], np.float32).astype(np.float64)
        r_e = np.linalg.norm(y - ye) / np.linalg.norm(ye)
        r_x = np.linalg.norm(y - yx) / np.linalg.norm(yx)
        key = (c["path"], c["K"])
        w = worst.setdefault(key, [0, 0, 0])
        w[0] = max(w[0], r_e)
        w[1] = max(w[1], r_x)
        w[2] += 1
    for (p, K), (re, rx, n) in sorted(worst.items()):
        print(f"{p:5s} K{K:<2d} {n:3d} cases: max rel vs kernel-arithmetic ref {re:.2e}, vs f64 exact {rx:.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["make", "report"])
    ap.add_argument("--pack", default=refenv.ijah_paths.pack())
    ap.add_argument("--layers", default="0-3")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    make(a) if a.mode == "make" else report(a)
