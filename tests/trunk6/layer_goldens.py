"""lab-trunk6: the T6 trunk kernels on REAL layer activations (lab-ref's goldens, layers 0 / 1 / 3), vs the bf16 base.

  $PY -I -B tests/trunk6/layer_goldens.py make --dir D         # CPU: pairs, references, cases.json for t6-check
  LOCK_OWNER=lab-trunk6 tools/with_lock.sh "t6 goldens" HH:MM -- build/t6-check cases D
  $PY -I -B tests/trunk6/layer_goldens.py report --dir D

A pair = (golden Linear input, golden Linear output) of the MLX reference on the K5 pack (8-bit trunk, bf16
activations). A pair is used only if the 8-bit weight of the source pack reproduces the golden output from the golden
input (rel L2 < 2e-2: the bf16 rounding of the reference), which proves the input / output mapping. Per pair:
  q8 err   = || x W_q8 - x W_bf16 || / || x W_bf16 ||        the served 8-bit trunk vs the bf16 base weight (f64)
  t6 err   = || y_engine - x W_bf16 || / || x W_bf16 ||      the engine's EXL3 kernel on the T6 pack
  kernel   = || y_engine - x W_hat || / || x W_hat ||        engine vs the numpy decode of the same pack (f16 input)
Rows: decode1 = 1 row (SIMD), verify4 = 4 rows (NAX).
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests/trunk6"))
import refenv  # noqa: E402
import convert_trunk_enc as E  # noqa: E402
import numpy as np  # noqa: E402
from check_kernels import st_get  # noqa: E402

GOLD = ROOT / "tests/golden/data"
T6 = Path(refenv.ijah_paths.pack())                     # the T6 pack (the published one)
Q8 = Path(refenv.ijah_paths.golden_pack())              # its 8-bit-trunk source pack (IJAH_GOLDEN_PACK)
BASE = Path(os.environ.get("NE_TEACHER_RUN", Path.home() / ".cache/ijah/teacher-run")) / "base"   # bf16 base trunks
PFX = "language_model.model.layers."
PAIRS = [  # (linear, golden input, golden output)
    ("attn_hyper_connection.input_mix_weight_down", "attn_hc.normed", "attn_hc.down"),
    ("attn_hyper_connection.input_mix_weight_up", "attn_hc.mix", "attn_hc.up"),
    ("linear_attn.in_proj_qkv", "attn_hc.block_in", "gdn.qkv"),
    ("linear_attn.in_proj_z", "attn_hc.block_in", "gdn.z"),
    ("linear_attn.in_proj_a", "attn_hc.block_in", "gdn.a"),
    ("linear_attn.in_proj_b", "attn_hc.block_in", "gdn.b"),
    ("linear_attn.out_proj", "gdn.gated_norm", "gdn.out"),
    ("self_attn.q_proj", "attn_hc.block_in", "attn.q_proj"),
    ("self_attn.k_proj", "attn_hc.block_in", "attn.k_proj"),
    ("self_attn.v_proj", "attn_hc.block_in", "attn.v_proj"),
    ("self_attn.indexer.index_qk_proj", "attn_hc.block_in", "attn.idx_qk"),
    ("self_attn.o_proj", "attn.gated", "attn.out"),
    ("mlp_hyper_connection.input_mix_weight_down", "mlp_hc.normed", "mlp_hc.down"),
    ("mlp_hyper_connection.input_mix_weight_up", "mlp_hc.mix", "mlp_hc.up"),
    ("mlp.shared_expert.gate_proj", "mlp_hc.block_in", "moe.shared_gate_proj"),
    ("mlp.shared_expert.up_proj", "mlp_hc.block_in", "moe.shared_up_proj"),
    ("mlp.shared_expert.down_proj", "moe.shared_act", "moe.shared_out"),
    ("ple.key_proj", "ple.embeddings", "ple.keys"),
    ("ple.value_proj", "ple.embeddings", "ple.values"),
]
CASES = ["L00_decode1", "L00_verify4", "L01_decode1", "L01_verify4", "L03_decode1", "L03_verify4"]


def wm_of(p):
    return json.loads((p / "model.safetensors.index.json").read_text())["weight_map"]


def q8_weight(pack, wm, nm):
    """MLX 8-bit affine -> f64 [out, in]."""
    import struct

    def raw(key, dt):
        fk = pack / wm[key]
        with open(fk, "rb") as fh:
            nn = struct.unpack("<Q", fh.read(8))[0]
            hh = json.loads(fh.read(nn))
        v = hh[key]
        a, b = v["data_offsets"]
        return np.array(np.memmap(fk, dtype=dt, mode="r", offset=8 + nn + a, shape=tuple(v["shape"])))
    q = raw(nm + ".weight", np.uint32).view(np.uint8).astype(np.float64)          # [out, in]
    s = (raw(nm + ".scales", np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    b = (raw(nm + ".biases", np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    G = q.shape[1] // s.shape[1]
    return q * np.repeat(s, G, axis=1) + np.repeat(b, G, axis=1)


def bf16_weight(L, path):
    from convert_trunk import st_read
    return st_read(BASE / f"L{L:02d}-trunk.safetensors", f"model.language_model.layers.{L}.{path}.weight").astype(np.float64)


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def make(a):
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    wt, wq = wm_of(T6), wm_of(Q8)
    cases, meta = [], []
    for case in CASES:
        cd = GOLD / case
        L = int(case[1:3])
        for path, xin, yout in PAIRS:
            nm = f"{PFX}{L}.{path}"
            if not (cd / f"{xin}.npy").exists() or not (cd / f"{yout}.npy").exists() or nm + ".trellis" not in wt:
                continue
            x = np.load(cd / f"{xin}.npy").astype(np.float64)
            yg = np.load(cd / f"{yout}.npy").astype(np.float64)
            Wq = q8_weight(Q8, wq, nm)                                     # [out, in]
            if x.size % Wq.shape[1] or yg.size % Wq.shape[0]:
                continue
            x = x.reshape(-1, Wq.shape[1])        # [rows, in] (e.g. the gated norm [1, 1, 48, 128])
            yg = yg.reshape(-1, Wq.shape[0])
            if x.shape[0] != yg.shape[0]:
                print(f"skip {case} {path}: rows {x.shape[0]} vs {yg.shape[0]}")
                continue
            chk = rel(x @ Wq.T, yg)
            if chk > 2e-2:
                print(f"skip {case} {path}: x W_q8 vs golden rel {chk:.3f} (mapping not a plain Linear)")
                continue
            f = T6 / wt[nm + ".trellis"]
            Wh = E.decode(st_get(f, nm + ".trellis"), st_get(f, nm + ".suh"), st_get(f, nm + ".svh"))   # [in, out]
            Wb = bf16_weight(L, path)
            tag = f"{case}_{path.replace('.', '_')}"
            x.astype(np.float32).tofile(d / f"{tag}.x.bin")
            (x @ Wh).astype(np.float32).tofile(d / f"{tag}.y.bin")
            (x @ Wb.T).astype(np.float32).tofile(d / f"{tag}.yb.bin")
            yg.astype(np.float32).tofile(d / f"{tag}.yg.bin")
            (x @ Wq.T).astype(np.float32).tofile(d / f"{tag}.yq.bin")
            R = x.shape[0]
            cases.append(dict(name=nm, tag=tag, R=R, path="auto", x=f"{tag}.x.bin", y=f"{tag}.y.bin", tol=2e-3))
            meta.append(dict(tag=tag, case=case, linear=path, R=R, map_check=chk))
            print(f"{case} {path}: R {R}, mapping check {chk:.1e}", flush=True)
    (d / "cases.json").write_text(json.dumps(dict(pack=str(T6), cases=cases), indent=1))
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(cases)} cases")


def report(a):
    d = Path(a.dir)
    rows = []
    for m in json.loads((d / "meta.json").read_text()):
        t = m["tag"]
        y = np.fromfile(d / f"{t}.out.bin", np.float32).astype(np.float64)
        yh = np.fromfile(d / f"{t}.y.bin", np.float32).astype(np.float64)
        yb = np.fromfile(d / f"{t}.yb.bin", np.float32).astype(np.float64)
        yq = np.fromfile(d / f"{t}.yq.bin", np.float32).astype(np.float64)
        rows.append((m["case"], m["linear"], m["R"], rel(yq, yb), rel(y, yb), rel(y, yh)))
    print(f"| case | linear | rows | q8 err vs bf16 | t6 err vs bf16 | t6 / q8 | kernel vs decode |")
    print("|---|---|---|---|---|---|---|")
    for c, l, R, eq, et, ek in rows:
        print(f"| {c} | {l} | {R} | {eq:.2e} | {et:.2e} | {et / eq:.2f} | {ek:.1e} |")
    r = np.array([[x[3], x[4]] for x in rows])
    print(f"geomean t6/q8 = {np.exp(np.mean(np.log(r[:, 1] / r[:, 0]))):.2f} over {len(rows)} pairs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["make", "report"])
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    make(a) if a.mode == "make" else report(a)
