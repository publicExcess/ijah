"""Convert the Flash-Next trunk to EXL3 (lead 5's recipe) and write an aligned pack that reuses the K5 experts (lab-trunk6).
This is how the published packs' trunks were made. It needs the MLX / numpy EXL3 reference (fn_exl3, exl3_codec, encode,
the imatrix calibration and the bf16 base fetcher) under IJAH_REF_DIR, which is not part of this repository; it is
shipped for reference and does not run without it.

  PY=.venv/bin/python
  LOCK_OWNER=lab-trunk6 NEED_GB=40 LOCK_WAIT=60 tools/with_lock.sh "trunk6 night (~35 GB)" HH:MM -- \
      $PY -I -B tools/convert_trunk.py --layers 0-47            # the night run (docs/trunk6.md section 5, ~65 min)
  $PY -I -B tools/convert_trunk.py --status                     # what is done, per-layer times, estimate (CPU)

Stages (each resumable; rerun the same command after a stop):
  skeleton  OUT/<every file of SRC> -> symlink to its real target (the K5 experts, n-gram table, MTP, lm_head, ...)
  hessians  full H = X^T X / rows of every trunk Linear input, layer groups of --group layers: fn_exl3 (MLX) on SRC,
            layer-sequential (see capture()), over the imatrix calibration text (279.5K tokens, 2048-token windows
            from a fresh cache; bit-identical to lead 5's capture), hooks on the group's QuantizedLinears; the inputs that are
            shared (GDN in_proj_z/a/b = in_proj_qkv; attention k/v/indexer = q_proj; shared up = gate) are captured
            once (--verify-aliases captures all of them and checks they are bit-identical) -> WORK/hessians/Lxx/
  encode    per layer, per Linear: the HF bf16 base weight (BASE/Lxx-trunk.safetensors, fetched by range reads if
            missing) -> EXL3 MCG w15 at the recipe's rate (convert_trunk_enc.encode), decode check, ->
            OUT/model-t6-Lxx.safetensors (64-byte aligned); the layer's H is deleted unless --keep-h
  mixer     the final hyper-connection mixer in bf16 (BASE/globals.safetensors) -> OUT/model-t6-mixer.safetensors
  index     OUT/model.safetensors.index.json = SRC's with every converted Linear's .weight/.scales/.biases replaced
            by .trellis/.suh/.svh (or the bf16 .weight for the mixer); rewritten after every layer, so a partial
            pack is valid (unconverted layers keep the 8-bit trunk)
Recipes (rate per Linear; lead 5): t6 = K6 everywhere, K8 for the HC up-mixes (input_mix_weight_up) and the PLE
key/value projections, final mixer bf16 (2.80 GB/token); t7 = the same with K7 bulk (3.22 GB/token).
RAM: a watchdog thread exits (status 3) if system free memory < --min-free-pct or MLX memory > --mem-limit GiB.
Not converted: the MTP layer (stays 8-bit), routers / norms / conv / block_inject (bf16 as packed), embed, lm_head.

Fine-tunes that changed only some trunk Linears (lab-swift; Swift-1.5: 300 Linears, docs/trunk6.md section 9):
  --delta DIR   a Linear's bf16 weight is DIR/<pack prefix>.bf16 (raw bf16, HF [out, in] layout, e.g.
                written by the fine-tune's own packaging) when that file
                exists, else the base's; --src = the fine-tune's own (aligned) 8-bit pack, so the Hessians are
                captured through the fine-tune. Each delta Linear logs its relative change vs the base weight.
  $PY -I -B tools/convert_trunk.py --src <Swift K5 pack> --out <Swift T6 pack> --work <work dir> \
      --delta <bf16 delta dir> --layers 0-47
"""
import argparse
import gc
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import convert_trunk_enc as E  # noqa: E402  (sets sys.path: IJAH_REF_DIR convert/, trunk/)

import numpy as np  # noqa: E402

LAB = E.LAB                                  # IJAH_REF_DIR: pack/fetch_base.py, imatrix/, model/ (fn_exl3)
PY = E.ijah_paths.python()
CONV = Path.home() / ".cache/ijah/convert"
SRC = CONV / "Qwen3.8-Flash-Next-EXL3-K5-a16"   # the aligned K5 pack (8-bit trunk)
OUT = CONV / "Qwen3.8-Flash-Next-EXL3-K5-T6"
WORK = CONV / "trunk6-work"
BASE = WORK / "base"                         # the HF bf16 base, fetched per layer by the reference's fetch_base.py
PFX = "language_model.model.layers."
HF = "model.language_model.layers."
MIXER = "language_model.model.hyper_connection_mixer."
HCW = 4 * 2560   # residual stream width (4 hyper-connection streams)
ALIAS = {"linear_attn.in_proj_z": "linear_attn.in_proj_qkv", "linear_attn.in_proj_a": "linear_attn.in_proj_qkv",
         "linear_attn.in_proj_b": "linear_attn.in_proj_qkv", "self_attn.k_proj": "self_attn.q_proj",
         "self_attn.v_proj": "self_attn.q_proj", "self_attn.indexer.index_qk_proj": "self_attn.q_proj",
         "mlp.shared_expert.up_proj": "mlp.shared_expert.gate_proj"}
NEVER_HOOKED = ("linear_attn.in_proj_a", "linear_attn.in_proj_b")   # mlx-vlm runs them through gather_qmm in prefill


def rate(recipe, path):
    if recipe not in ("t6", "t7"):
        raise ValueError(recipe)
    if path.endswith("input_mix_weight_up") or path.startswith("ple."):
        return 8
    return 6 if recipe == "t6" else 7


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.rename(path)


# ---------------------------------------------------------------- RAM watchdog
def free_pct():
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=20).stdout
        return int(re.search(r"free percentage: (\d+)%", out).group(1))
    except Exception:
        return 100


def watchdog(min_free_pct, mem_limit_gib):
    def run():
        while True:
            f = free_pct()
            used = 0.0
            if "mlx.core" in sys.modules:
                mx = sys.modules["mlx.core"]
                used = (mx.get_active_memory() + mx.get_cache_memory()) / 2**30
            if f < min_free_pct or used > mem_limit_gib:
                log(f"WATCHDOG: system free {f}% (min {min_free_pct}), MLX {used:.1f} GiB (max {mem_limit_gib}); exiting")
                os._exit(3)
            time.sleep(3)
    threading.Thread(target=run, daemon=True).start()


# ---------------------------------------------------------------- safetensors
def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


_DT = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "U16": np.uint16, "U32": np.uint32}


def st_read(path, key, raw=False):
    h, start = st_header(path)
    v = h[key]
    a, b = v["data_offsets"]
    arr = np.array(np.memmap(path, dtype=_DT[v["dtype"]], mode="r", offset=start + a, shape=tuple(v["shape"])))
    if v["dtype"] == "BF16" and not raw:
        arr = (arr.astype(np.uint32) << 16).view(np.float32)
    return arr


def st_write(path, tensors, metadata, align=64):
    """tensors: list of (name, dtype string, np.ndarray); every data offset (absolute) a multiple of `align`."""
    hdr, off, blobs = {}, 0, []
    for name, dt, arr in tensors:
        b = np.ascontiguousarray(arr).tobytes()
        pad = -off % align
        off += pad
        blobs.append((pad, b))
        hdr[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
    hdr["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    hb = json.dumps(hdr).encode()
    hb += b" " * (-(8 + len(hb)) % align)       # data start (8 + header) aligned too
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for pad, b in blobs:
            f.write(b"\0" * pad)
            f.write(b)
    tmp.rename(path)


# ---------------------------------------------------------------- pack structure
def src_index(src):
    return json.loads((src / "model.safetensors.index.json").read_text())


def layer_linears(wm, L):
    """Trunk Linears of layer L = the 8-bit affine tensors of the source pack (prefix has .scales), sorted."""
    pre = f"{PFX}{L}."
    return sorted(k[len(pre):-len(".scales")] for k in wm if k.startswith(pre) and k.endswith(".scales")
                  and ".switch_mlp." not in k)


def skeleton(src, out):
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src.iterdir()):
        if f.name == "model.safetensors.index.json":
            continue
        dst = out / f.name
        if dst.is_symlink() or dst.exists() or (out / f"model-t6-rest-{f.name}").exists():
            continue
        dst.symlink_to(f.resolve())
        n += 1
    if n:
        log(f"skeleton: {n} symlinks {out} -> real files of {src}")


def write_index(src, out, args, final=False):
    """Index of the current state. final: also the bf16 mixer, and `slim`: every source shard that still holds a
    replaced tensor (the 8-bit trunk shards; the mixer) is rewritten as OUT/model-t6-rest-<shard> with only the
    tensors the index still uses, and its symlink removed, so the pack has no stale or duplicate tensor names (the
    engine's Pack maps every tensor of every referenced shard; a queue residency set over all of them would page the
    unused 8-bit trunk in). Run once per invocation, at the end (rereads the source shards: ~5 GB for a partial pack)."""
    idx = src_index(src)
    wm = dict(idx["weight_map"])
    done = []
    for f in sorted(out.glob("model-t6-L*.safetensors")):
        L = int(f.name[10:12])
        h, _ = st_header(f)
        for path in {k[len(f"{PFX}{L}."):].rsplit(".", 1)[0] for k in h if k != "__metadata__"}:
            p = f"{PFX}{L}.{path}"
            for s in ("weight", "scales", "biases"):
                wm.pop(f"{p}.{s}", None)
            for s in ("trellis", "suh", "svh"):
                wm[f"{p}.{s}"] = f.name
        done.append(L)
    mixf = out / "model-t6-mixer.safetensors"
    use_mix = final and mixf.exists()
    if use_mix:
        h, _ = st_header(mixf)
        for k in h:
            if k == "__metadata__":
                continue
            p = k.rsplit(".", 1)[0]
            for s in ("weight", "scales", "biases"):
                wm.pop(f"{p}.{s}", None)
            wm[k] = mixf.name
    if final:
        slim(src, out, idx["weight_map"], wm)
    else:   # mid-run: shards already slimmed by an earlier run stay as they are (layers converted since keep their
        rest = {f.name[len("model-t6-rest-"):]: f.name for f in out.glob("model-t6-rest-*.safetensors")}
        for k, v in wm.items():       # stale 8-bit copies there until the final slim; names never clash)
            if v in rest:
                wm[k] = rest[v]
        if (out / "model.safetensors.index.json").exists():
            old = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
            if any(v == mixf.name for v in old.values()):   # keep an earlier run's final mixer state
                for k, v in old.items():
                    if v == mixf.name:
                        wm[k] = v
                        for s in ("scales", "biases"):
                            wm.pop(k.rsplit(".", 1)[0] + "." + s, None)
    meta = dict(idx.get("metadata", {}))
    meta["trunk6"] = {"recipe": args.recipe, "layers": done, "mixer_bf16": any(v == mixf.name for v in wm.values()),
                      "source": str(src), "producer": "ijah tools/convert_trunk.py"}
    if args.delta:
        meta["trunk6"]["delta"] = str(args.delta)
    meta.pop("total_size", None)
    jdump(out / "model.safetensors.index.json", {"metadata": meta, "weight_map": dict(sorted(wm.items()))})


def slim(src, out, src_wm, wm):
    """Rewrite source shards whose tensors are partly replaced (see write_index); updates wm in place."""
    used = {}
    for k, v in wm.items():
        used.setdefault(v, set()).add(k)
    for fname in sorted(set(src_wm.values())):
        total = {k for k, v in src_wm.items() if v == fname}
        keep = {k for k in total if wm.get(k) == fname}
        restname = f"model-t6-rest-{fname}"
        if keep == total and not (out / restname).exists():
            continue                                          # untouched shard: stays a symlink
        sf = src / fname
        h, start = st_header(sf)
        if keep:
            ts = []
            with open(sf, "rb") as f:
                for k in sorted(keep):
                    v = h[k]
                    a, b = v["data_offsets"]
                    f.seek(start + a)
                    ts.append((k, v["dtype"], np.frombuffer(f.read(b - a), np.uint8), v["shape"]))
            st_write_raw(out / restname, ts, {"source": str(sf), "producer": "ijah tools/convert_trunk.py"})
            for k in keep:
                wm[k] = restname
            log(f"slim: {fname} -> {restname} ({len(keep)} of {len(total)} tensors)")
        elif (out / restname).exists():
            (out / restname).unlink()
        link = out / fname
        if link.is_symlink():
            link.unlink()


def st_write_raw(path, tensors, metadata, align=64):
    """tensors: (name, dtype, raw uint8 bytes, shape) -> aligned safetensors (as st_write)."""
    hdr, off, blobs = {}, 0, []
    for name, dt, b, shape in tensors:
        pad = -off % align
        off += pad
        blobs.append((pad, b))
        hdr[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + b.size]}
        off += b.size
    hdr["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    hb = json.dumps(hdr).encode()
    hb += b" " * (-(8 + len(hb)) % align)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for pad, b in blobs:
            f.write(b"\0" * pad)
            f.write(b.tobytes())
    tmp.rename(path)


# ---------------------------------------------------------------- base weights
def base_file(L, args):
    for d in (Path(args.base), WORK / "base"):
        f = d / f"L{L:02d}-trunk.safetensors"
        if f.exists():
            return f
    d = WORK / "base"
    d.mkdir(parents=True, exist_ok=True)
    log(f"L{L}: base trunk missing, fetching by range reads into {d}")
    subprocess.run([PY, "-I", "-B", str(LAB / "pack/fetch_base.py"), "--out", str(d), "--layer", str(L),
                    "--no-experts"], check=True)
    return d / f"L{L:02d}-trunk.safetensors"


def base_globals(args):
    for d in (Path(args.base), WORK / "base"):
        if (d / "globals.ok").exists():
            return d / "globals.safetensors"
    d = WORK / "base"
    subprocess.run([PY, "-I", "-B", str(LAB / "pack/fetch_base.py"), "--out", str(d), "--globals"], check=True)
    return d / "globals.safetensors"


# ---------------------------------------------------------------- Hessians
def hdir(L):
    return WORK / "hessians" / f"L{L:02d}"


def h_done(L):
    return (hdir(L) / "meta.json").exists()


def hstate(L):
    return WORK / f"hstate-L{L:02d}.npy"     # residual stream (bf16 bits) entering layer L, all windows concatenated


def build_layers(pack, lo, hi):
    """fn_exl3.build_model restricted to layers [lo, hi) (+ the embedding when lo == 0): the other layers' parameters
    stay lazy and are never evaluated."""
    import mlx.core as mx
    import fn_exl3 as F
    F._patch_classes(pack)
    cfg = F.text_config(pack, True, hi)
    mcfg = F.ModelConfig.from_dict(pack.config)
    mcfg.text_config = cfg
    model = F.FlashNextExl3LM(cfg, mcfg)
    F._quantize_like_pack(model, pack, "language_model.")
    if lo == 0:
        F.load_module(model.model.embed_tokens, pack, "language_model.model.embed_tokens.", [])
        mx.eval(model.model.embed_tokens.parameters())
    for i in range(lo, hi):
        layer = model.model.layers[i]
        F.fit_exl3_banks(layer, pack, f"language_model.model.layers.{i}.")
        F.load_module(layer, pack, f"language_model.model.layers.{i}.", [])
        mx.eval(layer.parameters())
    pack.drop_files()
    model.eval()
    return model


def capture(layers, args, wm):
    """Full H for the Linears of `layers`, LAYER-SEQUENTIAL: each calibration window (2048 tokens, fresh cache) runs
    through layers [s, max + 1) one layer at a time, as Qwen4ExpModel.__call__ does (tiled embedding, the layer's own
    masks, its own fresh cache), starting from the stored residual stream entering layer s (hstate, written by the
    previous group) or from the embedding; hooks only on `layers`. Writes hstate for the next group. Each layer is
    loaded and run once per night instead of once per group (a truncated whole-model forward per group re-ran every
    earlier layer: ~3.5 h of forwards for 48 layers)."""
    import mlx.core as mx
    import mlx.nn as nn
    sys.path.insert(0, str(LAB / "imatrix"))
    sys.path.insert(0, str(LAB / "model"))
    import common as Cm
    import fn_exl3 as F
    from mlx_vlm.models.cache import ArraysCache
    Lm = F.L
    t0 = time.time()
    tok = Cm.tokenizer()
    wins, desc = Cm.calib_windows(tok, Cm.WINDOW, args.tokens)
    offs = np.cumsum([0] + [len(w) for w in wins])
    a, b = min(layers), max(layers) + 1
    s = max([k for k in range(a, 0, -1) if hstate(k).exists()] + [0])
    if s and np.load(hstate(s), mmap_mode="r").shape[0] != offs[-1]:
        raise RuntimeError(f"{hstate(s)}: token count differs from this calibration set (--tokens?)")
    model = build_layers(F.Pack(Path(args.src)), s, b)
    tload = time.time() - t0
    acc, names = {}, {}

    class CapQL(nn.QuantizedLinear):
        def __call__(self, x):
            name = names[id(self)]
            xf = x.reshape(-1, x.shape[-1]).astype(mx.float32)
            sm, n = acc.get(name, (0, 0))
            acc[name] = (sm + xf.T @ xf, n + xf.shape[0])
            return super().__call__(x)

    for L in layers:
        for path, m in model.model.layers[L].named_modules():
            if type(m) is nn.QuantizedLinear and (args.verify_aliases or path not in ALIAS):
                m.__class__ = CapQL
                names[id(m)] = (L, path)
    hin = np.load(hstate(s), mmap_mode="r") if s else None
    hout = np.lib.format.open_memmap(str(hstate(b)) + ".tmp.npy", mode="w+", dtype=np.uint16,
                                     shape=(int(offs[-1]), HCW)) if b < args.num_layers else None
    log(f"capture L{a}-{b - 1}: layers {s}-{b - 1} loaded in {tload:.0f} s ({'hstate L%d' % s if s else 'embedding'}), "
        f"{len(names)} hooks, {desc['tokens']} tokens in {len(wins)} windows")
    t1 = time.time()
    for wi, w in enumerate(wins):
        ids = mx.array([w])
        if hin is None:
            h = mx.tile(model.model.embed_tokens(ids), (1, 1, model.model.args.hc_count))
        else:
            h = mx.array(np.ascontiguousarray(hin[offs[wi]:offs[wi + 1]]))[None].view(mx.bfloat16)
        for L in range(s, b):
            layer = model.model.layers[L]
            cache = (ArraysCache(size=4 if "ple" in layer else 2) if layer.is_linear else Lm.QSAKVCache())
            mask = (Lm._create_qwen3_5_ssm_mask(h, cache) if layer.is_linear
                    else Lm._create_qwen4_exp_attention_mask(h, cache))
            h = layer(h, ids, mask=mask, cache=cache, position_ids=None)   # as Qwen4ExpModel.__call__ passes it
            mx.eval(h, *[v[0] for v in acc.values()])
            del cache
        if hout is not None:
            hout[offs[wi]:offs[wi + 1]] = np.array(h.view(mx.uint16))[0]
        del h
        if wi % 16 == 15 or wi == len(wins) - 1:
            log(f"  {wi + 1}/{len(wins)} windows, {offs[wi + 1] / (time.time() - t1):.0f} tok/s, "
                f"peak {mx.get_peak_memory() / 2**30:.1f} GiB")
            mx.clear_cache()
    tfwd = time.time() - t1
    if hout is not None:
        hout.flush()
        del hout
        os.replace(str(hstate(b)) + ".tmp.npy", hstate(b))
    for L in layers:
        d = hdir(L)
        d.mkdir(parents=True, exist_ok=True)
        rows, aliases = {}, {}
        lin = layer_linears(wm, L)
        for (LL, path), (sm, n) in sorted(acc.items()):
            if LL != L:
                continue
            np.save(d / f"{path}.npy", np.array(sm / n).astype(np.float32))
            rows[path] = int(n)
        for path in lin:
            if path not in rows:
                src = ALIAS.get(path)
                assert src in rows, (L, path, "no H and no alias source")
                aliases[path] = src
                rows[path] = rows[src]
        if args.verify_aliases:
            for path, src in ALIAS.items():
                if path not in NEVER_HOOKED and path in acc_paths(acc, L) and src in acc_paths(acc, L):
                    x1, x2 = np.load(d / f"{path}.npy"), np.load(d / f"{src}.npy")
                    log(f"  alias check L{L} {path} == {src}: {bool(np.array_equal(x1, x2))} "
                        f"(max |diff| {float(np.abs(x1 - x2).max()):.2e})")
                    aliases[path] = src
        jdump(d / "meta.json", dict(rows=rows, aliases=aliases, calibration=desc, capture_layers=list(layers),
                                    run_layers=[s, b], load_s=round(tload, 1), forward_s=round(tfwd, 1)))
    for k in range(1, a + 1):            # older hand-offs are no longer needed
        if k != b and hstate(k).exists() and k < a:
            hstate(k).unlink()
    log(f"capture L{a}-{b - 1} done: load {tload:.0f} s, forward+H {tfwd:.0f} s ({b - s} layers), "
        f"peak {mx.get_peak_memory() / 2**30:.1f} GiB")
    del model, acc
    gc.collect()
    mx.clear_cache()
    return dict(layers=list(layers), run=[s, b], load_s=round(tload, 1), forward_s=round(tfwd, 1))


def acc_paths(acc, L):
    return {p for (LL, p) in acc if LL == L}


def load_H(L, path):
    meta = json.loads((hdir(L) / "meta.json").read_text())
    src = meta["aliases"].get(path, path)
    if not (hdir(L) / f"{src}.npy").exists():
        src = ALIAS.get(path, path)
    return np.load(hdir(L) / f"{src}.npy").astype(np.float64)


# ---------------------------------------------------------------- encode
def t6_file(L):
    return OUT / f"model-t6-L{L:02d}.safetensors"


def encode_layer(L, args, wm):
    t0 = time.time()
    bf = base_file(L, args)
    tensors, rows = [], []
    for path in layer_linears(wm, L):
        K = rate(args.recipe, path)
        t1 = time.time()
        W, moved = weight(bf, L, path, args)                                 # [in, out]
        H = load_H(L, path)
        r = E.encode(W, H, K)
        Wd = E.decode(r["trellis"], r["suh"], r["svh"])
        dec = float(np.abs(Wd - r["W_hat"]).max() / (np.abs(r["W_hat"]).max() + 1e-30))
        assert dec < 1e-6, (L, path, dec)
        p = f"{PFX}{L}.{path}"
        tensors += [(f"{p}.trellis", "U16", r["trellis"]), (f"{p}.suh", "F16", r["suh"]), (f"{p}.svh", "F16", r["svh"])]
        row = dict(layer=L, linear=path, K=K, shape=list(W.shape), had=[E.had_size(W.shape[0]), E.had_size(W.shape[1])],
                   proxy=E.proxy_err(W, r["W_hat"], H), rel_l2=float(np.linalg.norm(r["W_hat"] - W) / np.linalg.norm(W)),
                   g_scale=round(r["g_scale"], 4), decode_check=dec, secs=round(time.time() - t1, 1))
        if moved is not None:
            row.update(source="delta", moved_vs_base=round(moved, 5))
        rows.append(row)
        log(f"  L{L:02d} {path:48s} K{K} {W.shape[0]}x{W.shape[1]} proxy {row['proxy']:.2e} rel {row['rel_l2']:.4f} "
            f"{row['secs']:.1f}s" + (f"  [delta: moved {moved:.4f} vs base]" if moved is not None else ""))
        del W, H, r, Wd
    if args.delta:
        want = sorted(f.name[len(f"{PFX}{L}."):-len(".bf16")] for f in Path(args.delta).glob(f"{PFX}{L}.*.bf16"))
        used = sorted(r["linear"] for r in rows if r.get("source") == "delta")
        assert want == used, (L, "delta files not all used", sorted(set(want) - set(used)))
        log(f"L{L:02d}: {len(used)} Linears from the delta, {len(rows) - len(used)} from the base")
    st_write(t6_file(L), tensors, {"format": "exl3", "codebook": "mcg", "window": 15, "recipe": args.recipe,
                                   "had": "128 else largest pow2 <= 64 dividing the dim", "layer": L,
                                   "producer": "ijah tools/convert_trunk.py"})
    with open(WORK / "encode.jsonl", "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    jdump(WORK / "layers" / f"L{L:02d}.json", dict(layer=L, secs=round(time.time() - t0, 1), linears=len(rows),
                                                  bytes=t6_file(L).stat().st_size))
    log(f"L{L:02d} encoded: {len(rows)} linears, {t6_file(L).stat().st_size / 2**20:.0f} MiB, {time.time() - t0:.0f} s")
    if not args.keep_h:
        shutil.rmtree(hdir(L))


def weight(bf, L, path, args):
    """The Linear's weight as float64 [in, out]: the --delta file if there is one (then also its relative L2 change vs
    the base weight, which must be > 0), else the base's."""
    W = st_read(bf, f"{HF}{L}.{path}.weight")
    moved = None
    if args.delta:
        f = Path(args.delta) / f"{PFX}{L}.{path}.bf16"
        if f.exists():
            raw = np.fromfile(f, dtype=np.uint16)
            assert raw.size == W.size, (L, path, raw.size, W.shape)
            D = (raw.astype(np.uint32) << 16).view(np.float32).reshape(W.shape)
            moved = float(np.linalg.norm(D - W) / np.linalg.norm(W))
            assert moved > 0, (L, path, "delta equals the base weight")
            W = D
    return W.T.astype(np.float64), moved


def write_mixer(args):
    f = OUT / "model-t6-mixer.safetensors"
    if f.exists():
        return
    g = base_globals(args)
    ts = []
    for nm in ("input_mix_weight_down", "input_mix_weight_up"):
        a = st_read(g, f"model.language_model.hyper_connection_mixer.{nm}.weight", raw=True)
        ts.append((f"{MIXER}{nm}.weight", "BF16", a))
    st_write(f, ts, {"format": "bf16", "producer": "ijah tools/convert_trunk.py"})
    log(f"mixer: bf16 final hyper-connection mixer -> {f.name}")


# ---------------------------------------------------------------- driver
def parse_layers(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def status(args, wm):
    enc = [json.loads(p.read_text()) for p in sorted((WORK / "layers").glob("L*.json"))] if (WORK / "layers").exists() else []
    caps = []
    for d in sorted((WORK / "hessians").glob("L*")) if (WORK / "hessians").exists() else []:
        if (d / "meta.json").exists():
            caps.append(json.loads((d / "meta.json").read_text()))
    done = [e["layer"] for e in enc if t6_file(e["layer"]).exists()]
    print(f"encoded layers: {done}")
    print(f"hessians on disk: {[int(d.name[1:]) for d in sorted((WORK / 'hessians').glob('L*'))] if (WORK / 'hessians').exists() else []}")
    if enc:
        s = np.mean([e["secs"] for e in enc])
        print(f"encode: {s:.0f} s per layer (mean of {len(enc)}); 48 layers ~ {48 * s / 60:.0f} min")
    logf = WORK / "capture.jsonl"
    if logf.exists():
        for line in logf.read_text().splitlines():
            print("capture pass:", line)


def main():
    global OUT, WORK
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0-47")
    ap.add_argument("--recipe", default="t6", choices=["t6", "t7"])
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--work", default=str(WORK))
    ap.add_argument("--base", default=str(BASE))
    ap.add_argument("--group", type=int, default=8, help="layers per Hessian capture pass")
    ap.add_argument("--num-layers", type=int, default=48)
    ap.add_argument("--tokens", type=int, default=None, help="calibration tokens (default all 279.5K)")
    ap.add_argument("--keep-h", action="store_true")
    ap.add_argument("--verify-aliases", action="store_true")
    ap.add_argument("--min-free-pct", type=int, default=8)
    ap.add_argument("--mem-limit", type=float, default=100.0, help="MLX active + cache GiB")
    ap.add_argument("--min-disk-gb", type=float, default=150.0)
    ap.add_argument("--delta", default="", help="dir of fine-tuned bf16 Linears (<pack prefix>.bf16) overriding the base")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-mixer", action="store_true")
    ap.add_argument("--capture-only", action="store_true", help="Hessians only (tests)")
    a = ap.parse_args()
    OUT, WORK = Path(a.out), Path(a.work)
    WORK.mkdir(parents=True, exist_ok=True)
    wm = src_index(Path(a.src))["weight_map"]
    if a.status:
        status(a, wm)
        return
    watchdog(a.min_free_pct, a.mem_limit)
    skeleton(Path(a.src), OUT)
    if not a.no_mixer:
        write_mixer(a)
    write_index(Path(a.src), OUT, a)
    todo = [L for L in parse_layers(a.layers) if not t6_file(L).exists() or (a.capture_only and not h_done(L))]
    log(f"recipe {a.recipe}: {len(todo)} layers to encode: {todo}")
    t0 = time.time()
    while todo:
        free = shutil.disk_usage(OUT).free / 1e9
        if free < a.min_disk_gb:
            log(f"disk: {free:.0f} GB free < {a.min_disk_gb}; stopping")
            sys.exit(4)
        group = todo[:a.group]
        need = [L for L in group if not h_done(L)]
        if need:
            r = capture(need, a, wm)
            with open(WORK / "capture.jsonl", "a") as f:
                f.write(json.dumps(r) + "\n")
        for L in group:
            if a.capture_only:
                continue
            encode_layer(L, a, wm)
            write_index(Path(a.src), OUT, a)
        todo = todo[len(group):]
        log(f"elapsed {(time.time() - t0) / 60:.1f} min, {len(todo)} layers left")
    if not a.capture_only:
        write_index(Path(a.src), OUT, a, final=not a.no_mixer)
    log(f"done: {OUT}")


if __name__ == "__main__":
    main()
