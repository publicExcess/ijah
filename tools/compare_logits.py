"""Whole-model logits: engine vs the MLX reference saved by tools/ref_logits.py (rung 3). lab-forward.

    tools/with_lock.sh "engine logits vs reference" HH:MM -- $PY tools/compare_logits.py [--ref DIR] [--pack DIR]
        [--f32] (engine without the reference rounding points)

Teacher-forced: the engine is fed the reference's token sequence one token per step (prompt, then the reference's
greedy continuation) and its logits at every position are compared with the reference's decode-path logits:
KL(ref || engine) per position (f64 log-softmax), top-1 agreement, top-5 overlap, max |dlogit|. The reference's own
prefill-vs-decode numbers (same model, its two code paths) are printed as the noise floor. Writes <ref>/engine.json.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refenv  # noqa: E402

refenv.add_paths()
import engine as E  # noqa: E402
import numpy as np  # noqa: E402


def f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def logsm(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def stats(ref, got):
    """ref, got: [n, V] logits -> dict of per-position arrays"""
    lr, lg = logsm(ref), logsm(got)
    pr = np.exp(lr)
    kld = (pr * (lr - lg)).sum(-1)
    t1r, t1g = ref.argmax(-1), got.argmax(-1)
    top5 = [len(set(np.argsort(-ref[i])[:5]) & set(np.argsort(-got[i])[:5])) / 5 for i in range(len(ref))]
    return {"kld": kld, "top1": (t1r == t1g).astype(np.float64), "top5": np.array(top5),
            "maxdl": np.abs(ref.astype(np.float64) - got).max(-1)}


def summary(s):
    return {"kld_mean": float(s["kld"].mean()), "kld_max": float(s["kld"].max()), "top1": float(s["top1"].mean()),
            "top5_overlap": float(s["top5"].mean()), "max_dlogit": float(s["maxdl"].max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=str(refenv.GOLDEN / "WHOLE"))
    ap.add_argument("--pack", default=refenv.ijah_paths.golden_pack())  # IJAH_GOLDEN_PACK: the pack the goldens / MLX reference were made from
    ap.add_argument("--f32", action="store_true")
    ap.add_argument("--tmax", type=int, default=1024)
    ap.add_argument("--only", default="chat,code,prose,facts,long")  # + long8k if saved
    ap.add_argument("--prefill4", action="store_true", help="long prompts: 4-row-chunk prefill (ne_fn_model_prefill)")
    ap.add_argument("--save", default="", help="lab-t6wire: write the engine logits (<name>.engine.npy, bf16 bits) and "
                    "the json to this directory instead of the reference directory (engine-vs-engine KLD: "
                    "tests/trunk6/engine_kld.py A B)")
    a = ap.parse_args()
    ref = Path(a.ref)
    man = json.loads((ref / "manifest.json").read_text())
    if a.f32:  # before the library loads: the layer graph and lab-attn's QSA kernels both read it
        os.environ["NE_FN_ROUND"] = "f32"
    eng = E.Engine()
    eng.set_ref_rounding(not a.f32)  # vs the MLX reference: rounding ON unless --f32 (library default: f32)
    pack = eng.pack_open(a.pack)
    m = eng.fn_model(pack, a.tmax)
    m.set_sampler(0.0, 1.0, 1, 0)
    out = {"rounding": "f32" if a.f32 else "reference bf16 points", "prompts": {},
           "attention": os.environ.get("NE_FN_ATTN", "qsa")}
    allk, allt, allp = [], [], []
    tag = "qsa" if os.environ.get("NE_FN_ATTN", "qsa") == "qsa" else "dense"
    if a.only != ap.get_default("only"):
        tag += "_" + a.only.replace(",", "-")
    if a.prefill4:
        tag += "_prefill4"
    for name, pm in man["prompts"].items():
        if name not in a.only.split(","):
            continue
        toks = np.load(ref / f"{name}.tokens.npy")
        dec = f32(np.load(ref / f"{name}.decode.npy"))
        n = len(dec)
        first = pm.get("first_pos", 0)   # long prompt: the engine prefills [0, first) without the head
        if first + n + 1 > a.tmax:
            m.free()
            m = eng.fn_model(pack, (first + n + 64 + 63) // 64 * 64)
            m.set_sampler(0.0, 1.0, 1, 0)
            a.tmax = m.tmax
        m.reset()
        got = np.empty_like(dec)
        t0 = time.time()
        i0 = 0
        if first > 0 and a.prefill4:   # 4-row chunks (+ 1-row remainder), the head on the last prompt token
            _, got[0], _ = m.prefill(toks[:first + 1], logits=True)
            i0 = 1
        else:
            for i in range(first):
                m.step(int(toks[i]), head=False)
        for i in range(i0, n):
            _, _, _, lg = m.step(int(toks[first + i]), head=True, logits=True)
            got[i] = lg
        dt = time.time() - t0
        s = stats(dec, got)
        if a.save:
            Path(a.save).mkdir(parents=True, exist_ok=True)
            np.save(Path(a.save) / f"{name}.engine.npy", (got.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16))
        pf = ref / f"{name}.prefill.npy"
        floor = stats(dec, f32(np.load(pf))) if pf.exists() else {k: np.zeros(1) for k in s}
        P = max(pm["prompt_len"] - first, 1)
        out["prompts"][name] = {"n": n, "prompt_len": P, "engine_vs_ref": summary(s),
                                "engine_vs_ref_generated_part": summary({k: v[P - 1:] for k, v in s.items()}),
                                "ref_prefill_vs_ref_decode": summary(floor), "engine_s": dt,
                                "kld_per_pos": [round(float(x), 6) for x in s["kld"]]}
        allk.append(s["kld"]); allt.append(s["top1"]); allp.append(floor["kld"])
        e, f = out["prompts"][name]["engine_vs_ref"], out["prompts"][name]["ref_prefill_vs_ref_decode"]
        print(f"{name:6s} n={n}: KLD mean {e['kld_mean']:.2e} max {e['kld_max']:.2e} top1 {e['top1']:.3f} "
              f"top5 {e['top5_overlap']:.3f} max|dlogit| {e['max_dlogit']:.3f}   | ref prefill vs decode: KLD mean "
              f"{f['kld_mean']:.2e} max {f['kld_max']:.2e} top1 {f['top1']:.3f}", flush=True)
    k, t, p = np.concatenate(allk), np.concatenate(allt), np.concatenate(allp)
    out["all"] = {"positions": int(len(k)), "kld_mean": float(k.mean()), "kld_max": float(k.max()),
                  "top1": float(t.mean()), "ref_floor_kld_mean": float(p.mean())}
    print(f"ALL {len(k)} positions: KLD mean {k.mean():.2e} max {k.max():.2e} top1 {t.mean():.4f} "
          f"(reference prefill-vs-decode floor KLD mean {p.mean():.2e})")
    out["pack"] = a.pack
    (Path(a.save) if a.save else ref).joinpath(f"engine_{tag}{'_f32' if a.f32 else ''}.json").write_text(
        json.dumps(out, indent=1))
    m.free()
    pack.close()


if __name__ == "__main__":
    main()
