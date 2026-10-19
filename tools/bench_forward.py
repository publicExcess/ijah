"""Whole-model decode speed (lab-forward): GPU ms per step at context lengths (positions faked; attention per NE_FN_ATTN, default QSA),
dispatches per step, and end-to-end tok/s of the token loop (greedy + temp 1 / top-p 0.95 / top-k 20).

    tools/with_lock.sh "bench forward" HH:MM -- $PY tools/bench_forward.py [--ctx 64,2048,8192] [--gen 128]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refenv  # noqa: E402

# Generation default: plain f32 (no reference bf16 rounding points): closest to the MLX reference at whole-model
# level (docs/forward.md section 2). --ref-round keeps the emulation. Set before the library loads (the layer graph
# and lab-attn's QSA kernels read NE_FN_ROUND).
os.environ.setdefault("NE_FN_ROUND", "ref" if "--ref-round" in sys.argv else "f32")  # (library default: f32)

refenv.add_paths()
import engine as E  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=refenv.ijah_paths.pack())
    ap.add_argument("--ctx", default="64,2048,8192")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--json", default=None)
    ap.add_argument("--ref-round", action="store_true", help="keep the reference bf16 rounding points")
    a = ap.parse_args()
    ctx = [int(x) for x in a.ctx.split(",")]
    tmax = ((max(ctx) + 64) + 31) // 32 * 32
    eng = E.Engine()
    pack = eng.pack_open(a.pack)
    t0 = time.time()
    m = eng.fn_model(pack, tmax)
    res = {"tmax": tmax, "load_s": time.time() - t0, **m.info(), "ctx": []}
    tok = Tokenizer.from_file(str(Path(a.pack) / "tokenizer.json"))
    ids = tok.encode("Photosynthesis is the process by which green plants", add_special_tokens=False).ids
    # warm: one short generation pages every weight in (first touch of the wired pack)
    t = time.time()
    m.set_sampler(0.0, 1.0, 1, 0)
    m.generate(ids, 4)
    m.reset()
    res["warm_s"] = time.time() - t
    for T in ctx:
        r = m.time(T, a.reps)
        res["ctx"].append(r)
        print(f"T={T:6d}: GPU {r['gpu_ms_median']:.2f} ms (min {r['gpu_ms_min']:.2f}), wall {r['wall_ms_median']:.2f} ms "
              f"per synchronous step", flush=True)
    for name, samp in (("greedy", (0.0, 1.0, 1, 0)), ("temp1_p0.95_k20", (1.0, 0.95, 20, 7))):
        m.reset()
        m.set_sampler(*samp)
        out, lps, st = m.generate(ids, a.gen, -1)
        n = st["decode_steps"]
        tps = n / (st["decode_ms"] / 1e3)
        res[name] = {"tok_s": tps, "ms_per_tok": st["decode_ms"] / n, **st,
                     "text": tok.decode(out, skip_special_tokens=False)}
        print(f"{name}: {tps:.1f} tok/s ({st['decode_ms'] / n:.2f} ms/token wall, GPU {st['gpu_ms_mean']:.2f} ms, "
              f"gather {st['gather_us_mean']:.0f} us), prefill {len(ids) / (st['prefill_ms'] / 1e3):.1f} tok/s; "
              f"{res[name]['text'][:100]!r}", flush=True)
    print(json.dumps({k: v for k, v in res.items() if k not in ("greedy", "temp1_p0.95_k20")}))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    m.free()
    pack.close()


if __name__ == "__main__":
    main()
