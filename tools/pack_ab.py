"""Engine-vs-engine A/B of whole packs in ONE job (lab-swift): teacher-forced logits, greedy agreement, decode speed.

    tools/with_lock.sh "pack A/B (~81 GB, whole model)" HH:MM -- $PY -I -B tools/pack_ab.py \
        --pack base=<IJAH_PACKS>/Qwen3.8-Flash-Next-EXL3-Ijah --pack swift=<IJAH_PACKS>/Swift-1.5-Flash-Next-EXL3-Ijah \
        --out DIR [--gen 128] [--time 64]

Packs run one after the other (each opened, used, freed). Per pack:
  - teacher-forced logits on the golden WHOLE prompts (tests/golden/data/WHOLE: chat, code, prose, facts, long; the
    tokens are the MLX K5 reference's prompt + greedy continuation; long prefills 2599 tokens without the head),
    saved as DIR/<name>/<prompt>.engine.npy (bf16 bits, the format of compare_logits.py --save / engine_kld.py)
  - greedy generation (chat template, thinking off) of --gen tokens on PROMPTS, tokens + text + decode tok/s
  - GPU ms per decode step at T = --time (ne_fn_model_time, 20 reps)
Then for every pair (first pack vs each other): KL(first || other) per position (mean / median / max), top-1
agreement, and per greedy prompt the common prefix length (first divergence). f32 (NE_FN_ROUND=f32), QSA.
Writes DIR/pack_ab.json.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refenv  # noqa: E402

os.environ.setdefault("NE_FN_ROUND", "f32")
refenv.add_paths()
import engine as E  # noqa: E402
import numpy as np  # noqa: E402
from generate import chat_prompt  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

PROMPTS = {
    "mul": "What is 17 x 23? Answer with the number and one sentence.",
    "palin": "Write a Python function that checks whether a string is a palindrome, ignoring case and punctuation.",
    "explain": "Explain in three sentences why the sky is blue.",
    "story": "Write the opening paragraph of a short story about a lighthouse keeper who finds a message in a bottle.",
}
TF = ("chat", "code", "prose", "facts", "long")


def f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def logsm(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def run_pack(eng, name, path, a, out):
    ref = refenv.GOLDEN / "WHOLE"
    man = json.loads((ref / "manifest.json").read_text())
    tok = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
    stop = tok.token_to_id("<|im_end|>")
    t0 = time.time()
    pack = eng.pack_open(path)
    tmax = 4096
    m = eng.fn_model(pack, tmax)
    res = {"pack": str(path), "load_s": round(time.time() - t0, 1), "tf": {}, "gen": {}}
    m.set_sampler(0.0, 1.0, 1, 0)
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    for p in TF:
        pm = man["prompts"][p]
        toks = np.load(ref / f"{p}.tokens.npy")
        n = len(np.load(ref / f"{p}.decode.npy", mmap_mode="r"))
        first = pm.get("first_pos", 0) or 0
        m.reset()
        got = np.empty((n, E.FnModel.VOCAB), np.float32)
        for i in range(first):
            m.step(int(toks[i]), head=False)
        for i in range(n):
            got[i] = m.step(int(toks[first + i]), head=True, logits=True)[3]
        np.save(d / f"{p}.engine.npy", (got.view(np.uint32) >> 16).astype(np.uint16))
        res["tf"][p] = n
    for p, text in PROMPTS.items():
        ids = tok.encode(chat_prompt(path, text, False), add_special_tokens=False).ids
        m.reset()
        gen, _, st = m.generate(ids, a.gen, stop if stop is not None else -1)
        n = st["decode_steps"]
        res["gen"][p] = {"tokens": gen, "text": tok.decode(gen, skip_special_tokens=False),
                         "decode_tok_s": round(n / (st["decode_ms"] / 1e3), 2) if n else 0.0,
                         "gpu_ms_step": round(st["gpu_ms_mean"], 3)}
        print(f"[{name}] {p}: {res['gen'][p]['decode_tok_s']} tok/s | {res['gen'][p]['text'][:160]!r}", flush=True)
    m.reset()
    res["time"] = m.time(a.time, 20)
    print(f"[{name}] GPU ms/step T {a.time}: {res['time']['gpu_ms_median']:.3f} (load {res['load_s']} s)", flush=True)
    m.free()
    pack.close()
    return res


def compare(out, A, B, ra, rb):
    ks, ts, per = [], [], {}
    for p in TF:
        la, lb = logsm(f32(np.load(out / A / f"{p}.engine.npy"))), logsm(f32(np.load(out / B / f"{p}.engine.npy")))
        k = (np.exp(la) * (la - lb)).sum(-1)
        t = (la.argmax(-1) == lb.argmax(-1)).astype(np.float64)
        per[p] = {"n": len(k), "kld_mean": float(k.mean()), "kld_median": float(np.median(k)), "top1": float(t.mean())}
        ks.append(k)
        ts.append(t)
    k, t = np.concatenate(ks), np.concatenate(ts)
    g = {}
    for p in PROMPTS:
        x, y = ra["gen"][p]["tokens"], rb["gen"][p]["tokens"]
        c = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
        g[p] = {"common_prefix": c, "len": [len(x), len(y)]}
    r = {"kld_mean": float(k.mean()), "kld_median": float(np.median(k)), "kld_p99": float(np.quantile(k, 0.99)),
         "kld_max": float(k.max()), "top1": float(t.mean()), "positions": int(len(k)), "per_prompt": per, "greedy": g}
    print(f"{A} vs {B}: KL({A}||{B}) mean {r['kld_mean']:.3e} median {r['kld_median']:.3e} p99 {r['kld_p99']:.3e} "
          f"max {r['kld_max']:.3e}, top-1 {r['top1']:.4f} over {r['positions']} positions; greedy common prefix "
          + ", ".join(f"{p} {v['common_prefix']}/{min(v['len'])}" for p, v in g.items()), flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", action="append", required=True, help="NAME=DIR (first = the reference side)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--time", type=int, default=64)
    a = ap.parse_args()
    packs = [(s.split("=", 1)[0], Path(s.split("=", 1)[1]).expanduser()) for s in a.pack]
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    eng = E.Engine()
    eng.set_ref_rounding(False)
    runs = {n: run_pack(eng, n, p, a, out) for n, p in packs}
    A = packs[0][0]
    cmp = {f"{A}_vs_{n}": compare(out, A, n, runs[A], runs[n]) for n, _ in packs[1:]}
    (out / "pack_ab.json").write_text(json.dumps({"runs": runs, "compare": cmp}, indent=1))


if __name__ == "__main__":
    main()
