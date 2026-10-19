"""MTP token loop on the native engine: tok/s, acceptance, and greedy identity vs no-MTP. lab-loop.

    NEED_GB=85 LOCK_OWNER=lab-loop tools/with_lock.sh "bench_mtp (~83 GB)" HH:MM -- \
        $PY tools/bench_mtp.py [--n 1024] [--ctx short|8k] [--modes greedy,sampled] [--depths 1,2,3,4,0]
        [--prompts think|fixed] [--bits 4] [--json OUT]

Prompts (--prompts think, default): the three "thinking" decode prompts of the oMLX / Sushi head-to-head harness
(the lab tree's omlx_h2h/h2h.py THINK: code_lru, text_trains, text_crisis), chat template with
thinking on (reasoning_effort medium), n tokens each, no stop (always n). --ctx 8k puts ~7.9K tokens of repository
docs (prompts.py's DOCS) in front of each question. Sampled = temperature 1, top-p 0.95, top-k 20 (the harness's).
Per prompt: no-MTP greedy and sampled (ne_fn_model_generate), MTP at each depth (0 = adaptive) greedy and sampled;
greedy MTP tokens are compared with greedy no-MTP (first mismatch). tok/s = tokens after the prefill's first one /
decode wall time (the same definition as tools/generate.py). Results also as JSON (--json).
"""
import argparse
import ctypes as C
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refenv  # noqa: E402

os.environ.setdefault("NE_FN_ROUND", "ref" if "--ref-round" in sys.argv else "f32")  # (library default: f32)
refenv.add_paths()
import numpy as np  # noqa: E402
import engine as E  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

THINK = {  # = omlx_h2h/h2h.py THINK (the decode prompts of the head-to-heads)
    "code_lru": "Implement a thread-safe LRU cache in Python with O(1) get and put, a per-entry TTL, and an eviction "
                "callback. Use type hints, explain the design briefly, and include pytest tests covering expiry, "
                "eviction order and concurrent access.",
    "text_trains": "Two trains leave stations 420 km apart at 9:00, heading toward each other at 80 km/h and "
                   "100 km/h. A bird starts at the first train at 9:30, flying at 150 km/h back and forth between "
                   "the trains until they meet. How far does the bird fly in total, and at what time and where do the "
                   "trains meet? Then: if the faster train had left 20 minutes late instead, how would each answer "
                   "change?",
    "text_crisis": "Explain the main causes of the 2008 financial crisis to a smart high-school student, then draw "
                   "three lessons for regulators today, with one concrete policy example for each.",
}
FIXED = {"chat17x23": "What is 17 multiplied by 23? Answer briefly."}
DOCS = [refenv.ROOT / p for p in (   # long-context prompts: this repo's own notes (the lab's planning docs before)
    "ARCHITECTURE.md", "docs/research-prefill.md", "docs/prefill.md", "docs/kernels.md")]


def chat(pack, text):
    import jinja2

    def raise_exception(msg):
        raise jinja2.TemplateError(msg)

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda x, **k: json.dumps(x, ensure_ascii=False)
    t = env.from_string((Path(pack) / "chat_template.jinja").read_text())
    return t.render(messages=[{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=True,
                    reasoning_effort="medium")


class Mtp:
    def __init__(self, eng, model, max_depth, bits):
        lib = eng.lib
        lib.ne_fn_mtp_create.restype = C.c_void_p
        lib.ne_fn_mtp_create.argtypes = [C.c_void_p, C.c_int, C.c_int]
        lib.ne_fn_mtp_free.argtypes = [C.c_void_p]
        lib.ne_fn_mtp_generate.restype = C.c_int
        lib.ne_fn_mtp_generate.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_uint64,
                                           C.c_void_p, C.c_void_p, C.c_void_p]
        self.eng, self.lib, self.maxd = eng, lib, max_depth
        self.ptr = lib.ne_fn_mtp_create(C.c_void_p(model.ptr), max_depth, bits)
        if not self.ptr:
            raise E.EngineError(f"ne_fn_mtp_create: {eng.last_error()}")

    def generate(self, ids, n, depth, seed=0, stop=-1):
        p = np.ascontiguousarray(ids, np.int32)
        out = np.empty(n, np.int32)
        st = np.zeros(10, np.float64)
        M1 = self.maxd + 1
        hist = np.zeros(M1 * M1, np.int32)
        k = self.lib.ne_fn_mtp_generate(C.c_void_p(self.ptr), p.ctypes.data, len(p), n, stop, depth, seed,
                                        out.ctypes.data, st.ctypes.data, hist.ctypes.data)
        if k < 0:
            raise E.EngineError(f"ne_fn_mtp_generate: {self.eng.last_error()}")
        keys = ("prefill_ms", "head_prefill_ms", "decode_ms", "tokens", "rounds", "draft_gpu_ms", "verify_gpu_ms",
                "gather_us", "host_us", "max_depth")
        s = dict(zip(keys, st.tolist()))
        s["hist"] = hist.reshape(M1, M1).tolist()
        return out[:k].tolist(), s


def accept_table(hist):
    """rounds by (depth, accepted) -> per depth position: P(draft i accepted | drafts < i accepted), tokens/round"""
    h = np.array(hist)
    reach = np.zeros(h.shape[0])
    acc = np.zeros(h.shape[0])
    for d in range(1, h.shape[0]):
        for a in range(d + 1):
            c = h[d, a]
            for i in range(min(a + 1, d)):  # positions 0..min(a, d-1) were reached
                reach[i] += c
                if i < a:
                    acc[i] += c
    cond = [round(acc[i] / reach[i], 3) if reach[i] else None for i in range(h.shape[0] - 1)]
    rounds = h.sum()
    toks = sum((a + 1) * h[d, a] for d in range(h.shape[0]) for a in range(h.shape[1]))
    return cond, (toks / rounds if rounds else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=refenv.ijah_paths.pack())
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--ctx", default="short", choices=["short", "8k"])
    ap.add_argument("--prompts", default="think", choices=["think", "fixed"])
    ap.add_argument("--only", default="", help="comma list of prompt names")
    ap.add_argument("--modes", default="greedy,sampled")
    ap.add_argument("--depths", default="1,2,3,4,0")
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--bits", type=int, default=4, help="draft readout bits (0 = full head)")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--json", default="")
    ap.add_argument("--ref-round", action="store_true")
    ap.add_argument("--time", action="store_true", help="only GPU ms of the verify (R = 2 .. max_depth + 1) and draft")
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(Path(a.pack) / "tokenizer.json"))
    prompts = THINK if a.prompts == "think" else FIXED
    if a.only:
        prompts = {k: v for k, v in prompts.items() if k in a.only.split(",")}
    prefix = ""
    if a.ctx == "8k":
        text = "\n\n".join(p.read_text() for p in DOCS if p.exists())
        ids = tok.encode(text, add_special_tokens=False).ids
        while len(ids) < 7900:
            ids = ids + ids
        prefix = tok.decode(ids[:7900]) + "\n\n---\n\nIgnoring the documents above, answer this: "
    tmax = ((8192 + a.n + 256 if a.ctx == "8k" else 1024 + a.n + 256) + 63) // 64 * 64
    eng = E.Engine()
    pack = eng.pack_open(a.pack)
    t0 = time.time()
    m = eng.fn_model(pack, tmax)
    mtp = Mtp(eng, m, a.max_depth, a.bits)
    print(f"load {time.time() - t0:.1f} s, tmax {tmax}, draft bits {a.bits}", flush=True)
    if a.time:
        f = eng.lib.ne_fn_mtp_time
        f.restype = C.c_int
        f.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p]
        one = m.time(64, 20)["gpu_ms_median"]
        print(f"1-row decode step at T 64: {one:.2f} ms GPU")
        for T in [t for t in (64, 2048, 8192) if t + 16 <= tmax]:
            for R in range(2, a.max_depth + 2):
                row = []
                for P in (0, R):
                    v, d = C.c_double(0), C.c_double(0)
                    eng._ok(f(C.c_void_p(mtp.ptr), R, T, P, 10, C.byref(v), C.byref(d)), "mtp_time")
                    row.append((v.value, d.value))
                print(f"  T {T:5d} R {R}: verify {row[0][0]:.2f} ms ({row[0][0] / one:.2f}x), with {R} pending rows "
                      f"{row[1][0]:.2f}; draft depth {R - 1}: {row[0][1]:.2f} ms ({row[0][1] / (R - 1):.2f} per step)",
                      flush=True)
        return
    stop = -1
    modes = a.modes.split(",")
    depths = [int(x) for x in a.depths.split(",")]
    res = []
    for name, q in prompts.items():
        ids = tok.encode(chat(a.pack, prefix + q), add_special_tokens=False).ids
        print(f"\n== {name}: {len(ids)} prompt tokens, {a.n} new", flush=True)
        base = {}
        for mode in modes:
            greedy = mode == "greedy"
            if greedy:
                m.set_sampler(0.0, 1.0, 1, 0)
            else:
                m.set_sampler(1.0, 0.95, 20, a.seed)
            if not a.no_baseline or greedy:
                m.reset()
                out, _, st = m.generate(ids, a.n, stop)
                tps = st["decode_steps"] / (st["decode_ms"] / 1e3)
                base[mode] = out
                r = dict(prompt=name, mode=mode, depth=None, tps=tps, prefill_ms=st["prefill_ms"],
                         gpu_ms=st["gpu_ms_mean"], n=len(out))
                res.append(r)
                print(f"  no-MTP {mode:7s}: {tps:6.1f} tok/s (GPU {st['gpu_ms_mean']:.2f} ms/token, prefill "
                      f"{st['prefill_ms'] / 1e3:.1f} s)", flush=True)
            for d in depths:
                out, s = mtp.generate(ids, a.n, d, seed=a.seed + d)
                tps = s["tokens"] / (s["decode_ms"] / 1e3)
                cond, tpr = accept_table(s["hist"])
                r = dict(prompt=name, mode=mode, depth=d, tps=tps, tokens_per_round=tpr, accept=cond,
                         rounds=s["rounds"], draft_ms=s["draft_gpu_ms"], verify_ms=s["verify_gpu_ms"],
                         gather_us=s["gather_us"], host_us=s["host_us"], hist=s["hist"], n=len(out),
                         prefill_ms=s["prefill_ms"], head_prefill_ms=s["head_prefill_ms"])
                extra = ""
                if greedy and "greedy" in base:
                    b = base["greedy"]
                    mm = next((i for i in range(min(len(b), len(out))) if b[i] != out[i]), None)
                    r["first_mismatch"] = mm
                    extra = "identical" if mm is None and len(b) == len(out) else f"FIRST MISMATCH at {mm}"
                    if mm is not None:
                        r["mismatch_ctx"] = tok.decode(out[max(0, mm - 8):mm + 4])
                        extra += f" ({tok.decode([b[mm]])!r} vs {tok.decode([out[mm]])!r})"
                        # the no-MTP logits at the mismatch (1-row path, teacher-forced): gap between the two tokens
                        m.reset()
                        m.prefill(ids)
                        lg = None
                        for t in b[:mm]:
                            _, _, _, lg = m.step(t, True, logits=True)
                        if lg is not None:
                            srt = np.sort(lg)
                            r["mismatch_gap"] = float(lg[b[mm]] - lg[out[mm]])
                            extra += f", no-MTP logit gap {r['mismatch_gap']:.3f} (top-2 gap {srt[-1] - srt[-2]:.3f})"
                if d == 0:
                    depth_counts = [sum(row) for row in s["hist"]]
                    r["depth_rounds"] = depth_counts
                    extra += f" depths {depth_counts}"
                res.append(r)
                print(f"  MTP d{d} {mode:7s}: {tps:6.1f} tok/s, {tpr:.2f} tok/round, accept {cond[:max(d, 1) if d else a.max_depth]}, "
                      f"draft {s['draft_gpu_ms']:.2f} + verify {s['verify_gpu_ms']:.2f} ms GPU/round, gather "
                      f"{s['gather_us']:.0f} us, host {s['host_us']:.0f} us; {extra}", flush=True)
    print("\n== medians over prompts (tok/s)")
    keys = sorted({(r["mode"], -1 if r["depth"] is None else r["depth"]) for r in res})
    for mode, d in keys:
        v = [r["tps"] for r in res if r["mode"] == mode and (r["depth"] if r["depth"] is not None else -1) == d]
        lab = "no-MTP" if d < 0 else f"MTP d{d}"
        tpr = [r["tokens_per_round"] for r in res if r["mode"] == mode and r["depth"] == d] if d >= 0 else []
        print(f"  {mode:7s} {lab:7s}: median {statistics.median(v):6.1f}  ({', '.join(f'{x:.1f}' for x in v)})"
              + (f"  tok/round {statistics.median(tpr):.2f}" if tpr else ""))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    pack.close()


if __name__ == "__main__":
    main()
