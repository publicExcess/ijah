"""lab-integrate: time to first token of generate() and of the MTP loop (prefill + the first sampled token).

    NEED_GB=85 LOCK_OWNER=... LOCK_WAIT=30 tools/with_lock.sh "bench_ttft (~83 GB)" HH:MM -- $PY tools/bench_ttft.py \
        [--lens 2048,8192,32768] [--steps 2048] [--mtp 2048,8192]

Prompts: repository text (tools/prefill_tokens.py's files), N tokens. TTFT = ne_fn_model_generate's prefill wall
(prefill() = chunked prefill of N - 1 tokens + the decode step with the head) with max_new = 1, run twice (the first
run includes the chunk recordings and pipeline compiles). --steps: the same with NE_FN_PREFILL=steps (the old 4-row
+ 1-row path) for comparison. --mtp: ne_fn_mtp_generate's prefill (target + the MTP head's prefill over every
position) at those lengths.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refenv  # noqa: E402

refenv.add_paths()
os.environ.setdefault("NE_FN_ROUND", "f32")
import engine as E  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
import bench_mtp as B  # noqa: E402


def real_tokens(pack, n):
    root = refenv.ROOT
    files = sorted(root.glob("docs/*.md")) + sorted(root.glob("models/flash_next/*.mm")) + \
        sorted(root.glob("kernels/*.metal")) + sorted(root.glob("tools/*.py"))
    tok = Tokenizer.from_file(str(Path(pack) / "tokenizer.json"))
    ids = tok.encode("\n\n".join(f.read_text(errors="replace") for f in files), add_special_tokens=False).ids
    while len(ids) < n:
        ids = ids + ids
    return ids[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=refenv.ijah_paths.pack())
    ap.add_argument("--lens", default="2048,8192,32768")
    ap.add_argument("--steps", default="2048")
    ap.add_argument("--mtp", default="2048,8192")
    a = ap.parse_args()
    lens = [int(x) for x in a.lens.split(",") if x]
    steps = [int(x) for x in a.steps.split(",") if x]
    mtpl = [int(x) for x in a.mtp.split(",") if x]
    nmax = max(lens + steps + mtpl)
    eng = E.Engine()
    pack = eng.pack_open(a.pack)
    t0 = time.time()
    m = eng.fn_model(pack, (nmax + 128 + 63) // 64 * 64)
    m.set_sampler(0.0, 1.0, 1, 0)
    print(f"load {time.time() - t0:.1f} s", flush=True)
    toks = real_tokens(a.pack, nmax)
    first = {}
    for n in lens:
        r = []
        for _ in range(2):
            m.reset()
            out, _, st = m.generate(toks[:n], 1, -1)
            r.append(st["prefill_ms"])
        first[n] = out[0]
        print(f"chunked  N {n:6d}: TTFT {r[1] / 1e3:6.2f} s ({n / r[1] * 1e3:6.0f} tok/s); first run {r[0] / 1e3:.2f} s; "
              f"first token {out[0]}", flush=True)
    for n in steps:
        os.environ["NE_FN_PREFILL"] = "steps"
        m.reset()
        out, _, st = m.generate(toks[:n], 1, -1)
        os.environ.pop("NE_FN_PREFILL")
        print(f"steps    N {n:6d}: TTFT {st['prefill_ms'] / 1e3:6.2f} s ({n / st['prefill_ms'] * 1e3:6.0f} tok/s); "
              f"first token {out[0]} ({'same' if first.get(n) == out[0] else 'differs from chunked'})", flush=True)
    if mtpl:
        mtp = B.Mtp(eng, m, 4, 4)
        for n in mtpl:
            r = []
            for _ in range(2):
                out, s = mtp.generate(toks[:n], 1, 2)
                r.append(s)
            s = r[1]
            print(f"MTP d2   N {n:6d}: TTFT {s['prefill_ms'] / 1e3:6.2f} s (head prefill {s['head_prefill_ms'] / 1e3:.2f} s "
                  f"of it); first run {r[0]['prefill_ms'] / 1e3:.2f} s; first token {out[0]}", flush=True)


if __name__ == "__main__":
    main()
