"""Unconstrained decode speed through HTTP, step and MTP paths (lab-api A/B of the baseline server vs this build), stdlib
only. Against a server that is up: one essay request per path and repeat, same seed, engine decode tok/s from the
response's timings. Prints one JSON line.

    $PY tests/server/speed_ab.py --url http://127.0.0.1:8141 --key-file KEY [--tokens 1024] [--reps 2]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_client import Client, merge  # noqa: E402,F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8141")
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--label", default="")
    ap.add_argument("--knobs", action="store_true", help="also requests with presence / frequency penalties + logit_bias + "
                    "min_p on MTP and on the step path, and logprobs 5 (step) (lab-sampler: what a knob costs)")
    a = ap.parse_args()
    cl = Client(a.url, Path(a.key_file).expanduser().read_text().strip())
    st, m = cl.req("GET", "/v1/models")
    model = m["data"][0]["id"]
    q = "Write a detailed, long essay about the history of the printing press, its technology and its effects on Europe."
    out = {"label": a.label, "tokens": a.tokens}
    runs = [("step", {}), ("mtp", {})]
    if a.knobs:
        kn = {"presence_penalty": 0.5, "frequency_penalty": 0.2, "logit_bias": {"5": -1}, "min_p": 0.02}
        runs += [("mtp+knobs", kn), ("step+knobs", kn), ("step+logprobs", {"logprobs": True, "top_logprobs": 5}),
                 ("mtp+presence1.5", {"presence_penalty": 1.5}), ("mtp+frequency0.2", {"frequency_penalty": 0.2}),
                 ("mtp+min_p0.05", {"min_p": 0.05}), ("mtp+bias", {"logit_bias": {"5": -1}})]
    for path, extra in runs:
        v = []
        for rep in range(a.reps + 1):   # the first run of each path is a warm-up
            chunks, _, first, wall = cl.stream(dict({"model": model, "reasoning_effort": "none", "max_tokens": a.tokens,
                                                     "ne_path": "mtp" if path.startswith("mtp") else "step", "temperature": 1.0, "seed": 7 + rep,
                                                     "messages": [{"role": "user", "content": q}],
                                                     "stream_options": {"include_usage": True}}, **extra))
            tim = next((ch.get("timings") for ch in chunks if ch.get("timings")), {}) or {}
            if rep and tim.get("decode_tok_s"):
                v.append(tim["decode_tok_s"])
        out[path] = {"tok_s": v, "median": round(statistics.median(v), 2) if v else None, "path": tim.get("path")}
    print("SPEED_AB " + json.dumps(out))


if __name__ == "__main__":
    main()
