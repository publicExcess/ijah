"""HTTP speed and model-switch probes against a running server (lab-srvperf). Prints one JSON line per probe.

    $PY tests/server/bench_http.py --url http://127.0.0.1:8140 --key-file KEY speed [--model ID] [--n 1024]
        [--paths step,mtp] [--greedy]                  # streamed tok/s through HTTP vs the engine's own timing
    $PY tests/server/bench_http.py ... switch --models A,B [--rounds 3]
        # short fresh-prefill requests A, A, B, B, then A, B alternating:
    $PY tests/server/bench_http.py ... turns --models A,B
        # A turn 1 (long system prompt), B (its own conversation), A turn 2: greedy texts, cached tokens, TTFT time to first token after a model switch vs the same model

speed: tok/s = (tokens - 1) / (last content chunk - first content chunk), the client's view; `timings` of the server
(engine decode tok/s, host_us_per_token on the step path) alongside.
"""
import argparse
import http.client
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ESSAY = ("Write a detailed, long essay about the history of the printing press, its technology and its effects on "
         "Europe.")


def stream(url, key, body):
    u = urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=900)
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", json.dumps(dict(body, stream=True,
                                                             stream_options={"include_usage": True})),
              {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        raise SystemExit(f"HTTP {r.status}: {r.read()[:300]!r}")
    first = last = None
    usage = tim = None
    text = []
    for line in r:
        if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
            continue
        o = json.loads(line[6:])
        if o.get("usage"):
            usage, tim = o["usage"], o.get("timings") or {}
            continue
        if not o.get("choices"):
            continue
        d = o["choices"][0]["delta"]
        piece = d.get("content") or d.get("reasoning_content")
        if piece:
            text.append(piece)
            now = time.perf_counter()
            first = first or now
            last = now
    c.close()
    end = time.perf_counter()
    return {"tokens": usage["completion_tokens"] if usage else 0, "ttft_s": (first - t0) if first else None,
            "wall_s": end - t0, "first": first, "last": last, "timings": tim, "text": "".join(text),
            "cached": (usage or {}).get("prompt_tokens_details", {}).get("cached_tokens")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8140")
    ap.add_argument("--key-file", default=str(Path.home() / ".config/ijah/api-key"))
    ap.add_argument("probe", choices=["speed", "switch", "turns"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--models", default="")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--paths", default="step,mtp")
    ap.add_argument("--prompt", default=ESSAY)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    key = Path(a.key_file).expanduser().read_text().strip()
    if a.probe == "speed":
        for path in a.paths.split(","):
            body = {"messages": [{"role": "user", "content": a.prompt}], "max_tokens": a.n, "ne_path": path,
                    "reasoning_effort": "none", "seed": 7, "temperature": 0.0 if a.greedy else 1.0}
            if a.model:
                body["model"] = a.model
            r = stream(a.url, key, body)
            tps = (r["tokens"] - 1) / (r["last"] - r["first"]) if r["tokens"] > 1 and r["last"] > r["first"] else 0
            t = r["timings"]
            print("BENCH " + json.dumps({"tag": a.tag, "model": a.model, "path": path, "tokens": r["tokens"],
                                         "http_tok_s": round(tps, 2), "engine_tok_s": t.get("decode_tok_s"),
                                         "engine_path": t.get("path"), "host_us_per_token": t.get("host_us_per_token"),
                                         "ttft_s": round(r["ttft_s"] or 0, 3), "prefill_ms": t.get("prefill_ms"),
                                         "text_head": r["text"][:160]}), flush=True)
    elif a.probe == "turns":
        A, Bm = a.models.split(",")[:2]
        sysp = Path(__file__).resolve().parents[2].joinpath("docs/mtp.md").read_text()[:12000]
        m1 = [{"role": "system", "content": sysp}, {"role": "user", "content": "In one sentence: what is MTP here?"}]
        rows = []

        def turn(model, msgs, n):
            r = stream(a.url, key, {"model": model, "messages": msgs, "max_tokens": n, "reasoning_effort": "none",
                                    "temperature": 0.0})
            rows.append({"model": model, "ttft_ms": round(r["ttft_s"] * 1e3, 1), "prefill_ms": r["timings"].get(
                "prefill_ms"), "cached": r["cached"], "path": r["timings"].get("path"), "text": r["text"]})
            return r["text"]
        t1 = turn(A, m1, 32)
        turn(Bm, [{"role": "system", "content": "You are terse. " * 300}, {"role": "user", "content": "Name a colour."}], 16)
        turn(A, m1 + [{"role": "assistant", "content": t1}, {"role": "user", "content": "And in five words?"}], 24)
        print("TURNS " + json.dumps({"tag": a.tag, "rows": rows}), flush=True)
    else:
        ms = a.models.split(",")
        seq = [ms[0], ms[0], ms[1], ms[1]] + [ms[0], ms[1]] * a.rounds   # same-model pairs, then alternating
        prev = None
        rows = []
        for i, m in enumerate(seq):
            body = {"model": m, "messages": [{"role": "user", "content": f"Say hello in one short sentence. ({i})"}],
                    "max_tokens": 16, "reasoning_effort": "none", "temperature": 0.0, "ne_cache": False}
            r = stream(a.url, key, body)
            rows.append({"model": m, "switch": prev is not None and prev != m, "ttft_ms": round(r["ttft_s"] * 1e3, 1),
                         "wall_ms": round(r["wall_s"] * 1e3, 1), "prefill_ms": r["timings"].get("prefill_ms")})
            prev = m
        sw = [x["ttft_ms"] for x in rows[1:] if x["switch"]]
        same = [x["ttft_ms"] for x in rows[1:] if not x["switch"]]
        print("SWITCH " + json.dumps({"tag": a.tag, "rows": rows,
                                      "ttft_ms_after_switch": sw, "ttft_ms_same_model": same}), flush=True)


if __name__ == "__main__":
    main()
