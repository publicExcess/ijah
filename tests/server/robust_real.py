"""Whole-model check of the review fixes (lab-fixsrv), against a running server (tests/server/robust_job.sh starts one):
streamed + non-streamed answers, a streamed tool call that closes properly, a client that leaves mid-reply (the slot is
free again within ~1 s, streamed and not), one that leaves during a long prefill (free within a piece; the pieces fed
are reused by the next request).

    $PY tests/server/robust_real.py --url http://127.0.0.1:8144 --key-file KEY
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_client import FAILS, TOOLS, check  # noqa: E402
from test_robust_http import Srv, sse_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--key-file", required=True)
    a = ap.parse_args()
    S = Srv(a.url, Path(a.key_file).read_text().strip())
    base = {"model": "x", "reasoning_effort": "none", "temperature": 0}
    # 1. an answer, non-streamed and streamed
    t0 = time.perf_counter()
    st, r = S.c.chat(dict(base, max_tokens=60, messages=[{"role": "user", "content": "Name the capital of France in one word."}]))
    txt = r["choices"][0]["message"]["content"] if st == 200 else ""
    check(st == 200 and "paris" in txt.lower(), f"non-streamed answer {txt.strip()[:40]!r} ({time.perf_counter() - t0:.2f} s, "
          f"{r.get('timings', {}).get('path')})")
    raw = S.stream_raw(dict(base, max_tokens=60, messages=[{"role": "user", "content": "Name the capital of Italy in one word."}]))
    chunks = sse_json(raw)
    txt = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices"))
    fin = [c["choices"][0]["finish_reason"] for c in chunks if c.get("choices") and c["choices"][0]["finish_reason"]]
    check("rome" in txt.lower() and fin and raw.rstrip().endswith(b"data: [DONE]"), f"streamed answer {txt.strip()[:40]!r}, finish {fin}")
    # 2. a tool call, streamed: closes properly, valid JSON arguments, finish tool_calls
    msgs = [{"role": "system", "content": "You are a helpful assistant. Use tools when they help."},
            {"role": "user", "content": "What's the weather in Paris for the next 3 days? Use the tool."}]
    chunks = sse_json(S.stream_raw({"model": "x", "tools": TOOLS, "reasoning_effort": "low", "messages": msgs}))
    fin = [c["choices"][0]["finish_reason"] for c in chunks if c.get("choices") and c["choices"][0]["finish_reason"]]
    calls = {}
    for c in chunks:
        for ch in c.get("choices", []):
            for tc in ch["delta"].get("tool_calls", []) or []:
                e = calls.setdefault(tc["index"], {"name": "", "args": ""})
                e["name"] += (tc.get("function") or {}).get("name", "") or ""
                e["args"] += (tc.get("function") or {}).get("arguments", "") or ""
    try:
        args = json.loads(calls[0]["args"])
    except Exception:   # noqa: BLE001
        args = None
    check(fin == ["tool_calls"] and calls and calls[0]["name"] == "get_weather" and isinstance(args, dict)
          and "paris" in str(args.get("city", "")).lower(), f"streamed tool call {calls.get(0)}, finish {fin}")
    # 3. a client leaving mid-reply frees the slot within ~1 s
    long_q = {"model": "x", "reasoning_effort": "none", "temperature": 0.7, "max_tokens": 3000,
              "messages": [{"role": "user", "content": "Write a very long, detailed essay about the history of the printing press."}]}
    for stream in (False, True):
        check(S.wait_idle(20) is not None, "idle before")
        s = S.post("/v1/chat/completions", dict(long_q, stream=stream))
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 30 and S.health()["active"] == 0:
            time.sleep(0.05)
        time.sleep(2.0)   # generating
        h = S.health()
        s.close()
        dt = S.wait_idle(20)
        check(h["active"] == 1 and dt is not None and dt < 1.5,
              f"{'streamed' if stream else 'non-streamed'}: client left mid-generation (active {h['active']}) -> slot free after "
              f"{dt and round(dt, 2)} s")
    # 4. a client leaving during a long prefill: free within a piece; the next identical request reuses what was fed
    words = open(Path(__file__).resolve().parents[2] / "docs/server.md").read().split()[:9000]
    big = dict(base, max_tokens=20, messages=[{"role": "user", "content": "Summarise in a few words: " + " ".join(words)}])
    check(S.wait_idle(20) is not None, "idle before the prefill test")
    s = S.post("/v1/chat/completions", dict(big, stream=True))
    time.sleep(2.0)
    h = S.health()
    s.close()
    dt = S.wait_idle(30)
    check(h["active"] == 1 and dt is not None and dt < 4.0, f"client left during a long prefill (active {h['active']}) -> slot free after "
          f"{dt and round(dt, 2)} s")
    t0 = time.perf_counter()
    st, r = S.c.chat(big)
    n, c = r["usage"]["prompt_tokens"], r["usage"]["prompt_tokens_details"]["cached_tokens"]
    check(st == 200 and c > 0, f"the next identical request reuses the pieces already fed: {c} of {n} prompt tokens cached "
          f"({time.perf_counter() - t0:.1f} s)")
    st, r = S.c.chat(dict(base, max_tokens=30, messages=[{"role": "user", "content": "Say hi."}]))
    check(st == 200 and S.health()["status"] == "ok", "the server is healthy afterwards")
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
