"""Client test of the Ijah OpenAI server (lab-server), stdlib only. Runs against a server that is up:

    $PY tests/server/test_client.py --url http://127.0.0.1:8140 --key-file ~/.config/ijah/api-key [--mock]
        [--bench N]      # + tok/s through HTTP vs the engine's own decode timing, step and MTP paths, N tokens

Checks: auth refusal (no key / wrong key), /v1/models, /health, streaming text, reasoning separation, a tool-call round
trip with a dummy tool (streamed tool_calls, then the tool result back; the second turn must reuse the prefix),
stop strings, usage counts, two concurrent requests (queued). The key is read from the file and never printed.
"""
import argparse
import http.client
import json
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

FAILS = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


class Client:
    def __init__(self, url, key):
        u = urlparse(url)
        self.host, self.port, self.key = u.hostname, u.port, key

    def _conn(self):
        return http.client.HTTPConnection(self.host, self.port, timeout=900)

    def req(self, method, path, body=None, key=True, raw_key=None):
        c = self._conn()
        h = {"Content-Type": "application/json"}
        if key:
            h["Authorization"] = "Bearer " + (raw_key if raw_key is not None else self.key)
        c.request(method, path, json.dumps(body) if body is not None else None, h)
        r = c.getresponse()
        data = r.read()
        c.close()
        try:
            return r.status, json.loads(data)
        except json.JSONDecodeError:
            return r.status, data

    def chat(self, body):
        return self.req("POST", "/v1/chat/completions", body)

    def stream(self, body):
        """-> (chunks, done_seen, first_token_s, wall_s)"""
        c = self._conn()
        t0 = time.perf_counter()
        c.request("POST", "/v1/chat/completions", json.dumps(dict(body, stream=True)),
                  {"Content-Type": "application/json", "Authorization": "Bearer " + self.key})
        r = c.getresponse()
        assert r.status == 200, (r.status, r.read()[:300])
        chunks, done, first = [], False, None
        buf = b""
        while True:
            line = r.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data: "):
                continue
            d = line[6:]
            if d == b"[DONE]":
                done = True
                break
            ch = json.loads(d)
            if first is None and ch.get("choices") and any(
                    ch["choices"][0]["delta"].get(k) for k in ("content", "reasoning_content", "tool_calls")):
                first = time.perf_counter() - t0
            chunks.append(ch)
        c.close()
        return chunks, done, first, time.perf_counter() - t0


def merge(chunks):
    content, reasoning, calls, finish, usage = "", "", {}, None, None
    for ch in chunks:
        if ch.get("usage"):
            usage = ch["usage"]
        for c in ch.get("choices", []):
            d = c["delta"]
            content += d.get("content") or ""
            reasoning += d.get("reasoning_content") or ""
            for tc in d.get("tool_calls") or []:
                e = calls.setdefault(tc["index"], {"id": None, "name": "", "arguments": ""})
                e["id"] = tc.get("id") or e["id"]
                f = tc.get("function") or {}
                e["name"] += f.get("name") or ""
                e["arguments"] += f.get("arguments") or ""
            finish = c.get("finish_reason") or finish
    return content, reasoning, [calls[k] for k in sorted(calls)], finish, usage


TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "city name"},
                                                    "days": {"type": "integer", "description": "forecast days"}},
                   "required": ["city"]}}}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8140")
    ap.add_argument("--key-file", default=str(Path.home() / ".config/ijah/api-key"))
    ap.add_argument("--mock", action="store_true", help="expect the mock engine's scripted answers")
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--bench-think", action="store_true",
                    help="bench on bench_mtp's code_lru prompt, thinking on at medium effort (docs/mtp.md 10.3)")
    ap.add_argument("--json", default="")
    ap.add_argument("--expect-model", default="", help="the model id /v1/models must report (lab-swift)")
    a = ap.parse_args()
    key = Path(a.key_file).expanduser().read_text().strip()
    cl = Client(a.url, key)
    res = {}

    # -- auth + listing
    st, _ = cl.req("GET", "/v1/models", key=False)
    check(st == 401, f"no key -> 401 ({st})")
    st, _ = cl.req("GET", "/v1/models", raw_key="sk-wrong-" + "x" * 40)
    check(st == 401, f"wrong key -> 401 ({st})")
    st, _ = cl.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}, key=False)
    check(st == 401, f"chat without key -> 401 ({st})")
    st, m = cl.req("GET", "/v1/models")
    check(st == 200 and m["data"][0]["context_length"] > 0, f"/v1/models ({m['data'][0]['id']}, ctx "
                                                           f"{m['data'][0]['context_length']})")
    model = m["data"][0]["id"]
    if a.expect_model:
        check(model == a.expect_model, f"/v1/models id is {a.expect_model} ({model})")
    st, h = cl.req("GET", "/health", key=False)
    check(st == 200 and h["status"] == "ok", "/health")
    st, e = cl.chat({"model": model, "messages": [], "stream": False})
    check(st == 400 and "error" in e, "empty messages -> 400")

    # -- streaming text (thinking off)
    q = "Name the capital of France in one word." if not a.mock else "hello"
    chunks, done, first, wall = cl.stream({"model": model, "reasoning_effort": "none", "stream_options":
                                           {"include_usage": True}, "messages": [{"role": "user", "content": q}]})
    content, reasoning, calls, finish, usage = merge(chunks)
    check(done and finish == "stop" and content.strip() and not reasoning and "<" not in content,
          f"stream text: {content.strip()[:60]!r} ({len(chunks)} chunks, first token {first:.2f} s)")
    check(a.mock or "paris" in content.lower(), "stream text answers Paris")
    check(usage and usage["prompt_tokens"] > 0 and usage["completion_tokens"] > 0
          and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], f"stream usage {usage}")

    # -- reasoning separation (thinking on, low effort), non-streamed
    q = "What is 17 * 23? Reply with just the number." if not a.mock else "hello"
    st, r = cl.chat({"model": model, "reasoning_effort": "low", "messages": [{"role": "user", "content": q}]})
    msg = r["choices"][0]["message"] if st == 200 else {}
    check(st == 200 and msg.get("reasoning_content") and msg.get("content")
          and "think>" not in msg["content"] and "think>" not in msg["reasoning_content"],
          f"reasoning separated: reasoning {len(msg.get('reasoning_content') or '')} chars, content "
          f"{(msg.get('content') or '').strip()[:40]!r}")
    check(a.mock or "391" in (msg.get("content") or ""), "17 * 23 = 391")
    u = r.get("usage", {})
    check(u.get("completion_tokens_details", {}).get("reasoning_tokens", 0) > 0
          and u["completion_tokens"] > u["completion_tokens_details"]["reasoning_tokens"], f"usage reasoning tokens {u}")
    res["reasoning_timings"] = r.get("timings")

    # -- tool call round trip
    q = "What's the weather in Paris for the next 3 days? Use the tool." if not a.mock else "TOOLTEST"
    msgs = [{"role": "system", "content": "You are a helpful assistant. Use tools when they help."},
            {"role": "user", "content": q}]
    # turn 1 on the step path (the MTP loop keeps no checkpoints, and the auto policy picks it for short prompts)
    chunks, done, first, wall = cl.stream({"model": model, "tools": TOOLS, "reasoning_effort": "low", "ne_path": "step",
                                           "messages": msgs, "stream_options": {"include_usage": True}})
    content, reasoning, calls, finish, usage1 = merge(chunks)
    args = {}
    try:
        args = json.loads(calls[0]["arguments"]) if calls else {}
    except json.JSONDecodeError:
        pass
    check(finish == "tool_calls" and calls and calls[0]["name"] == "get_weather" and calls[0]["id"]
          and "paris" in str(args.get("city", "")).lower(), f"streamed tool call: {calls}")
    check("<tool_call>" not in content and "<function" not in content, "no tool markup in content")
    check(not args or "days" not in args or isinstance(args["days"], int), "integer parameter typed by the schema")
    msgs2 = msgs + [{"role": "assistant", "content": content, "reasoning_content": reasoning,
                     "tool_calls": [{"id": c["id"], "type": "function",
                                     "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]}]
    msgs2 += [{"role": "tool", "tool_call_id": c["id"], "content": json.dumps(
        {"city": "Paris", "forecast": ["21 C sunny", "19 C cloudy", "17 C rain"]})} for c in calls]
    st, r2 = cl.chat({"model": model, "tools": TOOLS, "reasoning_effort": "low", "messages": msgs2, "ne_path": "step"})
    m2 = r2["choices"][0]["message"] if st == 200 else {}
    u2 = r2.get("usage", {})
    check(st == 200 and r2["choices"][0]["finish_reason"] == "stop" and (m2.get("content") or "").strip()
          and not m2.get("tool_calls"), f"tool result -> answer: {(m2.get('content') or '').strip()[:80]!r}")
    check(a.mock or "21" in (m2.get("content") or ""), "answer uses the tool result")
    cached = u2.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    check(cached >= 0.9 * (usage1 or {}).get("prompt_tokens", 1), f"turn 2 reuses the prefix: {cached} cached of "
                                                                    f"{u2.get('prompt_tokens')} (turn 1 prompt "
                                                                    f"{(usage1 or {}).get('prompt_tokens')})")
    res["tool_turn2_timings"] = r2.get("timings")
    # greedy turn 2 from the restored checkpoint vs from an empty state: the same tokens up to bf16 ties (the
    # prefill's chunk boundaries differ, so a late fork at a tie is expected; an early one means a bad restore)
    g = {"model": model, "tools": TOOLS, "reasoning_effort": "low", "messages": msgs2, "ne_path": "step",
         "temperature": 0, "max_tokens": 64}
    st_a, ra = cl.chat(dict(g))
    st_b, rb = cl.chat(dict(g, ne_cache=False))
    ta = (ra["choices"][0]["message"].get("reasoning_content") or "") + (ra["choices"][0]["message"].get("content") or "")
    tb = (rb["choices"][0]["message"].get("reasoning_content") or "") + (rb["choices"][0]["message"].get("content") or "")
    same = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
    ca = ra["usage"]["prompt_tokens_details"]["cached_tokens"]
    cb = rb["usage"]["prompt_tokens_details"]["cached_tokens"]
    check(ca > 0 and cb == 0 and same >= min(40, len(ta), len(tb)),
          f"greedy from checkpoint ({ca} cached) vs fresh ({cb}): first {same} of {len(ta)} chars identical")
    res["greedy_reuse_vs_fresh"] = {"cached": ca, "same_chars": same, "len": [len(ta), len(tb)]}

    # -- stop strings
    if a.mock:
        body = {"model": model, "messages": [{"role": "user", "content": "STOPTEST"}], "stop": ["STOP_HERE"]}
    else:
        body = {"model": model, "reasoning_effort": "none", "stop": [" 7", ",7", "7"], "messages": [
            {"role": "user", "content": "Count from 1 to 15, numbers separated by single spaces, nothing else."}]}
    chunks, done, _, _ = cl.stream(body)
    content, _, _, finish, _ = merge(chunks)
    check(finish == "stop" and content.strip() and "7" not in content and "STOP" not in content
          and "four" not in content, f"stop string: {content.strip()!r}")
    st, r = cl.chat(body)
    c = r["choices"][0]["message"]["content"] if st == 200 else ""
    check(st == 200 and "7" not in c and "STOP" not in c, f"stop string, non-streamed: {c.strip()!r}")

    # -- two concurrent requests (queued, both complete)
    out = [None, None]

    def go(i):
        b = {"model": model, "reasoning_effort": "none",
             "messages": [{"role": "user", "content": ["Say hi.", "Say bye."][i] if not a.mock else "hello"}]}
        out[i] = cl.chat(b)

    ths = [threading.Thread(target=go, args=(i,)) for i in range(2)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    check(all(o and o[0] == 200 and o[1]["choices"][0]["message"]["content"] for o in out),
          "two concurrent requests both answered (queued)")

    # -- bench: tok/s through HTTP vs the engine's own decode timing
    if a.bench:
        q = "LONGTEST" if a.mock else ("Write a detailed, long essay about the history of the printing press, its "
                                       "technology and its effects on Europe.")
        eff = "none"
        if a.bench_think and not a.mock:
            eff, q = "medium", ("Implement a thread-safe LRU cache in Python with O(1) get and put, a per-entry TTL, "
                                "and an eviction callback. Use type hints, explain the design briefly, and include "
                                "pytest tests covering expiry, eviction order and concurrent access.")
        for path in ("step", "mtp"):
            chunks, done, first, wall = cl.stream({
                "model": model, "reasoning_effort": eff, "max_tokens": a.bench, "ne_path": path,
                "temperature": 1.0, "seed": 7, "messages": [{"role": "user", "content": q}],
                "stream_options": {"include_usage": True}})
            _, _, _, finish, usage = merge(chunks)
            tim = next((ch.get("timings") for ch in chunks if ch.get("timings")), {}) or {}
            n = usage["completion_tokens"] if usage else 0
            http_tps = (n - 1) / (wall - first) if wall > first and n > 1 else 0
            print(f"bench {path}: {n} tokens; through HTTP {http_tps:.1f} tok/s (first token {first:.2f} s); engine "
                  f"{tim.get('decode_tok_s')} tok/s ({tim.get('path')}); prefill {tim.get('prefill_ms')} ms")
            res[f"bench_{path}"] = {"tokens": n, "http_tok_s": round(http_tps, 2), "first_token_s": round(first, 3),
                                    "engine": tim}
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    print(f"\n{len(FAILS)} failed: {FAILS}" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
