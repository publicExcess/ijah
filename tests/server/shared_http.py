"""lab-snap: shared-prefix snapshots through a running server's HTTP API (docs/server.md 3f).

    $PY tests/server/shared_http.py --url http://127.0.0.1:8144 --key-file KEY --phase first|restart --state S.json

A synthetic coding-agent-like first turn (no real client's content is read): 25 tool schemas (~5K tokens) + a long agent system prompt
(identical across sessions) + a short per-session environment tail (directory, date, session id) inside the system
message, then a different user message: ~7.8K tokens, the system message's end never shared. TTFT = request sent ->
first streamed delta. Greedy (temperature 0), thinking on (the server's default effort), max_tokens 96.

phase first:   warm-up; arm OFF (request extension ne_shared false = the earlier server) A cold, B; arm ON (another nonce:
               nothing shared with OFF) A cold, B (restores the tools block snapshot, captures the learned one), C
               (restores the learned one), C's next turn; cold references (ne_cache false, ne_shared false) of B and C:
               outputs must match; a second cold A per arm (cold-path cost). Writes the state file.
phase restart: (a new server process on the same --cache-dir) session D of arm ON: restored from the SSD tier;
               its cold reference.
"""
import argparse
import http.client
import json
import sys
import time
import zlib
from pathlib import Path
from urllib.parse import urlparse

TOOL_NAMES = ["bash", "read_file", "write_file", "edit_file", "multi_edit", "glob", "grep", "list_dir", "web_fetch",
              "web_search", "task", "todo_write", "todo_read", "notebook_read", "notebook_edit", "ask_user",
              "plan_enter", "plan_exit", "memory_read", "memory_write", "git_status", "git_diff", "run_tests",
              "open_url", "screenshot"]
VERBS = ["Reads", "Writes", "Searches", "Lists", "Runs", "Edits", "Fetches", "Records", "Inspects", "Summarises"]


def tools(nonce):
    out = []
    for i, nm in enumerate(TOOL_NAMES):
        v = VERBS[i % len(VERBS)]
        desc = (f"{v} {nm.replace('_', ' ')} for the current task. " + (f"[{nonce}] " if i == 0 else "") +
                f"Use this tool when the user asks for work that involves {nm.replace('_', ' ')}; prefer it over "
                "shell commands that do the same thing. The tool returns plain text. Errors are reported as text "
                "starting with 'error:'. Paths are absolute. Never pass secrets. Keep calls small and specific; "
                f"call it again with a narrower argument when the output is truncated. Tool number {i + 1} of 25. ")
        props = {"path": {"type": "string", "description": "Absolute path of the file or directory to operate on."},
                 "query": {"type": "string", "description": "Pattern, command or text the tool works with."},
                 "limit": {"type": "integer", "description": "Largest number of results or lines to return."},
                 "options": {"type": "object", "description": "Tool specific options; see the description.",
                             "properties": {"recursive": {"type": "boolean"}, "timeout_ms": {"type": "integer"}}}}
        out.append({"type": "function", "function": {"name": nm, "description": desc, "parameters": {
            "type": "object", "properties": props, "required": ["path"] if i % 2 else ["query"]}}})
    return out


SYSTEM = "\n\n".join(
    f"## Rule {i + 1}\nYou are a careful software engineering agent working in the user's terminal. "
    f"Before you change file number {i + 1} of a task, read it; keep edits minimal and explain them briefly. "
    "Prefer the dedicated tools over shell commands; run the project's tests after a change; never commit unless "
    "asked; never print credentials; ask when a request is ambiguous; report what you verified and what you did not."
    for i in range(16))


def env_tail(tag):
    return (f"\n\n# Environment\nWorking directory: ~/Projects/app-{tag}\nPlatform: darwin\n"
            f"Date: 2026-10-{10 + (sum(map(ord, tag)) % 18):02d} {sum(map(ord, tag)) % 24:02d}:17\n"
            f"Session: {zlib.crc32(tag.encode()):010d}-{tag}\nGit branch: main")


USER = {"A": "Look at the build scripts and tell me which targets are slow to compile.",
        "B": "Find every place where the config file is parsed and list the keys that are read.",
        "C": "Write a short plan for adding retries to the HTTP client, with the files you would touch.",
        "D": "Explain what the test runner does when a test times out, citing the source.",
        "E": "Which modules import the logging setup, and in which order are they loaded?",
        "F": "Propose names for the three new commands and say where their code would live.",
        "G": "Check whether the cache directory is cleaned up when the process exits early."}


SYSTEM2 = "You are a different agent with other rules.\n\n" + SYSTEM.replace("careful", "patient")


# (the user message is >= 33 tokens: a cold prompt's piece between the system message's end and the generation prompt
# then takes the chunk path, as every piece of a restored one; 16-32-row pieces run split-K, another valid arithmetic
# whose logits differ at the ~0.05-0.4 logprob level: docs/server.md 3e, 3f)
ASK = (" Work in small steps, show the commands you would run, and stop to ask me before you change anything outside "
       "the source tree or touch the network.")


def first_turn(nonce, sess, system=SYSTEM):
    return {"messages": [{"role": "system", "content": system + env_tail(f"{nonce}-{sess}")},
                         {"role": "user", "content": USER[sess] + ASK + " (" + sess + ")"}],
            "tools": tools(nonce)}


def call(url, key, req, **ext):
    u = urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=1800)
    body = dict(req, max_tokens=96, temperature=0.0, stream=True, stream_options={"include_usage": True}, **ext)
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", json.dumps(body),
              {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        raise SystemExit(f"HTTP {r.status}: {r.read()[:300]!r}")
    first = None
    reason, content, calls, usage, tim, lps = [], [], [], None, None, []
    for line in r:
        if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
            continue
        d = json.loads(line[6:])
        usage = d.get("usage") or usage
        tim = d.get("timings") or tim
        for ch in d.get("choices", []):
            dl = ch.get("delta", {})
            for lp in ((ch.get("logprobs") or {}).get("content") or []):
                lps.append((lp["token"], lp["logprob"], [(t["token"], t["logprob"]) for t in lp.get("top_logprobs", [])]))
            if dl.get("reasoning_content") or dl.get("content") or dl.get("tool_calls"):
                if first is None:
                    first = time.perf_counter()
                reason.append(dl.get("reasoning_content") or "")
                content.append(dl.get("content") or "")
                if dl.get("tool_calls"):
                    calls.append(json.dumps(dl["tool_calls"]))
    c.close()
    usage = usage or {}
    tim = tim or {}
    return {"ttft_ms": (first - t0) * 1e3 if first else None, "reasoning": "".join(reason),
            "content": "".join(content), "calls": "".join(calls), "prompt": usage.get("prompt_tokens"),
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"), "timings": tim, "lps": lps}


def show(label, r):
    t = r["timings"]
    print(f"{label:<28} prompt {r['prompt']} cached {r['cached']}: TTFT {r['ttft_ms']:.0f} ms (prefill "
          f"{t.get('prefill_ms')} ms, restore {t.get('restore_ms')}, cache {t.get('cache')}, shared_prefix "
          f"{t.get('shared_prefix', False)}, made {t.get('shared_made', 0)}, path {t.get('path')})", flush=True)


def same(a, b):
    return (a["reasoning"], a["content"], a["calls"]) == (b["reasoning"], b["content"], b["calls"])


def lp_check(nm, x, y, exact=False):
    """step path with logprobs: restored vs cold. -> (ok, line). exact: every token and logprob equal (bit-identical);
    else every token equal (or the first difference a tie: the two top logprobs there within 0.05) and the chosen
    tokens' logprobs within 0.05"""
    a, b = x["lps"], y["lps"]
    n = min(len(a), len(b))
    d = 0.0
    for i in range(n):
        if a[i][0] != b[i][0]:
            top = sorted(lp for _, lp in b[i][2])[-2:] if len(b[i][2]) >= 2 else [0, -9]
            tie = abs(top[1] - top[0]) < 0.05 and not exact
            return tie, (f"session {nm} (step path): first {i} tokens equal, then {a[i][0]!r} vs {b[i][0]!r} "
                         f"(cold top-2 logprob gap {abs(top[1] - top[0]):.3f}: {'tie' if tie else 'NOT a tie'}); "
                         f"max |dlogprob| before {d:.2e}")
        d = max(d, abs(a[i][1] - b[i][1]))
    ok = n > 0 and n >= min(64, len(b)) and (d == 0.0 if exact else d < 0.05)
    return ok, f"session {nm} (step path): all {n} tokens equal, max |dlogprob| {d:.2e}"


def diverge(x, y):
    p, q = x["reasoning"] + "\0" + x["content"], y["reasoning"] + "\0" + y["content"]
    return next((j for j in range(min(len(p), len(q))) if p[j] != q[j]), min(len(p), len(q)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8144")
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--phase", choices=["first", "restart"], required=True)
    ap.add_argument("--state", required=True)
    a = ap.parse_args()
    key = Path(a.key_file).read_text().strip()
    st = json.loads(Path(a.state).read_text()) if a.phase == "restart" else {"nonce": f"{time.time_ns() % 10**9:09d}"}
    N = st["nonce"]
    res = st.setdefault("res", {})
    off = {"ne_shared": False}
    cold = {"ne_shared": False, "ne_cache": False}
    ok = True

    def go(label, req, **ext):
        r = call(a.url, key, req, **ext)
        res[label] = r
        show(label, r)
        return r

    if a.phase == "first":
        go("warmup", {"messages": [{"role": "user", "content": "Say hi."}]})
        go("off A cold", first_turn(N + "off", "A"), **off)
        go("off B", first_turn(N + "off", "B"), **off)
        go("on A cold", first_turn(N + "on", "A"))
        b = go("on B", first_turn(N + "on", "B"))
        c = go("on C", first_turn(N + "on", "C"))
        req2 = first_turn(N + "on", "C")
        req2["messages"] += [{"role": "assistant", "content": c["content"] or "I will look.",
                              **({"reasoning_content": c["reasoning"]} if c["reasoning"] else {})},
                             {"role": "user", "content": "Thanks. Now also list the tests that cover it."}]
        go("on C turn 2", req2)
        rb = go("ref B (cold)", first_turn(N + "on", "B"), **cold)
        rc = go("ref C (cold)", first_turn(N + "on", "C"), **cold)
        go("off A2 cold", first_turn(N + "off2", "A"), **off)
        go("on A2 cold", first_turn(N + "on2", "A"))
        # exactness on the step path (deterministic given the prefill state): E restores the learned snapshot, F
        # (other system text, same tools) the tools block's
        stp = {"ne_path": "step", "logprobs": True, "top_logprobs": 5}
        # two cold references: "this server" = reuse off, captures on: the cold schedule with this change (the tools
        # block's end replaces the system message's end as a piece boundary; every piece behind it >= ~130 rows, as in
        # the restored request) -> must be bit-identical; "earlier schedule" = reuse and captures off (the system
        # checkpoint leaves a ~50-row piece before the generation prompt, which runs other kernels: another valid
        # arithmetic; two cold prefills with these two schedules differ as much) -> info
        e = go("on E (step, logprobs)", first_turn(N + "on", "E"), **stp, ne_shared=False)
        re_ = go("ref E (cold, this server)", first_turn(N + "on", "E"), **stp, ne_cache=False)
        re0 = go("ref E (cold, earlier sched)", first_turn(N + "on", "E"), **stp, **cold)
        f = go("on F (step, logprobs)", first_turn(N + "on", "F", SYSTEM2), **stp, ne_shared=False)
        rf = go("ref F (cold, this server)", first_turn(N + "on", "F", SYSTEM2), **stp, ne_cache=False)
        rf0 = go("ref F (cold, earlier sched)", first_turn(N + "on", "F", SYSTEM2), **stp, **cold)
        for nm, x, y, y0 in (("E", e, re_, re0), ("F", f, rf, rf0)):
            m, line = lp_check(nm, x, y, exact=True)
            ok &= m and (x["cached"] or 0) > 0
            print(("ok   " if m else "FAIL ") + line + f" vs a cold prefill on this server (cached {x['cached']})", flush=True)
            print("info " + lp_check(nm, x, y0)[1] + " vs the earlier schedule cold", flush=True)
            print("info " + lp_check(nm, y, y0)[1].replace("(step path)", "cold this server") +
                  " vs the earlier schedule cold (no restore involved)", flush=True)
        for nm, x, y in (("B", b, rb), ("C", c, rc)):  # MTP path: the drafts' history differs, ties may resolve apart
            print(f"info session {nm} (MTP) restored vs cold: {'identical' if same(x, y) else 'first difference at char %d of %d' % (diverge(x, y), len(x['reasoning']) + len(x['content']))}")
        ok &= (b["cached"] or 0) > 0 and (c["cached"] or 0) > (b["cached"] or 0)
        ok &= res["on C"]["timings"].get("path") in ("mtp", "mtp+step")
    else:
        go("restart warmup", {"messages": [{"role": "user", "content": "Say hello."}]})
        d = go("restart: on D", first_turn(N + "on", "D"))
        stp = {"ne_path": "step", "logprobs": True, "top_logprobs": 5}
        g = go("restart: on G (step)", first_turn(N + "on", "G"), **stp, ne_shared=False)
        rg = go("ref G (cold, this server)", first_turn(N + "on", "G"), **stp, ne_cache=False)
        m, line = lp_check("G", g, rg, exact=True)
        ok &= m and d["timings"].get("cache") == "ssd" and (d["cached"] or 0) > 0 and (g["cached"] or 0) > 0
        print(("ok   " if m else "FAIL ") + line + f" vs a cold prefill on this server (restored from the SSD tier: cached "
              f"{g['cached']}, cache {g['timings'].get('cache')})")
    Path(a.state).write_text(json.dumps(st, indent=1))
    print("shared_http:", "all passed" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
