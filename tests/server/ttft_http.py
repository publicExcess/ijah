"""lab-ttft: time to first streamed token of small turns at long context, through a running server's HTTP API.

    $PY tests/server/ttft_http.py --url http://127.0.0.1:8141 --key-file KEY [--levels 32768,100000,160000]
        [--sizes 50,150,500] [--reps 2] [--pack DIR]

One conversation: a system prompt + a user message of repository text up to each level (tokens counted with the pack's
tokenizer), then per size s a turn that appends the previous reply (as the client sees it) and a user message of
~s tokens of new text; greedy, max_tokens 8, streamed. TTFT = request sent -> first data chunk with a non-empty delta
(reasoning or content). Prints one line per turn (prompt / cached tokens from the server's usage and timings) and a
markdown table.
"""
import argparse
import http.client
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import refenv  # noqa: E402
import ijah_paths  # noqa: E402

refenv.add_paths()
from tokenizers import Tokenizer  # noqa: E402


def turn(url, key, messages):
    u = urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=1800)
    body = {"messages": messages, "max_tokens": 8, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", json.dumps(body),
              {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        raise SystemExit(f"HTTP {r.status}: {r.read()[:300]!r}")
    first = None
    reason, content, usage, tim = [], [], None, None
    for line in r:
        if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
            continue
        d = json.loads(line[6:])
        usage = d.get("usage") or usage
        tim = d.get("timings") or tim
        for ch in d.get("choices", []):
            dl = ch.get("delta", {})
            if dl.get("reasoning_content") or dl.get("content"):
                if first is None:
                    first = time.perf_counter()
                reason.append(dl.get("reasoning_content") or "")
                content.append(dl.get("content") or "")
    c.close()
    return (first - t0) * 1e3 if first else None, "".join(reason), "".join(content), usage or {}, tim or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8141")
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--pack", default=ijah_paths.pack())
    ap.add_argument("--levels", default="32768,100000,160000")
    ap.add_argument("--sizes", default="50,150,500")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    key = Path(a.key_file).read_text().strip()
    tok = Tokenizer.from_file(str(Path(a.pack) / "tokenizer.json"))
    files = sorted(ROOT.glob("docs/*.md")) + sorted(ROOT.glob("models/flash_next/*.mm")) + \
        sorted(ROOT.glob("kernels/**/*.metal")) + sorted(ROOT.glob("server/*.py")) + sorted(ROOT.glob("tools/*.py"))
    ids = tok.encode("\n\n".join(f.read_text(errors="replace") for f in files), add_special_tokens=False).ids
    pos = [0]

    def text(n):  # n tokens of repository text (decoded back: ~n tokens again)
        out = []
        while len(out) < n:
            k = min(n - len(out), len(ids) - pos[0])
            out += ids[pos[0]:pos[0] + k]
            pos[0] = (pos[0] + k) % len(ids)
        return tok.decode(out)

    msgs = [{"role": "system", "content": "You are a coding agent. Reference material follows.\n\n" + text(2000)}]
    rows = []
    total = 2000
    for L in [int(x) for x in a.levels.split(",") if x]:
        msgs.append({"role": "user", "content": text(max(16, L - total - 64))})
        total = L
        for kind, s in [("grow", 0)] + [("turn", s) for _ in range(a.reps) for s in
                                         [int(x) for x in a.sizes.split(",") if x]]:
            if kind == "turn":
                msgs.append({"role": "user", "content": text(max(8, s - 24))})
            ttft, r, c, usage, tim = turn(a.url, key, msgs)
            msgs.append({"role": "assistant", "content": c, **({"reasoning_content": r} if r else {})})
            pt = usage.get("prompt_tokens", 0)
            ct = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", tim.get("cached_tokens", 0))
            total = pt
            row = (f"{L // 1000}K", kind, s, pt, ct, pt - ct, ttft, tim.get("prefill_ms"), tim.get("path"))
            rows.append(row)
            print(f"{a.label} {row[0]} {kind} s{s}: prompt {pt} cached {ct} new {pt - ct}: TTFT {ttft:.0f} ms "
                  f"(server prefill_ms {tim.get('prefill_ms')}, path {tim.get('path')})", flush=True)
    print(f"\n| {a.label} level | turn | prompt | cached | new | TTFT ms (HTTP) | server prefill ms | path |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r[1] == "turn":
            pf = f"{r[7]:.0f}" if isinstance(r[7], (int, float)) else "?"
            print(f"| {r[0]} | s{r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]:.0f} | {pf} | {r[8]} |")


if __name__ == "__main__":
    main()
