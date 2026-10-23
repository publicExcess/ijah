"""pfast: pi-shaped agent turns through a running ne_server over HTTP (same client for every arm).

    $PY tools/bench_turn_http.py --port 8143 --key-file KEY --model Qwen3.8-Flash-Next-native-dflash \
        [--levels 16384,65536,90000] [--sizes 50,500,2000] [--reps 2] [--max-tokens 16] [--tag A] [--out FILE.jsonl]

A conversation like a coding agent's: a system prompt with a tools list, one long user message of repository text up to
each context level, then turns that each add ~s new tokens (the previous reply as the assistant message + a user
message of fresh repository text), streamed, the server's default sampling (no seed: the MTP / drafter path pi
gets). Per turn: client TTFT (request sent -> first SSE delta, reasoning or content) and the prompt size. The server's
own split (prefill ms, cached tokens; NE_SRV_TRACE lines) is in its log. Out: one JSON line per turn.
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import refenv  # noqa: E402

refenv.add_paths()
from tokenizers import Tokenizer  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters": {
    "type": "object", "properties": {k: {"type": "string", "description": k} for k in ks}, "required": list(ks)}}}
    for n, d, ks in (("read", "Read a file from the workspace", ("path",)),
                     ("bash", "Run a shell command in the workspace and return its output", ("command",)),
                     ("edit", "Replace an exact string in a file", ("path", "old", "new")),
                     ("write", "Write a file", ("path", "content")))]


class Filler:
    def __init__(self, tok):
        files = sorted(ROOT.glob("docs/*.md")) + sorted(ROOT.glob("models/flash_next/*.mm")) + \
            sorted(ROOT.glob("kernels/**/*.metal")) + sorted(ROOT.glob("server/*.py")) + sorted(ROOT.glob("tools/*.py"))
        self.text = "\n\n".join(f.read_text(errors="replace") for f in files)
        self.tok = tok
        self.i = 0

    def take(self, n):
        """~n tokens of the text, from where the last call stopped (wraps)"""
        out, have = [], 0
        while have < n:
            j = min(len(self.text), self.i + max(64, int((n - have) * 3.2)))
            seg = self.text[self.i:j]
            k = len(self.tok.encode(seg, add_special_tokens=False).ids)
            if have + k > n * 1.03 and len(seg) > 64:
                seg = seg[: max(64, int(len(seg) * (n - have) / max(k, 1)))]
                k = len(self.tok.encode(seg, add_special_tokens=False).ids)
            out.append(seg)
            have += k
            self.i += len(seg)
            if self.i >= len(self.text):
                self.i = 0
        return "".join(out)


def request(a, key, msgs):
    body = json.dumps({"model": a.model, "messages": msgs, "tools": TOOLS, "stream": True,
                       "stream_options": {"include_usage": True}, "max_tokens": a.max_tokens,
                       **({"temperature": a.temperature} if a.temperature is not None else {})}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.perf_counter()
    ttft = None
    text, reason = [], []
    usage = {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            d = line[5:].strip()
            if d == b"[DONE]":
                break
            o = json.loads(d)
            if o.get("usage"):
                usage = o["usage"]
            for ch in o.get("choices") or []:
                de = ch.get("delta") or {}
                c, rc = de.get("content"), de.get("reasoning_content") or de.get("reasoning")
                if (c or rc) and ttft is None:
                    ttft = (time.perf_counter() - t0) * 1e3
                if c:
                    text.append(c)
                if rc:
                    reason.append(rc)
    return ttft, (time.perf_counter() - t0) * 1e3, "".join(text), usage, "".join(reason)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8143)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-native-dflash")
    ap.add_argument("--pack", default=(ROOT / "PACK_READY").read_text().strip())
    ap.add_argument("--levels", default="16384,65536,90000")
    ap.add_argument("--sizes", default="50,500,2000")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=None, help="0 = greedy (identity checks); default: the server's")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    key = Path(a.key_file).read_text().strip()
    tok = Tokenizer.from_file(str(Path(a.pack) / "tokenizer.json"))
    fill = Filler(tok)
    out = open(a.out, "a") if a.out else None
    msgs = [{"role": "system", "content": "You are a coding agent working in a repository. Use the tools to read, "
             "run and edit. Reference material follows.\n\n" + fill.take(2000)},
            {"role": "user", "content": fill.take(1500)}]

    def turn(label, size):
        ttft, wall, text, usage, reason = request(a, key, msgs)
        pt = usage.get("prompt_tokens")
        rec = dict(tag=a.tag, turn=label, size=size, prompt=pt, ttft_ms=round(ttft or -1, 1), wall_ms=round(wall, 1),
                   text=text, reasoning=reason)
        print(json.dumps(rec), flush=True)
        if out and label.startswith("T"):
            out.write(json.dumps(rec) + "\n")
            out.flush()
        msgs.append({"role": "assistant", "content": text or "ok"})
        return pt or 0

    n = turn("warm", 0)
    for s in (50, 500):
        msgs.append({"role": "user", "content": fill.take(s)})
        n = turn("warm", s)
    for L in [int(x) for x in a.levels.split(",") if x]:
        msgs.append({"role": "user", "content": fill.take(max(64, L - n))})
        n = turn(f"grow{L // 1000}K", L)
        for r in range(a.reps):
            for s in [int(x) for x in a.sizes.split(",") if x]:
                msgs.append({"role": "user", "content": fill.take(s)})
                n = turn(f"T{L // 1000}K", s)


if __name__ == "__main__":
    sys.exit(main())
