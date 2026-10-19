"""Client of tests/server/persist_job.sh (lab-persist): one long conversation across server restarts.

--phase A   (fresh SSD tier) turn 1 cold, turn 2 reusing turn 1's checkpoint (from RAM), turn 2 regenerated (restores
            turn 2's own generation-prompt checkpoint from RAM: the state a next request after a restart restores)
--phase B   (restarted server, same SSD tier) turn 2 again: restored from SSD at the same checkpoint, output must equal
            phase A's regenerate bit for bit; then the same prompt with ne_cache false (cold reference)
--phase C   (restarted after every KV segment file was corrupted) turn 2: the entry is rejected, cold prefill, output
            = phase B's cold reference
Greedy (temperature 0), thinking off, 160 tokens. State between phases: --state JSON.
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def post(url, key, body):
    r = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                               {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    t0 = time.perf_counter()
    j = json.load(urllib.request.urlopen(r, timeout=900))
    j["_wall_ms"] = (time.perf_counter() - t0) * 1e3
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--phase", choices="ABC", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--max-tokens", type=int, default=160)
    a = ap.parse_args()
    key = Path(a.key_file).read_text().strip()
    st = json.loads(Path(a.state).read_text()) if Path(a.state).exists() else {}
    if "system" not in st:  # frozen in phase A: the docs may change between phases
        st["system"] = "\n\n".join((ROOT / f"docs/{d}.md").read_text() for d in ("mtp", "server", "kernels"))
    system = st["system"]
    msgs = [{"role": "system", "content": "You answer questions about these engine notes.\n\n" + system},
            {"role": "user", "content": "In two sentences: what does the server's prefix cache keep per conversation?"}]
    base = {"max_tokens": a.max_tokens, "temperature": 0, "reasoning_effort": "none"}
    fails = []

    def turn(m, label, **kw):
        j = post(a.url, key, {"messages": m, **base, **kw})
        t, u = j["timings"], j["usage"]
        row = {"label": label, "prompt": u["prompt_tokens"], "cached": u["prompt_tokens_details"]["cached_tokens"],
               "cache": t.get("cache"), "prefill_ms": t["prefill_ms"], "restore_ms": t.get("restore_ms"),
               "persist_ms": t.get("persist_ms"), "path": t["path"], "decode_tok_s": t["decode_tok_s"],
               "wall_ms": round(j["_wall_ms"]), "text": j["choices"][0]["message"]["content"]}
        print(json.dumps({k: v for k, v in row.items() if k != "text"}))
        return row

    if a.phase == "A":
        t1 = turn(msgs, "A turn 1 (cold)")
        st["t1_text"] = t1["text"]
        st["rows"] = [t1]
    m2 = msgs + [{"role": "assistant", "content": st["t1_text"]},
                 {"role": "user", "content": "And what happens to it when the RAM budget is exceeded?"}]
    if a.phase == "A":
        t2 = turn(m2, "A turn 2 (RAM reuse)")
        st["t2_text"] = t2["text"]
        st["rows"].append(t2)
        if t2["cache"] not in ("ram", "live") or t2["cached"] <= 0:
            fails.append("phase A turn 2 did not reuse from RAM")
        rg = turn(m2, "A turn 2 regenerated (RAM restore)")
        st["regen_text"], st["regen_cached"] = rg["text"], rg["cached"]
        st["rows"].append(rg)
    elif a.phase == "B":
        t2 = turn(m2, "B turn 2 after restart (SSD)")
        cold = turn(m2, "B turn 2 cold reference", ne_cache=False)
        st["cold_text"] = cold["text"]
        st["rows"] += [t2, cold]
        if t2["cache"] != "ssd" or t2["cached"] != st["regen_cached"]:
            fails.append(f"phase B: not restored from SSD at the same checkpoint ({t2['cache']}, {t2['cached']})")
        if t2["text"] != st["regen_text"]:
            fails.append("phase B: output after the restart differs from the never-restarted server's")
        print(f"restart restore: output {'IDENTICAL' if t2['text'] == st['regen_text'] else 'DIFFERENT'} to phase A; "
              f"prefill {t2['prefill_ms']:.0f} ms (restore {t2['restore_ms']} ms) vs cold {cold['prefill_ms']:.0f} ms; "
              f"cold output {'=' if cold['text'] == t2['text'] else '!='} reused output")
    else:
        t2 = turn(m2, "C turn 2, corrupted SSD entry")
        st["rows"].append(t2)
        if t2["cache"] != "none" or t2["cached"] != 0:
            fails.append(f"phase C: a corrupted entry was used ({t2['cache']}, {t2['cached']})")
        if t2["text"] != st["cold_text"]:
            fails.append("phase C: output differs from the cold reference")
        print(f"corrupted entry: cache {t2['cache']}, cached {t2['cached']}, output "
              f"{'=' if t2['text'] == st['cold_text'] else '!='} cold reference")
    Path(a.state).write_text(json.dumps(st, indent=1))
    for f in fails:
        print("FAIL", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
