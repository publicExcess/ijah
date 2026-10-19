"""Several models + the SSD prefix cache across a restart (lab-srvperf x lab-persist). Mock or real engine, greedy.

    --phase A   both ids answer; model A runs turn 1 of a conversation (system prompt > 256 tokens: its checkpoint is
                written through to A's SSD directory); state -> --state JSON
    (restart the server with the same --cache-dir / --cache-disk)
    --phase B   A's turn 2 restores from SSD (cached tokens > 0) and equals the same request with ne_cache false on the
                mock (STATETEST digest); B with A's prompt reuses nothing (per-model keys)

    $PY tests/server/models_restart.py --url URL --key-file KEY --models A,B --phase A|B --state S.json [--mock]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_client import FAILS, Client, check  # noqa: E402

SYS = "You answer questions about a fictional city called Lumen; keep answers short. " * 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--phase", choices=["A", "B"], required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    A, Bm = a.models.split(",")[:2]
    c = Client(a.url, Path(a.key_file).read_text().strip())
    q2 = "And its river? STATETEST" if a.mock else "And its river? One sentence."

    def chat(model, msgs, **kw):
        st, r = c.chat(dict({"model": model, "temperature": 0, "max_tokens": 48, "reasoning_effort": "none",
                             "messages": msgs}, **kw))
        assert st == 200, (st, r)
        return r["choices"][0]["message"]["content"], r["usage"]["prompt_tokens_details"]["cached_tokens"], r["model"]

    m1 = [{"role": "system", "content": SYS}, {"role": "user", "content": "Name Lumen's main square."}]
    if a.phase == "A":
        for m in (A, Bm):
            _, _, rm = chat(m, [{"role": "user", "content": "Say hi."}])
            check(rm == m, f"phase A: {m} answers ({rm})")
        t1, k1, _ = chat(A, m1)
        Path(a.state).write_text(json.dumps({"t1": t1}))
        print(f"phase A: {A} turn 1 ({len(t1)} chars, cached {k1})")
    else:
        t1 = json.loads(Path(a.state).read_text())["t1"]
        m2 = m1 + [{"role": "assistant", "content": t1}, {"role": "user", "content": q2}]
        r_b, k_b, _ = chat(Bm, m2)
        t2, k2, _ = chat(A, m2)
        ref, _, _ = chat(A, m2, ne_cache=False)
        check(k2 > 0, f"phase B: {A} turn 2 restored from the SSD tier after the restart (cached {k2})")
        check(k_b == 0, f"phase B: {Bm} with {A}'s conversation reuses nothing (cached {k_b}: per-model keys)")
        if a.mock:
            check(t2 == ref, f"phase B: restored state = a fresh prefill's ({t2} vs {ref})")
        else:
            print(f"phase B: restored {t2[:100]!r} / cold {ref[:100]!r} (equal: {t2 == ref})")
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
