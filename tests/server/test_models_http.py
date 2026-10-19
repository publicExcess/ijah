"""Two models in one server (lab-srvperf), against the server on its mock engine started with
`--model-id A,B` [--share-state] (tests/server/run_mock.sh): /v1/models lists both, requests are routed by `model`
(none -> the first; unknown -> 400), the prefix cache never crosses models and a model's conversation is restored after
the other model ran (with --share-state: on the same state buffers, the mock asserting every row), both models'
requests answered as alone when sent at once.

    $PY tests/server/test_models_http.py --url http://127.0.0.1:8149 --key-file KEY --models A,B
"""
import argparse
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_client import FAILS, Client, check  # noqa: E402

SYS = "Two-model routing test. " * 40   # > 256 tokens: a shared system-prompt checkpoint


def body(model, q, sys_prompt=SYS):
    b = {"temperature": 0, "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": q}]}
    if model is not None:
        b["model"] = model
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--models", required=True, help="the server's --model-id list, comma-separated")
    a = ap.parse_args()
    A, Bm = a.models.split(",")[:2]
    c = Client(a.url, Path(a.key_file).read_text().strip())

    st, r = c.req("GET", "/v1/models")
    ids = [m["id"] for m in r["data"]] if st == 200 else []
    check(ids == [A, Bm], f"/v1/models lists both ids in order ({ids})")

    def ask(model, q, sys_prompt=SYS):
        st, r = c.chat(body(model, q, sys_prompt))
        return st, r

    for m in (A, Bm):
        st, r = ask(m, "WHOAMI")
        check(st == 200 and r["choices"][0]["message"]["content"] == f"I am {m}." and r["model"] == m,
              f"model {m}: routed to its own engine ({st}, {r.get('choices', [{}])[0].get('message', {}).get('content')!r}, "
              f"model {r.get('model')})")
    st, r = ask(None, "WHOAMI")
    check(st == 200 and r["choices"][0]["message"]["content"] == f"I am {A}." and r["model"] == A,
          "no model field -> the first id")
    st, r = ask(A.lower(), "WHOAMI")
    check(st == 200 and r["model"] == A, "model id matched case-insensitively")
    st, r = ask("no-such-model", "WHOAMI")
    check(st == 400 and "unknown model" in r.get("error", {}).get("message", ""), f"unknown model -> 400 ({st})")

    # prefix cache per model: a fresh system prompt, A twice, then B (must not reuse A's), then B again
    sp = "Per-model cache test. " * 100
    _, r1 = ask(A, "first question", sp)
    _, r2 = ask(A, "second question", sp)
    # B: the same prompts (must not reuse A's checkpoints), then its own system prompt (other KV rows: a missed
    # handover under --share-state leaves them in the shared buffers and A's restore below fails the mock's row check)
    _, r3 = ask(Bm, "first question", sp)
    _, r4 = ask(Bm, "second question", sp)
    ask(Bm, "first question", "Other system prompt for model B only. " * 60)
    st5, r5 = ask(A, "third question STATETEST", sp)   # back to A after B ran (--share-state: B had the buffers)
    _, r6 = c.chat(dict(body(A, "third question STATETEST", sp), ne_cache=False))   # the same prompt, fresh prefill
    k = [r["usage"]["prompt_tokens_details"]["cached_tokens"] for r in (r1, r2, r3, r4)]
    check(k[0] == 0 and k[1] > 0 and k[2] == 0 and k[3] > 0,
          f"prefix cache keyed per model: cached tokens A {k[0]} -> {k[1]}, then B {k[2]} -> {k[3]}")
    k5 = r5["usage"]["prompt_tokens_details"]["cached_tokens"] if st5 == 200 else None
    got, want = (r5["choices"][0]["message"]["content"] if st5 == 200 else None), r6["choices"][0]["message"]["content"]
    check(st5 == 200 and k5 and k5 > 0 and got == want,
          f"back to A after B: A's checkpoint restored, state = a fresh prefill's ({st5}, cached {k5}, {got} vs {want})")

    # a conversation continued on A after B used the slot (--share-state: the newest checkpoint of A's resident
    # conversation is reused in place unless the handover parked it; a missed handover reads B's KV rows)
    sp3 = "Multi-turn handover test. " * 80
    m1 = [{"role": "system", "content": sp3}, {"role": "user", "content": "turn one"}]
    _, t1 = c.chat({"model": A, "temperature": 0, "messages": m1})
    ask(Bm, "something else", "Model B's own conversation. " * 70)
    m2 = m1 + [{"role": "assistant", "content": t1["choices"][0]["message"]["content"]},
               {"role": "user", "content": "turn two STATETEST"}]
    st7, t2 = c.chat({"model": A, "temperature": 0, "messages": m2})
    _, t3 = c.chat({"model": A, "temperature": 0, "messages": m2, "ne_cache": False})
    got, want = (t2["choices"][0]["message"]["content"] if st7 == 200 else t2), t3["choices"][0]["message"]["content"]
    check(st7 == 200 and t2["usage"]["prompt_tokens_details"]["cached_tokens"] > 0 and got == want,
          f"A's next turn after B ran: reused and state = a fresh prefill's ({got} vs {want})")

    # both models at once (one slot each): both answer their own, alone-identical replies
    ref = {m: ask(m, "LONGTEST")[1]["choices"][0]["message"]["content"] for m in (A, Bm)}
    res = {}

    def run(m):
        res[m] = ask(m, "LONGTEST")
    ts = [threading.Thread(target=run, args=(m,)) for m in (A, Bm)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    ok = all(res[m][0] == 200 and res[m][1]["choices"][0]["message"]["content"] == ref[m] for m in (A, Bm))
    paths = [res[m][1]["timings"]["path"] for m in (A, Bm)]
    check(ok, f"concurrent requests to both models: replies = alone ({paths})")
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
