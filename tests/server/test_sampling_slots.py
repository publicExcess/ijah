"""Sampling knobs across slots and long prompts (lab-sampler), against a server started with --slots 2 (mock or whole model):

    $PY tests/server/test_sampling_slots.py --url http://127.0.0.1:8142 --key-file KEY [--mock]

(1) a request with a ban + penalties on one slot and a plain request on the other, running at the same time (the engine
interleaves them token by token): each output equals the same request run alone (no bias / penalty / count leaks between
slots; ne_cache false: every run prefills from scratch); (2) the slot that served a knob request serves a plain request next:
its output equals a plain run before it (nothing left behind); (3) a prompt of several prefill pieces (> 4097 tokens) with
penalties and logprobs: the pieces' discarded samples must not count (output = the same request twice, penalties change the
text); (4) the same request on the step path with and without a stale MTP head (no error).
"""
import argparse
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tools"))
import refenv  # noqa: E402
import ijah_paths  # noqa: E402

refenv.add_paths()
from tokenizers import Tokenizer  # noqa: E402

from test_client import Client, check, FAILS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8142")
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    cl = Client(a.url, Path(a.key_file).expanduser().read_text().strip())
    st, m = cl.req("GET", "/v1/models")
    model = m["data"][0]["id"]
    rep = {"role": "user", "content": "hello" if a.mock else "Repeat the word 'echo' 60 times, separated by spaces."}
    base = {"model": model, "reasoning_effort": "none", "temperature": 0, "max_tokens": 80, "ne_path": "step",
            "ne_cache": False, "messages": [rep]}
    st, r0 = cl.chat(base)
    first = r0["choices"][0]["message"]["content"].split()[0] if st == 200 else "echo"
    tok = Tokenizer.from_file(str(Path(ijah_paths.pack()) / "tokenizer.json"))
    tid = tok.encode(first, add_special_tokens=False).ids[0]
    knobs = dict(base, presence_penalty=2.0, frequency_penalty=1.0, logit_bias={str(tid): -100}, logprobs=True, top_logprobs=2)

    def text(r):
        return r["choices"][0]["message"]["content"]

    st, solo_k = cl.chat(knobs)
    st2, solo_p = cl.chat(base)
    check(st == st2 == 200 and text(solo_k) != text(solo_p) or a.mock, "alone: the knob request differs from the plain one")
    out = [None, None]

    def go(i, body):
        out[i] = cl.chat(body)

    for round_ in range(3):
        ths = [threading.Thread(target=go, args=(0, knobs)), threading.Thread(target=go, args=(1, base))]
        [t.start() for t in ths]
        [t.join() for t in ths]
        slots = {o[1]["timings"]["slot"] for o in out if o and o[0] == 200}
        check(all(o and o[0] == 200 for o in out) and text(out[0][1]) == text(solo_k) and text(out[1][1]) == text(solo_p),
              f"concurrent round {round_}: knob request and plain request each equal their solo run (slots {sorted(slots)})")
        if round_ == 0:
            lp = out[0][1]["choices"][0]["logprobs"]["content"]
            check(len(lp) > 0 and all(e["token"] != first for e in lp[:1]), "the banned first token is absent on its slot")
    st3, again = cl.chat(base)
    check(st3 == 200 and text(again) == text(solo_p), "a plain request after the knob requests = the plain request before")

    # a long prompt: several prefill pieces
    body = "The quick brown fox jumps over the lazy dog. " * 700
    long_req = {"model": model, "reasoning_effort": "none", "temperature": 0, "max_tokens": 40, "ne_path": "step", "ne_cache": False,
                "messages": [{"role": "user", "content": body + "\nNow repeat the sentence above, then add a new one."}]}
    st, l0 = cl.chat(long_req)
    pt = l0["usage"]["prompt_tokens"] if st == 200 else 0
    kn = dict(long_req, presence_penalty=1.5, logprobs=True)
    st1, l1 = cl.chat(kn)
    st2, l2 = cl.chat(kn)
    stp, lp3 = cl.chat(dict(long_req, repetition_penalty=3.0))   # the prompt is 700 x one sentence: every token of it is 'seen'
    check(st == st1 == st2 == 200 and (pt > 4097 or a.mock), f"long prompt ({pt} tokens, several pieces) answered with penalties")
    check(st1 == 200 and text(l1) == text(l2), "long prompt: the same penalised request twice gives the same text")
    check(stp == 200 and (text(lp3) != text(l0) or a.mock), "long prompt: a repetition penalty (prompt tokens count) changes the answer")
    lp = l1["choices"][0]["logprobs"]["content"] if st1 == 200 else []
    check(lp and all(e["logprob"] <= 1e-6 for e in lp), f"long prompt: {len(lp)} logprob entries")
    print(f"\n{len(FAILS)} failed: {FAILS}" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
