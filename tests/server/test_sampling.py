"""HTTP test of the sampling knobs and the extra endpoints of the Ijah server (lab-sampler), stdlib + tokenizers.
Against a server that is up (mock or the whole model):

    $PY tests/server/test_sampling.py --url http://127.0.0.1:8141 --key-file KEY [--mock] [--json out.json]

n > 1 (sequential choices, streamed and not), logprobs / top_logprobs (chat and legacy), logit_bias (bans), the path a
request takes with each knob (timings.path / timings.sampling: knobs and seeds step, plain requests keep MTP), seeds,
validation errors, /v1/completions (prompt string / ids / list, stop, stream, logprobs) and the Anthropic /v1/messages
(text, system, tools, thinking, stop_sequences, max_tokens, streaming events, count_tokens, x-api-key, error body).
Keys are read from the file and never printed.
"""
import argparse
import http.client
import json
import math
import sys
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "server"))
sys.path.insert(0, str(HERE.parents[1] / "tools"))
import refenv  # noqa: E402
import ijah_paths  # noqa: E402

refenv.add_paths()
from tokenizers import Tokenizer  # noqa: E402

from test_client import Client, TOOLS, check, merge, FAILS  # noqa: E402

PY_PACK = ijah_paths.pack()


class Raw:
    """a client that can set headers and read named SSE events"""

    def __init__(self, url, key):
        u = urlparse(url)
        self.host, self.port, self.key = u.hostname, u.port, key

    def post(self, path, body, headers=None, stream=False):
        c = http.client.HTTPConnection(self.host, self.port, timeout=900)
        h = {"Content-Type": "application/json"}
        h.update(headers if headers is not None else {"Authorization": "Bearer " + self.key})
        c.request("POST", path, json.dumps(body), h)
        r = c.getresponse()
        if not stream or r.status != 200:
            data = r.read()
            c.close()
            try:
                return r.status, json.loads(data)
            except json.JSONDecodeError:
                return r.status, data
        ev, cur = [], None
        while True:
            line = r.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith(b"event: "):
                cur = line[7:].decode()
            elif line.startswith(b"data: "):
                d = line[6:]
                if d == b"[DONE]":
                    ev.append(("[DONE]", None))
                    break
                ev.append((cur, json.loads(d)))
                cur = None
        c.close()
        return 200, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8141")
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--models", default="", help="id1,id2: run everything once per model id of a multi-model server, "
                    "then the cross-model leak checks (default: the first model)")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    mock = a.mock
    key = Path(a.key_file).expanduser().read_text().strip()
    cl = Client(a.url, key)
    raw = Raw(a.url, key)
    tok = Tokenizer.from_file(str(Path(PY_PACK) / "tokenizer.json"))
    res = {}
    st, m = cl.req("GET", "/v1/models")
    ids = [x for x in a.models.split(",") if x] or [m["data"][0]["id"]]

    def run_model(model):
        Q = "hello" if mock else "Write one short sentence about the sea."
        base = {"model": model, "reasoning_effort": "none", "messages": [{"role": "user", "content": Q}], "max_tokens": 48}

        # ------------------------------------------------------------------ validation
        bad = [({"logit_bias": {"x": 1}}, "non-integer token id"), ({"logit_bias": {"5": 101}}, "bias above 100"),
               ({"logit_bias": {"5": -101}}, "bias below -100"), ({"logit_bias": {"99999999": 1}}, "token id outside the vocab"),
               ({"logit_bias": [1]}, "logit_bias not an object"), ({"presence_penalty": 3}, "presence_penalty 3"),
               ({"frequency_penalty": -2.5}, "frequency_penalty -2.5"), ({"repetition_penalty": 0}, "repetition_penalty 0"),
               ({"min_p": 2}, "min_p 2"), ({"n": 0}, "n 0"), ({"n": 65}, "n 65"), ({"top_logprobs": 3}, "top_logprobs without logprobs"),
               ({"logprobs": True, "top_logprobs": 21}, "top_logprobs 21"), ({"logprobs": "yes"}, "logprobs not a bool")]
        for extra, what in bad:
            s, r = cl.chat(dict(base, **extra))
            check(s == 400 and "error" in r, f"400 on {what} ({s})")

        # ------------------------------------------------------------------ n > 1
        s, r = cl.chat(dict(base, n=3))
        ch = r["choices"] if s == 200 else []
        u = r.get("usage", {})
        check(s == 200 and [c["index"] for c in ch] == [0, 1, 2] and all(c["message"]["content"] and c["finish_reason"] == "stop" for c in ch),
              f"n = 3: three choices {[c['message']['content'][:24] for c in ch]}")
        one = cl.chat(base)[1]["usage"]
        check(u.get("prompt_tokens") == one["prompt_tokens"] and u.get("completion_tokens", 0) >= 3 * 2 and
              u.get("total_tokens") == u["prompt_tokens"] + u["completion_tokens"],
              f"n = 3 usage: the prompt once ({u.get('prompt_tokens')}), completions summed ({u.get('completion_tokens')})")
        if mock:
            check(len({c["message"]["content"] for c in ch}) == 1, "mock: the three choices are the scripted answer")
        else:
            check(len({c["message"]["content"] for c in ch}) >= 2, "n = 3 at temperature 1: the choices differ")
        # (ne_cache false: a prefix hit prefills in other pieces, which can flip a late token; the draws themselves repeat)
        s, ra = cl.chat(dict(base, n=2, seed=11, temperature=1.0, ne_cache=False))
        s2, rb = cl.chat(dict(base, n=2, seed=11, temperature=1.0, ne_cache=False))
        ta = [c["message"]["content"] for c in ra["choices"]]
        tb = [c["message"]["content"] for c in rb["choices"]]
        check(s == s2 == 200 and ta == tb, f"n = 2 with a seed reproduces both choices ({ta[0][:30]!r}, {ta[1][:30]!r})")
        if not mock:
            check(ta[0] != ta[1], "seed + i: the two choices of one request differ")
        chunks, done, _, _ = cl.stream(dict(base, n=2, stream_options={"include_usage": True}))
        idxs = [c["index"] for ch_ in chunks for c in ch_.get("choices", [])]
        order_ok = idxs == sorted(idxs) and set(idxs) == {0, 1}
        fins = [c["index"] for ch_ in chunks for c in ch_.get("choices", []) if c.get("finish_reason")]
        check(done and order_ok and fins == [0, 1], f"n = 2 streamed: choice 0 then 1, one finish each ({fins})")
        usage = [ch_["usage"] for ch_ in chunks if ch_.get("usage")]
        check(len(usage) == 1 and usage[0]["completion_tokens"] > 0, f"n = 2 streamed usage {usage[:1]}")

        # ------------------------------------------------------------------ logprobs
        s, r = cl.chat(dict(base, logprobs=True, top_logprobs=4, temperature=0))
        c0 = r["choices"][0] if s == 200 else {}
        lp = (c0.get("logprobs") or {}).get("content") or []
        text = c0.get("message", {}).get("content", "")
        n_tok = r.get("usage", {}).get("completion_tokens", -1)
        check(s == 200 and lp and len(lp) in (n_tok, n_tok - 1) or (s == 200 and lp and abs(len(lp) - n_tok) <= 1),
              f"logprobs: {len(lp)} entries for {n_tok} completion tokens")
        check(all(set(e) >= {"token", "logprob", "bytes", "top_logprobs"} and e["logprob"] <= 1e-6 and len(e["top_logprobs"]) <= 4
                  and all(t["logprob"] <= 1e-6 for t in e["top_logprobs"]) for e in lp), "logprob entries well-formed")
        check("".join(bytes(e["bytes"]).decode("utf-8", "replace") for e in lp).strip() == text.strip(),
              f"the entries' bytes spell the text ({text.strip()[:40]!r})")
        if not mock:
            check(all(e["top_logprobs"] and e["top_logprobs"][0]["token"] == e["token"] and
                      abs(e["top_logprobs"][0]["logprob"] - e["logprob"]) < 1e-6 for e in lp),
                  "greedy: the sampled token is the top candidate with the same logprob")
            check(all(sum(math.exp(t["logprob"]) for t in e["top_logprobs"]) <= 1.0001 for e in lp),
                  "top-4 probabilities sum to at most 1")
            check(all([t["logprob"] for t in e["top_logprobs"]] == sorted([t["logprob"] for t in e["top_logprobs"]], reverse=True)
                      for e in lp), "top candidates sorted by probability")
        check(cl.chat(dict(base, logprobs=True))[1]["choices"][0]["logprobs"]["content"][0]["top_logprobs"] == [],
              "logprobs without top_logprobs: no alternatives")
        chunks, done, _, _ = cl.stream(dict(base, logprobs=True, top_logprobs=2, temperature=0))
        slp = [e for ch_ in chunks for c in ch_.get("choices", []) for e in (c.get("logprobs") or {}).get("content", [])]
        scontent = merge(chunks)[0]
        check(done and len(slp) == len(lp) and "".join(bytes(e["bytes"]).decode("utf-8", "replace") for e in slp).strip() == scontent.strip(),
              f"streamed logprobs: {len(slp)} entries, bytes spell the streamed text")
        if not mock:
            check([e["token"] for e in slp] == [e["token"] for e in lp] and
                  all(abs(x["logprob"] - y["logprob"]) < 1e-3 for x, y in zip(slp, lp)),
                  "greedy streamed = non-streamed tokens and logprobs")
        tk = r["timings"]
        check(tk["path"] == "step" and "logprobs" in tk.get("sampling", []), f"logprobs take the step path ({tk['path']}, {tk.get('sampling')})")

        # ------------------------------------------------------------------ logit_bias: bans
        s, r = cl.chat(dict(base, temperature=0, logprobs=True, top_logprobs=1))
        first_tok = r["choices"][0]["logprobs"]["content"][0]["token"]
        ids0 = tok.encode(first_tok, add_special_tokens=False).ids
        if len(ids0) == 1:
            tid = ids0[0]
            s, rb2 = cl.chat(dict(base, temperature=0, logit_bias={str(tid): -100}, logprobs=True, top_logprobs=1))
            c2 = rb2["choices"][0]
            e0 = c2["logprobs"]["content"][0]
            banned_txt = "".join(e["token"] for e in c2["logprobs"]["content"])
            check(s == 200 and e0["token"] != first_tok and rb2["timings"]["path"] == "step" and
                  "logit_bias" in rb2["timings"]["sampling"], f"logit_bias -100 bans {first_tok!r}: first token now {e0['token']!r}")
            check(all(e["token"] != first_tok for e in c2["logprobs"]["content"]) or first_tok.strip() == "",
                  "the banned token never appears")
            chunks, done, _, _ = cl.stream(dict(base, temperature=0, logit_bias={str(tid): -100}, logprobs=True))
            sl = [e["token"] for ch_ in chunks for c in ch_.get("choices", []) for e in (c.get("logprobs") or {}).get("content", [])]
            check(done and first_tok not in sl[:1], "ban, streamed")
            if not mock:
                # +100 on a token: exclusive selection
                s, rx = cl.chat(dict(base, temperature=0, logit_bias={str(tid): 100}, max_tokens=6, logprobs=True))
                toks = [e["token"] for e in rx["choices"][0]["logprobs"]["content"]]
                check(toks and all(t == first_tok for t in toks[:5]), f"logit_bias +100 selects {first_tok!r} every time: {toks[:6]}")
            res["ban"] = {"token": first_tok, "id": tid, "next": e0["token"]}
        else:
            check(False, f"first token {first_tok!r} is not one id")

        # ------------------------------------------------------------------ the path of each knob
        # (ne_path "mtp" asks for MTP wherever the request can have it: the auto policy follows measured rates)
        plain = cl.chat(dict(base, max_tokens=64, ne_path="mtp"))[1]["timings"]
        check(plain["path"] in ("mtp", "mtp+step") and "sampling" not in plain, f"plain request keeps MTP ({plain['path']})")
        for knob, extra in (("presence_penalty", {"presence_penalty": 0.5}), ("frequency_penalty", {"frequency_penalty": 0.5}),
                            ("repetition_penalty", {"repetition_penalty": 1.1}), ("min_p", {"min_p": 0.05}),
                            ("logit_bias", {"logit_bias": {"5": -5}})):
            s, r = cl.chat(dict(base, max_tokens=64, ne_path="mtp", **extra))   # bias, penalties and min_p run on MTP too
            tk = r["timings"] if s == 200 else {}
            check(s == 200 and tk.get("path") in ("mtp", "mtp+step") and tk.get("sampling") == [knob],
                  f"{knob}: {tk.get('path')}, timings.sampling {tk.get('sampling')}")
        s, r = cl.chat(dict(base, max_tokens=64, ne_path="mtp", logprobs=True))   # logprobs: the MTP loop has none
        tk = r["timings"] if s == 200 else {}
        check(s == 200 and tk.get("path") == "step" and tk.get("sampling") == ["logprobs"], f"logprobs beat a forced MTP ({tk.get('path')})")
        s, r = cl.chat(dict(base, max_tokens=64, ne_path="mtp", seed=3))
        tk = r["timings"] if s == 200 else {}
        check(s == 200 and tk.get("path") in ("mtp", "mtp+step") and tk.get("sampling") == ["seed"], f"a seed with ne_path mtp: MTP ({tk.get('path')})")
        s, r = cl.chat(dict(base, max_tokens=64, seed=3))
        tk = r["timings"] if s == 200 else {}
        check(s == 200 and tk.get("path") == "step" and tk.get("sampling") == ["seed"],
              f"a seed alone steps (reproducible): {tk.get('path')}, {tk.get('sampling')}")
        after = cl.chat(dict(base, max_tokens=64, ne_path="mtp"))[1]["timings"]
        check(after["path"] in ("mtp", "mtp+step") and "sampling" not in after, f"after the knobs: a plain request is plain again ({after['path']})")

        # ------------------------------------------------------------------ seeds
        sb = dict(base, temperature=1.0, max_tokens=40, seed=1234, ne_cache=False)
        t1 = cl.chat(sb)[1]["choices"][0]["message"]["content"]
        t2 = cl.chat(sb)[1]["choices"][0]["message"]["content"]
        check(t1 == t2, f"same seed, same text ({t1[:40]!r})")
        if not mock:
            t3 = cl.chat(dict(sb, seed=1235))[1]["choices"][0]["message"]["content"]
            check(t3 != t1, "another seed, another text")
            # the same request with the prefix reused (another prefill piece layout): the draws are the same, the logits
            # differ by rounding, so a late token may differ
            t4 = cl.chat(dict(sb, ne_cache=True))[1]["choices"][0]["message"]["content"]
            same = next((i for i, (x, y) in enumerate(zip(t1, t4)) if x != y), min(len(t1), len(t4)))
            check(same >= min(len(t1), len(t4)) * 0.5, f"seed, prefix reused vs fresh: first {same} of {len(t1)} characters identical")
            # penalties: presence penalty reduces repeats on a repetitive request
            rq = {"model": model, "reasoning_effort": "none", "temperature": 0, "max_tokens": 120,
                  "messages": [{"role": "user", "content": "Repeat the word 'echo' 60 times, separated by spaces."}]}
            r0 = cl.chat(rq)[1]["choices"][0]["message"]["content"]
            r1 = cl.chat(dict(rq, presence_penalty=2.0, frequency_penalty=1.0))[1]["choices"][0]["message"]["content"]
            check(r0.lower().count("echo") > r1.lower().count("echo") or r0 != r1,
                  f"penalties change a repetitive greedy answer ({r0.lower().count('echo')} echoes -> {r1.lower().count('echo')})")
            r15 = cl.chat(dict(rq, repetition_penalty=1.5))[1]["choices"][0]["message"]["content"]   # (a confident 'echo' survives /1.5)
            rr = cl.chat(dict(rq, repetition_penalty=10.0))[1]["choices"][0]["message"]["content"]
            check(rr != r0, f"repetition_penalty 10 changes a repetitive greedy answer (1.5: {'same' if r15 == r0 else 'changed'}; "
                            f"{r0.lower().count('echo')} echoes -> {rr.lower().count('echo')})")
            after = cl.chat(dict(rq))[1]["choices"][0]["message"]["content"]
            check(after[:60] == r0[:60], "no leak: the plain greedy answer after penalty requests starts as the one before")
            # min_p: a high min_p at temperature 1 narrows to the top token = greedy
            mp = cl.chat(dict(rq, temperature=1.0, min_p=1.0, max_tokens=40))[1]["choices"][0]["message"]["content"]
            check(mp == r0[:len(mp)] or mp[:30] == r0[:30], "min_p = 1 at temperature 1 is greedy")

        # ------------------------------------------------------------------ /v1/completions
        cb = {"model": model, "temperature": 0, "max_tokens": 12}
        P = "hello" if mock else "The capital of France is"
        s, r = raw.post("/v1/completions", dict(cb, prompt=P))
        txt = r["choices"][0]["text"] if s == 200 else ""
        check(s == 200 and r["object"] == "text_completion" and txt and r["choices"][0]["finish_reason"] in ("stop", "length")
              and r["usage"]["prompt_tokens"] > 0 and "<|im_start|>" not in txt,
              f"/v1/completions: {txt!r} ({r['choices'][0]['finish_reason'] if s == 200 else r})")
        check(mock or "Paris" in txt, "completion continues the raw prompt")
        ids = tok.encode(P, add_special_tokens=False).ids
        s, r2 = raw.post("/v1/completions", dict(cb, prompt=ids))
        check(s == 200 and r2["choices"][0]["text"] == txt, "prompt as token ids = the same completion")
        s, r3 = raw.post("/v1/completions", dict(cb, prompt=[P, P + " "], n=2))
        check(s == 200 and [c["index"] for c in r3["choices"]] == [0, 1, 2, 3] and r3["usage"]["prompt_tokens"] >= 2,
              f"two prompts x n = 2: indexes {[c['index'] for c in r3['choices']]}")
        s, r4 = raw.post("/v1/completions", dict(cb, prompt=P, logprobs=3))
        lg = r4["choices"][0]["logprobs"] if s == 200 else {}
        check(s == 200 and lg and len(lg["tokens"]) == len(lg["token_logprobs"]) == len(lg["top_logprobs"]) == len(lg["text_offset"])
              and all(len(t) <= 3 for t in lg["top_logprobs"]) and "".join(lg["tokens"]).strip() == r4["choices"][0]["text"].strip(),
              f"completions logprobs: {len(lg.get('tokens', []))} tokens, 3 alternatives")
        s, r5 = raw.post("/v1/completions", dict(cb, prompt=P))
        check(s == 200 and r5["choices"][0]["logprobs"] is None, "completions without logprobs: null")
        if mock:
            s, r6 = raw.post("/v1/completions", dict(cb, prompt="STOPTEST", stop=["STOP_HERE"], max_tokens=64))
            check(s == 200 and "STOP" not in r6["choices"][0]["text"] and "four" not in r6["choices"][0]["text"],
                  f"completions stop string: {r6['choices'][0]['text']!r}")
        s, ev = raw.post("/v1/completions", dict(cb, prompt=P, stream=True, stream_options={"include_usage": True}), stream=True)
        sc = "".join(e[1]["choices"][0]["text"] for e in ev if e[1] and e[1].get("choices"))
        fin = [e[1]["choices"][0]["finish_reason"] for e in ev if e[1] and e[1].get("choices") and e[1]["choices"][0]["finish_reason"]]
        check(s == 200 and ev[-1][0] == "[DONE]" and sc == txt and len(fin) == 1, f"completions streamed = non-streamed ({sc!r})")
        for bad_body in (dict(cb, prompt=P, echo=True), dict(cb, prompt=P, best_of=2), dict(cb), dict(cb, prompt=[]),
                         dict(cb, prompt=P, response_format={"type": "json_object"})):
            s, r = raw.post("/v1/completions", bad_body)
            check(s == 400, f"/v1/completions 400 on {sorted(set(bad_body) - set(cb))} ({s})")
        s, r = raw.post("/v1/completions", dict(cb, prompt=P, logit_bias={str(ids[0]): -100}, logprobs=1, max_tokens=4))
        check(s == 200 and r["timings"]["path"] == "step", "completions take the same knobs")

        # ------------------------------------------------------------------ Anthropic /v1/messages
        H = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        ab = {"model": model, "max_tokens": 64, "messages": [{"role": "user", "content": "hello" if mock else "Say hello in one word."}]}
        s, r = raw.post("/v1/messages", ab, H)
        blocks = r.get("content", []) if s == 200 else []
        check(s == 200 and r["type"] == "message" and r["role"] == "assistant" and r["id"].startswith("msg_") and
              blocks and blocks[0]["type"] == "text" and blocks[0]["text"].strip() and r["stop_reason"] == "end_turn" and
              r["usage"]["output_tokens"] > 0 and r["usage"]["input_tokens"] > 0, f"/v1/messages: {blocks[:1]} {r.get('stop_reason')}")
        check(not any(b["type"] == "thinking" for b in blocks), "no thinking block unless requested")
        s, r = raw.post("/v1/messages", ab, {"Authorization": "Bearer " + key})
        check(s == 200, "/v1/messages also takes Authorization: Bearer")
        s, r = raw.post("/v1/messages", ab, {"x-api-key": "wrong-" + "x" * 30})
        check(s == 401 and r.get("type") == "error" and r["error"]["type"] == "authentication_error", f"wrong x-api-key -> 401 in Anthropic's error body ({s})")
        s, r = raw.post("/v1/messages", dict(ab, max_tokens=None), H)
        check(s == 400 and r.get("type") == "error" and r["error"]["type"] == "invalid_request_error", f"missing max_tokens -> 400 ({s})")
        s, r = raw.post("/v1/messages", dict(ab, messages=[]), H)
        check(s == 400, "empty messages -> 400")
        s, r = raw.post("/v1/messages", dict(ab, system="You are terse.", max_tokens=1), H)
        check(s == 200 and r["stop_reason"] == "max_tokens", f"max_tokens -> stop_reason max_tokens ({r.get('stop_reason')})")
        s, r = raw.post("/v1/messages", dict(ab, thinking={"type": "enabled", "budget_tokens": 2048}, max_tokens=4000), H)
        bl = r.get("content", []) if s == 200 else []
        check(s == 200 and bl and bl[0]["type"] == "thinking" and bl[0]["thinking"].strip() and any(b["type"] == "text" for b in bl),
              f"thinking enabled -> a thinking block then text ({[b['type'] for b in bl]})")
        s, r = raw.post("/v1/messages", dict(ab, content=None, messages=[
            {"role": "user", "content": [{"type": "text", "text": "STOPTEST" if mock else "Count from 1 to 15, separated by spaces."}]}],
            stop_sequences=["STOP_HERE"] if mock else [" 7", "7"], max_tokens=64), H)
        check(s == 200 and r["stop_reason"] == "stop_sequence" and r["stop_sequence"] in ("STOP_HERE", " 7", "7"),
              f"stop_sequences -> stop_reason stop_sequence ({r.get('stop_reason')}, {r.get('stop_sequence')!r})")
        tool = {"name": "get_weather", "description": "Current weather for a city.",
                "input_schema": TOOLS[0]["function"]["parameters"]}
        tq = "TOOLTEST" if mock else "What's the weather in Paris for the next 3 days? Use the tool."
        s, r = raw.post("/v1/messages", {"model": model, "max_tokens": 600, "tools": [tool], "tool_choice": {"type": "auto"},
                                         "messages": [{"role": "user", "content": tq}]}, H)
        bl = r.get("content", []) if s == 200 else []
        tu = [b for b in bl if b["type"] == "tool_use"]
        check(s == 200 and r["stop_reason"] == "tool_use" and tu and tu[0]["name"] == "get_weather" and tu[0]["id"] and
              isinstance(tu[0]["input"], dict) and "paris" in str(tu[0]["input"].get("city", "")).lower(),
              f"tool use: {tu[:1]}")
        if tu:
            msgs = [{"role": "user", "content": tq}, {"role": "assistant", "content": bl},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu[0]["id"],
                                                  "content": "21 C and sunny"}]}]
            s, r2 = raw.post("/v1/messages", {"model": model, "max_tokens": 300, "tools": [tool], "messages": msgs}, H)
            t2 = "".join(b.get("text", "") for b in r2.get("content", [])) if s == 200 else ""
            check(s == 200 and r2["stop_reason"] == "end_turn" and t2.strip() and (mock or "21" in t2),
                  f"tool_result back -> answer {t2.strip()[:60]!r}; cached {r2.get('usage', {}).get('cache_read_input_tokens')}")
        s, r = raw.post("/v1/messages", {"model": model, "max_tokens": 600, "tools": [tool], "tool_choice": {"type": "tool", "name": "get_weather"},
                                         "messages": [{"role": "user", "content": "FORCETEST hello" if mock else "Hi there."}]}, H)
        tu = [b for b in r.get("content", []) if b["type"] == "tool_use"] if s == 200 else []
        check(s == 200 and len(tu) == 1 and tu[0]["name"] == "get_weather", f"tool_choice tool: forced call ({[b['type'] for b in r.get('content', [])]})")
        # streaming events
        s, ev = raw.post("/v1/messages", dict(ab, stream=True, thinking={"type": "enabled", "budget_tokens": 2048}, max_tokens=4000), H, stream=True)
        names = [e[0] for e in ev]
        check(s == 200 and names[0] == "message_start" and names[-2:] == ["message_delta", "message_stop"] and "ping" in names[:3],
              f"stream events: {names[:4]} ... {names[-3:]}")
        starts = [e[1] for e in ev if e[0] == "content_block_start"]
        stops = [e[1]["index"] for e in ev if e[0] == "content_block_stop"]
        check([b["index"] for b in starts] == list(range(len(starts))) and stops == list(range(len(starts))) and
              [b["content_block"]["type"] for b in starts][:1] == ["thinking"] and "text" in [b["content_block"]["type"] for b in starts],
              f"stream blocks open/close in order: {[b['content_block']['type'] for b in starts]}")
        sth = "".join(e[1]["delta"]["thinking"] for e in ev if e[0] == "content_block_delta" and e[1]["delta"]["type"] == "thinking_delta")
        stx = "".join(e[1]["delta"]["text"] for e in ev if e[0] == "content_block_delta" and e[1]["delta"]["type"] == "text_delta")
        md = next(e[1] for e in ev if e[0] == "message_delta")
        check(sth.strip() and stx.strip() and md["delta"]["stop_reason"] == "end_turn" and md["usage"]["output_tokens"] > 0,
              f"stream text {stx.strip()[:30]!r}, thinking {len(sth)} chars, {md['usage']}")
        s, ev = raw.post("/v1/messages", {"model": model, "max_tokens": 600, "tools": [tool], "stream": True,
                                          "messages": [{"role": "user", "content": tq}]}, H, stream=True)
        jj = "".join(e[1]["delta"]["partial_json"] for e in ev if e[0] == "content_block_delta" and e[1]["delta"]["type"] == "input_json_delta")
        ts = [e[1]["content_block"] for e in ev if e[0] == "content_block_start" and e[1]["content_block"]["type"] == "tool_use"]
        try:
            inp = json.loads(jj)
        except json.JSONDecodeError:
            inp = None
        md = next((e[1] for e in ev if e[0] == "message_delta"), {"delta": {}})
        check(ts and inp and "paris" in str(inp.get("city", "")).lower() and md["delta"].get("stop_reason") == "tool_use",
              f"streamed tool_use: input_json {jj} -> stop_reason {md['delta'].get('stop_reason')}")
        s, r = raw.post("/v1/messages/count_tokens", {"model": model, "messages": ab["messages"], "system": "x"}, H)
        check(s == 200 and r["input_tokens"] > 3, f"count_tokens {r}")
        s, r = raw.post("/v1/messages", dict(ab, temperature=0.7, top_k=10, top_p=0.9, seed=5), H)
        check(s == 200, "temperature / top_k / top_p / seed pass through")
        s, r = raw.post("/v1/messages", dict(ab, tools=[{"type": "bash_20250124", "name": "bash"}]), H)
        check(s == 400, "a server tool type is refused")


    for model in ids:
        print(f"=== model {model}", flush=True)
        run_model(model)
    if len(ids) > 1:   # knobs are per model instance: a request on one model leaves nothing on the other
        qb = {"reasoning_effort": "none", "temperature": 0, "max_tokens": 48, "ne_cache": False, "ne_path": "step",   # (one path: MTP vs step greedy can differ at a tie)
              "messages": [{"role": "user", "content": "hello" if mock else "Write one short sentence about the sea."}]}
        ref = {i: cl.chat(dict(qb, model=i))[1]["choices"][0]["message"]["content"] for i in ids}
        first = cl.chat(dict(qb, model=ids[0], logprobs=True))[1]["choices"][0]["logprobs"]["content"][0]["token"]
        tid = tok.encode(first, add_special_tokens=False).ids[0]
        for i, j in ((ids[0], ids[1]), (ids[1], ids[0])):
            knob = {"presence_penalty": 2.0, "logit_bias": {str(tid): -100}, "logprobs": True, "top_logprobs": 2}
            st1, r1 = cl.chat(dict(qb, model=i, **knob))
            check(st1 == 200 and r1["model"] == i and r1["choices"][0]["logprobs"]["content"], f"knob request on {i}: answered, logprobs present")
            st2, r2 = cl.chat(dict(qb, model=j))
            check(st2 == 200 and r2["model"] == j and r2["choices"][0]["message"]["content"] == ref[j],
                  f"plain request on {j} right after a knob request on {i} = its own earlier answer")
            st3, r3 = cl.chat(dict(qb, model=i))
            check(st3 == 200 and r3["choices"][0]["message"]["content"] == ref[i], f"... and {i} itself is plain again")
        outm = [None, None]

        def go(k, body):
            outm[k] = cl.chat(body)
        import threading
        ths = [threading.Thread(target=go, args=(0, dict(qb, model=ids[0], presence_penalty=2.0, logit_bias={str(tid): -100}))),
               threading.Thread(target=go, args=(1, dict(qb, model=ids[1])))]
        [t.start() for t in ths]
        [t.join() for t in ths]
        check(all(o and o[0] == 200 for o in outm) and outm[1][1]["choices"][0]["message"]["content"] == ref[ids[1]],
              "concurrent: a knob request on one model next to a plain one on the other: the plain one is untouched")
        s2, ev = raw.post("/v1/messages", {"model": ids[1], "max_tokens": 32, "messages": [{"role": "user", "content": "hello"}]},
                          {"x-api-key": key})
        check(s2 == 200 and ev["model"] == ids[1], f"/v1/messages routes by model ({ev.get('model')})")
        s3, cv = raw.post("/v1/completions", {"model": ids[1], "prompt": "hello", "max_tokens": 8, "logprobs": 1})
        check(s3 == 200 and cv["model"] == ids[1] and cv["choices"][0]["logprobs"], f"/v1/completions routes by model, logprobs ({cv.get('model')})")

    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    print(f"\n{len(FAILS)} failed: {FAILS}" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
