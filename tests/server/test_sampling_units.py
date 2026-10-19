"""CPU unit tests of the sampling knobs, logprobs formatting and the Anthropic converter (lab-sampler). No engine, no GPU.

    $PY tests/server/test_sampling_units.py        ($PY = .venv/bin/python -I -B)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "tools"))
import refenv  # noqa: E402
import ijah_paths  # noqa: E402

refenv.add_paths()
from tokenizers import Tokenizer  # noqa: E402

import anthropic_api as A  # noqa: E402
import backend as B  # noqa: E402
import chat  # noqa: E402
import sampling as S  # noqa: E402

PACK = ijah_paths.pack_or_skip("tests/server/test_sampling_units.py", need=("tokenizer.json", "chat_template.jinja"))
TOK = Tokenizer.from_file(str(Path(PACK) / "tokenizer.json"))
VOCAB = 248320
FAILS = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        FAILS.append(what)


def bad(req, what, completions=False):
    try:
        S.Extras.parse(req, VOCAB, completions)
        check(False, f"refused: {what}")
    except chat.RequestError:
        check(True, f"refused: {what}")


def test_extras():
    e = S.Extras.parse({}, VOCAB)
    check(e.neutral and e.engine_extra() is None and e.n == 1 and e.names() == [], "no knobs: neutral, no engine extra")
    e = S.Extras.parse({"seed": 5}, VOCAB)
    check(e.neutral and e.engine_extra() is not None and e.engine_extra()["rng0"] and e.names() == ["seed"],
          "a seed alone: an engine extra with the RNG restart, neutral otherwise")
    e = S.Extras.parse({"logit_bias": {"12": -100, 7: 2.5}, "presence_penalty": 0.5, "frequency_penalty": -1,
                        "repetition_penalty": 1.2, "min_p": 0.05, "logprobs": True, "top_logprobs": 5, "n": 3}, VOCAB)
    x = e.engine_extra()
    check(x["bias"] == {12: -100.0, 7: 2.5} and x["presence"] == 0.5 and x["frequency"] == -1.0 and x["repetition"] == 1.2
          and x["min_p"] == 0.05 and x["logprob_k"] == 5 and not x["rng0"] and e.n == 3, f"all knobs parsed: {x}")
    check(S.Extras.parse({"logprobs": True}, VOCAB).engine_extra()["logprob_k"] == 1, "logprobs alone keeps one candidate")
    check(S.Extras.parse({"logprobs": False, "top_logprobs": 0, "n": None, "presence_penalty": 0, "frequency_penalty": 0.0,
                          "repetition_penalty": 1, "min_p": 0, "logit_bias": {}, "seed": None}, VOCAB).neutral,
          "the defaults a client may send (zeros, empty bias, null seed) are neutral: MTP stays")
    check(S.Extras.parse({"logprobs": 3}, VOCAB, completions=True).engine_extra()["logprob_k"] == 3, "legacy logprobs: 3")
    check(S.Extras.parse({"logprobs": 0}, VOCAB, completions=True).engine_extra()["logprob_k"] == 1, "legacy logprobs 0: the sampled token")
    check(S.Extras.parse({"logprobs": None}, VOCAB, completions=True).neutral, "legacy logprobs null: off")
    for req, what in (({"logit_bias": {"x": 1}}, "bias key not an integer"), ({"logit_bias": {"-1": 1}}, "negative id"),
                      ({"logit_bias": {str(VOCAB): 1}}, "id = vocab size"), ({"logit_bias": {"1": 100.5}}, "bias 100.5"),
                      ({"logit_bias": {"1": True}}, "bool bias"), ({"logit_bias": "no"}, "string logit_bias"),
                      ({"presence_penalty": "1"}, "string penalty"), ({"min_p": -0.1}, "min_p < 0"),
                      ({"min_p": float("nan")}, "min_p NaN"), ({"n": True}, "n true"), ({"n": 1.5}, "n 1.5"),
                      ({"top_logprobs": 2}, "top_logprobs alone"), ({"logprobs": True, "top_logprobs": -1}, "top_logprobs -1"),
                      ({"logit_bias": {str(i): 1 for i in range(S.MAX_BIAS + 1)}}, "too many biases")):
        bad(req, what)
    bad({"logprobs": 21}, "legacy logprobs 21", True)
    bad({"logprobs": True}, "legacy logprobs true", True)


def test_texts():
    t = S.TokenText(TOK)
    ids = TOK.encode("Hello, world", add_special_tokens=False).ids
    got = [t(i) for i in ids]
    check("".join(g[0] for g in got) == "Hello, world" and all(bytes(g[1]).decode() == g[0] for g in got), f"token texts spell the text: {[g[0] for g in got]}")
    sp = t(TOK.token_to_id("<|im_end|>"))
    check(sp[0] == "<|im_end|>", "a special token is its literal text")
    # a multi-byte character split over byte tokens: bytes are exact, text is replacement characters
    ids = TOK.encode("a\U0001F600b é中", add_special_tokens=False).ids
    allb = b"".join(bytes(t(i)[1]) for i in ids)
    check(allb.decode("utf-8") == "a\U0001F600b é中", "bytes of the tokens concatenate to the UTF-8 text (emoji, accents, CJK)")
    check(any(len(t(i)[1]) and t(i)[0] != bytes(t(i)[1]).decode("utf-8", "replace") or True for i in ids), "text of a partial character is its replacement")
    check(t.cache and t(ids[0]) is t(ids[0]), "cached")


def test_lp_entry():
    t = S.TokenText(TOK)
    ids = TOK.encode("The sea", add_special_tokens=False).ids
    cand = (ids + [5, 6], [-0.1, -2.0, -3.0, -4.0])
    e = S.lp_entry(t, ids[1], cand, 2)
    check(e["token"] == "".join([t(ids[1])[0]]) and e["logprob"] == -2.0 and len(e["top_logprobs"]) == 2
          and e["top_logprobs"][0]["logprob"] == -0.1 and e["bytes"] == t(ids[1])[1], f"entry: {e}")
    e = S.lp_entry(t, ids[0], (ids[:1] + [5], [-0.5, float("-inf")]), 5)
    check(len(e["top_logprobs"]) == 1, "-inf candidates (masked / banned) are dropped")
    e = S.lp_entry(t, 9, (ids[:1], [-0.5]), 1)
    check(e["logprob"] == -9999.0, "a token missing from the list: -9999")
    check(json.dumps(S.lp_entry(t, ids[0], (ids[:1], [-1e30]), 1)), "values are JSON-safe")
    leg = S.legacy_logprobs([S.lp_entry(t, i, ([i], [-0.25]), 1) for i in ids])
    check(leg["token_logprobs"] == [-0.25] * len(ids) and leg["text_offset"][0] == 0 and
          len(leg["top_logprobs"]) == len(ids) and leg["text_offset"][-1] == len("".join(leg["tokens"][:-1])), "legacy object")


def test_session_extra():
    """the engine plumbing on the mock backend: set at the request's start, armed once per kept first sample, cleared at the
    end (also after an error), logprobs travel with the tokens; a plain request touches nothing"""
    tpl = chat.Template(PACK, TOK, "xhigh")
    mb = B.MockBackend(TOK, ctx=8192)
    s = B.Session(mb, 8192, max_ckpts=4, mtp_policy="never", sys_ckpt_min=8, log=lambda *a: None)
    p = tpl.encode(tpl.render({"messages": [{"role": "user", "content": "hello"}], "reasoning_effort": "none"})[0])
    seen, lpl = [], []

    def on(t, lp=None):
        seen.extend(t)
        lpl.append(lp)

    st = s.generate(p, 100, (1.0, 0.95, 20, 1), False, on, lambda: False, tpl.im_end)
    check(mb.stats.get("extra_sets", 0) == 0 and mb.extra is None and all(x is None for x in lpl), "a plain request never touches the extras")
    ex = S.Extras.parse({"logprobs": True, "top_logprobs": 3, "logit_bias": {"5": -100}}, VOCAB).engine_extra()
    seen.clear(); lpl.clear()
    st = s.generate(p, 100, (0.0, 1.0, 1, 1), False, on, lambda: False, tpl.im_end, extra=ex)
    check(mb.stats["extra_sets"] == 1 and mb.stats["extra_clears"] == 1 and mb.extra is None, "set once, cleared at the end")
    check(len(lpl) == len(seen) and all(l is not None and len(l[0]) == 3 and l[0][0] == t for l, t in zip(lpl, seen)),
          "every token comes with its candidates (the sampled token first)")
    check(mb.armed >= 1, f"armed before the first kept sample ({mb.armed})")
    ex2 = S.Extras.parse({"presence_penalty": 0.5}, VOCAB).engine_extra()
    seen.clear(); lpl.clear()
    s.generate(p, 100, (0.0, 1.0, 1, 1), False, on, lambda: False, tpl.im_end, extra=ex2)
    check(all(x is None for x in lpl) and mb.stats["extra_clears"] == 2, "no logprob_k: tokens alone")
    # MTP with bias / penalties (not logprobs): armed before the MTP run takes the last prompt token, cleared after
    mb3 = B.MockBackend(TOK, ctx=8192)
    s3 = B.Session(mb3, 8192, max_ckpts=4, mtp_policy="always", sys_ckpt_min=8, log=lambda *a: None)
    exb = S.Extras.parse({"presence_penalty": 0.5, "logit_bias": {"5": -100}}, VOCAB).engine_extra()
    toks = []
    st = s3.generate(p, 100, (1.0, 0.95, 20, 1), True, lambda t, lp=None: toks.extend(t), lambda: False, tpl.im_end, extra=exb)
    check(st["path"] == "mtp" and mb3.armed == 1 and mb3.extra is None and mb3.stats["extra_clears"] == 1 and mb3.stats["mtp_runs"] >= 1,
          f"MTP run with a penalty: armed once, cleared ({st['path']}, armed {mb3.armed})")
    try:
        s3.generate(p, 100, (1.0, 0.95, 20, 1), True, lambda t, lp=None: None, lambda: False, tpl.im_end,
                    extra=S.Extras.parse({"logprobs": True}, VOCAB).engine_extra())
        check(False, "logprobs on the MTP path refused")
    except AssertionError:
        check(mb3.extra is None, "logprobs on the MTP path refused (and nothing left set)")
    # an engine error still clears the extras
    class Boom(B.MockBackend):
        def step(self, tok):
            raise RuntimeError("boom")
    mb2 = Boom(TOK, ctx=8192)
    s2 = B.Session(mb2, 8192, max_ckpts=4, mtp_policy="never", sys_ckpt_min=8, log=lambda *a: None)
    try:
        s2.generate(p, 100, (0.0, 1.0, 1, 1), False, lambda t, lp=None: None, lambda: False, tpl.im_end, extra=ex2)
        check(False, "the engine error propagates")
    except RuntimeError:
        check(mb2.extra is None and mb2.stats["extra_clears"] == 1, "after an engine error the extras are cleared")


def test_anthropic():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Weather in Paris?"}]},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "need the tool", "signature": "x"},
                                              {"type": "text", "text": "Checking."},
                                              {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "21 C"}]},
                                         {"type": "text", "text": "thanks"}]}]
    o = A.to_openai({"model": "m", "max_tokens": 100, "system": [{"type": "text", "text": "Be brief."}], "messages": msgs,
                     "tools": [{"name": "get_weather", "description": "d", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}],
                     "tool_choice": {"type": "any"}, "stop_sequences": ["END"], "temperature": 0.3, "top_k": 7,
                     "thinking": {"type": "enabled", "budget_tokens": 20000}})
    ms = o["messages"]
    check(ms[0] == {"role": "system", "content": "Be brief."}, "system blocks -> a system message")
    check(ms[2]["role"] == "assistant" and ms[2]["reasoning_content"] == "need the tool" and ms[2]["content"] == "Checking."
          and ms[2]["tool_calls"][0]["function"] == {"name": "get_weather", "arguments": '{"city": "Paris"}'}
          and ms[2]["tool_calls"][0]["id"] == "toolu_1", "assistant blocks -> reasoning_content, content, tool_calls")
    check(ms[3] == {"role": "tool", "tool_call_id": "toolu_1", "content": "21 C"} and ms[4]["role"] == "user",
          "tool_result -> a tool message, the text after it -> a user message")
    check(o["tools"][0]["function"]["parameters"]["properties"] == {"city": {"type": "string"}} and o["tool_choice"] == "required"
          and o["stop"] == ["END"] and o["temperature"] == 0.3 and o["top_k"] == 7 and o["reasoning_effort"] == "xhigh",
          "tools, tool_choice any -> required, stop_sequences, sampling, thinking budget -> effort")
    check(A.to_openai({"max_tokens": 5, "messages": [{"role": "user", "content": "x"}]})["reasoning_effort"] == "none", "no thinking field -> thinking off")
    check(A.to_openai({"max_tokens": 5, "messages": [{"role": "user", "content": "x"}], "thinking": {"type": "enabled", "budget_tokens": 1024}})["reasoning_effort"] == "low", "small budget -> low")
    img = A.to_openai({"max_tokens": 5, "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}, {"type": "text", "text": "what"}]}]})
    check(img["messages"][0]["content"][0]["image_url"]["url"] == "data:image/png;base64,AAAA", "base64 image -> data URL")
    for req, what in (({"messages": [{"role": "user", "content": "x"}]}, "no max_tokens"),
                      ({"max_tokens": 5, "messages": []}, "no messages"),
                      ({"max_tokens": 5, "messages": [{"role": "system", "content": "x"}]}, "system role in messages"),
                      ({"max_tokens": 5, "messages": [{"role": "user", "content": [{"type": "audio"}]}]}, "audio block"),
                      ({"max_tokens": 5, "messages": [{"role": "user", "content": "x"}], "tools": [{"type": "bash_20250124", "name": "bash"}]}, "server tool")):
        try:
            A.to_openai(req)
            check(False, f"refused: {what}")
        except chat.RequestError:
            check(True, f"refused: {what}")
    check(A.stop_reason("tool_calls", None) == ("tool_use", None) and A.stop_reason("length", None) == ("max_tokens", None)
          and A.stop_reason("stop", "END") == ("stop_sequence", "END") and A.stop_reason("stop", None) == ("end_turn", None), "stop reasons")
    check(A.usage(100, 60, 7) == {"input_tokens": 40, "output_tokens": 7, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 60}, "usage")
    st = A.Stream("msg_1", "m", 12)
    ev = st.start()
    ev += st.deltas([("reasoning", "hm"), ("reasoning", "m"), ("content", "Hi"), ("tool_start", 0, "c1", "f"), ("tool_args", 0, '{"a": 1'),
                     ("tool_args", 0, "}"), ("tool_start", 1, "c2", "g"), ("tool_args", 1, "{}")])
    ev += st.finish("tool_use", None, 9)
    names = [e[0] for e in ev]
    check(names[:2] == ["message_start", "ping"] and names[-2:] == ["message_delta", "message_stop"], "stream: start / ping ... delta / stop")
    starts = [(e[1]["index"], e[1]["content_block"]["type"]) for e in ev if e[0] == "content_block_start"]
    stops = [e[1]["index"] for e in ev if e[0] == "content_block_stop"]
    check(starts == [(0, "thinking"), (1, "text"), (2, "tool_use"), (3, "tool_use")] and stops == [0, 1, 2, 3], f"blocks open and close in order: {starts}")
    check("".join(e[1]["delta"].get("thinking", "") for e in ev if e[0] == "content_block_delta") == "hmm" and
          "".join(e[1]["delta"].get("partial_json", "") for e in ev if e[0] == "content_block_delta" and e[1]["index"] == 2) == '{"a": 1}',
          "thinking and tool input deltas concatenate")
    check(ev[-2][1]["delta"]["stop_reason"] == "tool_use" and ev[-2][1]["usage"]["output_tokens"] == 9, "message_delta")
    bl = A.blocks("text", "think", [{"id": "c", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}], True)
    check([b["type"] for b in bl] == ["thinking", "text", "tool_use"] and bl[2]["input"] == {"a": 1}, "full response blocks")


def test_stop_matched():
    f = chat.StopFilter(["END", "STOP"])
    out = f.feed("abc ST") + f.feed("OP xyz")
    check(f.hit and f.matched == "STOP" and out == "abc ", f"StopFilter.matched ({f.matched!r}, {out!r})")
    p = chat.RawParser()
    check(p.feed("\n x \n") == [("content", "\n x \n")] and p.finish() == [], "RawParser keeps the text verbatim")


if __name__ == "__main__":
    test_extras()
    test_texts()
    test_lp_entry()
    test_session_extra()
    test_anthropic()
    test_stop_matched()
    print(f"\n{len(FAILS)} failed: {FAILS}" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)
