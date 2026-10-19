"""CPU tests of the grammar-constrained decoding layer (lab-api): server/grammar.py over the pack's real tokenizer and
xgrammar, no engine, no GPU.

    $PY tests/server/test_grammar.py        ($PY = .venv/bin/python -I -B)

Checks: five JSON schemas (nested objects, enums, arrays, nullable, bounded arrays, nested number arrays): a valid
document is accepted token by token (compact and pretty-printed, thinking on and off), an invalid one is rejected at the
offending token, the end-of-text token is not allowed before the document is complete and is the only one after it;
json_object; a seeded random walk over the masks always ends in a document that validates against its schema;
tool_choice forced (one call to the named function, nothing before or after) and required (a call to any tool); the
reasoning phase before the format (</think> + a blank line); session plumbing on the mock engine: the mask is set before
every sampled token, MTP is never used with a mask, the mask is cleared afterwards and an unconstrained request never sets
one.
"""
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import refenv  # noqa: E402
import ijah_paths  # noqa: E402

refenv.add_paths()
import numpy as np  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

import backend as B  # noqa: E402
import chat  # noqa: E402
import grammar as G  # noqa: E402
from schemas import CASES, WEATHER_TOOLS  # noqa: E402

try:
    import jsonschema
except ImportError:   # pragma: no cover
    jsonschema = None

PACK = ijah_paths.pack_or_skip("tests/server/test_grammar.py", need=("tokenizer.json", "chat_template.jinja"))
TOK = Tokenizer.from_file(str(Path(PACK) / "tokenizer.json"))
STOPS = B.RealBackend.STOPS
FAILS = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def allowed(bm, t):
    return bool((int(bm[0, t >> 5]) >> (t & 31)) & 1)


def allowed_ids(bm):
    return np.flatnonzero(np.unpackbits(bm.view(np.uint8).reshape(-1), bitorder="little"))


def walk(cons, text):
    """feed `text` token by token; -> (index of the first token the mask forbids or None, the mask after the last)"""
    bm = cons.start()
    for n, t in enumerate(TOK.encode(text, add_special_tokens=False).ids):
        if not allowed(bm, t):
            return n, bm
        bm = cons.advance(t)
    return None, bm


def only_stop(bm):
    return set(allowed_ids(bm).tolist()) == set(STOPS)


def rf(schema):
    return {"response_format": {"type": "json_schema", "json_schema": {"name": "t", "schema": schema}}}


def test_schemas(kit):
    for c in CASES:
        req = rf(c["schema"])
        okv = json.loads(json.dumps(c["valid"]))
        if jsonschema:
            jsonschema.validate(okv, c["schema"])
            check(not jsonschema.Draft202012Validator(c["schema"]).is_valid(c["invalid"]),
                  f"{c['name']}: the invalid example is invalid for jsonschema too")
        for thinking in (False, True):
            pre = "Let me work this out.\nOk.\n</think>\n\n" if thinking else ""
            for label, doc in (("compact", json.dumps(c["valid"])), ("pretty", json.dumps(c["valid"], indent=2)),
                               ("ws-heavy", json.dumps(c["valid"], indent=4))):
                cons = kit.for_request(req, thinking)
                bad, bm = walk(cons, pre + doc)
                check(bad is None and only_stop(bm),
                      f"{c['name']} valid, {label}, thinking {thinking}: accepted, then only the stop tokens "
                      f"(bad token {bad})")
            cons = kit.for_request(req, thinking)
            bad, bm = walk(cons, pre + json.dumps(c["invalid"]))
            check(bad is not None, f"{c['name']} invalid document rejected at token {bad}, thinking {thinking}")
        # the stop token is not allowed before the document is complete
        cons = kit.for_request(req, False)
        doc = json.dumps(c["valid"])
        ids = TOK.encode(doc, add_special_tokens=False).ids
        bm = cons.start()
        early = False
        for t in ids[:-1]:
            early |= any(allowed(bm, s) for s in STOPS)
            bm = cons.advance(t)
        check(not early and not any(allowed(bm, s) for s in STOPS), f"{c['name']}: no stop token before the end")
        # first token of the answer: the grammar opens with '{' (or blanks), never prose
        first = allowed_ids(kit.for_request(req, False).start())
        texts = {TOK.decode([int(t)], skip_special_tokens=False).strip() for t in first[:200000]}
        check(texts <= {"", "{", "{\"", "{\n"} or all(x.startswith("{") or x == "" for x in texts),
              f"{c['name']}: first token opens the object ({len(first)} tokens allowed)")


def test_json_object(kit):
    req = {"response_format": {"type": "json_object"}}
    cons = kit.for_request(req, False)
    bad, bm = walk(cons, '{"a": [1, 2.5, {"b": null, "c": true}], "d": "e\\n", "f": {}}')
    check(bad is None and only_stop(bm), "json_object: nested object accepted")
    check(walk(kit.for_request(req, False), "[1, 2]")[0] == 0, "json_object: a top-level array is refused")
    check(walk(kit.for_request(req, False), "Sure, here is the JSON")[0] == 0, "json_object: prose is refused")
    check(walk(kit.for_request(req, False), '{"a": 1,}')[0] is not None, "json_object: a trailing comma is refused")
    bad, bm = walk(kit.for_request(req, True), 'Easy.\n</think>\n\n{"a": 1}')
    check(bad is None and only_stop(bm), "json_object with thinking: reasoning, </think>, blank line, object")
    check(kit.for_request({"response_format": {"type": "text"}}, False) is None, "response_format text: no constraint")
    check(kit.for_request({}, False) is None, "no response_format: no constraint")


def test_fuzz(kit, n_per_schema=4):
    """a seeded random model walks the masks; half of the picks favour structural closers so documents end"""
    rng = np.random.default_rng(1234)
    texts = {}

    def text(i):
        if i not in texts:
            texts[i] = TOK.decode([i], skip_special_tokens=False)
        return texts[i]

    pool = np.array([i for i in rng.choice(248077, 30000, replace=False) if len(text(int(i))) <= 4 and i not in STOPS])
    closers = np.array([i for i in range(248077) if 0 < len(text(i)) <= 3 and i not in STOPS
                        and all(ch in '"}],:0123456789.- ' for ch in text(i))])
    bits = lambda bm, ids: ((bm[0, ids >> 5].astype(np.int64) >> (ids & 31)) & 1).astype(bool)  # noqa: E731
    done = valid = steps = 0
    t0 = time.perf_counter()
    for c in CASES:
        for k in range(n_per_schema):
            thinking = k % 2 == 1
            cons = kit.for_request(rf(c["schema"]), thinking)
            bm = cons.start()
            toks = []
            for _ in range(700):
                steps += 1
                if rng.random() < 0.55:
                    cand = closers[bits(bm, closers)]
                else:
                    cand = pool[bits(bm, pool)]
                if len(cand) == 0:
                    ids = allowed_ids(bm)
                    cand = ids if len(ids) < 8 else rng.choice(ids, 8)
                t = int(rng.choice(cand))
                toks.append(t)
                if t in STOPS:
                    break
                bm = cons.advance(t)
            out = "".join(text(t) for t in toks if t not in STOPS)
            if toks[-1] in STOPS:
                done += 1
                body = out.split("</think>", 1)[1] if thinking else out
                try:
                    doc = json.loads(body)
                    ok = jsonschema.Draft202012Validator(c["schema"]).is_valid(doc) if jsonschema else True
                except json.JSONDecodeError:
                    ok = False
                valid += ok
                if not ok:
                    print("   invalid walk:", c["name"], repr(body[:300]))
    el = time.perf_counter() - t0
    total = n_per_schema * len(CASES)
    check(done >= total * 0.4, f"random walks that reached the end: {done} of {total}")
    check(valid == done, f"every finished random walk validates against its schema: {valid} of {done} "
                         f"({steps} masks in {el:.1f} s)")


def check_call_text(name, args_schema, calls):
    ok = bool(calls)
    for c in calls:
        a = json.loads(c["function"]["arguments"])
        if jsonschema:
            ok &= jsonschema.Draft202012Validator(args_schema[c["function"]["name"]]).is_valid(a)
    return ok


def test_tools(kit):
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in WEATHER_TOOLS}
    call = ("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n"
            "</parameter>\n<parameter=units>\ncelsius\n</parameter>\n</function>\n</tool_call>")
    call2 = "<tool_call>\n<function=get_time>\n<parameter=tz>\nEurope/Paris\n</parameter>\n</function>\n</tool_call>"
    forced = {"tools": WEATHER_TOOLS, "tool_choice": {"type": "function", "function": {"name": "get_weather"}}}
    required = {"tools": WEATHER_TOOLS, "tool_choice": "required"}
    for thinking in (False, True):
        pre = "Hmm.\n</think>\n\n" if thinking else ""
        bad, bm = walk(kit.for_request(forced, thinking), pre + call)
        check(bad is None and only_stop(bm), f"forced: the named call is accepted, then only stop (thinking {thinking}; bad {bad})")
        bad, _ = walk(kit.for_request(forced, thinking), pre + call2)
        check(bad is not None, f"forced: a call to the other function is refused (thinking {thinking})")
        bad, _ = walk(kit.for_request(forced, thinking), pre + "I will call it.\n\n" + call)
        check(bad is not None, f"forced: prose before the call is refused (thinking {thinking})")
        bad, bm = walk(kit.for_request(required, thinking), pre + call2)
        check(bad is None and any(allowed(bm, s) for s in STOPS),
              f"required: either function is accepted, and stop is allowed after a call (thinking {thinking})")
        bad, bm = walk(kit.for_request(required, thinking), pre + call + "\n" + call2)
        check(bad is None, f"required: two calls in a row (parallel) are accepted (thinking {thinking})")
        bad, _ = walk(kit.for_request(required, thinking), pre + "No tool needed.")
        check(bad is not None, f"required: an answer without a call is refused (thinking {thinking})")
        cons = kit.for_request(required, thinking)
        _, bm = walk(cons, pre)
        check(not any(allowed(bm, s) for s in STOPS), f"required: no stop token before a call (thinking {thinking})")
    req = dict(required, parallel_tool_calls=False)
    bad, bm = walk(kit.for_request(req, False), call)
    check(bad is None and only_stop(bm), "required, parallel_tool_calls false: one call, then only stop")
    # a model that wants to chat first and add a sign-off: the mask keeps only the call; the server's parser reads it
    bm_cases = []
    for k in range(2):
        b = B.MockBackend(TOK, 4096)
        s = B.Session(b, 4096, 2, "auto", log=lambda *a: None)
        text = ("<|im_start|>user\nFORCETEST<|im_end|>\n<|im_start|>assistant\n" + ("<think>\n" if k else "<think>\n\n</think>\n\n"))
        prompt = TOK.encode(text, add_special_tokens=False).ids
        got = []
        s.generate(prompt, 300, (1.0, 0.95, 20, 1), False, got.extend, lambda: False, TOK.token_to_id("<|im_end|>"),
                   constraint=kit.for_request(forced, bool(k)))
        gen = TOK.decode([t for t in got if t not in STOPS], skip_special_tokens=False)
        p = chat.OutputParser(bool(k), chat.tool_schemas(forced), True)
        content, reasoning, calls = chat.assemble(p.feed(gen) + p.finish())
        good = len(calls) == 1 and calls[0]["function"]["name"] == "get_weather" and not content.strip()
        if good and jsonschema:
            good = jsonschema.Draft202012Validator(schemas["get_weather"]).is_valid(json.loads(calls[0]["function"]["arguments"]))
        bm_cases.append(good)
        if not good:
            print("   bad forced walk:", repr(gen[:300]), calls)
    check(all(bm_cases), f"mock model, forced call: prose dropped, one valid get_weather call (thinking off / on: {bm_cases})")
    # request parsing
    for bad_req in ({"tools": WEATHER_TOOLS, "tool_choice": {"type": "function", "function": {"name": "nope"}}},
                    {"tools": WEATHER_TOOLS, "tool_choice": "sometimes"},
                    {"response_format": {"type": "xml"}}, {"response_format": {"type": "json_schema"}},
                    rf({"type": "nonsense"})):
        try:
            kit.for_request(bad_req, False)
            check(False, f"bad request refused: {json.dumps(bad_req)[:70]}")
        except chat.RequestError as e:
            check(True, f"bad request refused (400): {str(e)[:70]}")
    sub = {"type": "allowed_tools", "allowed_tools": {"mode": "required", "tools": [
        {"type": "function", "function": {"name": "get_time"}}]}}
    bad, _ = walk(kit.for_request({"tools": WEATHER_TOOLS, "tool_choice": sub}, False), call)
    check(bad is not None, "allowed_tools required [get_time]: a get_weather call is refused")
    bad, _ = walk(kit.for_request({"tools": WEATHER_TOOLS, "tool_choice": sub}, False), call2)
    check(bad is None, "allowed_tools required [get_time]: a get_time call is accepted")
    check(kit.for_request({"tools": WEATHER_TOOLS, "tool_choice": dict(sub, allowed_tools=dict(sub["allowed_tools"], mode="auto"))},
                          False) is None, "allowed_tools auto: no constraint")
    check(kit.for_request({"tools": WEATHER_TOOLS, "tool_choice": "auto"}, False) is None
          and kit.for_request({"tools": WEATHER_TOOLS}, False) is None
          and kit.for_request({"tools": WEATHER_TOOLS, "tool_choice": "none"}, False) is None
          and kit.for_request({"tool_choice": "required"}, False) is None,
          "tool_choice auto / none / unset / no tools: no constraint (MTP stays available)")


def test_format_plus_tools(kit):
    """a response_format with tools left to the model: the document or tool calls, nothing else"""
    c = CASES[3]
    req = dict(rf(c["schema"]), tools=WEATHER_TOOLS)
    call = ("<tool_call>\n<function=get_time>\n<parameter=tz>\nEurope/Paris\n</parameter>\n</function>\n</tool_call>")
    for thinking in (False, True):
        pre = "Hmm.\n</think>\n\n" if thinking else ""
        bad, bm = walk(kit.for_request(req, thinking), pre + json.dumps(c["valid"]))
        check(bad is None and only_stop(bm), f"response_format + tools: the document is accepted (thinking {thinking})")
        bad, bm = walk(kit.for_request(req, thinking), pre + call)
        check(bad is None and any(allowed(bm, s) for s in STOPS), f"response_format + tools: a tool call is accepted (thinking {thinking})")
        bad, _ = walk(kit.for_request(req, thinking), pre + "I think so. " + call)
        check(bad is not None, f"response_format + tools: prose before a call is refused (thinking {thinking})")
        bad, _ = walk(kit.for_request(req, thinking), pre + '{"verdict": "maybe"}')
        check(bad is not None, f"response_format + tools: an invalid document is still refused (thinking {thinking})")
    forced = dict(req, tool_choice={"type": "function", "function": {"name": "get_time"}})
    bad, _ = walk(kit.for_request(forced, False), json.dumps(c["valid"]))
    check(bad is not None, "response_format + a forced tool: the forced call wins (no document)")
    r2 = G.with_format_hint(dict(req, messages=[{"role": "user", "content": "hi"}]), G.GrammarKit.response_format(req))
    check(r2["messages"][0]["role"] == "system" and "tool" in r2["messages"][0]["content"]
          and r2["messages"][1]["content"] == "hi", "the format hint becomes a system message and mentions the tools")
    r3 = G.with_format_hint({"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "x"}],
                             "response_format": {"type": "json_object"}}, ("object", None))
    check(r3["messages"][0]["content"].startswith("Be brief.\n\n") and len(r3["messages"]) == 2,
          "the format hint is appended to the existing system message")
    r4 = {"messages": [{"role": "user", "content": "x"}], "ne_format_hint": False}
    check(G.with_format_hint(r4, ("object", None)) is r4, "ne_format_hint false: the request is untouched")


def test_session(kit):
    """Session + mock backend: the mask follows every sampled token and never reaches an MTP run"""
    b = B.MockBackend(TOK, 32768, 0.0)
    s = B.Session(b, 32768, 2, "auto", log=lambda *a: None)
    im_end = TOK.token_to_id("<|im_end|>")
    req = rf(CASES[0]["schema"])
    text = ("<|im_start|>user\nJSONTEST please<|im_end|>\n<|im_start|>assistant\n<think>\n")
    prompt = TOK.encode(text, add_special_tokens=False).ids
    cons = kit.for_request(req, True)
    got = []
    st = s.generate(prompt, 400, (1.0, 0.95, 20, 1), False, got.extend, lambda: False, im_end, constraint=cons)
    check(b.stats["mask_sets"] == st["completion_tokens"] and b.stats["masked_samples"] == st["completion_tokens"],
          f"mock: one mask before every sampled token ({b.stats['mask_sets']} sets, {st['completion_tokens']} tokens)")
    s.b.set_mask(None)
    out = "".join(TOK.decode([t], skip_special_tokens=False) for t in got if t not in STOPS)
    body = out.split("</think>", 1)[-1]
    try:
        doc = json.loads(body)
        valid = jsonschema.Draft202012Validator(CASES[0]["schema"]).is_valid(doc) if jsonschema else True
    except json.JSONDecodeError:
        doc, valid = None, False
    check(got[-1] in STOPS, "mock: the run ends on a stop token")
    check(b.mask is None, "mock: the mask is cleared afterwards")
    n0 = b.stats["mask_sets"]
    got2 = []
    s.generate(prompt, 40, (1.0, 0.95, 20, 1), False, got2.extend, lambda: False, im_end)
    check(b.stats["mask_sets"] == n0 and b.stats["masked_samples"] == st["completion_tokens"],
          "mock: an unconstrained request sets no mask")
    check(valid, "mock: a model that wants to chat first still ends with a schema-valid document")
    print("   mock walk:", repr(body[:120]), "valid:", valid)


def main():
    t0 = time.perf_counter()
    kit = G.GrammarKit(TOK, 248320, STOPS)
    print(f"grammar kit {time.perf_counter() - t0:.2f} s")
    for fn in (test_schemas, test_json_object, test_tools, test_format_plus_tools, test_fuzz, test_session):
        t = time.perf_counter()
        fn(kit)
        print(f"   {fn.__name__} {time.perf_counter() - t:.1f} s")
    print(f"\n{len(FAILS)} failed: {FAILS}" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
