"""CPU unit tests of the server's chat layer and prefix-reuse session (lab-server). No engine, no GPU.

    $PY tests/server/test_units.py        ($PY = .venv/bin/python -I -B)
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "tools"))
import refenv  # noqa: E402
import ijah_paths  # noqa: E402

refenv.add_paths()
from tokenizers import Tokenizer  # noqa: E402
import numpy as np  # noqa: E402

import backend as B  # noqa: E402
import chat  # noqa: E402
from prefix_cache import lcp  # noqa: E402

PACK = ijah_paths.pack_or_skip("tests/server/test_units.py", need=("tokenizer.json", "chat_template.jinja"))
TOK = Tokenizer.from_file(str(Path(PACK) / "tokenizer.json"))
FAILS = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        FAILS.append(what)


def run_parser(text, thinking=True, schemas=None, split=None):
    p = chat.OutputParser(thinking, schemas or {}, True)
    d = []
    rng = random.Random(split or 0)
    i = 0
    while i < len(text):  # random piece sizes: the parser must not depend on token boundaries
        n = rng.randint(1, 7) if split is not None else len(text)
        d += p.feed(text[i:i + n])
        i += n
    d += p.finish()
    return chat.assemble(d), p, d


def test_parser():
    gen = "Need weather.\n</think>\n\nI will check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n" \
          "</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"
    sch = {"get_weather": {"city": "string", "days": "integer"}}
    for split in (None, 1, 2, 3):
        (content, reasoning, calls), p, d = run_parser(gen, True, sch, split)
        check(reasoning == "Need weather." and content == "I will check.", f"reasoning / content split (split {split})")
        check(len(calls) == 1 and calls[0]["function"]["name"] == "get_weather"
              and json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris", "days": 3},
              f"tool call parsed + typed (split {split}): {calls and calls[0]['function']}")
    # streamed argument pieces are valid JSON once concatenated, name first
    (_, _, calls), p, d = run_parser(gen, True, sch, 5)
    kinds = [x[0] for x in d]
    check(kinds.index("tool_start") < kinds.index("tool_args"), "tool name streamed before arguments")
    check(sum(1 for k in kinds if k == "tool_args") == 3, "arguments in 3 pieces ({ city, , days, })")
    # two calls, no params, multi-line string value
    g2 = "</think>\n\n<tool_call>\n<function=a>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n<parameter=x>\n" \
         "line1\nline2\n</parameter>\n</function>\n</tool_call>"
    (content, reasoning, calls), _, _ = run_parser(g2, True, {}, 2)
    check(content == "" and [c["function"]["name"] for c in calls] == ["a", "b"]
          and calls[0]["function"]["arguments"] == "{}"
          and json.loads(calls[1]["function"]["arguments"]) == {"x": "line1\nline2"}, "two calls, empty + multiline")
    # JSON body inside <tool_call>
    g3 = '<tool_call>\n{"name": "f", "arguments": {"q": 1}}\n</tool_call>'
    (content, _, calls), _, _ = run_parser(g3, False, {}, 3)
    check(len(calls) == 1 and json.loads(calls[0]["function"]["arguments"]) == {"q": 1}, "JSON tool-call body")
    # thinking off: everything is content; '<' that is not a tag passes
    (content, reasoning, calls), _, _ = run_parser("a < b and <tool c", False, {}, 1)
    check(content == "a < b and <tool c" and reasoning == "" and not calls, "thinking off, partial tags released")
    # truncated reasoning (length): reasoning only
    (content, reasoning, _), _, _ = run_parser("still thinking\n", True, {}, 2)
    check(reasoning == "still thinking" and content == "", "truncated reasoning")
    # tools disabled: tags stay text
    p = chat.OutputParser(False, {}, tools_enabled=False)
    d = p.feed("<tool_call>x</tool_call>") + p.finish()
    check(chat.assemble(d)[0] == "<tool_call>x</tool_call>", "tool tags as text without tools")


def test_cut_calls():
    """a tool call that never closed (cut off by length / EOS / a stop string) is not a tool call (lab-fixsrv)"""
    sch = {"get_weather": {"city": "string", "days": "integer"}}
    head = "Checking.\n\n<tool_call>\n<function=get_weather>\n"
    one = "<parameter=city>\nParis\n</parameter>\n"
    cases = {   # text -> (calls kept, cut, content contains)
        "mid-parameter value": (head + one + "<parameter=days>\n3", 0, True, "<parameter=days>\n3"),
        "mid-parameter tag": (head + one + "<parame", 0, True, "<parame"),
        "mid first parameter": (head + "<parameter=city>\nPa", 0, True, "<parameter=city>\nPa"),
        "after a finished parameter": (head + one, 0, True, "</parameter>\n"),
        "no parameter yet": (head, 0, True, "<function=get_weather>"),
        "inside the name": ("<tool_call>\n<function=get_wea", 0, True, "<function=get_wea"),
        "text then EOF": ("<tool_call>", 0, True, "<tool_call>"),
        "after </function>": (head + one + "</function>\n", 1, False, None),
        "partial </tool_call>": (head + one + "</function>\n</tool_c", 1, False, None),
        "complete": (head + one + "</function>\n</tool_call>", 1, False, None),
    }
    for name, (text, ncalls, cut, has) in cases.items():
        for split in (None, 1, 3):
            (content, _, calls), p, d = run_parser(text, False, sch, split)
            ok = len(calls) == ncalls and p.cut == cut and len(p.finished_calls) == ncalls and \
                (has is None or has in content)
            if ncalls:
                ok = ok and json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
            check(ok, f"cut call, {name} (split {split}): {ncalls} call(s), cut {cut}: calls {len(calls)}, cut {p.cut}")
    # the text comes back whole: content = prose + the raw call text, byte for byte
    text = head + one + "<parameter=days>\n3"
    (content, _, calls), p, d = run_parser(text, False, sch, 2)
    check(content == "Checking." + text[len("Checking.\n\n"):] and not calls,   # (the blank line before a call is dropped)
          f"cut call: content is the generated text, the call's raw text included ({content!r})")
    # a streaming client sees the start and the finished parameters but never a closing brace
    pieces = [x for x in d if x[0] == "tool_args"]
    check(d[-1][0] == "tool_abort" and "".join(x[2] for x in pieces) == '{"city": "Paris"}'[:-1],
          f"streamed arguments stay unterminated, ended by tool_abort: {pieces}")
    # two calls, the second cut: the first is kept
    two = head + one + "</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nLy"
    (content, _, calls), p, d = run_parser(two, False, sch, 4)
    check(len(calls) == 1 and p.cut and len(p.finished_calls) == 1 and "<parameter=city>\nLy" in content
          and json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"},
          "two calls, the second cut: one kept, the cut one given back as content")
    # a tagged end after an unfinished parameter: malformed, given back (not cut: generation went on)
    bad = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</tool_call>\nOK."
    (content, _, calls), p, d = run_parser(bad, False, sch, 3)
    check(not calls and not p.cut and content.startswith("<tool_call>") and content.endswith("</tool_call>OK."),
          f"malformed (tagged) call given back as content ({content!r})")
    # a JSON body that parses stays a call; a cut one is content
    (content, _, calls), p, _ = run_parser('<tool_call>\n{"name": "f", "arguments": {"q": 1}}', False, {}, 3)
    check(len(calls) == 1 and p.finished_calls and not p.cut, "JSON body, parsed before the end: a call")
    (content, _, calls), p, _ = run_parser('<tool_call>\n{"name": "f", "argum', False, {}, 3)
    check(not calls and not p.finished_calls and content.startswith("<tool_call>"), "JSON body cut off: content")


def test_cancel_prefill():
    """a cancel is seen between prefill pieces (lab-fixsrv): a long prompt is not prefilled to the end for a gone client,
    the pieces fed stay a valid prefix, the slot is free, the next request reuses them"""
    tpl = chat.Template(PACK, TOK, "xhigh")
    for use in (False, True):
        mb = B.MockBackend(TOK, ctx=32768)
        s = B.Session(mb, 32768, max_ckpts=4, mtp_policy="always", sys_ckpt_min=8, log=lambda *a: None)
        text, _ = tpl.render({"messages": [{"role": "system", "content": "sys " * 30},
                                           {"role": "user", "content": "word " * 20000}]})
        p = tpl.encode(text)
        calls = [0]

        def cancelled():
            calls[0] += 1
            return calls[0] > 2   # two pieces, then the client is gone
        st = s.generate(p, 100, (1.0, 0.95, 20, 0), use, lambda t: None, cancelled, tpl.im_end)
        fed = mb.stats["fed_tokens"]
        check(st.get("cancelled") == "prefill" and st["completion_tokens"] == 0 and 0 < fed < len(p) - 4000
              and not s.slots[0].busy and mb.fed == p[:len(mb.fed)] and s.kv == mb.fed,
              f"cancel between prefill pieces (use_mtp {use}): fed {fed} of {len(p)} tokens, slot free, state = a prefix")
        out = []
        st2 = s.generate(p, 100, (1.0, 0.95, 20, 0), use, lambda t: out.extend(t), lambda: False, tpl.im_end)
        mb2 = B.MockBackend(TOK, ctx=32768)
        s2 = B.Session(mb2, 32768, max_ckpts=4, mtp_policy="always", sys_ckpt_min=8, log=lambda *a: None)
        out2 = []
        s2.generate(p, 100, (1.0, 0.95, 20, 0), use, lambda t: out2.extend(t), lambda: False, tpl.im_end)
        check(st2["cached_tokens"] >= fed - 1 and out == out2 and mb.stats["fed_tokens"] - fed <= len(p) - fed + 1,
              f"the next request reuses the {st2['cached_tokens']} tokens already fed; output = a fresh run")
    mb = B.MockBackend(TOK, ctx=32768)
    s = B.Session(mb, 32768, max_ckpts=4, mtp_policy="never", sys_ckpt_min=8, log=lambda *a: None)
    st = s.generate(p, 100, (1.0, 0.95, 20, 0), False, lambda t: None, lambda: True, tpl.im_end)
    check(st.get("cancelled") == "prefill" and mb.stats["fed_tokens"] == 0 and not s.slots[0].busy,
          "cancelled before the first piece: nothing is fed")


def test_stop():
    for split in (1, 2, 4, 100):
        s = chat.StopFilter(["STOP_HERE", "xyz"])
        text = "one two three STOP_HERE four"
        out = "".join(s.feed(text[i:i + split]) for i in range(0, len(text), split)) + s.flush()
        check(out == "one two three " and s.hit, f"stop string (split {split})")
    s = chat.StopFilter("zz")
    out = s.feed("a z") + s.feed("b") + s.flush()
    check(out == "a zb" and not s.hit, "partial stop released")


def test_template():
    tpl = chat.Template(PACK, TOK, "xhigh")
    req = {"messages": [{"role": "user", "content": "hi"}]}
    text, think = tpl.render(req)
    check(think and text.endswith("<|im_start|>assistant\n<think>\n") and "Reasoning effort is set to xhigh" in text,
          "default: thinking on, effort xhigh")
    text, think = tpl.render(dict(req, reasoning_effort="low"))
    check("effort is set to low" in text, "reasoning_effort low")
    text, think = tpl.render(dict(req, reasoning_effort="medium"))
    check("Reasoning effort" not in text and think, "medium = no instruction (the template's)")
    text, think = tpl.render(dict(req, reasoning_effort="none"))
    check(not think and text.endswith("<think>\n\n</think>\n\n"), "reasoning_effort none = thinking off")
    text, think = tpl.render(dict(req, chat_template_kwargs={"enable_thinking": False}))
    check(not think, "chat_template_kwargs.enable_thinking false")
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "w",
                                               "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "w?"},
            {"role": "assistant", "content": "", "reasoning_content": "r",
             "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "21 C"}]
    text, _ = tpl.render({"messages": msgs, "tools": tools})
    check("<function=get_weather>\n<parameter=city>\nParis\n</parameter>" in text and "<tool_response>\n21 C" in text
          and '"name": "get_weather"' in text, "tool round trip renders")
    try:
        tpl.render({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]})
        check(False, "image part refused")
    except chat.RequestError:
        check(True, "image part refused")


def test_detok():
    ids = TOK.encode("Grüße, 世界! <think>x</think> 🙂", add_special_tokens=False).ids
    d = chat.Detok(TOK)
    check("".join(d(t) for t in ids) == TOK.decode(ids, skip_special_tokens=False), "incremental detok = decode")


def _conv_runner(s, mbs, tpl, use=False):
    """run(msgs) on the session; checks every slot's engine state against what the session believes"""
    im_end = tpl.im_end

    def run(msgs, **kw):
        text, _ = tpl.render({"messages": msgs, **kw})
        p = tpl.encode(text)
        out = []
        st = s.generate(p, 1000, (1.0, 0.95, 20, 0), use, out.extend, lambda: False, im_end)
        mb = mbs[st["slot"]]
        check(mb.fed == p + out[:-1], f"engine state == prompt + output[:-1] ({len(p)} + {len(out)})")
        return p, out, st
    return run


def _next_turn(msgs, out, text, send_reasoning=False):
    r, c = TOK.decode(out[:-1], skip_special_tokens=False).split("</think>")
    a = {"role": "assistant", "content": c.strip()}
    if send_reasoning:
        a["reasoning_content"] = r
    return msgs + [a, {"role": "user", "content": text}]


def test_session():
    mb = B.MockBackend(TOK, ctx=8192, cont=False)  # the older library: MTP only from an empty state
    s = B.Session(mb, 8192, max_ckpts=4, mtp_policy="never", sys_ckpt_min=8, log=lambda *a: None)
    tpl = chat.Template(PACK, TOK, "xhigh")
    sysmsg = {"role": "system", "content": "You are a test assistant. " * 20}
    im_end = tpl.im_end
    run = _conv_runner(s, [mb], tpl)

    m1 = [sysmsg, {"role": "user", "content": "first question"}]
    p1, o1, st1 = run(m1)
    check(st1["cached_tokens"] == 0, "turn 1: nothing cached")
    m2 = _next_turn(m1, o1, "second")
    p2, o2, st2 = run(m2)  # reasoning not sent back: rendered as <think>\n\n</think>: reuse = end-of-prompt ckpt
    check(st2["cached_tokens"] == s.last_start(p1), f"turn 2 reuses the turn-1 checkpoint at its generation prompt "
          f"({st2['cached_tokens']} of {len(p1)})")
    m3 = _next_turn(m2, o2, "third", send_reasoning=True)
    p3, o3, st3 = run(m3)  # reasoning sent back: the whole previous turn is reusable (live state)
    check(st3["cached_tokens"] >= len(p2) + len(o2) - 2, f"turn 3 reuses the live state ({st3['cached_tokens']}, "
          f"turn 2 ended at {len(p2) + len(o2) - 1})")
    # another conversation with the same system prompt: the system checkpoint
    p4, o4, st4 = run([sysmsg, {"role": "user", "content": "other conversation"}])
    b = s.system_boundary(p4, im_end)
    check(b > 0 and st4["cached_tokens"] == b, f"new conversation reuses the system checkpoint ({st4['cached_tokens']} = {b})")
    # back to conversation 1 (turn 3 again): it was parked (KV rows copied out) -> its turn-3 checkpoint
    p5, o5, st5 = run(m3)
    check(st5["cached_tokens"] == s.last_start(p3), f"parked conversation 1 restored at its turn-3 checkpoint "
          f"({st5['cached_tokens']} = {s.last_start(p3)})")
    # same prompt twice (regenerate): the end-of-prompt ckpt has all n tokens, so an earlier one is used
    p6, o6, st6 = run(m3)
    check(0 < st6["cached_tokens"] < len(p6), f"regenerate falls back to an earlier checkpoint ({st6['cached_tokens']})")
    # MTP run: the resident conversation is parked first, the MTP conversation keeps no checkpoint
    s.mtp_policy = "always"
    text, _ = tpl.render({"messages": [sysmsg, {"role": "user", "content": "mtp question"}]})
    p7 = tpl.encode(text)
    out = []
    st7 = s.generate(p7, 1000, (1.0, 0.95, 20, 0), True, out.extend, lambda: False, im_end)
    check(st7["path"] == "mtp" and not s.live and not s.slots[0].conv.snaps, "MTP: live off, no checkpoints")
    p8, o8, st8 = run([sysmsg, {"role": "user", "content": "after mtp"}])
    check(st8["cached_tokens"] == b, f"after MTP the system checkpoint is reused ({st8['cached_tokens']})")
    p9, o9, st9 = run(m3)
    check(st9["cached_tokens"] == s.last_start(p3), f"after MTP conversation 1 still restores ({st9['cached_tokens']})")
    # policy: long prompt with reuse -> step path; short prompt without -> MTP
    s.mtp_policy = "auto"
    check(not s.choose_mtp(20000, 19000, 10 ** 6, True), "auto: long prompt with reuse -> step path")
    check(s.choose_mtp(300, 0, 10 ** 6, True), "auto: short fresh prompt -> MTP")
    check(not s.choose_mtp(300, 0, 10 ** 6, False), "auto: MTP not allowed (stop strings / top_k > 32) -> step")
    # lab-ttft: full chunks, then the rest as one piece (each piece = a pass over the weights)
    check(B.pieces(5000) == [4097, 903], f"prefill pieces {B.pieces(5000)}")
    check(B.pieces(5000, 1025) == [1025] * 4 + [900], f"capped pieces {B.pieces(5000, 1025)}")
    check(B.pieces(150) == [150] and B.pieces(8194) == [4097, 4097] and B.pieces(8195) == [4097, 4097, 1],
          f"small / exact pieces {B.pieces(150)} {B.pieces(8194)} {B.pieces(8195)}")


def _three_convs(s, mbs, tpl, turns=4, use=False):
    """3 conversations (distinct system prompts), round robin, `turns` turns each -> stats per (conv, turn)"""
    run = _conv_runner(s, mbs, tpl, use)
    convs = [[{"role": "system", "content": f"System prompt of conversation {c}. " * 30},
              {"role": "user", "content": f"question 1 of conversation {c}"}] for c in range(3)]
    res = []
    for t in range(turns):
        for c in range(3):
            p, o, st = run(convs[c])
            res.append((c, t, p, st))
            convs[c] = _next_turn(convs[c], o, f"question {t + 2} of conversation {c} " + "pad " * 40)
    return res


def test_multi():
    tpl = chat.Template(PACK, TOK, "xhigh")
    # 1 slot, ample RAM: every turn after the first of each conversation restores its last generation prompt
    mb = B.MockBackend(TOK, ctx=16384)
    s = B.Session(mb, 16384, max_ckpts=4, mtp_policy="never", sys_ckpt_min=8, log=lambda *a: None)
    res = _three_convs(s, [mb], tpl)
    ok = all(st["cached_tokens"] == (0 if t == 0 else s.last_start(prev))
             for (c, t, p, st), prev in zip(res, [None] * 3 + [r[2] for r in res[:-3]]))
    check(ok, "3 conversations x 4 turns on 1 slot: each turn > 1 reuses its conversation's last checkpoint " +
          str([st["cached_tokens"] for *_, st in res]))
    check(mb.stats["kv_loaded"] > 0 and s.cache.stats["parks"] >= 9, f"parks {s.cache.stats['parks']}, KV rows "
          f"loaded {mb.stats['kv_loaded']} (incremental: saved {mb.stats['kv_saved']})")
    tot = sum(len(r[2]) for r in res)
    check(mb.stats["kv_saved"] < tot, f"parking copies only new rows ({mb.stats['kv_saved']} saved < {tot} prompt "
          "tokens)")
    # tight RAM budget: one parked conversation fits; the others are evicted (cold) but results stay correct
    mb2 = B.MockBackend(TOK, ctx=16384)
    s2 = B.Session(mb2, 16384, max_ckpts=4, mtp_policy="never", sys_ckpt_min=8, log=lambda *a: None, cache_ram=1)
    res2 = _three_convs(s2, [mb2], tpl, turns=2)
    check(all(st["cached_tokens"] == 0 for c, t, p, st in res2 if t > 0), "budget 1 byte: nothing parked survives")
    check(s2.cache.ram_bytes() <= 4 * mb2.snap_bytes + 4 * 16384, "RAM within budget (+ the resident conversation)")
    # budget for ~1 parked conversation (snapshots 256 B + rows 4 B each in the mock)
    mb3 = B.MockBackend(TOK, ctx=16384)
    s3 = B.Session(mb3, 16384, max_ckpts=4, mtp_policy="never", sys_ckpt_min=8, log=lambda *a: None,
                   cache_ram=6000)
    res3 = _three_convs(s3, [mb3], tpl, turns=3)
    check(max(s3.cache.ram_bytes() for _ in [0]) <= 6000, f"RAM {s3.cache.ram_bytes()} <= budget 6000")
    # disk tier: same tight budget, everything evicted goes to disk and comes back
    import tempfile
    d = tempfile.mkdtemp(prefix="ne-cache-test")
    mb4 = B.MockBackend(TOK, ctx=16384)
    s4 = B.Session(mb4, 16384, max_ckpts=4, mtp_policy="never", sys_ckpt_min=8, log=lambda *a: None,
                   cache_ram=1, disk_dir=d + "/c", disk_bytes=1 << 30)
    res4 = _three_convs(s4, [mb4], tpl, turns=3)
    ok = all(st["cached_tokens"] == (0 if t == 0 else s4.last_start(prev))
             for (c, t, p, st), prev in zip(res4, [None] * 3 + [r[2] for r in res4[:-3]]))
    check(ok and s4.cache.stats["disk_hits"] >= 6, f"disk tier: every turn restored ({s4.cache.stats['disk_hits']} "
          "disk hits)")
    s4.cache.disk.close()
    # 2 slots: the 3 conversations spread over both; still every turn reuses
    mbs = [B.MockBackend(TOK, ctx=16384), B.MockBackend(TOK, ctx=16384)]
    s5 = B.Session(mbs, 16384, max_ckpts=4, mtp_policy="never", sys_ckpt_min=8, log=lambda *a: None)
    res5 = _three_convs(s5, mbs, tpl)
    ok = all(st["cached_tokens"] == (0 if t == 0 else s5.last_start(prev))
             for (c, t, p, st), prev in zip(res5, [None] * 3 + [r[2] for r in res5[:-3]]))
    check(ok and {st["slot"] for *_, st in res5} == {0, 1}, "2 slots, 3 conversations: every turn reuses, both slots used")


def test_multi_mtp():
    """3 conversations rotating on one slot, every turn on the MTP-continuation path: parking carries the MTP head's
    KV rows and bookkeeping, so each restored conversation continues with MTP (the mock checks the head rows)"""
    tpl = chat.Template(PACK, TOK, "xhigh")
    for budget, disk in ((1 << 30, False), (1, True)):
        import tempfile
        d = tempfile.mkdtemp(prefix="ne-cache-mtp") if disk else None
        mb = B.MockBackend(TOK, ctx=16384)
        s = B.Session(mb, 16384, max_ckpts=4, mtp_policy="always", sys_ckpt_min=8, log=lambda *a: None,
                      cache_ram=budget, disk_dir=d and d + "/c", disk_bytes=(1 << 30) if disk else 0)
        res = _three_convs(s, [mb], tpl, use=True)
        ok = all(st["cached_tokens"] == (0 if t == 0 else s.last_start(prev)) and st["path"] == "mtp"
                 for (c, t, p, st), prev in zip(res, [None] * 3 + [r[2] for r in res[:-3]]))
        check(ok and mb.stats["head_restores"] >= 9, f"MTP turns on 1 slot ({'disk tier' if disk else 'RAM'}): every "
              f"turn > 1 restores its conversation with the head in step ({mb.stats['head_restores']} head restores; "
              f"paths {sorted({st['path'] for *_, st in res})})")
        if s.cache.disk:
            s.cache.disk.close()


def test_concurrent():
    """two requests interleaved token by token on two slots (the worker's loop); a third waits for a slot"""
    tpl = chat.Template(PACK, TOK, "xhigh")
    mbs = [B.MockBackend(TOK, ctx=16384), B.MockBackend(TOK, ctx=16384)]
    s = B.Session(mbs, 16384, max_ckpts=4, mtp_policy="never", sys_ckpt_min=8, log=lambda *a: None)
    sysm = {"role": "system", "content": "Shared system prompt for the main agent and its subagent. " * 20}
    prompts = []
    for q in ("main agent: LONGTEST", "subagent: LONGTEST", "third request"):
        text, _ = tpl.render({"messages": [sysm, {"role": "user", "content": q}]})
        prompts.append(tpl.encode(text))
    outs = [[], [], []]
    gens = []
    order = []
    for i in range(2):
        sl, cand = s.pick(prompts[i])
        gens.append((i, sl, s.run(sl, cand, prompts[i], 1000, (1.0, 0.95, 20, 0), False,
                                  lambda t, i=i: (outs[i].extend(t), order.append(i)), lambda: False, tpl.im_end,
                                  others=lambda: True)))
        for _ in range(5):  # the main agent is decoding when the subagent's request arrives
            next(gens[0][2])
    sl3, _ = s.pick(prompts[2])
    check(sl3 is None, "third request: no free slot while two run")
    stats = {}
    while gens:
        for g in list(gens):
            try:
                next(g[2])
            except StopIteration as e:
                stats[g[0]] = e.value
                gens.remove(g)
    check(stats[0]["slot"] != stats[1]["slot"], f"two slots used ({stats[0]['slot']}, {stats[1]['slot']})")
    check(stats[1]["cached_tokens"] > 0, f"the subagent reused the main agent's system checkpoint from the other "
          f"slot ({stats[1]['cached_tokens']})")
    switches = sum(1 for a, b in zip(order, order[1:]) if a != b)
    check(switches > 100, f"token-level interleaving ({switches} switches over {len(order)} tokens)")
    for i in range(2):
        mb = mbs[stats[i]["slot"]]
        check(mb.fed == prompts[i] + outs[i][:-1], f"stream {i}: engine state == prompt + output[:-1]")
        ref = B.MockBackend(TOK, ctx=16384)
        r = []
        B.Session(ref, 16384, mtp_policy="never", log=lambda *a: None).generate(
            prompts[i], 1000, (1.0, 0.95, 20, 0), False, r.extend, lambda: False, tpl.im_end)
        check(r == outs[i], f"stream {i}: output = the same request alone ({len(r)} tokens)")


def test_concurrent_mask():
    """a constrained request (token mask) next to an unconstrained one on the other slot: the mask is the slot's own,
    set before each of its samples, never seen by the other slot, cleared at the end"""
    tpl = chat.Template(PACK, TOK, "xhigh")
    mbs = [B.MockBackend(TOK, ctx=16384), B.MockBackend(TOK, ctx=16384)]
    s = B.Session(mbs, 16384, max_ckpts=4, mtp_policy="never", sys_ckpt_min=8, log=lambda *a: None)
    allow = np.full((B.MockBackend.VOCAB + 31) // 32, -1, np.int32)

    class Cons:
        kind = "object"
        n = 0

        def start(self):
            return allow

        def advance(self, t):
            self.n += 1
            return allow
    ps = [tpl.encode(tpl.render({"messages": [{"role": "user", "content": q}]})[0])
          for q in ("JSONTEST constrained", "LONGTEST free")]
    outs, gens, seen = [[], []], [], []
    cons = Cons()
    for i in range(2):
        sl, cand = s.pick(ps[i])
        gens.append([i, sl, s.run(sl, cand, ps[i], 300, (1.0, 0.95, 20, 0), False, outs[i].extend, lambda: False,
                                  tpl.im_end, others=lambda: True, constraint=cons if i == 0 else None)])
    while gens:
        for g in list(gens):
            try:
                next(g[2])
                if g[0] == 1:
                    seen.append(g[1].b.mask is None)
            except StopIteration:
                gens.remove(g)
    m0, m1 = mbs[0].stats["masked_samples"], mbs[1].stats["masked_samples"]
    check(m0 == len(outs[0]) and m1 == 0 and all(seen) and mbs[0].mask is None and mbs[1].mask is None,
          f"mask per slot: {m0} masked samples on the constrained slot ({len(outs[0])} tokens), {m1} on the other; "
          "both cleared after")
    ref = []
    B.Session(B.MockBackend(TOK, ctx=16384), 16384, mtp_policy="never", log=lambda *a: None).generate(
        ps[1], 300, (1.0, 0.95, 20, 0), False, ref.extend, lambda: False, tpl.im_end)
    check(ref == outs[1], "the unconstrained stream = the same request alone")


def test_random_mix(seed=3, rounds=120):
    """random workload: 5 conversations (2 share a system prompt), new turns, regenerates, edited earlier turns,
    MTP requests, 1-3 slots, budgets from tiny to ample, 0-2 interleaved requests, a disk tier: the mock asserts that
    the KV rows always match the recurrent state's tokens; every output must equal the same request run alone"""
    import tempfile
    rng = random.Random(seed)
    tpl = chat.Template(PACK, TOK, "xhigh")
    bad = 0
    total = 0
    allr = [0, 0]  # restarts, SSD hits
    for case in range(4):
        nslots = 1 + case % 3
        mbs = [B.MockBackend(TOK, ctx=16384) for _ in range(nslots)]
        budget = [1, 4000, 20000, 1 << 30][case]
        d = tempfile.mkdtemp(prefix="ne-cache-mix") if case in (1, 2) else None
        s = B.Session(mbs, 16384, max_ckpts=3, mtp_policy="never", sys_ckpt_min=8, log=lambda *a: None,
                      cache_ram=budget, disk_dir=d and d + "/c", disk_bytes=1 << 30 if d else 0)
        systems = ["Shared system prompt. " * 25, "Shared system prompt. " * 25] + \
                  [f"System prompt {c}. " * (20 + 5 * c) for c in range(2, 5)]
        convs = [[{"role": "system", "content": systems[c]}, {"role": "user", "content": f"hello {c}"}]
                 for c in range(5)]
        active = []
        restarts = 0
        for r in range(rounds // 4):
            if d and not active and rng.random() < 0.15:  # the server restarts: new engines, same SSD directory
                s.cache.close()
                mbs = [B.MockBackend(TOK, ctx=16384) for _ in range(nslots)]
                s = B.Session(mbs, 16384, max_ckpts=3, mtp_policy="never", sys_ckpt_min=8, log=lambda *a: None,
                              cache_ram=budget, disk_dir=d and d + "/c", disk_bytes=1 << 30)
                restarts += 1
            c = rng.randrange(5)
            kind = rng.random()
            msgs = convs[c]
            if kind < 0.1 and len(msgs) > 4:  # edit an earlier user turn
                msgs = msgs[:3] + [{"role": "user", "content": f"edited {r}"}]
            text, _ = tpl.render({"messages": msgs})
            prompt = tpl.encode(text)
            use_mtp = kind > 0.92
            if use_mtp:
                s.mtp_policy = "always"
            sl, cand = s.pick(prompt, reuse=rng.random() > 0.05)
            if sl is None or (use_mtp and active):
                s.mtp_policy = "never"
                continue
            out = []
            g = s.run(sl, cand, prompt, 300, (1.0, 0.95, 20, 0), use_mtp, out.extend, lambda: False, tpl.im_end,
                      others=lambda: len(active) > 1)
            active.append((c, prompt, out, g, msgs, kind))
            s.mtp_policy = "never"
            # advance the running requests a random number of turns, interleaved
            for _ in range(rng.randrange(0, 400)):
                if not active:
                    break
                e = active[rng.randrange(len(active))]
                try:
                    next(e[3])
                except StopIteration:
                    active.remove(e)
                    total += 1
                    ref = []
                    B.Session(B.MockBackend(TOK, ctx=16384), 16384, mtp_policy="never", log=lambda *a: None).generate(
                        e[1], 300, (1.0, 0.95, 20, 0), False, ref.extend, lambda: False, tpl.im_end)
                    if ref != e[2]:
                        bad += 1
                    if e[5] >= 0.1 and e[5] <= 0.92 and rng.random() < 0.8:  # continue the conversation
                        convs[e[0]] = _next_turn(e[4], e[2], f"turn {r} " + "x " * rng.randrange(1, 60),
                                                 send_reasoning=rng.random() < 0.3)
            if s.cache.ram_budget < (1 << 30):
                pass
        for e in active:  # drain
            for _ in e[3]:
                pass
        if s.cache.disk:
            s.cache.disk.close()
        allr[0] += restarts
        allr[1] += s.cache.stats["disk_hits"]
    check(bad == 0 and total > 30, f"random mix: {total} requests ({allr[0]} restarts; SSD hits after the last ones "
          f"{allr[1]}), {bad} outputs differ from the request alone "
          "(state / KV bookkeeping asserted on every feed)")



def test_session_cont():
    """MTP from the current state (docs/mtp.md 10.9): every turn on the MTP path, with the same reuse as the step path"""
    mb = B.MockBackend(TOK, ctx=8192)
    s = B.Session(mb, 8192, max_ckpts=4, mtp_policy="always", sys_ckpt_min=8, log=lambda *a: None)
    tpl = chat.Template(PACK, TOK, "xhigh")
    sysmsg = {"role": "system", "content": "You are a test assistant. " * 20}
    im_end = tpl.im_end

    def run(msgs, use=True, cancel_after=None):
        text, _ = tpl.render({"messages": msgs})
        p = tpl.encode(text)
        out = []
        calls = [0]

        def on(t):
            out.extend(t)
            calls[0] += 1

        cancelled = (lambda: calls[0] >= cancel_after) if cancel_after else (lambda: False)
        st = s.generate(p, 1000, (1.0, 0.95, 20, 0), use, on, cancelled, im_end)
        check(mb.fed == p + out[:-1] and s.kv == mb.fed and s.live,
              f"cont: engine state == prompt + output[:-1] == kv ({len(p)} + {len(out)}, path {st['path']})")
        return p, out, st

    def reply_msgs(o, with_reasoning):
        r, c = TOK.decode(o[:-1], skip_special_tokens=False).split("</think>")
        m = {"role": "assistant", "content": c.strip()}
        if with_reasoning:
            m["reasoning_content"] = r
        return m

    m1 = [sysmsg, {"role": "user", "content": "first question"}]
    p1, o1, st1 = run(m1)
    check(st1["path"] == "mtp" and st1["cached_tokens"] == 0, "cont turn 1: MTP, nothing cached")
    m2 = m1 + [reply_msgs(o1, False), {"role": "user", "content": "second"}]
    p2, o2, st2 = run(m2)
    check(st2["path"] == "mtp" and st2["cached_tokens"] == s.last_start(p1),
          f"cont turn 2: MTP from the turn-1 generation-prompt checkpoint ({st2['cached_tokens']})")
    m3 = m2 + [reply_msgs(o2, True), {"role": "user", "content": "third"}]
    p3, o3, st3 = run(m3)
    check(st3["path"] == "mtp" and st3["cached_tokens"] >= len(p2) + len(o2) - 2,
          f"cont turn 3: MTP from the live state an MTP run left ({st3['cached_tokens']})")
    p4, o4, st4 = run([sysmsg, {"role": "user", "content": "other conversation"}])
    b = s.system_boundary(p4, im_end)
    check(st4["path"] == "mtp" and b > 0 and st4["cached_tokens"] == b, "cont: new conversation, system checkpoint")
    # a step-path turn (e.g. top_k > 32) in between keeps the head in step: the next MTP turn reuses its live state
    m4 = [sysmsg, {"role": "user", "content": "other conversation"}, reply_msgs(o4, True),
          {"role": "user", "content": "LONGTEST please"}]
    p5, o5, st5 = run(m4, use=False)
    m5 = m4 + [reply_msgs(o5, True), {"role": "user", "content": "and after that"}]
    p6, o6, st6 = run(m5)
    check(st5["path"] == "step" and st6["path"] == "mtp" and st6["cached_tokens"] >= len(p5) + len(o5) - 2,
          f"cont: step turn, then MTP from its live state ({st6['cached_tokens']})")
    # a stale head (state changed outside the MTP calls): the turn falls back to the step path, the reuse kept
    mb.head = False
    m6 = m5 + [reply_msgs(o6, True), {"role": "user", "content": "stale"}]
    p8, o8, st8 = run(m6)
    check(st8["path"] == "step" and st8["cached_tokens"] >= len(p6) + len(o6) - 2,
          f"cont: stale head -> step path, reuse kept ({st8['path']}, {st8['cached_tokens']})")
    mb.head = True  # (the engine re-syncs a stale head only through reset / restore: the mock's head rows follow)
    mb.hrows = mb.fed[1:]
    # a cancelled MTP run (stop string / disconnect) ends after the round; state and kv still consistent
    p7, o7, st7 = run(m6 + [reply_msgs(o8, True), {"role": "user", "content": "LONGTEST again"}], cancel_after=2)
    check(st7["path"] == "mtp" and 1 < len(o7) <= 9 and st7["cached_tokens"] >= len(p8) + len(o8) - 2,
          f"cont: cancelled after 2 rounds ({len(o7)} tokens)")
    mb.head = True
    check(mb.stats["resets"] <= 2, f"cont: no resets beyond the first turns ({mb.stats['resets']})")
    s.mtp_policy = "auto"
    s.rates = B.Rates()  # the mock's 0-ms decode made the step rate meaningless
    check(s.choose_mtp(20000, 19000, 10 ** 6, True), "cont auto: long prompt with reuse -> MTP")
    check(not s.choose_mtp(20000, 19000, 10 ** 6, False), "cont auto: MTP not allowed -> step")
    check(B.pieces(5000, table=B.MTP_PIECES) == [4096, 904], f"MTP feed pieces {B.pieces(5000, table=B.MTP_PIECES)}")


def test_sched():
    """slot scheduler (lab-sched): an idle resident conversation on the other slot does not stop MTP; a request
    arriving during an MTP run makes it yield at the round boundary (both then step, interleaved) and MTP resumes when
    the other finishes; outputs = each request alone; slot choice = prefix holder, else empty, else least resident"""
    tpl = chat.Template(PACK, TOK, "xhigh")
    mbs = [B.MockBackend(TOK, ctx=16384), B.MockBackend(TOK, ctx=16384)]
    s = B.Session(mbs, 16384, max_ckpts=4, mtp_policy="always", sys_ckpt_min=8, log=lambda *a: None)
    sysm = {"role": "system", "content": "Scheduler test system prompt. " * 20}

    def enc(q):
        text, _ = tpl.render({"messages": [sysm, {"role": "user", "content": q}]})
        return tpl.encode(text)

    def alone(p):
        ref = B.MockBackend(TOK, ctx=16384)
        r = []
        B.Session(ref, 16384, mtp_policy="never", log=lambda *a: None).generate(
            p, 1000, (0.0, 1.0, 1, 0), False, r.extend, lambda: False, tpl.im_end)
        return r

    active = []    # [i, slot, gen] as the worker's list
    arrived = [False]

    def contended(sl):
        return any(e[1] is not sl for e in active) or (arrived[0] and any(not x.busy for x in s.slots if x is not sl))

    def start(i, p, on):
        sl, cand = s.pick(p)
        g = s.run(sl, cand, p, 1000, (0.0, 1.0, 1, 0), True, on, lambda: False, tpl.im_end,
                  contended=lambda sl=sl: contended(sl))
        active.append([i, sl, g])
        return sl

    def drain(stats, hook=None):
        while active:
            for e in list(active):
                try:
                    next(e[2])
                except StopIteration as x:
                    stats[e[0]] = x.value
                    active.remove(e)
            if hook:
                hook()

    # 1) X runs alone and stays resident (idle) on slot 0; Y (long) alone on the other slot: MTP, not the step path
    px, py = enc("short question X"), enc("conversation Y: LONGTEST")
    st = {}
    ox, oy = [], []
    start(0, px, ox.extend)
    drain(st)
    sly = start(1, py, oy.extend)
    drain(st)
    check(st[0]["slot"] == 0 and sly.i == 1, f"new conversation next to a resident one -> the empty slot ({sly.i})")
    check(st[1]["path"] == "mtp" and mbs[1].stats["mtp_runs"] == 1,
          f"one active + one idle resident conversation -> MTP ({st[1]['path']})")
    check(oy == alone(py) and mbs[1].fed == py + oy[:-1], "Y: output = alone, state = prompt + output[:-1]")

    # 2) A (long, MTP) runs alone; B arrives after ~30 tokens -> A yields at the round boundary, both step
    # interleaved; B finishes later, A finished first -> B resumes MTP
    pa, pb = enc("conversation Y: LONGTEST") , enc("request B: LONGTEST please")
    oa, ob, order = [], [], []

    def on_a(t):
        oa.extend(t)
        order.append(0)
        if len(oa) >= 30 and not began[0]:
            arrived[0] = True     # B is in the queue: the next round boundary hands over
    st = {}
    sla = start(0, pa, on_a)
    check(sla.i == 1, f"A continues where its prefix is resident (slot {sla.i})")
    began = [False]
    at_b = []

    def admit_b():
        if arrived[0] and not began[0]:
            began[0] = True
            arrived[0] = False
            at_b.append(len(oa))
            slb = start(1, pb, lambda t: (ob.extend(t), order.append(1)))
            check(slb.i == 0, f"B takes the other slot ({slb.i})")
    drain(st, admit_b)
    switches = sum(1 for a, b in zip(order, order[1:]) if a != b)
    check(began[0] and 30 <= at_b[0] <= 33 and mbs[sla.i].stats["mtp_runs"] >= 1,
          f"A ran MTP alone until B arrived, then yielded at the round boundary (A had {at_b and at_b[0]} tokens)")
    check(st[0]["path"] == "mtp+step" and st[1]["path"] == "mtp+step" and st[1].get("mtp_resumes", 0) >= 1,
          f"A: MTP then step ({st[0]['path']}); B: step then MTP resumed after A ({st[1]['path']}, "
          f"{st[1].get('mtp_resumes', 0)} resumes, {st[1].get('mtp_tokens', 0)} MTP tokens)")
    check(switches > 50, f"two active -> token-level interleaving ({switches} switches)")
    check(oa == alone(pa) and ob == alone(pb), f"outputs identical to each request alone ({len(oa)}, {len(ob)} tokens)")
    for i, p, o in ((0, pa, oa), (1, pb, ob)):
        mb = mbs[st[i]["slot"]]
        check(mb.fed == p + o[:-1], f"stream {i}: engine state == prompt + output[:-1]")

    # 3) slot choice: a new conversation with nothing reusable displaces the slot holding fewer tokens
    nt = [len(sl.conv.toks) for sl in s.slots]
    sl, c = s.pick(enc("brand new"), reuse=False)
    check(sl.i == nt.index(min(nt)), f"no reuse -> least resident state (slot {sl.i}, resident {nt})")


def test_models():
    """server/models.json resolution (lab-swift): first complete pack wins, partial T6 packs are skipped, --pack wins."""
    import tempfile
    import ne_server as S
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)

        def mk(name, t6_layers=None, mixer=True):
            d = t / name
            d.mkdir()
            (d / "tokenizer.json").write_text("{}")
            (d / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 48}}))
            meta = {} if t6_layers is None else {"trunk6": {"layers": list(range(t6_layers)), "mixer_bf16": mixer}}
            (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": meta, "weight_map": {}}))
            return str(d)
        k5, t6p, t6f = mk("k5"), mk("t6part", 4), mk("t6full", 48)
        reg = t / "models.json"
        reg.write_text(json.dumps({"default": "a", "models": {"a": {"packs": [t6p, k5]}, "b": {"packs": [t6f, k5]},
                                                              "c": {"packs": [str(t / "missing")]}}}))
        check(S.resolve_model(None, None, reg)[:2] == ("a", k5), "default id; partial T6 pack skipped -> fallback")
        check(S.resolve_model("b", None, reg)[1] == t6f, "complete T6 pack is the first choice")
        check(S.resolve_model("x", "/some/pack", reg)[:2] == ("x", "/some/pack"), "--pack overrides the registry")
        for mid in ("c", "zzz"):
            try:
                S.resolve_model(mid, None, reg)
                check(False, f"id {mid} refused")
            except SystemExit:
                check(True, f"id {mid} refused (no complete pack / unknown)")
        real = json.loads(S.MODELS.read_text())
        check({"Qwen3.8-Flash-Next-native", "Swift-1.5-Flash-Next-native"} <= set(real["models"]),
              "server/models.json has the base and Swift ids")


class _Quiet:
    """swallow stderr (the expected tracebacks of an injected failure)"""
    def __enter__(self):
        self.e, sys.stderr = sys.stderr, open("/dev/null", "w")

    def __exit__(self, *a):
        sys.stderr.close()
        sys.stderr = self.e


def test_post_reply_failures():
    """the SSD write-through or the budget failing after a finished reply (MemoryError copying the KV rows out ...) does
    not turn the reply into an error (lab-fixsrv); the next turn works and the mirror catches up"""
    import glob
    import shutil
    import tempfile
    import prefix_cache as PC
    tpl = chat.Template(PACK, TOK, "xhigh")
    d = tempfile.mkdtemp(prefix="ne-postfail-test")
    logs = []
    for use in (False, True):
        shutil.rmtree(d, ignore_errors=True)
        mb = B.MockBackend(TOK, ctx=16384)
        s = B.Session(mb, 16384, max_ckpts=4, mtp_policy="always" if use else "never", sys_ckpt_min=8, log=logs.append,
                      cache_ram=1 << 30, disk_dir=d, disk_bytes=1 << 30, ident={"format": PC.FORMAT, "test": "post"})
        run = _conv_runner(s, [mb], tpl, use)
        msgs = [{"role": "system", "content": "System prompt. " * 40}, {"role": "user", "content": "question one"}]
        p0, o0, st0 = run(msgs)
        s.cache.disk.flush()
        # write-through raises: the reply is intact
        orig = mb.kv_save
        mb.kv_save = lambda *a, **k: (_ for _ in ()).throw(MemoryError("cannot allocate"))
        msgs = _next_turn(msgs, o0, "question two " + "pad " * 40)
        logs.clear()
        with _Quiet():
            p1, o1, st1 = run(msgs)
        mb.kv_save = orig
        ref = []
        B.Session(B.MockBackend(TOK, ctx=16384), 16384, mtp_policy="never", log=lambda *a: None).generate(
            p1, 1000, (1.0, 0.95, 20, 0), False, ref.extend, lambda: False, tpl.im_end)
        check(o1 == ref and st1["completion_tokens"] == len(o1) and not s.slots[0].busy
              and any("write-through failed" in x and "MemoryError" in x for x in logs),
              f"kv_save raising MemoryError after the reply (use_mtp {use}): the reply is intact ({len(o1)} tokens), logged")
        # the budget raising
        orig_enf = s.cache.enforce
        s.cache.enforce = lambda *a, **k: (_ for _ in ()).throw(OSError("disk gone"))
        msgs = _next_turn(msgs, o1, "question three " + "pad " * 40)
        logs.clear()
        with _Quiet():
            p2, o2, st2 = run(msgs)
        s.cache.enforce = orig_enf
        check(st2["completion_tokens"] == len(o2) and o2 and any("enforcing the budget failed" in x for x in logs),
              f"cache.enforce raising after the reply (use_mtp {use}): the reply is intact, logged")
        # and the conversation goes on, the mirror caught up
        msgs = _next_turn(msgs, o2, "question four")
        p3, o3, st3 = run(msgs)
        s.cache.disk.flush()
        check(st3["cached_tokens"] > st2["prompt_tokens"] - 50 and glob.glob(f"{d}/*/e*/manifest.json"),
              f"the next turn reuses the state ({st3['cached_tokens']} cached) and the SSD mirror exists")
        s.cache.close()
    shutil.rmtree(d, ignore_errors=True)


def test_persist():
    """the SSD tier survives a restart (lab-persist): a new Session on the same directory and identity restores every
    conversation's last checkpoint from disk (fresh mock engine: its KV rows come only from the files); another
    identity reuses nothing and keeps the entries; a corrupted file is rejected (that turn prefills cold, output
    unchanged); crash leftovers and a damaged manifest are cleaned at start; the budget holds; a second tier on the
    same directory gets a private one"""
    import glob
    import shutil
    import tempfile
    import prefix_cache as PC
    tpl = chat.Template(PACK, TOK, "xhigh")
    d = tempfile.mkdtemp(prefix="ne-persist-test")
    ID = {"format": PC.FORMAT, "test": "A"}
    logs = []

    def sess(ident=ID, budget=1 << 30, ram=1 << 30, use=False):
        mb = B.MockBackend(TOK, ctx=16384)
        s = B.Session(mb, 16384, max_ckpts=4, mtp_policy="always" if use else "never", sys_ckpt_min=8,
                      log=logs.append, cache_ram=ram, disk_dir=d, disk_bytes=budget, ident=ident)
        return s, mb

    def alone(p):
        ref = []
        B.Session(B.MockBackend(TOK, ctx=16384), 16384, mtp_policy="never", log=lambda *a: None).generate(
            p, 1000, (1.0, 0.95, 20, 0), False, ref.extend, lambda: False, tpl.im_end)
        return ref

    for use in (False, True):
        shutil.rmtree(d, ignore_errors=True)
        s1, mb1 = sess(use=use)
        convs = [[{"role": "system", "content": f"System prompt of conversation {c}. " * 30},
                  {"role": "user", "content": f"question 1 of conversation {c}"}] for c in range(3)]
        run = _conv_runner(s1, [mb1], tpl, use)
        last = {}
        for t in range(2):
            for c in range(3):
                p, o, st = run(convs[c])
                last[c] = p
                convs[c] = _next_turn(convs[c], o, f"question {t + 2} of conversation {c} " + "pad " * 40)
        check(s1.cache.close(), "every SSD write landed at close")
        n_ent = len(glob.glob(f"{d}/*/e*/manifest.json"))
        check(n_ent >= 3, f"{n_ent} conversations on disk after the first server ({'MTP' if use else 'step'})")
        # restart: a new engine (no rows, no state) and a new session on the same directory
        s2, mb2 = sess(use=use)
        run2 = _conv_runner(s2, [mb2], tpl, use)
        ok = True
        for c in range(3):
            p, o, st = run2(convs[c])
            ok &= st["cached_tokens"] == s2.last_start(last[c]) and o == alone(p)
        check(ok and s2.cache.stats["disk_hits"] == 3 and mb2.stats["kv_loaded"] > 0,
              f"after a restart every conversation restores its last checkpoint from SSD, outputs = alone "
              f"({s2.cache.stats['disk_hits']} SSD hits, {mb2.stats['kv_loaded']} rows loaded"
              + (f", {mb2.stats['head_restores']} head restores)" if use else ")"))
        if use:
            check(mb2.stats["head_restores"] >= 3, "MTP head restored from SSD (the turn continues on MTP)")
        s2.cache.close()
    # another build / pack: nothing reused, the entries are kept
    s3, mb3 = sess(ident={"format": PC.FORMAT, "test": "B"})
    p, o, st = _conv_runner(s3, [mb3], tpl)(convs[0])
    check(st["cached_tokens"] == 0 and len(glob.glob(f"{d}/*/")) == 2,
          f"another identity: no reuse (cached {st['cached_tokens']}), the first identity's directory is kept")
    s3.cache.close()
    # damage: flip one byte of a KV segment of conversation 0's entry, leave crash debris in conversation 1's
    s4, _ = sess()
    ents = {}
    for e in s4.cache.disk.entries.values():
        for c in range(3):
            if f"conversation {c}." in TOK.decode(e.toks[:120]):
                ents.setdefault(c, e)
    s4.cache.close()
    seg = sorted(ents[0].dir.glob("seg*.bin"))[0]
    raw = bytearray(seg.read_bytes())
    raw[len(raw) // 2] ^= 0x40
    seg.write_bytes(bytes(raw))
    (ents[1].dir / "seg999.bin.tmp").write_bytes(b"partial")
    (ents[1].dir / "snap998.bin").write_bytes(b"orphan")
    (ents[2].dir / "manifest.json").write_text('{"magic": "native-engine prefix cache", "fo')
    logs.clear()
    s5, mb5 = sess()
    run5 = _conv_runner(s5, [mb5], tpl)
    check(not (ents[1].dir / "seg999.bin.tmp").exists() and not (ents[1].dir / "snap998.bin").exists()
          and not ents[2].dir.exists(), "start: crash debris deleted, the entry with a damaged manifest dropped")
    p0, o0, st0 = run5(convs[0])
    p1, o1, st1 = run5(convs[1])
    check(st0["cached_tokens"] == 0 and o0 == alone(p0) and s5.cache.disk.stats["rejected"] == 1
          and any("checksum mismatch" in x for x in logs),
          f"corrupted segment rejected: cold prefill, output = alone (cached {st0['cached_tokens']})")
    check(st1["cached_tokens"] > 0 and o1 == alone(p1), f"the entry next to the debris restores ({st1['cached_tokens']})")
    # a second server on the same directory: private tier, the shared one untouched
    s6, _ = sess()
    check(s6.cache.disk.private and s6.cache.disk.root.name.startswith("private-"), "second tier -> private directory")
    s6.cache.close()
    check(not s6.cache.disk.root.exists(), "the private directory is removed at close")
    s5.cache.close()
    # budget: ~2 conversations' bytes -> LRU eviction, other identities' directories first
    one = max(e.bytes for e in s5.cache.disk.entries.values())
    s7, mb7 = sess(budget=int(one * 1.5))
    s7.cache.disk.flush()
    check(s7.cache.disk.bytes() <= one * 1.5 and len(glob.glob(f"{d}/*/")) == 1,
          f"budget {int(one * 1.5)} B: {s7.cache.disk.bytes()} B kept, the other identity's directory removed first")
    run7 = _conv_runner(s7, [mb7], tpl)
    for c in range(3):
        run7(convs[c] + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "more " * 50}])
    s7.cache.disk.flush()
    on_disk = sum(f.stat().st_size for f in Path(d).rglob("*") if f.is_file() and f.suffix == ".bin")
    check(s7.cache.disk.bytes() <= one * 1.5 and on_disk <= one * 1.5 + 4096,
          f"after more turns: record {s7.cache.disk.bytes()} B, files {on_disk} B <= budget")
    s7.cache.close()
    shutil.rmtree(d, ignore_errors=True)



def test_shared_prefix():
    """lab-snap (docs/server.md 3f): sessions that share the tools block + most of the system prompt (a per-session
    tail differs: the system checkpoint never matches) restore a shared-prefix snapshot: the second session the tools
    block's end (captured during the first one's cold prefill), the third on the learned boundary (the prefix the
    first two share, captured during the second's prefill); outputs = a cold prefill; after a restart from the SSD
    tier; the snapshot is never adopted (stays the same size); the conversation's own later turns are unchanged; off
    = no reuse; the RAM / SSD caps hold"""
    import shutil
    import tempfile
    tpl = chat.Template(PACK, TOK, "xhigh")
    mk = [TOK.encode("</IMPORTANT>", add_special_tokens=False).ids]
    tools = [{"type": "function", "function": {"name": f"tool_{i}", "description": f"Does thing {i}. " * 20,
              "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "a path " * 10}}}}}
             for i in range(25)]
    base = "You are an agent working in a terminal. " * 120
    d = tempfile.mkdtemp(prefix="ne-shared-test")

    def sess(use, shared=True, disk=True, ram_n=None):
        mb = B.MockBackend(TOK, ctx=32768)
        s = B.Session(mb, 32768, max_ckpts=4, mtp_policy="always" if use else "never", log=lambda *a: None,
                      disk_dir=d if disk else None, disk_bytes=4 << 30, markers=mk, shared=shared)
        if ram_n is not None:
            s.cache.max_shared = ram_n
        return s, mb

    def msgs(tag, q=None):
        return [{"role": "system", "content": base + f"\n\nEnvironment: session {tag}, started 2026-10-{tag}"},
                {"role": "user", "content": q or f"question of session {tag} " * 30}]

    def alone(p, use):
        ref = []
        B.Session(B.MockBackend(TOK, ctx=32768), 32768, mtp_policy="always" if use else "never",
                  log=lambda *a: None, shared=False).generate(p, 1000, (1.0, 0.95, 20, 0), use, ref.extend,
                                                              lambda: False, tpl.im_end)
        return ref

    for use in (False, True):
        shutil.rmtree(d, ignore_errors=True)
        s, mb = sess(use)
        run = _conv_runner(s, [mb], tpl, use)
        pa, oa, sa = run(msgs("11"), tools=tools)
        tb = s.struct_boundary(pa, tpl.im_end)
        sysb = s.system_boundary(pa, tpl.im_end)
        check(0 < tb < sysb and sa["cached_tokens"] == 0 and sa.get("shared_made") == 1,
              f"[{'mtp' if use else 'step'}] cold session A: tools-block snapshot at {tb} (system ends at {sysb})")
        run([{"role": "system", "content": "short"}, {"role": "user", "content": "title " * 20}])  # in between
        pb, ob, sb = run(msgs("12"), tools=tools)
        L = lcp(pa, pb)
        x = (L - B.SHARED_MARGIN) // B.SHARED_ALIGN * B.SHARED_ALIGN
        check(sb["cached_tokens"] == tb and sb["shared_src"] and sb.get("shared_made") == 1 and ob == alone(pb, use),
              f"session B restores the tools-block snapshot ({sb['cached_tokens']}), captures the learned boundary, "
              f"output = cold")
        pc, oc, sc = run(msgs("13"), tools=tools)
        check(sc["cached_tokens"] == x and x > sysb - 64 and sc["shared_src"] and not sc.get("shared_made")
              and oc == alone(pc, use), f"session C restores the learned snapshot at {sc['cached_tokens']} "
              f"(sessions share {L} tokens), output = cold")
        if use:
            check(sc["path"] == "mtp" and mb.stats["head_restores"] > 0, "MTP continues after the shared restore")
        sh = sorted(len(c.toks) for c in s.cache.shared())
        check(sh == [tb, x] and all(c.slot is None for c in s.cache.shared()),
              f"shared snapshots are never adopted or extended ({sh})")
        s1, s2 = sorted(s.cache.shared(), key=lambda c: len(c.toks))
        check(len(s2.segs) == 2 and s2.segs[0][2] is s1.segs[0][2] and s2.segs[1][:2] == [tb, x],
              "the learned snapshot reuses the tools-block snapshot's KV copy (rows [0, tb) once in RAM)")
        mc = _next_turn(msgs("13"), oc, "a follow-up question")
        pc2, oc2, sc2 = run(mc, tools=tools)
        check(sc2["cached_tokens"] == s.last_start(pc) and not sc2.get("shared_made"),
              f"session C's next turn reuses its own checkpoint ({sc2['cached_tokens']})")
        check(s.cache.close(), "SSD writes landed")
        s2, mb2 = sess(use)
        run2 = _conv_runner(s2, [mb2], tpl, use)
        pd, od, sd = run2(msgs("14"), tools=tools)
        check(sd["cached_tokens"] == x and sd["cache_src"] == "ssd" and sd["shared_src"] and od == alone(pd, use),
              f"after a restart session D restores the learned snapshot from SSD ({sd['cached_tokens']})")
        s2.cache.close()
    # off: as before (the system checkpoint does not match: cold)
    shutil.rmtree(d, ignore_errors=True)
    s, mb = sess(False, shared=False, disk=False)
    run = _conv_runner(s, [mb], tpl)
    run(msgs("11"), tools=tools)
    _, _, sb = run(msgs("12"), tools=tools)
    check(sb["cached_tokens"] == 0 and not s.cache.shared(), "NE_SHARED_SNAPS off: session B prefills cold")
    # caps: one shared snapshot in RAM (the others go to the SSD tier), two on disk
    s, mb = sess(False, ram_n=1)
    s.cache.disk.shared_cap = 2
    run = _conv_runner(s, [mb], tpl)
    for i, t in enumerate(("Alpha", "Beta", "Gamma")):  # three distinct tool sets: three tools-block snapshots
        tl = [dict(x, function=dict(x["function"], name=f"{t}_{x['function']['name']}")) for x in tools]
        run(msgs(f"2{i}"), tools=tl)
    s.cache.disk.flush()
    nd = sum(1 for e in s.cache.disk.entries.values() if e.shared)
    check(len(s.cache.shared()) == 1 and nd == 2, f"caps: {len(s.cache.shared())} shared in RAM, {nd} on SSD")
    s.cache.close()
    shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    test_models()
    test_parser()
    test_cut_calls()
    test_stop()
    test_template()
    test_detok()
    test_session()
    test_session_cont()
    test_cancel_prefill()
    test_multi()
    test_multi_mtp()
    test_concurrent()
    test_concurrent_mask()
    test_random_mix()
    test_sched()
    test_persist()
    test_shared_prefix()
    test_post_reply_failures()
    print(f"\n{len(FAILS)} failed" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)
