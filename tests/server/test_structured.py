"""HTTP test of the structured-output features of the Ijah server (lab-api), stdlib + jsonschema. Against a
server that is up (mock engine or the real model):

    $PY tests/server/test_structured.py --url http://127.0.0.1:8141 --key-file KEY [--mock] [--json out.json]

Checks: /v1/models capabilities; response_format json_schema on five schemas (nested objects, enums, arrays, nullable,
bounded arrays) with thinking off, two of them with thinking on, one streamed, the answers validate against the schemas
and carry the content the prompt asked for; json_object; bad response_format / schema -> 400; max_tokens cutting a
document -> finish_reason length; tool_choice forced (exactly the named function, valid arguments, streamed and not) and
required (a call although the prompt gives no reason); tool_choice auto stays unconstrained; constrained requests report
timings.grammar and take the step path (the MTP loop has no mask); unconstrained requests report no grammar.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))
import refenv  # noqa: E402

refenv.add_paths()
import jsonschema  # noqa: E402

from schemas import CASES, WEATHER_TOOLS  # noqa: E402
from test_client import FAILS, Client, check, merge  # noqa: E402


def rf(schema, name="answer"):
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}


def parse(content):
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8141")
    ap.add_argument("--key-file", default=str(Path.home() / ".config/ijah/api-key"))
    ap.add_argument("--mock", action="store_true", help="the mock engine: person schema + scripted prompts only")
    ap.add_argument("--bench", type=int, default=0, help="tokens of an unconstrained speed sample (step and MTP)")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    key = Path(a.key_file).expanduser().read_text().strip()
    cl = Client(a.url, key)
    res = {"cases": {}}

    st, m = cl.req("GET", "/v1/models")
    mod = m["data"][0] if st == 200 else {}
    caps = mod.get("capabilities") or []
    check(st == 200 and all(c in caps for c in ("chat", "tool_use", "streaming", "reasoning", "json_schema")),
          f"/v1/models capabilities {caps}, input_modalities {mod.get('input_modalities')}")
    check("text" in (mod.get("input_modalities") or []), "/v1/models input_modalities has text")
    model = mod.get("id", "x")
    res["capabilities"] = caps

    cases = CASES[:1] if a.mock else CASES
    # -- five schemas, thinking off
    for c in cases:
        prompt = "JSONTEST please" if a.mock else c["prompt"]
        t0 = time.perf_counter()
        st, r = cl.chat({"model": model, "reasoning_effort": "none", "response_format": rf(c["schema"], c["name"]),
                         "messages": [{"role": "user", "content": prompt}], "max_tokens": 1500})
        ch = r["choices"][0] if st == 200 else {}
        msg = ch.get("message", {})
        doc = parse(msg.get("content"))
        valid = doc is not None and jsonschema.Draft202012Validator(c["schema"]).is_valid(doc)
        tim = r.get("timings", {}) if st == 200 else {}
        check(st == 200 and valid and ch.get("finish_reason") == "stop",
              f"{c['name']}: schema-valid JSON, finish stop ({r['usage']['completion_tokens'] if st == 200 else '-'} tokens, "
              f"{time.perf_counter() - t0:.1f} s, {tim.get('decode_tok_s')} tok/s)")
        check(tim.get("grammar") == "schema" and tim.get("path") == "step", f"{c['name']}: timings.grammar {tim.get('grammar')}, path {tim.get('path')}")
        if not a.mock and valid:
            ok = False
            try:
                ok = bool(c["expect"](doc))
            except Exception:   # noqa: BLE001
                pass
            check(ok, f"{c['name']}: the content answers the prompt: {json.dumps(doc)[:110]}")
        if not valid:
            print("   content:", repr(msg.get("content"))[:400], "| error:", r if st != 200 else "")
        res["cases"][c["name"]] = {"valid": valid, "timings": tim, "content": msg.get("content")}

    # -- thinking on (reasoning first, then the document)
    for c in cases[:1] if a.mock else [CASES[1], CASES[3]]:
        prompt = "JSONTEST please" if a.mock else c["prompt"]
        st, r = cl.chat({"model": model, "reasoning_effort": "low", "response_format": rf(c["schema"], c["name"]),
                         "messages": [{"role": "user", "content": prompt}], "max_tokens": 6000})
        msg = r["choices"][0]["message"] if st == 200 else {}
        doc = parse(msg.get("content"))
        valid = doc is not None and jsonschema.Draft202012Validator(c["schema"]).is_valid(doc)
        check(st == 200 and valid and msg.get("reasoning_content")
              and "think>" not in (msg.get("content") or "") and r["choices"][0]["finish_reason"] == "stop",
              f"{c['name']} with thinking: {len(msg.get('reasoning_content') or '')} chars of reasoning, then "
              f"schema-valid JSON ({r['usage']['completion_tokens_details']['reasoning_tokens'] if st == 200 else '-'} reasoning tokens)")
        if not valid:
            print("   content:", repr(msg.get("content"))[:400], "| error:", r if st != 200 else "")

    # -- streamed
    c = cases[0]
    chunks, done, first, wall = cl.stream({"model": model, "reasoning_effort": "none", "response_format": rf(c["schema"]),
                                           "messages": [{"role": "user", "content": "JSONTEST" if a.mock else c["prompt"]}],
                                           "max_tokens": 1500})
    content, _, calls, finish, _ = merge(chunks)
    doc = parse(content)
    check(done and finish == "stop" and doc is not None and jsonschema.Draft202012Validator(c["schema"]).is_valid(doc)
          and len(chunks) > 5, f"streamed json_schema: {len(chunks)} chunks, valid, finish {finish}")

    # -- json_object
    st, r = cl.chat({"model": model, "reasoning_effort": "none", "response_format": {"type": "json_object"},
                     "messages": [{"role": "user", "content": "JSONTEST" if a.mock else
                                   "List three primary colours as JSON with a key 'colours'."}], "max_tokens": 600})
    msg = r["choices"][0]["message"] if st == 200 else {}
    doc = parse(msg.get("content"))
    check(st == 200 and isinstance(doc, dict) and (a.mock or isinstance(doc.get("colours", doc.get("colors")), list)),
          f"json_object: {json.dumps(doc)[:100]}")

    # -- bad requests
    for what, body in (("unknown response_format type", {"response_format": {"type": "xml"}}),
                       ("json_schema without schema", {"response_format": {"type": "json_schema", "json_schema": {}}}),
                       ("unsupported schema type", {"response_format": rf({"type": "nonsense"})}),
                       ("tool_choice naming a missing tool", {"tools": WEATHER_TOOLS, "tool_choice": {
                           "type": "function", "function": {"name": "nope"}}})):
        st, e = cl.chat(dict({"model": model, "messages": [{"role": "user", "content": "hi"}]}, **body))
        check(st == 400 and "error" in e, f"400 for {what}: {(e.get('error') or {}).get('message', '')[:70] if st == 400 else st}")

    # -- the document cut by max_tokens
    st, r = cl.chat({"model": model, "reasoning_effort": "none", "response_format": rf(cases[0]["schema"]),
                     "messages": [{"role": "user", "content": "JSONTEST" if a.mock else cases[0]["prompt"]}],
                     "max_tokens": 12})
    check(st == 200 and r["choices"][0]["finish_reason"] == "length" and parse(r["choices"][0]["message"]["content"]) is None,
          f"max_tokens 12 cuts the document: finish_reason {r['choices'][0]['finish_reason'] if st == 200 else st}")

    # -- tool_choice forced / required / auto
    name = "get_weather" if a.mock else "get_time"
    other = "get_time" if a.mock else "get_weather"
    q_force = "FORCETEST" if a.mock else "What's the weather in Paris for the next 3 days?"   # forced to the other tool
    forced = {"type": "function", "function": {"name": name}}
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in WEATHER_TOOLS}
    for effort in ("none", "low"):
        st, r = cl.chat({"model": model, "reasoning_effort": effort, "tools": WEATHER_TOOLS, "tool_choice": forced,
                         "messages": [{"role": "user", "content": q_force}], "max_tokens": 4000})
        ch = r["choices"][0] if st == 200 else {}
        calls = (ch.get("message") or {}).get("tool_calls") or []
        args = parse(calls[0]["function"]["arguments"]) if calls else None
        check(st == 200 and len(calls) == 1 and calls[0]["function"]["name"] == name and ch.get("finish_reason") == "tool_calls"
              and args is not None and jsonschema.Draft202012Validator(schemas[name]).is_valid(args),
              f"forced {name} (effort {effort}): {json.dumps(calls[0]['function']) if calls else r}"[:230])
        tim = r.get("timings", {}) if st == 200 else {}
        check(tim.get("grammar") == "forced" and tim.get("path") == "step", f"forced: timings.grammar {tim.get('grammar')}, path {tim.get('path')}")
    chunks, done, first, wall = cl.stream({"model": model, "reasoning_effort": "none", "tools": WEATHER_TOOLS,
                                           "tool_choice": forced, "messages": [{"role": "user", "content": q_force}]})
    content, _, calls, finish, _ = merge(chunks)
    args = parse(calls[0]["arguments"]) if calls else None
    check(done and finish == "tool_calls" and len(calls) == 1 and calls[0]["name"] == name and args is not None
          and "<tool_call>" not in content, f"forced, streamed: {calls}"[:200])

    # a constrained turn (step path, token mask) keeps the MTP head in step: the follow-up turn is unconstrained and
    # continues with MTP from the reused prefix (lab-mtpcont's continuation)
    msgs = [{"role": "user", "content": q_force}]
    st, r1 = cl.chat({"model": model, "reasoning_effort": "none", "tools": WEATHER_TOOLS, "tool_choice": forced,
                      "messages": msgs, "max_tokens": 400})
    calls = (r1["choices"][0]["message"].get("tool_calls") or []) if st == 200 else []
    if calls:
        msgs2 = msgs + [{"role": "assistant", "content": r1["choices"][0]["message"].get("content") or "", "tool_calls": calls},
                        {"role": "tool", "tool_call_id": calls[0]["id"], "content": json.dumps({"result": "ok, 12:00"})}]
        st, r2 = cl.chat({"model": model, "reasoning_effort": "none", "tools": WEATHER_TOOLS, "messages": msgs2,
                          "ne_path": "mtp", "max_tokens": 60})
        tim = r2.get("timings", {}) if st == 200 else {}
        cached = r2.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0) if st == 200 else 0
        check(st == 200 and tim.get("path") == "mtp" and "grammar" not in tim and cached > 0,
              f"unconstrained follow-up after a forced call: MTP continues from the reused prefix (path {tim.get('path')}, "
              f"{cached} cached tokens)")
    else:
        check(False, "forced call for the follow-up turn")

    st, r = cl.chat({"model": model, "reasoning_effort": "none", "tools": WEATHER_TOOLS, "tool_choice": "required",
                     "messages": [{"role": "user", "content": "FORCETEST" if a.mock else "Hello! How are you today?"}],
                     "max_tokens": 4000})
    ch = r["choices"][0] if st == 200 else {}
    calls = (ch.get("message") or {}).get("tool_calls") or []
    ok = bool(calls) and ch.get("finish_reason") == "tool_calls"
    for cl_ in calls:
        a_ = parse(cl_["function"]["arguments"])
        ok &= cl_["function"]["name"] in schemas and a_ is not None and \
            jsonschema.Draft202012Validator(schemas[cl_["function"]["name"]]).is_valid(a_)
    check(st == 200 and ok, f"required: a call although the prompt gives no reason: "
                            f"{[c['function'] for c in calls]}"[:200])
    check((r.get("timings") or {}).get("grammar") == "required", f"required: timings.grammar {(r.get('timings') or {}).get('grammar')}")

    st, r = cl.chat({"model": model, "reasoning_effort": "none", "tools": WEATHER_TOOLS, "tool_choice": "auto",
                     "messages": [{"role": "user", "content": "FORCETEST" if a.mock else "Say hello in one word."}],
                     "max_tokens": 200})
    tim = r.get("timings", {}) if st == 200 else {}
    check(st == 200 and "grammar" not in tim, f"tool_choice auto: unconstrained (path {tim.get('path')})")

    # -- unconstrained speed sample
    if a.bench:
        q = "LONGTEST" if a.mock else ("Write a detailed, long essay about the history of the printing press, its "
                                       "technology and its effects on Europe.")
        for path in ("step", "mtp"):
            chunks, _, first, wall = cl.stream({"model": model, "reasoning_effort": "none", "max_tokens": a.bench,
                                                "ne_path": path, "temperature": 1.0, "seed": 7,
                                                "messages": [{"role": "user", "content": q}],
                                                "stream_options": {"include_usage": True}})
            _, _, _, _, usage = merge(chunks)
            tim = next((ch.get("timings") for ch in chunks if ch.get("timings")), {}) or {}
            print(f"bench {path}: {usage['completion_tokens']} tokens; engine {tim.get('decode_tok_s')} tok/s "
                  f"({tim.get('path')}); grammar {tim.get('grammar')}")
            check("grammar" not in tim, f"unconstrained {path} request reports no grammar")
            res[f"bench_{path}"] = {"tokens": usage["completion_tokens"], "engine": tim}
        # a constrained sample on the same request: the cost of the mask
        st, r = cl.chat({"model": model, "reasoning_effort": "none", "max_tokens": a.bench, "temperature": 1.0, "seed": 7,
                         "response_format": {"type": "json_object"},
                         "messages": [{"role": "user", "content": "Write a long JSON object listing 80 inventions with "
                                       "year and inventor, keys 'inventions' as array."}]})
        tim = r.get("timings", {}) if st == 200 else {}
        print(f"bench constrained json_object: {r['usage']['completion_tokens']} tokens; engine {tim.get('decode_tok_s')} tok/s "
              f"({tim.get('path')}, grammar {tim.get('grammar')})")
        res["bench_constrained"] = tim
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    print(f"\n{len(FAILS)} failed: {FAILS}" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
