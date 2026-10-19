"""Anthropic Messages API (/v1/messages) on top of the OpenAI-shaped core of the Ijah server (lab-sampler).

Pure CPU. `to_openai(req)` turns a Messages request into the chat-completions request the server already runs (system,
text / image / tool_use / tool_result / thinking blocks, tools, tool_choice, thinking, stop_sequences, sampling);
`message(...)` builds the non-streamed response and `Stream` the event sequence of a streamed one (message_start,
content_block_start / _delta / _stop, message_delta, message_stop) from the parser's deltas
(chat.OutputParser: ("reasoning", s) | ("content", s) | ("tool_start", idx, id, name) | ("tool_args", idx, s)).
Differences from Anthropic's service: no thinking unless `thinking.type` is "enabled" (the API's default) and the budget
only picks the reasoning effort; thinking blocks carry an empty signature and any signature is accepted back; server
tools (web search, bash ...) are refused; `input_tokens` excludes the reused prefix (`cache_read_input_tokens`).
"""
import json
import secrets

from chat import RequestError


def _text_of(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for b in c:
            if isinstance(b, str):
                out.append(b)
            elif isinstance(b, dict) and b.get("type") == "text":
                out.append(b.get("text", ""))
            else:
                raise RequestError("only text blocks are supported inside system / tool_result content")
        return "".join(out)
    if c is None:
        return ""
    raise RequestError("content must be a string or a list of blocks")


def _image_part(b):
    src = b.get("source")
    if not isinstance(src, dict):
        raise RequestError("image block without a source")
    if src.get("type") == "base64":
        return {"type": "image_url", "image_url": {"url": f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"}}
    if src.get("type") == "url" and isinstance(src.get("url"), str):
        return {"type": "image_url", "image_url": {"url": src["url"]}}
    raise RequestError("image source must be base64 or url")


def _effort(budget):
    if not isinstance(budget, (int, float)):
        return "xhigh"
    return "low" if budget <= 4096 else "medium" if budget <= 16384 else "xhigh"


def to_openai(req):
    """Anthropic Messages request -> chat.completions request (a new dict)"""
    msgs = req.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise RequestError("messages must be a non-empty list")
    if not isinstance(req.get("max_tokens"), int) or isinstance(req.get("max_tokens"), bool) or req["max_tokens"] < 1:
        raise RequestError("max_tokens is required (an integer >= 1)")
    out = []
    if req.get("system"):
        out.append({"role": "system", "content": _text_of(req["system"])})
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            raise RequestError("every message needs role user or assistant")
        c = m.get("content")
        if isinstance(c, str):
            out.append({"role": m["role"], "content": c})
            continue
        if not isinstance(c, list):
            raise RequestError("message content must be a string or a list of blocks")
        if m["role"] == "user":
            parts = []
            for b in c:
                t = b.get("type") if isinstance(b, dict) else None
                if t == "text":
                    parts.append({"type": "text", "text": b.get("text", "")})
                elif t == "image":
                    parts.append(_image_part(b))
                elif t == "tool_result":
                    if parts:   # text before a result: its own user turn
                        out.append({"role": "user", "content": parts})
                        parts = []
                    txt = _text_of(b.get("content"))
                    if b.get("is_error"):
                        txt = "Error: " + txt
                    out.append({"role": "tool", "tool_call_id": b.get("tool_use_id", ""), "content": txt})
                elif t in ("document", "search_result"):
                    raise RequestError(f"{t} blocks are not supported")
                else:
                    raise RequestError(f"unsupported user content block {t!r}")
            if parts:
                out.append({"role": "user", "content": parts})
        else:
            text, reasoning, calls = [], [], []
            for b in c:
                t = b.get("type") if isinstance(b, dict) else None
                if t == "text":
                    text.append(b.get("text", ""))
                elif t == "thinking":
                    reasoning.append(b.get("thinking", ""))
                elif t == "redacted_thinking":
                    pass
                elif t == "tool_use":
                    calls.append({"id": b.get("id") or "call_" + secrets.token_hex(12), "type": "function",
                                  "function": {"name": b.get("name", ""),
                                               "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)}})
                else:
                    raise RequestError(f"unsupported assistant content block {t!r}")
            am = {"role": "assistant", "content": "".join(text)}
            if reasoning:
                am["reasoning_content"] = "".join(reasoning)
            if calls:
                am["tool_calls"] = calls
            out.append(am)
    o = {"messages": out, "max_tokens": req["max_tokens"], "stream": bool(req.get("stream"))}
    if req.get("model") is not None:   # (the multi-model server routes by it; one model: ignored)
        o["model"] = req["model"]
    if req.get("tools"):
        tools = []
        for t in req["tools"]:
            if not isinstance(t, dict) or t.get("type") not in (None, "custom"):
                raise RequestError(f"tool type {t.get('type') if isinstance(t, dict) else t!r} is not supported "
                                   "(custom tools with an input_schema only)")
            tools.append({"type": "function", "function": {
                "name": t.get("name", ""), "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}})
        o["tools"] = tools
        tc = req.get("tool_choice")
        if isinstance(tc, dict):
            k = tc.get("type")
            if k == "any":
                o["tool_choice"] = "required"
            elif k == "tool":
                o["tool_choice"] = {"type": "function", "function": {"name": tc.get("name", "")}}
            elif k == "none":
                o["tool_choice"] = "none"
            elif k not in (None, "auto"):
                raise RequestError(f"tool_choice type {k!r}")
            if tc.get("disable_parallel_tool_use"):
                o["parallel_tool_calls"] = False
    th = req.get("thinking")
    if isinstance(th, dict) and th.get("type") == "enabled":
        o["reasoning_effort"] = _effort(th.get("budget_tokens"))
    else:
        o["reasoning_effort"] = "none"
    for k in ("temperature", "top_p", "top_k"):
        if req.get(k) is not None:
            o[k] = req[k]
    if req.get("stop_sequences"):
        o["stop"] = req["stop_sequences"]
    for k in ("seed", "min_p", "repetition_penalty", "presence_penalty", "frequency_penalty", "logit_bias"):
        if req.get(k) is not None:   # extensions
            o[k] = req[k]
    return o


def stop_reason(finish, stop_string):
    if finish == "tool_calls":
        return "tool_use", None
    if finish == "length":
        return "max_tokens", None
    if stop_string:
        return "stop_sequence", stop_string
    return "end_turn", None


def usage(prompt_tokens, cached, completion_tokens):
    return {"input_tokens": max(prompt_tokens - cached, 0), "output_tokens": completion_tokens,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": cached}


def blocks(content, reasoning, calls, thinking):
    """assembled output -> Anthropic content blocks (thinking, text, tool_use)"""
    out = []
    if reasoning or (thinking and not content and not calls):
        out.append({"type": "thinking", "thinking": reasoning, "signature": ""})
    if content or not (out or calls):
        out.append({"type": "text", "text": content})
    for c in calls:
        try:
            inp = json.loads(c["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            inp = {}
        out.append({"type": "tool_use", "id": c["id"], "name": c["function"]["name"], "input": inp})
    return out


def message(mid, model, content_blocks, reason, stop_string, use):
    return {"id": mid, "type": "message", "role": "assistant", "model": model, "content": content_blocks,
            "stop_reason": reason, "stop_sequence": stop_string, "usage": use}


class Stream:
    """parser deltas -> Anthropic stream events [(event name, data dict)]"""

    def __init__(self, mid, model, input_tokens):
        self.mid, self.model, self.input_tokens = mid, model, input_tokens
        self.index = -1       # index of the open block
        self.kind = None      # "thinking" | "text" | "tool" (open block)
        self.tool_idx = None  # parser's call index of the open tool_use block

    def start(self):
        return [("message_start", {"type": "message_start", "message": {
            "id": self.mid, "type": "message", "role": "assistant", "model": self.model, "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": self.input_tokens, "output_tokens": 1, "cache_creation_input_tokens": 0,
                      "cache_read_input_tokens": 0}}}),
                ("ping", {"type": "ping"})]

    def _close(self):
        if self.kind is None:
            return []
        ev = [("content_block_stop", {"type": "content_block_stop", "index": self.index})]
        if self.kind == "thinking":   # (a signature_delta ahead of the stop, as the service does)
            ev.insert(0, ("content_block_delta", {"type": "content_block_delta", "index": self.index,
                                                  "delta": {"type": "signature_delta", "signature": ""}}))
        self.kind = None
        return ev

    def _open(self, kind, block):
        ev = self._close()
        self.index += 1
        self.kind = kind
        ev.append(("content_block_start", {"type": "content_block_start", "index": self.index, "content_block": block}))
        return ev

    def deltas(self, ds):
        ev = []
        for d in ds:
            if d[0] == "reasoning":
                if self.kind != "thinking":
                    ev += self._open("thinking", {"type": "thinking", "thinking": "", "signature": ""})
                ev.append(("content_block_delta", {"type": "content_block_delta", "index": self.index,
                                                   "delta": {"type": "thinking_delta", "thinking": d[1]}}))
            elif d[0] == "content":
                if self.kind != "text":
                    ev += self._open("text", {"type": "text", "text": ""})
                ev.append(("content_block_delta", {"type": "content_block_delta", "index": self.index,
                                                   "delta": {"type": "text_delta", "text": d[1]}}))
            elif d[0] == "tool_start":
                ev += self._open("tool", {"type": "tool_use", "id": d[2], "name": d[3], "input": {}})
                self.tool_idx = d[1]
            elif d[0] == "tool_args":
                ev.append(("content_block_delta", {"type": "content_block_delta", "index": self.index,
                                                   "delta": {"type": "input_json_delta", "partial_json": d[2]}}))
        return ev

    def finish(self, reason, stop_string, output_tokens, use=None):
        ev = self._close()
        u = {"output_tokens": output_tokens}
        if use:
            u.update(use)
        ev.append(("message_delta", {"type": "message_delta", "delta": {"stop_reason": reason, "stop_sequence": stop_string},
                                     "usage": u}))
        ev.append(("message_stop", {"type": "message_stop"}))
        return ev
