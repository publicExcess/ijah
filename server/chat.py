"""Chat layer of the Ijah OpenAI server (lab-server): the pack's chat template, OpenAI message conversion,
incremental detokenisation, stop strings, reasoning / content separation and Qwen-style tool-call parsing.

Pure CPU (tokenizers + jinja2, requirements.txt); no engine import. Unit tests: tests/server/test_units.py.
"""
import json
import os
import re
import secrets
import threading
import time
from array import array
from bisect import bisect_right
from pathlib import Path

import jinja2
from tokenizers import Tokenizer
from jinja2.sandbox import ImmutableSandboxedEnvironment

IM_START, IM_END, EOT = "<|im_start|>", "<|im_end|>", "<|endoftext|>"
THINK_END = "</think>"
TC_OPEN, TC_CLOSE = "<tool_call>", "</tool_call>"

# OpenAI reasoning_effort -> the template's (xhigh | medium | low); None = thinking off
EFFORT = {"none": None, "off": None, "false": None, "minimal": "low", "low": "low", "medium": "medium",
          "high": "xhigh", "xhigh": "xhigh", "max": "xhigh", "maximum": "xhigh"}


class RequestError(ValueError):
    """-> HTTP 400 with an OpenAI error body"""


# ------------------------------------------------------------------------------------------------------ template
class Template:
    def __init__(self, pack_dir, tokenizer, default_effort="xhigh"):
        def raise_exception(msg):
            raise jinja2.TemplateError(msg)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = raise_exception
        env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False, **k: json.dumps(
            x, ensure_ascii=ensure_ascii, indent=indent)
        self.t = env.from_string((Path(pack_dir) / "chat_template.jinja").read_text())
        self.tok = tokenizer
        self.default_effort = default_effort
        self.im_end = tokenizer.token_to_id(IM_END)
        self.eot = tokenizer.token_to_id(EOT)
        self.im_start = tokenizer.token_to_id(IM_START)
        self._enc = []  # incremental encode (lab-ttft): LRU of (text, ids array, <|im_start|> char / token positions, lits)
        self._enc_lock = threading.Lock()
        self._enc_on = os.environ.get("NE_TOK_CACHE", "1") != "0" and self.im_start is not None
        # Literal special-token text in message content (a tool result or a file showing <|im_end|>, <|image_pad|>,
        # <tool_call> ...) must stay text: the tokenizer would split it out as the real token. render() swaps every
        # such literal in the request's strings for one private-use character, encode() restores it and tokenizes it
        # with a copy of the tokenizer that has no added tokens (plain BPE); only the template's own markup is special.
        self._plain = self._lit_re = self._split_re = None
        self._lits = []
        try:
            spec = json.loads(tokenizer.to_str())
            lits = sorted({t["content"] for t in spec.get("added_tokens", []) if t.get("content")}, key=lambda x: (-len(x), x))
            if lits:
                spec["added_tokens"] = []
                self._plain = Tokenizer.from_str(json.dumps(spec))
                self._lits = lits
                alt = "|".join(re.escape(t) for t in lits)
                self._lit_re, self._split_re = re.compile(alt), re.compile("(" + alt + ")")
        except Exception:   # a tokenizer without to_str (tests): no guarding
            self._plain = self._lit_re = self._split_re = None

    @staticmethod
    def _image_src(p):
        """the source string of an OpenAI image content part (image_url {url} | image_url str | input_image)"""
        u = p.get("image_url", p.get("image"))
        if isinstance(u, dict):
            u = u.get("url")
        if not isinstance(u, str) or not u:
            raise RequestError("image content part without a url")
        return u

    @staticmethod
    def _messages(msgs, images=None):
        """images: a list to collect the image sources into (in prompt order); None = images refused"""
        if not isinstance(msgs, list) or not msgs:
            raise RequestError("messages must be a non-empty list")
        out = []
        for m in msgs:
            if not isinstance(m, dict) or "role" not in m:
                raise RequestError("every message needs a role")
            m = dict(m)
            role = m["role"]
            if role == "developer":
                m["role"] = role = "system"
            c = m.get("content")
            if isinstance(c, list):  # OpenAI content parts: text, and images (lab-vision) when enabled
                parts, items, n_img = [], [], 0
                for p in c:
                    if isinstance(p, dict) and p.get("type") in ("text", "input_text"):
                        parts.append(p.get("text", ""))
                        items.append({"type": "text", "text": p.get("text", "")})
                    elif isinstance(p, str):
                        parts.append(p)
                        items.append({"type": "text", "text": p})
                    elif isinstance(p, dict) and p.get("type") in ("image_url", "input_image", "image"):
                        if images is None:
                            raise RequestError("image input is not enabled on this server")
                        if role == "system":
                            raise RequestError("system messages cannot contain images")
                        images.append(Template._image_src(p))
                        items.append({"type": "image"})
                        n_img += 1
                    else:
                        raise RequestError("only text and image_url content parts are supported")
                # text only: one string (as before); with images the template renders the items in order
                m["content"] = items if n_img else "".join(parts)
            elif c is None:
                m["content"] = ""
            if role == "assistant":
                if m.get("reasoning_content") is None and isinstance(m.get("reasoning"), str):
                    m["reasoning_content"] = m["reasoning"]
                tcs = []
                for tc in m.get("tool_calls") or []:
                    f = dict(tc.get("function", tc))
                    a = f.get("arguments", {})
                    if isinstance(a, str):
                        try:
                            a = json.loads(a) if a.strip() else {}
                        except json.JSONDecodeError:
                            raise RequestError(f"tool call {f.get('name')}: arguments are not JSON")
                    if not isinstance(a, dict):
                        raise RequestError(f"tool call {f.get('name')}: arguments must be an object")
                    f["arguments"] = a
                    tcs.append({"type": "function", "function": f})
                if tcs:
                    m["tool_calls"] = tcs
                else:
                    m.pop("tool_calls", None)
            out.append(m)
        return out

    def effort(self, req):
        """-> (enable_thinking, reasoning_effort or None)"""
        kw = req.get("chat_template_kwargs") or {}
        eff = req.get("reasoning_effort", kw.get("reasoning_effort"))
        if isinstance(req.get("reasoning"), dict):
            eff = req["reasoning"].get("effort", eff)
        think = kw.get("enable_thinking", req.get("enable_thinking"))
        if eff is not None:
            key = str(eff).lower()
            if key not in EFFORT:
                raise RequestError(f"reasoning_effort {eff!r}: use none, low, medium, high or xhigh")
            eff = EFFORT[key]
            if eff is None:
                think = False
        if think is False:
            return False, None
        return True, eff or self.default_effort

    # -- literal special-token text (see __init__)
    def _has_lit(self, o):
        if isinstance(o, str):
            return self._lit_re.search(o) is not None
        if isinstance(o, dict):
            return any(self._has_lit(k) or self._has_lit(v) for k, v in o.items())
        if isinstance(o, (list, tuple)):
            return any(self._has_lit(v) for v in o)
        return False

    @staticmethod
    def _pua(o, used):
        if isinstance(o, str):
            used.update(re.findall("[\ue000-\uf8ff]", o))
        elif isinstance(o, dict):
            for k, v in o.items():
                Template._pua(k, used)
                Template._pua(v, used)
        elif isinstance(o, (list, tuple)):
            for v in o:
                Template._pua(v, used)

    def _swap(self, o, idx, base):
        if isinstance(o, str):
            return self._lit_re.sub(lambda m: chr(base + idx[m.group()]), o)
        if isinstance(o, dict):
            return {self._swap(k, idx, base): self._swap(v, idx, base) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [self._swap(v, idx, base) for v in o]
        return o

    def _guard(self, *objs):
        """-> (objs with every literal special-token string replaced by a private-use character, base | None). base is
        the first code point of a block of len(lits) private-use characters that occur nowhere in the request."""
        if self._lit_re is None or not any(self._has_lit(o) for o in objs):
            return objs, None
        used = set()
        for o in objs:
            self._pua(o, used)
        base = 0xE000
        while any(chr(base + i) in used for i in range(len(self._lits))):
            base += len(self._lits)
            if base + len(self._lits) > 0xF8FF:
                raise RequestError("the request uses every private-use character")
        idx = {t: i for i, t in enumerate(self._lits)}
        return [self._swap(o, idx, base) for o in objs], base

    def _restore(self, text, base):
        if base is None:
            return text
        return re.sub("[%s-%s]" % (chr(base), chr(base + len(self._lits) - 1)),
                      lambda m: self._lits[ord(m.group()) - base], text)

    def _enc_raw(self, text, base):
        """text (with guard characters when base is set) -> ids: the template's special tokens are tokens, the
        restored literals are plain text (tokenized together with the text around them, as a tokenizer would)"""
        if base is None:
            return self.tok.encode(text, add_special_tokens=False).ids
        out = []
        for i, seg in enumerate(self._split_re.split(text)):   # odd parts: the template's own special tokens
            if i % 2:
                out.append(self.tok.token_to_id(seg))
            elif seg:
                out += self._plain.encode(self._restore(seg, base), add_special_tokens=False).ids
        return out

    def _render(self, req, images=None):
        msgs = self._messages(req.get("messages"), images)
        tools = req.get("tools") or None
        if tools is not None:
            if not isinstance(tools, list):
                raise RequestError("tools must be a list")
            # rendered in the OpenAI shape {"type": "function", "function": {...}}, as transformers / vLLM / Sushi /
            # oMLX pass them to the template (unwrapping them gave a 225-token shorter, off-reference prompt for an agent client's
            # 25 tools); a bare function dict is wrapped so every client renders alike
            tools = [{"type": "function", "function": t} if isinstance(t, dict) and "type" not in t and "name" in t
                     else t for t in tools]
        if req.get("tool_choice") == "none":
            tools = None
        think, eff = self.effort(req)
        kw = dict(req.get("chat_template_kwargs") or {})
        kw.pop("reasoning_effort", None)
        kw.pop("enable_thinking", None)
        (msgs, tools, kw), base = self._guard(msgs, tools, kw)
        ctx = dict(kw, messages=msgs, tools=tools, add_generation_prompt=True, enable_thinking=think)
        if eff:
            ctx["reasoning_effort"] = eff
        try:
            text = self.t.render(**ctx)
        except jinja2.TemplateError as e:
            raise RequestError(f"chat template: {e}")
        return text, think, base

    def render(self, req, images=None):
        """OpenAI request -> (prompt text, thinking on); images: list that receives the image sources (in the order
        of their <|image_pad|> placeholders), None = image parts are refused"""
        text, think, base = self._render(req, images)
        return self._restore(text, base), think

    def render_ids(self, req, images=None):
        """OpenAI request -> (prompt token ids, thinking on). Only the template's own markup becomes special tokens:
        special-token text inside the messages, tools and template kwargs is encoded as plain text."""
        text, think, base = self._render(req, images)
        return self.encode(text, base), think

    ENC_KEEP, ENC_MIN = 8, 8192  # texts remembered; shorter texts are always encoded whole

    def encode(self, text, lits=None):
        """text -> token ids (raw text: its special-token strings are tokens; lits: see render_ids, the text carries
        guard characters from `_guard`). Incremental (lab-ttft, docs/server.md 3e): the tokenizer splits on its added special
        tokens before normalising / BPE, so the text before an <|im_start|> encodes the same whatever follows it. An
        earlier text sharing a prefix with this one (the conversation's previous turn) gives its ids up to the last
        <|im_start|> lying wholly inside the common prefix; only the rest is encoded: a 160K-token conversation 93 ms
        -> ~2 ms per turn on the request's critical path. Exact by construction (tests/server/test_tokcache.py
        compares against the whole-text encode); NE_TOK_CACHE=0: always the whole text."""
        if not self._enc_on or len(text) < self.ENC_MIN:
            return self._enc_raw(text, lits)
        with self._enc_lock:
            best = (0, 0, None, 0)  # (char cut, token cut, entry, index of the cut's <|im_start|>)
            for e in self._enc:
                if e[4] != lits:   # (the guard characters of another request mean other literals)
                    continue
                c = _common_prefix(e[0], text)
                j = bisect_right(e[2], c - len(IM_START)) - 1  # last <|im_start|> with the whole token in the prefix
                if j >= 0 and e[2][j] > best[0]:
                    best = (e[2][j], e[3][j], e, j)
            cut, tcut, e, j = best
            tail = self._enc_raw(text[cut:], lits)
            if e is None:
                ids, cpos, tpos = tail, [], []
            else:
                ids = e[1][:tcut].tolist() + tail
                cpos, tpos = list(e[2][:j]), list(e[3][:j])
            # this text's <|im_start|> positions: the reused head's + the tail's
            i = text.find(IM_START, cut)
            while i >= 0:
                cpos.append(i)
                i = text.find(IM_START, i + 1)
            tpos += [tcut + k for k, t in enumerate(tail) if t == self.im_start]
            if len(cpos) == len(tpos):  # (else a literal the tokenizer did not split: keep it out of the cache)
                self._enc = [x for x in self._enc if x[0] != text][: self.ENC_KEEP - 1]
                self._enc.insert(0, (text, array("i", ids), cpos, tpos, lits))
            return ids


def _common_prefix(a, b):
    """length of the common prefix of two strings (slices compare in C)"""
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def tool_schemas(req):
    """{name: {param: json-schema type}} of the request's tools"""
    out = {}
    for t in req.get("tools") or []:
        f = t.get("function", t) if isinstance(t, dict) else {}
        props = ((f.get("parameters") or {}).get("properties") or {})
        out[f.get("name")] = {k: (v.get("type") if isinstance(v, dict) else None) for k, v in props.items()}
    return out


# ------------------------------------------------------------------------------------------------- detokeniser
class Detok:
    """incremental token ids -> text (tokenizers' DecodeStream; skip_special_tokens off: <think> etc. are text)"""

    def __init__(self, tokenizer):
        from tokenizers.decoders import DecodeStream

        self.tok = tokenizer
        self.ds = DecodeStream(skip_special_tokens=False)

    def __call__(self, token):
        s = self.ds.step(self.tok, token)
        return s or ""


def _partial_suffix(text, marker):
    """length of the longest suffix of text that is a proper prefix of marker"""
    for n in range(min(len(marker) - 1, len(text)), 0, -1):
        if text.endswith(marker[:n]):
            return n
    return 0


class StopFilter:
    """OpenAI stop strings over the raw generated text: holds back what could be the start of a stop string"""

    def __init__(self, stops):
        if isinstance(stops, str):
            stops = [stops]
        stops = [s for s in (stops or []) if isinstance(s, str) and s]
        if len(stops) > 16:
            raise RequestError("at most 16 stop strings")
        self.stops = stops
        self.hold = max((len(s) for s in stops), default=1) - 1
        self.buf = ""
        self.hit = False
        self.matched = None   # the stop string that ended the text (Anthropic stop_sequence)

    def feed(self, text):
        if self.hit:
            return ""
        if not self.stops:
            return text
        self.buf += text
        cut = min((i for i in (self.buf.find(s) for s in self.stops) if i >= 0), default=-1)
        if cut >= 0:
            self.hit = True
            self.matched = next(s for s in self.stops if self.buf.startswith(s, cut))
            out, self.buf = self.buf[:cut], ""
            return out
        keep = max((_partial_suffix(self.buf, s) for s in self.stops), default=0)
        out, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
        return out

    def flush(self):
        out, self.buf = self.buf, ""
        return "" if self.hit else out


# --------------------------------------------------------------------------------------------- output parser
def _convert(value, typ):
    if typ in (None, "string"):
        return value
    if isinstance(typ, list):
        if "string" in typ:
            return value
        typ = typ[0] if typ else None
    v = value.strip()
    if typ == "boolean" and v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return json.loads(v)
    except (json.JSONDecodeError, ValueError):
        return value


class OutputParser:
    """generated text -> deltas: ("reasoning", s), ("content", s), ("tool_start", idx, id, name), ("tool_args", idx, s),
    ("tool_abort", idx, raw text) (call idx never closed: a non-streamed reply drops it and keeps the text).

    Reasoning: the prompt ends with '<think>\\n' when thinking is on, so the text starts in reasoning until '</think>';
    the newlines around the tag are dropped. Tool calls (the template's format):
      <tool_call>\\n<function=NAME>\\n<parameter=K>\\nVALUE\\n</parameter>\\n...</function>\\n</tool_call>
    streamed as the name first, then the JSON arguments object in pieces (one per complete parameter) that
    concatenate to valid JSON. A JSON body {"name": .., "arguments": {..}} inside <tool_call> is accepted too.
    """

    def __init__(self, thinking, schemas=None, tools_enabled=True):
        self.mode = "reasoning" if thinking else "content"
        self.schemas = schemas or {}
        self.tools = tools_enabled
        self.buf = ""
        self.nl_hold = ""  # newlines held back (dropped before </think> / <tool_call>, emitted otherwise)
        self.lead = True   # strip newlines at the start of the current section
        self.calls = []    # tool calls seen: {"id", "name", "args": dict, "closed": bool}
        self.call = None   # open call parse state
        self.cut = False   # the generation ended inside a tool call (its text went back as content)
        self.reasoning_done_at = None

    @property
    def finished_calls(self):
        """the calls that ended properly (</function>): what a client may execute"""
        return [c for c in self.calls if c.get("closed")]

    # -- helpers
    def _emit_text(self, kind, s, out):
        if not s:
            return
        if self.lead:
            s = s.lstrip("\n")
            if not s:
                return
            self.lead = False
        body = s.rstrip("\n")
        tail = s[len(body):]
        if body:
            out.append((kind, self.nl_hold + body))
            self.nl_hold = ""
        self.nl_hold += tail

    def feed(self, text):
        out = []
        self.buf += text
        while self.buf:
            if self.mode == "reasoning":
                i = self.buf.find(THINK_END)
                if i >= 0:
                    self._emit_text("reasoning", self.buf[:i], out)
                    self.buf = self.buf[i + len(THINK_END):]
                    self.nl_hold, self.lead, self.mode = "", True, "content"
                    self.reasoning_done_at = True
                    continue
                k = _partial_suffix(self.buf, THINK_END)
                self._emit_text("reasoning", self.buf[:len(self.buf) - k], out)
                self.buf = self.buf[len(self.buf) - k:]
                break
            if self.mode == "content":
                i = self.buf.find(TC_OPEN) if self.tools else -1
                if i >= 0:
                    self._emit_text("content", self.buf[:i], out)
                    self.buf = self.buf[i + len(TC_OPEN):]
                    self.nl_hold = ""
                    self.mode, self.call = "call", {"body": "", "state": "head", "idx": len(self.calls)}
                    continue
                k = _partial_suffix(self.buf, TC_OPEN) if self.tools else 0
                self._emit_text("content", self.buf[:len(self.buf) - k], out)
                self.buf = self.buf[len(self.buf) - k:]
                break
            if self.mode == "call":
                i = self.buf.find(TC_CLOSE)
                if i >= 0:
                    self.call["body"] += self.buf[:i]
                    self.buf = self.buf[i + len(TC_CLOSE):]
                    self._call_progress(out, final=True, tagged=True)
                    self.mode, self.call, self.lead = "content", None, True
                    continue
                k = _partial_suffix(self.buf, TC_CLOSE)
                self.call["body"] += self.buf[:len(self.buf) - k]
                self.buf = self.buf[len(self.buf) - k:]
                self._call_progress(out, final=False)
                break
        return out

    def _call_progress(self, out, final, tagged=False):
        """advance the open call by the body text so far. final: no more text will come for it; tagged: its
        </tool_call> was seen (else the generation ended inside the call). A call is *closed* when its </function> was
        seen, or when a tagged end follows parameters that all finished. A call that ends unclosed (cut off by the
        length limit / EOS / a stop string mid-parameter, or markup that does not parse) is not a tool call: it is
        marked unclosed and gives its text back with ("tool_abort", idx, raw) -- the arguments were not finished, so
        they must not be reported as a call (finish_reason tool_calls)."""
        c = self.call
        body = c["body"]
        if c["state"] == "head":
            s = body.lstrip()
            if s.startswith("<function="):
                j = s.find(">")
                if j >= 0:
                    name = s[len("<function="):j].strip()
                    rec = {"id": "call_" + secrets.token_hex(12), "name": name, "args": {}, "closed": False}
                    c.update(state="params", name=name, id=rec["id"], pos=len(body) - len(s) + j + 1,
                             args=rec["args"], opened=False, rec=rec)
                    self.calls.append(rec)
                    out.append(("tool_start", c["idx"], c["id"], name))
        if c["state"] == "params":
            types = self.schemas.get(c["name"], {})
            while True:
                rest = body[c["pos"]:]
                p = rest.find("<parameter=")
                f = rest.find("</function>")
                if p >= 0 and (f < 0 or p < f):
                    j = rest.find(">", p)
                    e = rest.find("</parameter>", p)
                    if j < 0 or e < 0:   # a parameter still open (or cut off)
                        break
                    key = rest[p + len("<parameter="):j].strip()
                    val = rest[j + 1:e]
                    if val.startswith("\n"):
                        val = val[1:]
                    if val.endswith("\n"):
                        val = val[:-1]
                    v = _convert(val, types.get(key))
                    c["args"][key] = v
                    frag = ("{" if not c["opened"] else ", ") + json.dumps(key, ensure_ascii=False) + ": " + \
                        json.dumps(v, ensure_ascii=False)
                    c["opened"] = True
                    out.append(("tool_args", c["idx"], frag))
                    c["pos"] += e + len("</parameter>")
                    continue
                if f >= 0 or (final and tagged):
                    if c["state"] != "done":
                        out.append(("tool_args", c["idx"], "}" if c["opened"] else "{}"))
                        c["state"] = "done"
                        c["rec"]["closed"] = True
                break
            if final and c["state"] == "params":   # cut off inside the call (or a malformed one): not a tool call
                if not tagged:
                    self.cut = True
                out.append(("tool_abort", c["idx"], TC_OPEN + body + c.get("tail", "") + (TC_CLOSE if tagged else "")))
        if final and c["state"] == "head":  # no <function=...>: a JSON body, else give the text back as content
            try:
                obj = json.loads(body.strip())
                name, args = obj["name"], obj.get("arguments", obj.get("parameters", {}))
                if isinstance(args, str):
                    args = json.loads(args)
                cid = "call_" + secrets.token_hex(12)
                self.calls.append({"id": cid, "name": name, "args": args, "closed": True})
                out.append(("tool_start", c["idx"], cid, name))
                out.append(("tool_args", c["idx"], json.dumps(args, ensure_ascii=False)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                if not tagged:
                    self.cut = True
                out.append(("content", TC_OPEN + body + c.get("tail", "") + (TC_CLOSE if tagged else "")))

    def finish(self):
        """end of generation: flush held text; an open tool call is closed with what was parsed"""
        out = []
        if self.mode == "call":
            self.call["tail"] = self.buf   # a partial </tool_call> held back: part of the text if the call is given back
            self._call_progress(out, final=True)
        elif self.buf:
            self._emit_text(self.mode, self.buf, out)
        self.buf = ""
        return out


class RawParser:
    """/v1/completions: the generated text verbatim (no reasoning split, no tool calls, no newline trimming)"""
    mode = "content"
    cut = False
    finished_calls = ()

    def __init__(self):
        self.calls = []

    def feed(self, text):
        return [("content", text)] if text else []

    def finish(self):
        return []


def assemble(deltas):
    """deltas -> (content, reasoning, tool_calls) of a non-streamed response"""
    content, reasoning, calls = [], [], {}
    for d in deltas:
        if d[0] == "content":
            content.append(d[1])
        elif d[0] == "reasoning":
            reasoning.append(d[1])
        elif d[0] == "tool_start":
            calls[d[1]] = {"id": d[2], "type": "function", "function": {"name": d[3], "arguments": ""}}
        elif d[0] == "tool_args":
            calls[d[1]]["function"]["arguments"] += d[2]
        elif d[0] == "tool_abort":   # a call that never closed: not a call, its text is content
            calls.pop(d[1], None)
            content.append(d[2])
    return "".join(content), "".join(reasoning), [calls[k] for k in sorted(calls)]


def now():
    return int(time.time())
