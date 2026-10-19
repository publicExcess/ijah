"""Grammar-constrained decoding for the Ijah server (lab-api): response_format (json_object / json_schema)
and tool_choice (required / a named function), through xgrammar 0.2.8 (Apache-2.0, a pip dependency; no code of it
is copied) and the engine's allowed-token mask (ne_fn_model_set_mask, docs/server.md section 9).

    kit = GrammarKit(tokenizer, vocab_size=248320, stop_ids=(248046, 248044))     # once per process
    cons = kit.for_request(req, thinking)                                         # None = unconstrained
    mask = cons.start()          # int32 [1, 7760] words (bit set = token allowed) for the first sampled token
    mask = cons.advance(token)   # accept the sampled token, the mask for the next one (never None while running)

The model's output is one grammar from the first generated token on:
  thinking on  (the prompt ends with '<think>\\n'):  <any text but think tags> '</think>' '\\n\\n' FORMAT
  thinking off (the prompt ends with '</think>\\n\\n'):                                          FORMAT
FORMAT = a JSON value (json_object: an object; json_schema: the schema), or tool calls in the template's XML
(<tool_call>\\n<function=NAME>\\n<parameter=K>\\nV\\n</parameter>\\n...\\n</function>\\n</tool_call>): tool_choice
required = free text, then at least one call to any of the tools; a named function = exactly that call, nothing else.
While the reasoning runs the mask only keeps the special tokens out (and the stop token until the format is done).
JSON whitespace is bounded (32 in a row) so a run of blanks cannot trap the decoder after an early string end.
Pure CPU; imports xgrammar lazily (requirements.txt; xgrammar pulls in torch, which it imports at load).
"""
import hashlib
import json
import os
import re
import threading
from collections import OrderedDict

import numpy as np

from chat import RequestError

MAX_WHITESPACE = 32
THINK_TAGS = ["<think>", "</think>"]
TAG_TOKENS = THINK_TAGS + ["<tool_call>", "</tool_call>"]


class Unavailable(RuntimeError):
    """xgrammar could not be loaded (not installed: pip install -r requirements.txt)"""


_xgr = None
_xgr_lock = threading.Lock()


def load_xgrammar():
    """-> the xgrammar module (imported once)"""
    global _xgr
    with _xgr_lock:
        if _xgr is not None:
            return _xgr
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        try:
            import xgrammar
            from xgrammar import structural_tag  # noqa: F401
        except Exception as e:  # noqa: BLE001 - any import failure = feature off
            raise Unavailable(f"xgrammar not available: {e}") from e
        _xgr = xgrammar
        return _xgr


def mask_words(vocab_size):
    return (vocab_size + 31) // 32


class Constraint:
    """one request's grammar state (a GrammarMatcher + its reusable bitmask)"""

    def __init__(self, xgr, compiled, vocab_size, stop_ids, kind):
        self.m = xgr.GrammarMatcher(compiled)
        self.kind = kind
        self.bm = np.full((1, mask_words(vocab_size)), -1, dtype=np.int32)
        stop = np.zeros(mask_words(vocab_size), dtype=np.uint32)
        for t in stop_ids:
            stop[t >> 5] |= np.uint32(1 << (t & 31))
        self.stop = stop.view(np.int32).reshape(1, -1)   # only the stop tokens allowed
        self.dead = False   # the grammar rejected a sampled token (must not happen: the mask forbids it)
        self.tokens = 0

    def _fill(self):
        self.bm.fill(-1)
        self.m.fill_next_token_bitmask(self.bm)
        if not self.bm.any():     # dead end (a bounded whitespace run, a schema nothing satisfies): only stop is left
            return self.stop
        return self.bm

    def start(self):
        return self._fill()

    def advance(self, token):
        self.tokens += 1
        if self.m.is_terminated() or not self.m.accept_token(int(token)):
            self.dead = True
            return self.stop
        return self._fill()

    @property
    def done(self):
        """the grammar is complete: only a stop token is left"""
        return self.m.is_terminated()


class GrammarKit:
    """tokenizer info + compiler + a bounded cache of compiled grammars (process-wide)"""

    def __init__(self, tokenizer, vocab_size, stop_ids, cache=64):
        xgr = self.xgr = load_xgrammar()
        self.vocab_size, self.stop_ids = vocab_size, tuple(stop_ids)
        vocab = tokenizer.get_vocab(with_added_tokens=True)
        # the added tokens are markup (<|im_start|>, <|vision_pad|> ...): an empty entry is a special token to xgrammar,
        # never allowed (the stop ids aside). The four tags the grammars spell out stay: their literal text has to be
        # matched by their single token, and JSON strings exclude them by text (_json_format)
        added = {t.content for t in tokenizer.get_added_tokens_decoder().values()} - set(TAG_TOKENS)
        enc = [""] * vocab_size
        for s, i in vocab.items():
            if i < vocab_size and s not in added:
                enc[i] = s
        self.info = xgr.TokenizerInfo(enc, xgr.VocabType.BYTE_LEVEL, vocab_size=vocab_size,
                                      stop_token_ids=list(self.stop_ids), add_prefix_space=False)
        self.compiler = xgr.GrammarCompiler(self.info, max_threads=4, cache_enabled=True)
        self._cache = OrderedDict()
        self._cache_n = cache
        self._lock = threading.Lock()

    # -- request parsing
    @staticmethod
    def response_format(req):
        """-> ("object", None) | ("schema", dict) | None"""
        rf = req.get("response_format")
        if rf is None:
            return None
        if not isinstance(rf, dict) or "type" not in rf:
            raise RequestError("response_format must be an object with a type")
        t = rf["type"]
        if t == "text":
            return None
        if t == "json_object":
            return ("object", None)
        if t == "json_schema":
            js = rf.get("json_schema")
            if not isinstance(js, dict) or "schema" not in js:
                raise RequestError("response_format.json_schema.schema is required")
            schema = js["schema"]
            if isinstance(schema, str):
                try:
                    schema = json.loads(schema)
                except json.JSONDecodeError:
                    raise RequestError("response_format.json_schema.schema is not valid JSON")
            if not isinstance(schema, (dict, bool)):
                raise RequestError("response_format.json_schema.schema must be an object")
            return ("schema", schema)
        raise RequestError(f"response_format type {t!r}: use text, json_object or json_schema")

    @staticmethod
    def tool_mode(req):
        """-> ("required", [tools]) | ("forced", [tool]) | None (auto / none / no tools)"""
        tools = [t for t in (req.get("tools") or []) if isinstance(t, dict)]
        tc = req.get("tool_choice")
        if not tools or tc in (None, "auto", "none"):
            return None
        fns = [t.get("function", t) if t.get("type", "function") == "function" else None for t in tools]
        names = [f.get("name") for f in fns if isinstance(f, dict)]
        if tc == "required":
            return ("required", tools)
        if isinstance(tc, dict) and tc.get("type") == "allowed_tools":   # OpenAI: a subset of the tools + a mode
            at = tc.get("allowed_tools") or {}
            want = {(t.get("function") or {}).get("name", t.get("name")) for t in at.get("tools") or [] if isinstance(t, dict)}
            sub = [t for t, f in zip(tools, fns) if isinstance(f, dict) and f.get("name") in want]
            if not sub:
                raise RequestError("tool_choice allowed_tools names no tool of tools")
            return ("required", sub) if at.get("mode") == "required" else None
        if isinstance(tc, dict):
            name = (tc.get("function") or {}).get("name") if tc.get("type", "function") == "function" else None
            if name is None and isinstance(tc.get("name"), str):
                name = tc["name"]
            if name not in names:
                raise RequestError(f"tool_choice names {name!r}, which is not in tools")
            return ("forced", [t for t, f in zip(tools, fns) if isinstance(f, dict) and f.get("name") == name][:1])
        raise RequestError(f"tool_choice {tc!r}: use none, auto, required or {{\"type\": \"function\", "
                           "\"function\": {\"name\": ...}}")

    def wanted(self, req):
        return self.response_format(req) is not None or self.tool_mode(req) is not None

    @staticmethod
    def auto_tools(req):
        """the tools of a request that has a response_format and lets the model choose (tool_choice auto / unset): the
        answer is then the document or tool calls"""
        if req.get("tool_choice") not in (None, "auto"):
            return []
        return [t for t in (req.get("tools") or []) if isinstance(t, dict)]

    # -- compile
    def _tool_params(self, tools):
        out = []
        for t in tools:
            f = t.get("function", t)
            if not isinstance(f, dict) or not isinstance(f.get("name"), str):
                raise RequestError("every tool needs a function name")
            p = f.get("parameters")
            out.append({"type": "function", "function": {"name": f["name"], "description": f.get("description", ""),
                                                          "parameters": p if isinstance(p, dict) else
                                                          {"type": "object", "properties": {}}}})
        return out

    def _json_format(self, kind, schema):
        from xgrammar.structural_tag import JSONSchemaFormat
        if kind == "object":
            s = {"type": "object"}
        else:
            s = schema
        # (excludes=TAG_TOKENS would keep <think> / <tool_call> out of the strings, at +200 ms compile per schema)
        return JSONSchemaFormat(json_schema=s, max_whitespace_cnt=MAX_WHITESPACE)

    def _compile(self, req, thinking):
        from xgrammar.structural_tag import (AnyTextFormat, ConstStringFormat, JSONSchemaFormat, SequenceFormat,
                                             StructuralTag, TagFormat)
        xgr = self.xgr
        rf, tm = self.response_format(req), self.tool_mode(req)
        reasoning = None
        if thinking:   # the prompt ends with '<think>\n': the reasoning block is already open (begin = "")
            reasoning = SequenceFormat(elements=[
                TagFormat(begin="", content=AnyTextFormat(excludes=THINK_TAGS), end="</think>"),
                ConstStringFormat(value="\n\n")])
        if tm is not None:
            tag = xgr.get_model_structural_tag(
                "qwen_3_5", tools=self._tool_params(tm[1]), tool_choice="required" if tm[0] == "required" else {
                    "type": "function", "function": {"name": tm[1][0].get("function", tm[1][0])["name"]}},
                reasoning="enabled" if thinking else "disabled", max_whitespace_cnt=MAX_WHITESPACE,
                parallel_tool_calls=req.get("parallel_tool_calls", True) is not False)
            return self.compiler.compile_structural_tag(tag), tm[0]
        kind, schema = rf
        body = self._json_format(kind, schema)
        auto = self.auto_tools(req)
        if auto:   # the model may still call a tool instead of answering: the document OR calls (no prose in between)
            from xgrammar.structural_tag import OrFormat, TagsWithSeparatorFormat
            calls = TagsWithSeparatorFormat(
                tags=[TagFormat(begin=f"<tool_call>\n<function={t['function']['name']}>\n",
                                content=JSONSchemaFormat(json_schema=t["function"]["parameters"], style="qwen_xml",
                                                         max_whitespace_cnt=MAX_WHITESPACE),
                                end="\n</function>\n</tool_call>") for t in self._tool_params(auto)],
                separator="\n", at_least_one=True, stop_after_first=req.get("parallel_tool_calls", True) is False)
            body = OrFormat(elements=[body, calls])
            kind += "+tools"
        if reasoning is None:
            return self.compiler.compile_structural_tag(StructuralTag(format=body)), kind
        return self.compiler.compile_structural_tag(StructuralTag(format=SequenceFormat(elements=[reasoning, body]))), kind

    def for_request(self, req, thinking):
        """-> Constraint or None; RequestError (400) for a schema xgrammar rejects"""
        if not self.wanted(req):
            return None
        rf, tm = self.response_format(req), self.tool_mode(req)
        key = hashlib.sha256(json.dumps([rf, tm, bool(thinking), req.get("parallel_tool_calls", True),
                                         self.auto_tools(req) if rf is not None else None],
                                        sort_keys=True, default=str).encode()).hexdigest()
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
        if hit is None:
            try:
                hit = self._compile(req, bool(thinking))
            except RequestError:
                raise
            except Exception as e:  # noqa: BLE001 - xgrammar raises its own error types for bad schemas
                msg = re.sub(r"^\[\d\d:\d\d:\d\d\] \S+\.cc:\d+: ", "", (str(e).splitlines() or [""])[0])   # xgrammar's log prefix
                raise RequestError(f"cannot constrain the output: {msg[:300] or type(e).__name__}")
            with self._lock:
                self._cache[key] = hit
                while len(self._cache) > self._cache_n:
                    self._cache.popitem(last=False)
        compiled, kind = hit
        return Constraint(self.xgr, compiled, self.vocab_size, self.stop_ids, kind)


# ------------------------------------------------------------------------------------------- the model's brief
def format_hint(rf, tools=False):
    """the instruction the model gets with a response_format (OpenAI puts the schema in front of the model too)"""
    if rf[0] == "object":
        h = "Respond with a single JSON object and nothing else (no code fence, no commentary)."
    else:
        h = ("Respond with a single JSON value that conforms to this JSON schema, and nothing else (no code fence, no "
             "commentary):\n" + json.dumps(rf[1], ensure_ascii=False))
    return h + ("\nIf you need one of the tools, call it instead of answering." if tools else "")


def with_format_hint(req, rf):
    """req with the format hint appended to its first system / developer message (one is added if there is none);
    ne_format_hint: false in the request turns this off. The caller's request is not changed."""
    if req.get("ne_format_hint") is False:
        return req
    hint = format_hint(rf, bool(GrammarKit.auto_tools(req)))
    msgs = list(req.get("messages") or [])
    if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") in ("system", "developer"):
        m = dict(msgs[0])
        c = m.get("content")
        if isinstance(c, list):
            m["content"] = c + [{"type": "text", "text": "\n\n" + hint}]
        else:
            m["content"] = (c + "\n\n" if isinstance(c, str) and c else "") + hint
        msgs[0] = m
    else:
        msgs.insert(0, {"role": "system", "content": hint})
    return dict(req, messages=msgs)
