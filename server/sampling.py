"""Sampling knobs beyond temperature / top_p / top_k of the Ijah server (lab-sampler, docs/server.md section 11):
logit_bias, presence_penalty, frequency_penalty, repetition_penalty, min_p, seed, logprobs / top_logprobs, n.

Pure CPU. `Extras.parse(req, vocab)` validates the request fields (RequestError -> HTTP 400) and `engine_extra()` is
what the backend hands to the engine (None = the sampler as before: nothing extra is touched, MTP stays on). Bias,
penalties and min_p run on both paths (the MTP verify rows get the same transform: docs/server.md section 11b); logprobs
(the MTP loop has none) and a bare seed (a fixed path is what makes it reproducible) take the step path.
"""
from chat import RequestError

MAX_BIAS = 4096
MAX_TOP_LOGPROBS = 20
MAX_N = 64


def _num(req, key, lo, hi, default, what=None):
    v = req.get(key)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise RequestError(f"{key} must be a number")
    if not (lo <= v <= hi) or v != v:
        raise RequestError(f"{key} {v}: use {what or f'{lo}..{hi}'}")
    return float(v)


class Extras:
    """validated extra sampling parameters of one request"""

    def __init__(self):
        self.min_p = 0.0
        self.presence = 0.0
        self.frequency = 0.0
        self.repetition = 1.0
        self.bias = {}            # token id -> bias (-100 = banned)
        self.logprobs = False     # report the sampled token's log-probability
        self.top_logprobs = 0     # ... and this many alternatives (<= 20)
        self.seeded = False
        self.n = 1

    @classmethod
    def parse(cls, req, vocab, completions=False):
        e = cls()
        n = req.get("n", 1)
        n = 1 if n is None else n
        if isinstance(n, bool) or not isinstance(n, int) or not (1 <= n <= MAX_N):
            raise RequestError(f"n {n!r}: use an integer 1..{MAX_N}")
        e.n = n
        e.min_p = _num(req, "min_p", 0.0, 1.0, 0.0)
        e.presence = _num(req, "presence_penalty", -2.0, 2.0, 0.0)
        e.frequency = _num(req, "frequency_penalty", -2.0, 2.0, 0.0)
        e.repetition = _num(req, "repetition_penalty", 1e-3, 10.0, 1.0, "(0, 10], 1 = off")
        lb = req.get("logit_bias")
        if lb:
            if not isinstance(lb, dict):
                raise RequestError("logit_bias must be an object {token id: bias}")
            if len(lb) > MAX_BIAS:
                raise RequestError(f"logit_bias: at most {MAX_BIAS} tokens")
            for k, v in lb.items():
                try:
                    t = int(k)
                except (TypeError, ValueError):
                    raise RequestError(f"logit_bias: token id {k!r} is not an integer")
                if not (0 <= t < vocab):
                    raise RequestError(f"logit_bias: token id {t} is outside the vocabulary (0..{vocab - 1})")
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not (-100 <= v <= 100):
                    raise RequestError(f"logit_bias[{t}] = {v!r}: use -100..100 (-100 bans the token)")
                e.bias[t] = float(v)
        lp, tl = req.get("logprobs"), req.get("top_logprobs")
        if completions:   # legacy: logprobs = the number of alternatives (0 = the sampled token only), null = off
            if lp is not None and (isinstance(lp, bool) or not isinstance(lp, int) or not (0 <= lp <= MAX_TOP_LOGPROBS)):
                raise RequestError(f"logprobs {lp!r}: use an integer 0..{MAX_TOP_LOGPROBS}")
            e.logprobs, e.top_logprobs = lp is not None, lp or 0
        else:
            if lp is not None and not isinstance(lp, bool):
                raise RequestError("logprobs must be true or false")
            e.logprobs = bool(lp)
            if tl is not None:
                if isinstance(tl, bool) or not isinstance(tl, int) or not (0 <= tl <= MAX_TOP_LOGPROBS):
                    raise RequestError(f"top_logprobs {tl!r}: use an integer 0..{MAX_TOP_LOGPROBS}")
                if not e.logprobs and tl:
                    raise RequestError("top_logprobs needs logprobs: true")
                e.top_logprobs = tl if e.logprobs else 0
        e.seeded = req.get("seed") is not None
        return e

    @property
    def neutral(self):
        return (self.min_p == 0.0 and self.presence == 0.0 and self.frequency == 0.0 and self.repetition == 1.0
                and not self.bias and not self.logprobs)

    def names(self):
        """the active knobs (timings.sampling)"""
        n = [k for k, on in (("logit_bias", self.bias), ("presence_penalty", self.presence != 0.0),
                             ("frequency_penalty", self.frequency != 0.0), ("repetition_penalty", self.repetition != 1.0),
                             ("min_p", self.min_p != 0.0), ("logprobs", self.logprobs), ("seed", self.seeded)) if on]
        return n

    def engine_extra(self):
        """-> dict for backend.set_extra, or None when nothing but temperature / top_p / top_k applies (a seed counts:
        the RNG counter restarts at the request, so the draws depend on the seed and the token's index only)"""
        if self.neutral and not self.seeded:
            return None
        return {"min_p": self.min_p, "presence": self.presence, "frequency": self.frequency,
                "repetition": self.repetition, "bias": dict(self.bias), "rng0": self.seeded,
                "logprobs": self.logprobs, "logprob_k": max(1, self.top_logprobs) if self.logprobs else 0}


# ------------------------------------------------------------------------------------------- logprobs output
def _byte_decoder():
    """GPT-2 / Qwen byte-level BPE: the printable unicode character that stands for each byte, inverted"""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class TokenText:
    """token id -> (text, utf-8 bytes) for the logprobs output; text of a partial UTF-8 sequence is its replacement
    characters (OpenAI: the `bytes` list is the exact content)"""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.u2b = _byte_decoder()
        self.cache = {}

    def __call__(self, t):
        r = self.cache.get(t)
        if r is None:
            s = self.tok.id_to_token(t)
            if s is None:
                s = ""
            b = bytearray()
            for ch in s:
                x = self.u2b.get(ch)
                if x is None:
                    b += ch.encode("utf-8")
                else:
                    b.append(x)
            b = bytes(b)
            r = (b.decode("utf-8", errors="replace"), list(b))
            if len(self.cache) < 65536:
                self.cache[t] = r
        return r


def lp_value(x):
    """a JSON-safe log-probability (OpenAI uses -9999 for 'impossible')"""
    return max(float(x), -9999.0)


def lp_entry(texts, token, cand, top_n):
    """one generated token's chat `logprobs.content` item: token, logprob, bytes, top_logprobs. cand = (ids, logprobs)
    of the sampler's candidates (finite ones only are real); the sampled token is among them by construction"""
    ids, lps = cand
    own = None
    for i, lp in zip(ids, lps):
        if i == token:
            own = lp
            break
    s, b = texts(token)
    ent = {"token": s, "logprob": lp_value(own) if own is not None else -9999.0, "bytes": b, "top_logprobs": []}
    for i, lp in list(zip(ids, lps))[:top_n]:
        if lp == float("-inf") or lp != lp:
            break
        ts, tb = texts(i)
        ent["top_logprobs"].append({"token": ts, "logprob": lp_value(lp), "bytes": tb})
    return ent


def legacy_logprobs(entries, texts_of_top=None):
    """chat-style entries -> the /v1/completions `logprobs` object"""
    tokens, token_lps, tops, offs = [], [], [], []
    pos = 0
    for e in entries:
        tokens.append(e["token"])
        token_lps.append(e["logprob"])
        tops.append({t["token"]: t["logprob"] for t in e["top_logprobs"]})
        offs.append(pos)
        pos += len(e["token"])
    return {"tokens": tokens, "token_logprobs": token_lps, "top_logprobs": tops, "text_offset": offs}

