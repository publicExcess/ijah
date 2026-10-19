"""Whole-model check of the sampler's allowed-token mask (ne_fn_model_set_mask, lab-api) -- run under tools/with_lock.sh:

    NE_LIB=... NEED_GB=85 tools/with_lock.sh "mask check (~81 GB)" HH:MM -- $PY tests/server/mask_check.py

Checks, on a short chat prompt: (1) an all-ones mask changes nothing: greedy and seeded-sampled runs give the same tokens as
no mask; (2) a mask with one allowed token forces that token (greedy and sampled), logprob is 0 (the mask renormalises);
(3) a mask of the digit tokens: 40 sampled steps stay inside it, the unmasked run does not; (4) clearing the mask restores
the free run; (5) the mask is applied to the sample at the end of prefill; (6) an empty mask is refused; (7) a mask whose tokens logit_bias all bans
raises a clear error; (8) the MTP loop refuses to run while a mask is set (when it is available).
"""
import ctypes as C
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("NE_FN_ROUND", "f32")
import refenv  # noqa: E402
import ijah_paths  # noqa: E402

refenv.add_paths()
import numpy as np  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

import engine as E  # noqa: E402

FAILS = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def bits(ids, V=248320):
    a = np.zeros((V + 31) // 32, np.uint32)
    for t in ids:
        a[t >> 5] |= np.uint32(1 << (t & 31))
    return a.view(np.int32)


def main():
    pack_dir = ijah_paths.pack()
    tok = Tokenizer.from_file(str(Path(pack_dir) / "tokenizer.json"))
    eng = E.Engine()
    check(eng.has("ne_fn_model_set_mask"), "the library exports ne_fn_model_set_mask")
    m = eng.fn_model(eng.pack_open(pack_dir), 4096)
    prompt = tok.encode("<|im_start|>user\nCount from 1 to 20 separated by commas.<|im_end|>\n<|im_start|>assistant\n"
                        "<think>\n\n</think>\n\n", add_special_tokens=False).ids

    def run(n, temp, seed, mask=None):
        """prefill + n sampled steps -> tokens (the first from the prefill)"""
        m.reset()
        m.set_sampler(temp, 0.95 if temp else 1.0, 20 if temp else 1, seed)
        m.set_mask(mask)
        nx, _, _ = m.prefill(prompt)
        out = [nx]
        lps = []
        for _ in range(n - 1):
            nx, lp, _, _ = m.step(out[-1])
            out.append(nx)
            lps.append(lp)
        return out, lps

    ones = np.full((248320 + 31) // 32, -1, np.int32)
    free_g, _ = run(24, 0.0, 1)
    all_g, _ = run(24, 0.0, 1, ones)
    check(free_g == all_g, f"greedy: an all-ones mask changes nothing ({tok.decode(free_g)[:50]!r})")
    free_s, _ = run(24, 1.0, 5)
    all_s, _ = run(24, 1.0, 5, ones)
    check(free_s == all_s, "sampled (seed 5): an all-ones mask changes nothing")
    m.set_mask(None)
    again, _ = run(24, 0.0, 1)
    check(again == free_g, "greedy: the cleared mask gives the free run again")

    X = tok.encode("Zebra", add_special_tokens=False).ids[0]
    for temp in (0.0, 1.0):
        out, lps = run(8, temp, 3, bits([X]))
        check(out == [X] * 8 and all(abs(lp) < 1e-3 for lp in lps),
              f"temp {temp}: a one-token mask forces token {X} ({tok.decode([X])!r}), logprob ~0 ({[round(x, 4) for x in lps[:3]]})")
    ok_chars = set("0123456789")
    digits = [t for t in range(248077) if tok.decode([t]) and set(tok.decode([t])) <= ok_chars][:3000]
    dset = set(digits)
    out, _ = run(40, 1.0, 11, bits(digits))
    check(all(t in dset for t in out), f"40 sampled steps stay inside a digits-only mask: {tok.decode(out)[:60]!r}")
    check(not all(t in dset for t in free_s), "the unmasked sampled run leaves that set (the mask is what constrains)")
    out, _ = run(6, 0.0, 1, bits([X]))
    check(out[0] == X, "the sample at the end of the prefill obeys the mask")
    # greedy under a mask = argmax over the allowed tokens: the mask on the free run's own tokens keeps them
    keep = set(free_g)
    out, _ = run(24, 0.0, 1, bits(sorted(keep)))
    check(out == free_g, "greedy: a mask holding the free run's tokens reproduces it")
    # an empty mask is refused
    zeros = np.zeros((248320 + 31) // 32, np.int32)
    rc = eng.lib.ne_fn_model_set_mask(m.ptr, zeros.ctypes.data)
    check(rc != 0, f"an empty mask is refused (rc {rc}: {eng.last_error()[:60]})")
    m.set_mask(None)
    # the mask allows only tokens that logit_bias bans: a clear error (prefill and step), not a disallowed token or a
    # NaN logprob; the model stays usable once the bias is lifted
    def raises(f):
        try:
            f()
            return False, "no error"
        except E.EngineError as e:
            return "no token can be sampled" in str(e), str(e)[:70]
    for temp in (0.0, 1.0):
        m.reset()
        m.set_sampler(temp, 0.95 if temp else 1.0, 20 if temp else 1, 3)
        m.set_mask(bits([X]))
        m.set_logit_bias({X: -100.0})
        ok, msg = raises(lambda: m.prefill(prompt))
        check(ok, f"temp {temp}: mask {{X}} + logit_bias X = -100 -> prefill raises ({msg})")
        m.set_logit_bias(None)
        m.reset()
        nx, _, _ = m.prefill(prompt)
        m.set_logit_bias({X: -100.0})
        ok, msg = raises(lambda: m.step(nx))
        check(ok, f"temp {temp}: the same on a decode step ({msg})")
        m.set_logit_bias(None)
        nx2, lp2, _, _ = m.step(X)
        check(nx2 == X and abs(lp2) < 1e-3, f"temp {temp}: bias lifted, the next step samples the masked token again")
    m.set_logit_bias(None)
    m.set_mask(None)
    # MTP refuses while a mask is set
    if eng.has("ne_fn_mtp_create"):
        L = eng.lib
        L.ne_fn_mtp_create.restype = C.c_void_p
        L.ne_fn_mtp_create.argtypes = [C.c_void_p, C.c_int, C.c_int]
        L.ne_fn_mtp_generate.restype = C.c_int
        L.ne_fn_mtp_generate.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_uint64,
                                         C.c_void_p, C.c_void_p, C.c_void_p]
        mp = L.ne_fn_mtp_create(C.c_void_p(m.ptr), 4, 4)
        if mp:
            p = np.ascontiguousarray(prompt, np.int32)
            out = np.zeros(16, np.int32)
            st = np.zeros(10, np.float64)
            m.set_sampler(0.0, 1.0, 1, 1)
            m.set_mask(ones)
            rc = L.ne_fn_mtp_generate(C.c_void_p(mp), p.ctypes.data, len(p), 16, 248046, 2, 1, out.ctypes.data,
                                      st.ctypes.data, None)
            check(rc < 0 and "mask" in eng.last_error(), f"MTP refuses while a mask is set (rc {rc}: {eng.last_error()[:70]})")
            m.set_mask(None)
            rc = L.ne_fn_mtp_generate(C.c_void_p(mp), p.ctypes.data, len(p), 16, 248046, 2, 1, out.ctypes.data,
                                      st.ctypes.data, None)
            check(rc > 0, f"MTP runs again after the mask is cleared ({rc} tokens)")
    print(f"\n{len(FAILS)} failed: {FAILS}" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
