"""Whole-model check of the sampler's extended knobs (lab-sampler) -- run under tools/with_lock.sh:

    NE_LIB=... NEED_GB=85 tools/with_lock.sh "sampler model check (~81 GB)" HH:MM -- $PY tests/server/sampler_model_check.py

On a short chat prompt, against numpy on the model's own logits (step(logits=True): the raw logits the sample was drawn from):
(1) neutral extras change nothing; (2) the candidate list (logprob_k) = the top-k of the logits, logprobs = log-softmax;
(3) logit_bias: -100 bans, +100 forces, in between shifts the argmax/logprob as computed; (4) presence / frequency /
repetition penalties: every step's token is the argmax of the penalised logits (counts of the tokens sampled so far,
prompt tokens seen by the repetition penalty), logprob exact; the server's re-arm after a discarded prefill sample;
(5) min_p: sampled tokens keep to p >= min_p * p_max, min_p 1 = greedy; (6) seeds: same seed same tokens, other seed other
tokens, the draws do not depend on how the prompt was fed once the counter is restarted; (7) the MTP loop applies bias and penalties (exactness: tests/mtp/test_mtp_knobs.py) and
refuses logprobs; (8) GPU step time with each knob on (one indicative round each).
"""
import ctypes as C
import os
import statistics
import sys
from collections import Counter
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
V = 248320


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def log_softmax(v):
    v = v.astype(np.float64)
    m = v[np.isfinite(v)].max()
    return v - (m + np.log(np.exp(v[np.isfinite(v)] - m).sum()))


def transform(lg, bias=None, gen=None, prompt=None, presence=0.0, frequency=0.0, repetition=1.0):
    v = lg.astype(np.float64).copy()
    if bias:
        for t, b in bias.items():
            v[t] = -np.inf if b <= -100 else v[t] + b
    cnt = np.zeros(V)
    for t, c in (gen or {}).items():
        cnt[t] = c
    seen = cnt > 0
    if prompt is not None and repetition != 1.0:
        seen = seen.copy()
        seen[list(set(prompt))] = True
    if repetition != 1.0:
        v = np.where(seen, np.where(v < 0, v * repetition, v / repetition), v)
    v = np.where(cnt > 0, v - frequency * cnt - presence, v)
    return v


def main():
    pack_dir = ijah_paths.pack()
    tok = Tokenizer.from_file(str(Path(pack_dir) / "tokenizer.json"))
    eng = E.Engine()
    for f in ("ne_fn_model_set_sampler_ex", "ne_fn_model_set_logit_bias", "ne_fn_model_set_seen", "ne_fn_model_top",
              "ne_fn_model_set_rng_step"):
        check(eng.has(f), f"the library exports {f}")
    pack = eng.pack_open(pack_dir)
    m = eng.fn_model(pack, 4096)
    prompt = tok.encode("<|im_start|>user\nRepeat the word echo forty times, separated by spaces.<|im_end|>\n"
                        "<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False).ids
    prompt2 = tok.encode("<|im_start|>user\nWrite two sentences about the sea.<|im_end|>\n<|im_start|>assistant\n"
                         "<think>\n\n</think>\n\n", add_special_tokens=False).ids

    def run(n, temp=0.0, seed=1, top_p=1.0, top_k=1, pr=None, bias=None, ext=None, logits=False, lp_k=0, pieces=None,
            rng0=True):
        """reset, sampler (+ extras), prefill, n-1 steps -> dict(tokens, logits [raw, one per sample], tops, lps)"""
        pr = pr or prompt
        ext = ext or {}
        m.reset()
        m.set_sampler(temp, top_p, top_k, seed)
        m.set_logit_bias(bias)
        m.set_sampler_ex(ext.get("min_p", 0.0), ext.get("presence", 0.0), ext.get("frequency", 0.0),
                         ext.get("repetition", 1.0), lp_k)
        if ext.get("repetition", 1.0) != 1.0:
            m.set_seen(pr)
        if rng0:
            m.set_rng_step(0)
        out = dict(tokens=[], logits=[], tops=[], lps=[])
        if pieces:   # the server's plain prefill: pieces sample after their last token, only the final one is kept
            i = 0
            for p in pieces[:-1]:
                m.prefill(pr[i:i + p])
                i += p
            m.set_sampler_ex(ext.get("min_p", 0.0), ext.get("presence", 0.0), ext.get("frequency", 0.0),
                             ext.get("repetition", 1.0), lp_k)   # arm: counts zeroed
            if ext.get("repetition", 1.0) != 1.0:
                m.set_seen(pr)
            if rng0:
                m.set_rng_step(0)
            nx, lg, _ = m.prefill(pr[i:], logits=logits)
        else:
            nx, lg, _ = m.prefill(pr, logits=logits)
        out["tokens"].append(nx)
        out["logits"].append(lg)
        if lp_k:
            out["tops"].append(m.top())
        for _ in range(n - 1):
            nx, lp, _, lg = m.step(out["tokens"][-1], logits=logits)
            out["tokens"].append(nx)
            out["logits"].append(lg)
            out["lps"].append(lp)
            if lp_k:
                out["tops"].append(m.top())
        return out

    # ---------------------------------------------------------------- (1) neutral
    base = run(24)
    m.set_sampler_ex(0.0, 0.0, 0.0, 1.0, 0)
    again = run(24)
    check(base["tokens"] == again["tokens"], f"neutral extras: greedy unchanged ({tok.decode(base['tokens'])[:50]!r})")

    # ---------------------------------------------------------------- (2) candidate list
    r = run(14, lp_k=8, logits=True)
    ok_ids = ok_lp = True
    worst = 0.0
    for lg, (ids, lps), t in zip(r["logits"], r["tops"], r["tokens"]):
        order = np.argsort(-lg.astype(np.float64), kind="stable")[:8]
        ref = log_softmax(lg)
        ok_ids &= ids == order.tolist() and t == int(order[0])
        d = float(np.max(np.abs(np.array(lps) - ref[order])))
        worst = max(worst, d)
        ok_lp &= d < 2e-3
    check(ok_ids and ok_lp, f"logprob_k 8: candidates = top-8 of the logits, logprobs exact (max diff {worst:.1e})")
    r2 = run(14, lp_k=3, top_k=20, temp=1.0, seed=9, top_p=0.95, logits=True)
    check(all(len(t[0]) == 20 for t in r2["tops"]) and
          all(r2["tokens"][i] in r2["tops"][i][0][:20] for i in range(len(r2["tokens"]))),
          "sampled with top_k 20 and logprob_k 3: 20 candidates, the sampled token among them")
    g1 = run(24, lp_k=5)
    check(g1["tokens"] == base["tokens"], "logprob_k > 0 does not change the greedy tokens")

    # ---------------------------------------------------------------- (3) logit_bias
    t1 = base["tokens"][0]
    rb = run(24, bias={t1: -100}, lp_k=2, logits=True)
    check(rb["tokens"][0] != t1 and rb["tokens"][0] == int(np.argsort(-rb["logits"][0].astype(np.float64), kind="stable")[1]),
          f"-100 bans the first greedy token {tok.decode([t1])!r}: now {tok.decode([rb['tokens'][0]])!r} (the runner-up)")
    ref0 = log_softmax(transform(rb["logits"][0], {t1: -100}))
    check(abs(rb["tops"][0][1][0] - ref0[rb["tokens"][0]]) < 2e-3, "banned token: the logprob is over the rest of the vocabulary")
    banned = set(base["tokens"][:6])
    rb2 = run(30, bias={t: -100 for t in banned})
    check(not (set(rb2["tokens"]) & banned), f"{len(banned)} banned tokens never appear in 30 steps")
    force = int(tok.encode("echo", add_special_tokens=False).ids[0])
    rb3 = run(12, bias={force: 100})
    check(rb3["tokens"] == [force] * 12, "+100 forces the token every step")
    sh = {int(base["tokens"][1]): -0.75, force: 3.0}
    rb4 = run(12, bias=sh, logits=True, lp_k=2)
    exp = [int(np.argmax(transform(lg, sh))) for lg in rb4["logits"]]
    # (the path diverges once the bias changes a token: compare each step to the argmax of ITS OWN logits)
    check(rb4["tokens"] == exp, "additive bias: every step is the argmax of logits + bias")
    m.set_logit_bias(None)
    check(run(24)["tokens"] == base["tokens"], "after set_logit_bias(None) the greedy run is the original")

    # ---------------------------------------------------------------- (4) penalties vs numpy
    for ext in ({"presence": 1.0}, {"frequency": 0.5}, {"repetition": 1.3}, {"presence": 0.8, "frequency": 0.3, "repetition": 1.15}):
        rp = run(40, ext=ext, logits=True, lp_k=3)
        gen, bad, worst = Counter(), 0, 0.0
        for j, (lg, t, (ids, lps)) in enumerate(zip(rp["logits"], rp["tokens"], rp["tops"])):
            v = transform(lg, None, gen, prompt, ext.get("presence", 0.0), ext.get("frequency", 0.0), ext.get("repetition", 1.0))
            if v[t] < v.max() - 2e-3:
                bad += 1
            ref = log_softmax(v)
            worst = max(worst, abs(lps[ids.index(t)] - ref[t]) if t in ids else 0.0)
            gen[t] += 1
        check(bad == 0 and worst < 3e-3, f"penalties {ext}: 40 steps = argmax of the penalised logits (counts incl. prompt "
                                         f"for repetition), logprob diff {worst:.1e}; differs from plain greedy at "
                                         f"{next((i for i, (a, b) in enumerate(zip(rp['tokens'], base['tokens'])) if a != b), None)}")
    ext = {"presence": 0.8, "frequency": 0.3, "repetition": 1.15}
    single = run(32, ext=ext)
    multi = run(32, ext=ext, pieces=[10, len(prompt) - 10])
    same = next((i for i, (a, b) in enumerate(zip(single["tokens"], multi["tokens"])) if a != b), 32)
    check(same >= 24, f"two-piece prefill + re-arm = one-piece prefill (first {same} of 32 tokens identical)")
    check(run(24)["tokens"] == base["tokens"], "after penalties: the plain greedy run is the original (counts cleared)")

    # ---------------------------------------------------------------- (5) min_p
    for mp, T in ((0.2, 1.5), (0.05, 1.0)):
        rm = run(60, temp=T, top_k=64, top_p=1.0, seed=21, ext={"min_p": mp}, logits=True, pr=prompt2)
        bad = sum(1 for lg, t in zip(rm["logits"], rm["tokens"]) if lg[t] < lg.max() + T * np.log(mp) - 2e-3)
        check(bad == 0, f"min_p {mp} at temperature {T}: all 60 sampled tokens have p >= min_p * p_max")
    free = run(60, temp=1.5, top_k=64, top_p=1.0, seed=21, logits=True, pr=prompt2)
    badf = sum(1 for lg, t in zip(free["logits"], free["tokens"]) if lg[t] < lg.max() + 1.5 * np.log(0.2) - 2e-3)
    check(badf > 0, f"control: without min_p, {badf} of 60 draws at temperature 1.5 fall below the 0.2 line")
    g = run(30, pr=prompt2)
    mp1 = run(30, temp=1.0, top_k=20, top_p=1.0, seed=3, ext={"min_p": 1.0}, pr=prompt2)
    check(mp1["tokens"] == g["tokens"], "min_p 1.0 at temperature 1 = greedy")

    # ---------------------------------------------------------------- (6) seeds
    s1 = run(40, temp=1.0, top_k=20, top_p=0.95, seed=1234, pr=prompt2)
    s2 = run(40, temp=1.0, top_k=20, top_p=0.95, seed=1234, pr=prompt2)
    s3 = run(40, temp=1.0, top_k=20, top_p=0.95, seed=1235, pr=prompt2)
    check(s1["tokens"] == s2["tokens"] and s1["tokens"] != s3["tokens"], "same seed: same 40 tokens; another seed: other tokens")
    for k in range(3):   # unrelated steps first: the counter is restarted by set_rng_step(0)
        m.step(1000 + k)
    s4 = run(40, temp=1.0, top_k=20, top_p=0.95, seed=1234, pr=prompt2)
    check(s4["tokens"] == s1["tokens"], "history before the request does not matter once the counter is restarted")
    s5 = run(40, temp=1.0, top_k=20, top_p=0.95, seed=1234, pr=prompt2, pieces=[7, len(prompt2) - 7])
    same = next((i for i, (a, b) in enumerate(zip(s1["tokens"], s5["tokens"])) if a != b), 40)
    check(same >= 24, f"a discarded prefill sample before the restart does not shift the draws (first {same} of 40 identical)")
    print("   seed text:", repr(tok.decode(s1["tokens"])[:70]))

    # ---------------------------------------------------------------- (7) the MTP loop: bias, penalties, logprobs
    if eng.has("ne_fn_mtp_create") and eng.has("ne_fn_mtp_generate"):
        L = eng.lib
        L.ne_fn_mtp_create.restype = C.c_void_p
        L.ne_fn_mtp_create.argtypes = [C.c_void_p, C.c_int, C.c_int]
        L.ne_fn_mtp_generate.restype = C.c_int
        L.ne_fn_mtp_generate.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_uint64,
                                         C.c_void_p, C.c_void_p, C.c_void_p]
        mp_ptr = L.ne_fn_mtp_create(C.c_void_p(m.ptr), 4, 4)
        if mp_ptr:
            p = np.ascontiguousarray(prompt, np.int32)
            out = np.zeros(16, np.int32)
            st = np.zeros(10, np.float64)

            def gen():
                return L.ne_fn_mtp_generate(C.c_void_p(mp_ptr), p.ctypes.data, len(p), 8, 248046, 2, 1, out.ctypes.data,
                                            st.ctypes.data, None)
            m.reset()
            m.set_sampler(0.0, 1.0, 1, 0)
            m.set_logit_bias({t1: -100})
            rc = gen()
            check(rc > 0 and int(out[0]) != t1 and t1 not in out[:rc].tolist(),
                  f"the MTP loop applies a ban ({tok.decode(out[:6].tolist())!r}: the banned token never appears)")
            m.set_logit_bias(None)
            m.set_sampler_ex(0.0, 0.5, 0.2, 1.0, 0)
            rc = gen()
            check(rc > 0, f"the MTP loop runs with presence / frequency penalties ({rc} tokens)")
            m.set_sampler_ex(0.0, 0.0, 0.0, 1.0, 3)
            rc = gen()
            check(rc < 0 and "logprobs" in eng.last_error(), f"the MTP loop refuses logprobs ({eng.last_error()[:60]})")
            m.set_sampler_ex(0.0, 0.0, 0.0, 1.0, 0)
            m.reset()
            rc = gen()
            check(rc > 0 and out[:rc].tolist() == base["tokens"][:rc], f"the MTP loop with every knob cleared = the plain greedy run ({rc} tokens)")
    m.reset()

    # ---------------------------------------------------------------- (8) GPU step time
    T = 2048
    m.set_sampler(1.0, 0.95, 20, 7)
    m.time(T, 10)
    modes = [("off", {}), ("bias(1 token)", {"bias": {5: -1.0}}), ("penalties", {"presence": 0.5, "frequency": 0.2, "repetition": 1.1}),
             ("min_p", {"min_p": 0.05}), ("logprob_k 20", {"lp_k": 20}),
             ("all", {"bias": {5: -1.0}, "presence": 0.5, "frequency": 0.2, "repetition": 1.1, "min_p": 0.05, "lp_k": 20}),
             ("off", {})]
    times = {}
    for name, kw in modes:
        m.set_logit_bias(kw.get("bias"))
        m.set_sampler_ex(kw.get("min_p", 0.0), kw.get("presence", 0.0), kw.get("frequency", 0.0), kw.get("repetition", 1.0),
                         kw.get("lp_k", 0))
        r = [m.time(T, 60)["gpu_ms_median"] for _ in range(4)]
        times.setdefault(name, []).append(statistics.median(r))
        print(f"   step GPU ms with {name}: {statistics.median(r):.3f}  (rounds {[round(x, 3) for x in r]})", flush=True)
    m.set_logit_bias(None)
    m.set_sampler_ex(0.0, 0.0, 0.0, 1.0, 0)
    print("SAMPLER_STEP_MS " + str({k: [round(x, 3) for x in v] for k, v in times.items()}))
    print(f"\n{len(FAILS)} failed: {FAILS}" if FAILS else "\nall passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
