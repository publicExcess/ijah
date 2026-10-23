"""lab-loop: end-to-end distribution check of sampled MTP (exact speculative sampling with sampled drafts).

    NEED_GB=85 LOCK_OWNER=lab-loop tools/with_lock.sh "sampled-dist (~83 GB)" HH:MM -- $PY tests/mtp/test_sampled_dist.py
        [--runs 400] [--depth 3] [--prompt text_crisis]

The model's sampler seed is fixed, so the prefill's sample x_P is the same in every run; the MTP seed varies. The
first round's first emitted token (out[1]: d_1 if accepted, else the correction) must then be distributed exactly as
the target's filtered distribution p(. | prompt, x_P) (temperature 1, top-k 20, top-p 0.95), computed here from the
1-row decode logits. Reports the empirical vs p total variation, the TV that sampling noise alone gives at this
many runs (simulated from p), a chi-square over p's support, the share of rounds whose d_1 was accepted, and the
same for out[2] vs the target's distribution given (x_P, out[1]) (computed per observed out[1]).

--pool N (lab-dflfix): every output position, not just out[1] (always verified) -- N MTP runs (restore an MTP snapshot of
the prompt, generate_from 1 + K tokens, sampler and MTP seeds varied) against N plain step-path samples from the same
snapshot (the target itself), a two-sample chi-square per position t = 0 .. K (tokens with < 10 pooled samples merged)
and per consecutive pair (t - 1, t). A gate that verifies a draft depending on its own value shows up at t >= 2.
"""
import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
os.environ.setdefault("NE_FN_ROUND", "f32")
import refenv  # noqa: E402

refenv.add_paths()
import numpy as np  # noqa: E402
import engine as E  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
import bench_mtp as B  # noqa: E402

TEMP, TOP_P, TOP_K = 1.0, 0.95, 20


def filt(lg):
    """temperature -> top-k -> top-p (smallest prefix with mass >= p), lowest id on ties: {id: prob}"""
    idx = np.lexsort((np.arange(len(lg)), -lg.astype(np.float64)))[:TOP_K]
    w = np.exp((lg[idx].astype(np.float64) - lg[idx[0]]) / TEMP)
    c = np.cumsum(w / w.sum())
    n = int(np.searchsorted(c, TOP_P) + 1) if TOP_P < 1 else len(idx)
    n = min(n, len(idx))
    w = w[:n] / w[:n].sum()
    return dict(zip(idx[:n].tolist(), w.tolist()))


def compare(name, samples, p, rng):
    n = len(samples)
    cnt = Counter(samples)
    tv = 0.5 * sum(abs(cnt.get(t, 0) / n - p.get(t, 0.0)) for t in set(cnt) | set(p))
    ids = list(p)
    pr = np.array([p[t] for t in ids])
    sim = []
    for _ in range(2000):
        c = rng.multinomial(n, pr)
        sim.append(0.5 * np.abs(c / n - pr).sum())
    sim = np.array(sim)
    outside = sum(v for t, v in cnt.items() if t not in p)
    chi = sum((cnt.get(t, 0) - n * p[t]) ** 2 / (n * p[t]) for t in ids)
    pval = float((sim >= tv).mean())
    print(f"  {name}: {n} samples, {len(cnt)} distinct, TV {tv:.4f} (noise: median {np.median(sim):.4f}, 95% "
          f"{np.quantile(sim, 0.95):.4f}; P(noise >= TV) {pval:.2f}), chi2 {chi:.1f} over {len(ids)} ids, "
          f"{outside} outside p's support")
    return pval >= 0.01 and outside == 0


def chi2_p(x, k):
    """upper tail of chi-square(k) at x (Wilson-Hilferty)"""
    import math
    if k <= 0:
        return 1.0
    z = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    return 0.5 * math.erfc(z / math.sqrt(2))


def two_sample(a, b, minc=10):
    ca, cb = Counter(a), Counter(b)
    keys = [k for k in set(ca) | set(cb) if ca[k] + cb[k] >= minc]
    ra = len(a) - sum(ca[k] for k in keys)
    rb = len(b) - sum(cb[k] for k in keys)
    cells = [(ca[k], cb[k]) for k in keys] + ([(ra, rb)] if ra + rb > 0 else [])
    na, nb = len(a), len(b)
    chi = 0.0
    for x, y in cells:
        tot = x + y
        ea, eb = tot * na / (na + nb), tot * nb / (na + nb)
        chi += (x - ea) ** 2 / ea + (y - eb) ** 2 / eb
    return chi, len(cells) - 1, chi2_p(chi, len(cells) - 1)


def pooled(a, eng, m, mtp, ids, tok):
    """--pool: MTP / DFlash generations vs the step path from one snapshot, every position"""
    import ctypes as C
    L = eng.lib
    for f, r, at in (("ne_fn_mtp_feed", C.c_int, [C.c_void_p, C.c_void_p, C.c_int]),
                     ("ne_fn_mtp_snapshot", C.c_void_p, [C.c_void_p]),
                     ("ne_fn_mtp_restore", C.c_int, [C.c_void_p, C.c_void_p]),
                     ("ne_fn_mtp_generate_from", C.c_int, [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int,
                                                          C.c_uint64, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p,
                                                          C.c_void_p])):
        getattr(L, f).restype, getattr(L, f).argtypes = r, at
    P = C.c_void_p(mtp.ptr)
    K = a.pool_k
    m.reset()
    pre = np.ascontiguousarray(ids[:-1], np.int32)
    eng._ok(L.ne_fn_mtp_feed(P, pre.ctypes.data, len(pre)), "feed")
    snap = L.ne_fn_mtp_snapshot(P)
    assert snap, eng.last_error()
    last = np.ascontiguousarray(ids[-1:], np.int32)
    out = np.zeros(K + 1, np.int32)
    runs_m, runs_s = [], []
    for s in range(a.pool):
        eng._ok(L.ne_fn_mtp_restore(P, snap), "restore")
        m.set_sampler(TEMP, TOP_P, TOP_K, 100000 + s)
        k = L.ne_fn_mtp_generate_from(P, last.ctypes.data, 1, K + 1, -1, a.depth, 1000 + s, out.ctypes.data, None, None,
                                      None, None)
        assert k == K + 1, (k, eng.last_error())
        runs_m.append(tuple(out.tolist()))
        eng._ok(L.ne_fn_mtp_restore(P, snap), "restore")
        m.set_sampler(TEMP, TOP_P, TOP_K, 500000 + s)
        t, seq = int(ids[-1]), []
        for _ in range(K + 1):
            t = m.step(t, True)[0]
            seq.append(t)
        runs_s.append(tuple(seq))
        if (s + 1) % 2000 == 0:
            print(f"  {s + 1} / {a.pool} runs", flush=True)
    ok = True
    for t in range(K + 1):
        chi, dof, pv = two_sample([r[t] for r in runs_m], [r[t] for r in runs_s])
        chi2, dof2, pv2 = two_sample([r[t - 1:t + 1] for r in runs_m], [r[t - 1:t + 1] for r in runs_s]) if t else (0, 0, 1)
        print(f"  out[{t}]: MTP vs step chi2 {chi:.1f} / {dof} dof, P {pv:.3f}" +
              (f"; pair (out[{t - 1}], out[{t}]) chi2 {chi2:.1f} / {dof2} dof, P {pv2:.3f}" if t else ""), flush=True)
        ok &= pv >= 1e-3 and pv2 >= 1e-3
    print("sampled-dist pooled:", "OK" if ok else "CHECK")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=refenv.ijah_paths.pack())
    ap.add_argument("--runs", type=int, default=400)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--prompt", default="text_crisis")
    ap.add_argument("--dflash", default="", help="lab-dflash: DFlash2 checkpoint dir (its gated rounds draft)")
    ap.add_argument("--dflash-gate", type=int, default=1, help="lab-dfllook: 1 sequential exact gate, 2 lookahead")
    ap.add_argument("--pool", type=int, default=0, help="lab-dflfix: N runs per side of the pooled every-position check")
    ap.add_argument("--pool-k", type=int, default=6, help="tokens after the first one per pooled run")
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(Path(a.pack) / "tokenizer.json"))
    ids = tok.encode(B.chat(a.pack, B.THINK[a.prompt]), add_special_tokens=False).ids
    eng = E.Engine()
    pack = eng.pack_open(a.pack)
    m = eng.fn_model(pack, 1024)
    mtp = B.Mtp(eng, m, max(4, a.depth), 4)
    if a.dflash:  # lab-dflash (docs/mtp.md 13)
        import ctypes as C
        en = eng.lib.ne_fn_dfl_enable
        en.restype, en.argtypes = C.c_int, [C.c_void_p, C.c_char_p, C.c_int, C.c_int, C.c_int, C.c_int]
        if en(C.c_void_p(mtp.ptr), a.dflash.encode(), 8, max(4, a.depth), 8, a.dflash_gate):
            raise E.EngineError(eng.last_error())
    # find a spot with entropy: along the greedy continuation, position j whose token is certain under the filter
    # (so x_P = cont[j] whatever the seed) and whose next token has >= 5 ids and top <= 0.7
    m.set_sampler(0.0, 1.0, 1, 0)
    m.reset()
    cont, _, _ = m.generate(ids, 160, -1)
    m.reset()
    _, lg, _ = m.prefill(ids, logits=True)
    dists = [filt(lg)]
    for t in cont[:-1]:
        _, _, _, lg = m.step(t, True, logits=True)
        dists.append(filt(lg))
    j = next((j for j in range(1, len(cont) - 1) if len(dists[j]) == 1 and len(dists[j + 1]) >= 5
              and max(dists[j + 1].values()) <= 0.7), None)
    assert j is not None, "no position with a certain token followed by an uncertain one"
    ids = ids + cont[:j]
    print(f"prompt extended by {j} greedy tokens; next {tok.decode([cont[j]])!r} is certain, then "
          f"{len(dists[j + 1])} ids (top {max(dists[j + 1].values()):.2f})")
    if a.pool:
        ok = pooled(a, eng, m, mtp, ids, tok)
        pack.close()
        sys.exit(0 if ok else 1)
    m.set_sampler(TEMP, TOP_P, TOP_K, 1234)
    outs = []
    for s in range(a.runs):
        o, st = mtp.generate(ids, 3, a.depth, seed=1000 + s)
        outs.append(o)
    xp = {o[0] for o in outs}
    assert len(xp) == 1, f"prefill sample varies: {xp}"
    xp = xp.pop()
    # reference distributions from the 1-row decode
    m.reset()
    m.prefill(ids)
    _, _, _, lg1 = m.step(xp, True, logits=True)
    p1 = filt(lg1)
    rng = np.random.default_rng(0)
    print(f"== {a.prompt}: depth {a.depth}, x_P = {xp} {tok.decode([xp])!r}; p(. | x_P) has {len(p1)} ids, top "
          f"{max(p1.values()):.3f}")
    ok = compare("out[1] vs p(. | x_P)", [o[1] for o in outs], p1, rng)
    # out[2] given out[1]: pool the per-value comparisons through a mixture p2 = sum_v P(out1 = v) p(. | x_P, v)
    by = {}
    for o in outs:
        by.setdefault(o[1], []).append(o[2])
    for v, lst in sorted(by.items(), key=lambda kv: -len(kv[1]))[:2]:
        if len(lst) < 60:
            continue
        m.reset()
        m.prefill(ids)
        m.step(xp, True)
        _, _, _, lg2 = m.step(v, True, logits=True)
        ok &= compare(f"out[2] | out[1] = {tok.decode([v])!r}", lst, filt(lg2), rng)
    print("sampled-dist:", "OK" if ok else "CHECK")
    pack.close()


if __name__ == "__main__":
    main()
