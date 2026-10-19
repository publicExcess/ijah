"""lab-sampler: the MTP loop with logit_bias / presence / frequency / repetition penalties / min_p stays exact.

    NEED_GB=85 LOCK_OWNER=lab-sampler tools/with_lock.sh "mtp knobs (~83 GB)" HH:MM -- $PY tests/mtp/test_mtp_knobs.py
        [--runs 600] [--depth 2]

The verify rows get the sampler's transform (kernel mtp_logit_xform: bias, then penalties with the counts of the tokens
generated so far plus the row's own drafts), the acceptance compares the transformed target distributions with the
draft's proposals (which ignore the knobs: any proposal is exact), min_p is applied in the host distribution. Checks:
(1) greedy MTP == greedy step path under the knobs (a fork is accepted only where the two chosen tokens' transformed 1-row
logits are within 0.75: the verify rows' multi-row numerics differ from the 1-row decode's by a few bf16 steps, most on
special tokens after off-distribution text: the same prompt WITHOUT knobs forks at 46 with a 0.25 gap; the transform itself is
checked exactly by tests/server/sampler_check.py);
(2) sampled MTP, many seeds: out[0] (the step path's sample), out[1] | out[0] and out[2] | out[0], out[1] against the exact
filtered distribution computed from the 1-row decode logits and a float64 numpy transform (the same chi-square /
total-variation-noise test as tests/mtp/test_sampled_dist.py); a control against the untransformed distribution (the
test must be able to fail); (3) the counts after a run equal the generated tokens.
"""
import argparse
import os
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
import engine as E  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
import bench_mtp as B  # noqa: E402

V = 248320
TEMP, TOP_P, TOP_K = 1.0, 0.95, 20


def transform(lg, bias, gen, prompt, presence, frequency, repetition):
    v = lg.astype(np.float64).copy()
    for t, b in (bias or {}).items():
        v[t] = -np.inf if b <= -100 else v[t] + b
    cnt = np.zeros(V)
    for t, c in gen.items():
        cnt[t] = c
    seen = cnt > 0
    if repetition != 1.0:
        seen = seen.copy()
        seen[list(set(prompt))] = True
        v = np.where(seen, np.where(v < 0, v * repetition, v / repetition), v)
    return np.where(cnt > 0, v - frequency * cnt - presence, v)


def filt(v, min_p=0.0):
    """temperature -> top-k -> min_p -> top-p: {id: prob}"""
    idx = np.lexsort((np.arange(len(v)), -v))[:TOP_K]
    w = np.exp((v[idx] - v[idx[0]]) / TEMP)
    if min_p > 0:
        w = w[:max(1, int(np.sum(w >= min_p)))]
    c = np.cumsum(w / w.sum())
    n = min(int(np.searchsorted(c, TOP_P) + 1), len(w)) if TOP_P < 1 else len(w)
    w = w[:n] / w[:n].sum()
    return dict(zip(idx[:n].tolist(), w.tolist()))


def compare(name, samples, p, rng):
    n = len(samples)
    cnt = Counter(samples)
    tv = 0.5 * sum(abs(cnt.get(t, 0) / n - p.get(t, 0.0)) for t in set(cnt) | set(p))
    ids = list(p)
    pr = np.array([p[t] for t in ids])
    sim = np.array([0.5 * np.abs(rng.multinomial(n, pr) / n - pr).sum() for _ in range(2000)])
    outside = sum(v for t, v in cnt.items() if t not in p)
    pval = float((sim >= tv).mean())
    print(f"  {name}: {n} samples, {len(cnt)} distinct, TV {tv:.4f} (noise: median {np.median(sim):.4f}, 95% "
          f"{np.quantile(sim, 0.95):.4f}; P(noise >= TV) {pval:.3f}), {outside} outside p's support", flush=True)
    return pval, outside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=ijah_paths.pack())
    ap.add_argument("--runs", type=int, default=600)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--prompt", default="text_crisis")
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(Path(a.pack) / "tokenizer.json"))
    ids0 = tok.encode(B.chat(a.pack, B.THINK[a.prompt]), add_special_tokens=False).ids
    echo = tok.encode("<|im_start|>user\nRepeat the word echo forty times, separated by spaces.<|im_end|>\n"
                      "<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False).ids
    eng = E.Engine()
    pack = eng.pack_open(a.pack)
    m = eng.fn_model(pack, 1024)
    mtp = B.Mtp(eng, m, max(4, a.depth), 4)
    rng = np.random.default_rng(0)
    state = {"ok": True}

    def knobs_on(kn, ids):
        m.set_logit_bias(kn.get("bias"))
        m.set_sampler_ex(kn.get("min_p", 0.0), kn.get("presence", 0.0), kn.get("frequency", 0.0), kn.get("repetition", 1.0), 0)
        if kn.get("repetition", 1.0) != 1.0:
            m.set_seen(ids)

    def knobs_off():
        m.set_logit_bias(None)
        m.set_sampler_ex(0.0, 0.0, 0.0, 1.0, 0)

    def uncertain_spot(ids):
        """extend the prompt along the greedy continuation to a position whose next token has >= 6 ids under the filter"""
        m.set_sampler(0.0, 1.0, 1, 0)
        m.reset()
        cont, _, _ = m.generate(ids, 120, -1)
        m.reset()
        _, lg, _ = m.prefill(ids, logits=True)
        raw = [lg]
        for t in cont[:-1]:
            _, _, _, lg = m.step(t, True, logits=True)
            raw.append(lg)
        j = next(j for j in range(1, len(cont) - 2) if len(filt(raw[j].astype(np.float64))) >= 6)
        top = [int(t) for t in np.lexsort((np.arange(V), -raw[j].astype(np.float64)))[:4]]
        return ids + cont[:j], j, top

    def run_set(name, ids, kn, runs, seed0=7000, n_greedy=48):
        ok = True
        print(f"== {name}: {kn}", flush=True)
        # (1) greedy: MTP == step path under the knobs
        m.set_sampler(0.0, 1.0, 1, 0)
        knobs_on(kn, ids)
        g_mtp, _ = mtp.generate(ids, n_greedy, a.depth, seed=1)
        m.reset()
        knobs_on(kn, ids)
        g_step, _, _ = m.generate(ids, n_greedy, -1)
        knobs_off()
        m.reset()
        g_free, _, _ = m.generate(ids, n_greedy, -1)
        mism = next((i for i, (x, y) in enumerate(zip(g_mtp, g_step)) if x != y), None)
        print(f"  greedy MTP vs step with the knobs: {len(g_mtp)} tokens, first mismatch {mism} "
              f"({tok.decode(g_mtp[:12])!r})", flush=True)
        if mism is not None:   # the two chosen tokens' transformed logits at the fork (from the 1-row decode)
            m.set_sampler(0.0, 1.0, 1, 0)
            m.reset()
            _, lg, _ = m.prefill(ids, logits=True)
            for t in g_step[:mism]:
                _, _, _, lg = m.step(t, True, logits=True)
            v = transform(lg, kn.get("bias"), Counter(g_step[:mism]), ids, kn.get("presence", 0.0), kn.get("frequency", 0.0), kn.get("repetition", 1.0))
            gap = float(abs(v[g_step[mism]] - v[g_mtp[mism]]))
            ulp = 2.0 ** (np.floor(np.log2(max(abs(v[g_step[mism]]), 1e-3))) - 7)
            print(f"  fork at {mism}: step {g_step[mism]} ({v[g_step[mism]]:.4f}) vs MTP {g_mtp[mism]} ({v[g_mtp[mism]]:.4f}): "
                  f"gap {gap:.4f} = {gap / ulp:.1f} bf16 steps", flush=True)
            ok &= gap <= 0.76
        if runs == 0:
            same_as_free = next((i for i, (x, y) in enumerate(zip(g_step, g_free)) if x != y), None)
            print(f"  the knobs change the greedy text at token {same_as_free}: {tok.decode(g_step[:30])!r}", flush=True)
            state["ok"] &= ok and same_as_free is not None
            return
        # (2) sampled
        outs = []
        for s in range(runs):
            m.set_sampler(TEMP, TOP_P, TOP_K, seed0 + s)   # the prefill sample (step path) varies with the seed
            knobs_on(kn, ids)
            o, _ = mtp.generate(ids, 4, a.depth, seed=1000 + s)
            outs.append(o)
        knobs_off()

        def ref(prefix):
            """exact filtered distribution of the token after ids + prefix, from raw 1-row logits + numpy transform"""
            m.set_sampler(0.0, 1.0, 1, 0)
            m.reset()
            _, lg, _ = m.prefill(ids, logits=True)
            for t in prefix:
                _, _, _, lg = m.step(t, True, logits=True)
            v = transform(lg, kn.get("bias"), Counter(prefix), ids, kn.get("presence", 0.0), kn.get("frequency", 0.0), kn.get("repetition", 1.0))
            return filt(v, kn.get("min_p", 0.0)), filt(lg.astype(np.float64))

        p0, p0_raw = ref([])
        pv, outside = compare("out[0] (step sample) vs exact", [o[0] for o in outs], p0, rng)
        ok &= pv >= 0.01 and outside == 0
        controls = []
        tv0 = 0.5 * sum(abs(p0.get(t, 0.0) - p0_raw.get(t, 0.0)) for t in set(p0) | set(p0_raw))
        if tv0 > 0.05:
            controls.append(compare(f"  control: out[0] vs the UNtransformed p (TV between the two {tv0:.3f}; must fail)",
                                    [o[0] for o in outs], p0_raw, rng)[0])
        by1 = {}
        for o in outs:
            by1.setdefault(o[0], []).append(o)
        for v0, lst in sorted(by1.items(), key=lambda kv: -len(kv[1]))[:2]:
            if len(lst) < 50:
                continue
            p1, p1_raw = ref([v0])
            pv, outside = compare(f"out[1] | out[0] = {tok.decode([v0])!r}", [o[1] for o in lst], p1, rng)
            ok &= pv >= 0.01 and outside == 0
            tvc = 0.5 * sum(abs(p1.get(t, 0.0) - p1_raw.get(t, 0.0)) for t in set(p1) | set(p1_raw))
            if tvc > 0.05:   # the knobs move this distribution: the untransformed one must be rejected
                controls.append(compare(f"  control: out[1] vs the untransformed p (TV between the two {tvc:.3f}; must fail)",
                                        [o[1] for o in lst], p1_raw, rng)[0])
            by2 = {}
            for o in lst:
                by2.setdefault(o[1], []).append(o[2])
            for v1, l2 in sorted(by2.items(), key=lambda kv: -len(kv[1]))[:1]:
                if len(l2) < 40:
                    continue
                p2, p2_raw = ref([v0, v1])
                pv, outside = compare(f"out[2] | out[0], out[1] = {tok.decode([v0])!r}, {tok.decode([v1])!r}", l2, p2, rng)
                ok &= pv >= 0.01 and outside == 0
                tvc = 0.5 * sum(abs(p2.get(t, 0.0) - p2_raw.get(t, 0.0)) for t in set(p2) | set(p2_raw))
                if tvc > 0.05:
                    controls.append(compare(f"  control: out[2] vs the untransformed p (TV between the two {tvc:.3f}; must fail)",
                                            l2, p2_raw, rng)[0])
        rej = sum(1 for c in controls if c < 0.01)
        print(f"  controls rejected (P < 0.01): {rej} of {len(controls)}", flush=True)
        state["ok"] &= ok and rej >= 1

    ids, j, top = uncertain_spot(ids0)
    print(f"prompt extended by {j} greedy tokens", flush=True)
    run_set("bias + presence + frequency", ids, dict(bias={top[0]: -1.0, top[2]: 1.5}, presence=1.0, frequency=0.6), a.runs, 7000)
    run_set("repetition + min_p + ban", ids, dict(repetition=1.25, min_p=0.1, bias={top[1]: -100}), a.runs, 107000)
    # a repeating answer: the draft head predicts the repeats, so a verify row's counts must include the round's own
    # earlier drafts (extra in mtp_logit_xform); greedy MTP must equal the step path for every strength
    for kn in (dict(presence=0.5, frequency=0.3), dict(presence=2.0, frequency=1.0), dict(presence=6.0, frequency=1.5),
               dict(repetition=3.0), dict(presence=3.0, frequency=2.0, bias={tok.encode("echo", add_special_tokens=False).ids[0]: -2.0})):
        run_set("a repeating answer, greedy", echo, kn, 0, n_greedy=64)
    print("mtp-knobs:", "OK" if state["ok"] else "CHECK")
    pack.close()
    sys.exit(0 if state["ok"] else 1)


if __name__ == "__main__":
    main()
