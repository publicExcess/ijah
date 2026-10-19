"""lab-sampler: the sampler kernels' extended knobs on synthetic logits (no model; a few seconds on the GPU).

    $PY tests/server/sampler_check.py            # run under tools/with_lock.sh (GPU), no NEED_GB needed

fn_sample_p1 / p2 (kernels/fn_model.metal) with logit_bias (ban = bias <= -100), presence / frequency / repetition
penalties over a counts buffer (prompt flag + generated counts; pass 2 counts what it samples), min_p, the candidate
list for logprobs (cand_k, lp_on) and their order with the token mask, each against a float64 numpy reference.
Exit 0 = all pass.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import engine as E  # noqa: E402
import numpy as np  # noqa: E402

VOCAB = 248320
NB = (VOCAB + 1023) // 1024
FLAG = 0x80000000
ok_all = True


def check(name, ok, note=""):
    global ok_all
    ok_all &= bool(ok)
    print(("PASS " if ok else "FAIL ") + name + (f"  [{note}]" if note else ""), flush=True)


class Sampler:
    """one set of device buffers + the two kernels, parameters as the host writes them (models/flash_next/model.h)"""

    def __init__(self, eng, logits):
        self.eng = eng
        self.lg = eng.from_numpy(np.ascontiguousarray(logits, np.float32))
        self.cand = eng.alloc(NB * 64 * 8)
        self.lse = eng.alloc(NB * 8)
        self.res = eng.alloc(16)
        self.tok = eng.alloc(16)
        self.top = eng.alloc(64 * 8)
        self.mask = eng.from_numpy(np.full((VOCAB + 31) // 32, 0xFFFFFFFF, np.uint32))
        self.bias = eng.from_numpy(np.zeros(VOCAB, np.float32))
        self.cnt = eng.from_numpy(np.zeros(VOCAB, np.uint32))
        self.sp = eng.alloc(64)

    def set_bias(self, d):
        b = np.zeros(VOCAB, np.float32)
        for k, v in d.items():
            b[k] = v
        self._reup("bias", b)

    def _reup(self, name, arr):
        setattr(self, name, self.eng.from_numpy(np.ascontiguousarray(arr)))

    def set_counts(self, c):
        self._reup("cnt", np.asarray(c, np.uint32))

    def set_mask(self, allowed):
        bits = np.zeros((VOCAB + 31) // 32 * 32, np.uint8)
        bits[list(allowed)] = 1
        self._reup("mask", np.packbits(bits, bitorder="little").view(np.uint32))

    def run(self, temp=0.0, top_p=1.0, top_k=1, seed=0, step=1, mask_on=0, min_p=0.0, presence=0.0, frequency=0.0,
            repetition=1.0, bias_on=0, pen_on=0, cand_k=0, lp_on=0):
        w = np.zeros(16, np.uint32)
        w[0:2] = np.array([temp, top_p], np.float32).view(np.uint32)
        w[2:7] = [top_k, seed, step, VOCAB, mask_on]
        w[7:11] = np.array([min_p, presence, frequency, repetition], np.float32).view(np.uint32)
        w[11:15] = [bias_on, pen_on, cand_k, lp_on]
        self.sp = self.eng.from_numpy(w)
        e = self.eng
        e.run_kernel("fn_sample_p1", [self.lg, self.sp, self.cand, self.lse, self.mask, self.bias, self.cnt],
                     (NB, 1, 1), (1024, 1, 1))
        e.run_kernel("fn_sample_p2", [self.cand, self.lse, self.sp, self.res, self.tok, self.top, self.cnt],
                     (1, 1, 1), (1024, 1, 1))
        r = self.res.download(np.uint32, (4,))
        return int(r[0]), float(r[1:2].view(np.float32)[0]), int(r[2]), int(r[3])

    def top_list(self, n):
        t = self.top.download(np.uint32, (64, 2))[:n]
        return t[:, 0].astype(int).tolist(), t[:, 1].copy().view(np.float32).tolist()

    def counts(self):
        return self.cnt.download(np.uint32, (VOCAB,))


def transform(logits, bias=None, cnt=None, presence=0.0, frequency=0.0, repetition=1.0, allowed=None):
    """float64 reference of pass 1's logit transform"""
    v = logits.astype(np.float64).copy()
    if bias is not None:
        ban = bias <= -100
        v = v + bias
        v[ban] = -np.inf
    if cnt is not None:
        seen = cnt != 0
        g = (cnt & 0x0FFFFFFF).astype(np.float64)
        if repetition != 1.0:
            v = np.where(seen, np.where(v < 0, v * repetition, v / repetition), v)
        v = np.where(seen, v - frequency * g - presence * (g > 0), v)
    if allowed is not None:
        m = np.full(VOCAB, False)
        m[list(allowed)] = True
        v = np.where(m, v, -np.inf)
    return v


def log_softmax(v):
    m = v[np.isfinite(v)].max()
    return v - (m + np.log(np.exp(v[np.isfinite(v)] - m).sum()))


def main():
    eng = E.Engine()
    rng = np.random.default_rng(5)
    base = (rng.standard_normal(VOCAB) * 1.5).astype(np.float32)
    hot = rng.choice(VOCAB, 40, replace=False)
    base[hot] = np.linspace(9, 5, 40, dtype=np.float32)   # a clear top-40
    base = np.round(base * 16) / 16                         # bf16-like values
    s = Sampler(eng, base)
    order = np.argsort(-base.astype(np.float64), kind="stable")
    lp0 = log_softmax(base.astype(np.float64))

    # 1. neutral: greedy = argmax, logprob = log-softmax
    t, lp, am, K = s.run()
    check("neutral greedy = argmax, logprob exact", t == order[0] and am == order[0] and abs(lp - lp0[order[0]]) < 2e-4,
          f"{t} lp {lp:.4f} vs {lp0[order[0]]:.4f}")

    # 2. logit_bias
    top1, top2 = int(order[0]), int(order[1])
    s.set_bias({top1: -100})
    t, lp, am, _ = s.run(bias_on=1)
    ref = log_softmax(transform(base, np.where(np.arange(VOCAB) == top1, -100.0, 0.0)))
    check("bias -100 bans the top token (greedy -> second, logprob over the rest)",
          t == top2 and am == top2 and abs(lp - ref[top2]) < 2e-4, f"{t} lp {lp:.4f} vs {ref[top2]:.4f}")
    b = np.zeros(VOCAB)
    b[top2] = -0.75
    b[int(order[7])] = 3.0
    s.set_bias({top2: -0.75, int(order[7]): 3.0})
    t, lp, _, _ = s.run(bias_on=1)
    ref = log_softmax(transform(base, b))
    want = int(np.argmax(transform(base, b)))
    check("additive bias moves the argmax and the logprob", t == want and abs(lp - ref[want]) < 2e-4,
          f"{t} (ref {want}) lp {lp:.4f} vs {ref[want]:.4f}")
    t, _, _, _ = s.run(bias_on=0)
    check("bias_on = 0 ignores the buffer", t == top1)
    s.set_bias({})

    # 3. penalties: counts = prompt flag on the top 3, generated counts 2 / 5 on the next two
    c = np.zeros(VOCAB, np.uint32)
    c[order[:3]] = FLAG
    c[order[3]] = 2
    c[order[4]] = 5
    c[order[5]] = FLAG | 1
    s.set_counts(c)
    for (pres, freq, rep) in ((0.0, 0.0, 1.5), (1.0, 0.0, 1.0), (0.0, 0.7, 1.0), (0.5, 0.3, 1.3), (2.0, 1.0, 2.0)):
        s.set_counts(c)   # (pass 2 counts the token it samples: a fresh table per run)
        v = transform(base, None, c, pres, freq, rep)
        want = int(np.argmax(v))
        t, lp, am, _ = s.run(presence=pres, frequency=freq, repetition=rep, pen_on=1)
        ref = log_softmax(v)
        check(f"penalties presence {pres} frequency {freq} repetition {rep}: greedy and logprob",
              t == want and abs(lp - ref[want]) < 3e-4, f"{t} (ref {want}) lp {lp:.4f} vs {ref[want]:.4f}")
    # negative logits: repetition multiplies them (more negative)
    neg = np.full(VOCAB, -3.0, np.float32)
    neg[100] = -1.0
    neg[200] = -2.0
    sn = Sampler(eng, neg)
    cn = np.zeros(VOCAB, np.uint32)
    cn[100] = FLAG
    sn.set_counts(cn)
    t, _, _, _ = sn.run(repetition=4.0, pen_on=1)
    check("repetition on a negative logit multiplies it (-1 -> -4: token 200 wins)", t == 200, f"{t}")
    # pass 2 counts what it samples
    s.set_counts(np.zeros(VOCAB, np.uint32))
    toks = []
    for i in range(4):
        t, _, _, _ = s.run(presence=0.6, pen_on=1, step=i + 1)
        toks.append(t)
    cc = s.counts()
    check("pass 2 counts every sampled token; presence makes greedy move on",
          all(int(cc[t]) & 0x0FFFFFFF == toks.count(t) for t in set(toks)) and int((cc & 0x0FFFFFFF).sum()) == 4
          and len(set(toks)) >= 2, f"{toks}")
    s.set_counts(np.zeros(VOCAB, np.uint32))
    s.run(pen_on=0)
    check("pen_on = 0 neither reads nor counts", int(s.counts().sum()) == 0)

    # 4. order with the mask: banned/masked tokens, penalties never revive them
    allowed = [int(x) for x in order[:6]]
    s.set_mask(allowed)
    c = np.zeros(VOCAB, np.uint32)
    c[order[0]] = FLAG | 3
    s.set_counts(c)
    v = transform(base, None, c, 1.0, 1.0, 1.0, allowed)
    t, lp, _, _ = s.run(presence=1.0, frequency=1.0, pen_on=1, mask_on=1)
    ref = log_softmax(v)
    want = int(np.argmax(v))
    check("mask after penalties: logprob over the allowed set only", t == want and t in allowed and abs(lp - ref[want]) < 3e-4,
          f"{t} lp {lp:.4f} vs {ref[want]:.4f}")
    s.set_mask(range(VOCAB))
    s.set_counts(np.zeros(VOCAB, np.uint32))

    # 5. candidate list (logprobs) independent of the draw's top-k
    for (tk, ck) in ((1, 20), (20, 5), (5, 20), (64, 64)):
        t, lp, _, K = s.run(temp=1.0, top_p=0.95, top_k=tk, seed=3, step=1, cand_k=ck, lp_on=1)
        ids, lps = s.top_list(K)
        n = max(tk, ck)
        ok = K == n and ids == [int(x) for x in order[:n]] and np.max(np.abs(np.array(lps) - lp0[order[:n]])) < 3e-4
        check(f"top list: top_k {tk}, cand_k {ck} -> {K} candidates sorted, logprobs exact", ok)
    s.set_bias({top1: -100, top2: 2.0})
    t, lp, _, K = s.run(top_k=1, cand_k=6, lp_on=1, bias_on=1)
    ids, lps = s.top_list(K)
    b = np.zeros(VOCAB)
    b[top1] = -100
    b[top2] = 2.0
    v = transform(base, b)
    o2 = np.argsort(-v, kind="stable")[:6]
    ref = log_softmax(v)
    check("top list under bias: transformed order and logprobs", ids == [int(x) for x in o2] and
          np.max(np.abs(np.array(lps) - ref[o2])) < 3e-4 and t == int(o2[0]))
    s.set_bias({})

    # 6. min_p / top-p draw frequencies vs the exact distribution
    lg2 = (rng.standard_normal(VOCAB) * 0.5).astype(np.float32)
    idx = rng.choice(VOCAB, 30, replace=False)
    lg2[idx] = np.linspace(7, 4.5, 30, dtype=np.float32)
    s2 = Sampler(eng, lg2)
    o = np.argsort(-lg2.astype(np.float64), kind="stable")
    N = 3000
    for (temp, tk, mp, tp) in ((1.0, 20, 0.1, 1.0), (0.8, 30, 0.2, 0.95), (1.0, 20, 0.0, 0.9), (1.5, 64, 0.05, 0.9)):
        p = np.exp((lg2[o[:tk]].astype(np.float64) - lg2[o[0]]) / temp)
        nk = int(np.sum(p >= mp)) if mp > 0 else tk
        nk = max(nk, 1)
        p = p[:nk]
        keep = int(np.searchsorted(np.cumsum(p), tp * p.sum()) + 1) if tp < 1 else nk
        keep = min(keep, nk)
        q = p[:keep] / p[:keep].sum()
        cnt = {}
        for i in range(N):
            t, _, _, _ = s2.run(temp=temp, top_p=tp, top_k=tk, seed=99, step=i + 1, min_p=mp)
            cnt[t] = cnt.get(t, 0) + 1
        outside = sum(v for k, v in cnt.items() if k not in set(o[:keep].tolist()))
        f = np.array([cnt.get(int(i), 0) for i in o[:keep]]) / N
        z = float(np.max(np.abs(f - q) / np.sqrt(q * (1 - q) / N + 1e-12)))
        check(f"draw: temp {temp} top-k {tk} min_p {mp} top-p {tp}: {keep} kept, outside {outside}, max z {z:.2f}",
              outside == 0 and z < 4.5)
    # 7. the MTP loop's verify-row transform (models/flash_next/mtp.metal mtp_logit_xform): in place on [R][V] logits;
    # row r predicts the token after [t1, d_1 .. d_r], so its counts are cnt + the drafts d_1 .. d_r (t1 is in cnt already)
    R = 5
    lgs = (rng.standard_normal((R, VOCAB)) * 2).astype(np.float32)
    lgs = np.round(lgs * 8) / 8
    cnt = np.zeros(VOCAB, np.uint32)
    pool = [int(x) for x in rng.choice(VOCAB, 12, replace=False)]
    cnt[pool[0]] = 3
    cnt[pool[1]] = FLAG
    cnt[pool[2]] = FLAG | 2
    cnt[pool[3]] = 1
    vtok = np.array([pool[3], pool[0], pool[4], pool[0], pool[5]] + [0] * 11, np.uint32)   # t1, d_1..d_4 (d_1 = d_3, d_2 fresh)
    bias_t = np.zeros(VOCAB, np.float32)
    bias_t[pool[6]] = -100
    bias_t[pool[7]] = 2.5
    bias_t[pool[4]] = -0.5
    for (pres, freq, rep, bias_on) in ((0.7, 0.4, 1.0, 0), (0.0, 0.0, 1.3, 1), (1.5, 0.5, 1.2, 1), (0.0, 0.0, 1.0, 1)):
        lg_buf = eng.from_numpy(np.ascontiguousarray(lgs.reshape(-1)))
        w = np.zeros(16, np.uint32)
        w[0:2] = np.array([1.0, 0.95], np.float32).view(np.uint32)
        w[2:7] = [20, 1, 1, VOCAB, 0]
        w[7:11] = np.array([0.0, pres, freq, rep], np.float32).view(np.uint32)
        pen = pres != 0 or freq != 0 or rep != 1.0
        w[11:15] = [bias_on, 1 if pen else 0, 0, 0]
        sp = eng.from_numpy(w)
        eng.run_kernel("mtp_logit_xform", [lg_buf, eng.from_numpy(vtok), sp, eng.from_numpy(bias_t), eng.from_numpy(cnt)],
                       ((VOCAB + 1023) // 1024, R, 1), (1024, 1, 1))
        got = lg_buf.download(np.float32, (R, VOCAB))
        worst = 0.0
        for r in range(R):
            c = cnt.copy()
            for k in range(1, r + 1):
                c[vtok[k]] += 1
            ref = transform(lgs[r], bias_t if bias_on else None, c if pen else None, pres, freq, rep)
            fin = np.isfinite(ref)
            worst = max(worst, float(np.max(np.abs(got[r][fin] - ref[fin]))), 0.0 if np.array_equal(np.isfinite(got[r]), fin) else 1e9)
        check(f"verify-row transform (presence {pres}, frequency {freq}, repetition {rep}, bias {bool(bias_on)}): "
              f"5 rows with drafts in the counts, max diff {worst:.1e}", worst < 1e-4)
    print("sampler_check:", "PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
