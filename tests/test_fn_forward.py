"""lab-forward checks of the model-level pieces against lab-ref's goldens (engine via tools/engine.py).

    $PY tests/test_fn_forward.py [--pack DIR] [--only ngram,ple,head,sampler]
1. ngram:   C++ n-gram ids + table gather (ne_fn_ngram) vs L01_* ple.ngram_ids / ple.embeddings (CPU).
2. ple:     layer 1 (PLE + GDN + MoE) as one engine layer step from the golden hc_in + states (L01_decode1):
            hc_after_ple, PLE conv state, hc_out (chain tolerances x2 as test_fn_layer).
3. head:    final mixer + 8-bit lm_head kernels (fn_hc_pre / fn_hc_mix with the mixer weights, fn_lm_head) on
            HEAD_decode1's head.hc_in vs head.logits; argmax equal; then the sampler kernels (greedy = argmax,
            logprob = log_softmax) on those logits.
4. sampler: fn_sample_p1/p2 on synthetic logits: greedy ties -> lowest id; top-k/top-p/temperature draw frequencies
            vs the exact distribution (chi-square-ish bound).
Exit 0 = all pass. Needs the pack the goldens were made from (IJAH_GOLDEN_PACK, default IJAH_PACK) for GPU parts; run it under
tools/with_lock.sh.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import compare as Cm  # noqa: E402
import engine as E  # noqa: E402
import golden  # noqa: E402
import ijah_paths  # noqa: E402
import numpy as np  # noqa: E402

EOS = 248044
VOCAB = 248320
PACK_DEFAULT = ijah_paths.golden_pack()  # the pack the goldens were made from


def history_rows(c):
    """(p2, p1, t) per row of a PLE case from token_history_in + token_ids"""
    hist = [int(x) for x in c["ple.token_history_in"].reshape(-1)]
    out = []
    for t in c.m["token_ids"]:
        out.append((hist[-2], hist[-1], int(t)))
        hist.append(int(t))
    return out


def check_ngram(eng, pack):
    res = []
    for case in ("L01_decode1", "L01_verify4", "L01_prefill"):
        c = golden.load(case)
        rows = history_rows(c)
        ids_ref = c["ple.ngram_ids"].reshape(len(rows), 16).astype(np.int64)
        emb_ref = c["ple.embeddings"].reshape(len(rows), 2560)
        bad_ids = bad_emb = 0
        for i, (p2, p1, t) in enumerate(rows):
            ids, emb = eng.ngram(pack, p2, p1, t)
            bad_ids += int(not np.array_equal(ids, ids_ref[i]))
            bad_emb += int(not np.array_equal(emb, emb_ref[i]))
        res.append({"name": f"{case}: n-gram ids / gathered rows ({len(rows)} rows)", "kind": "exact", "tol": 0.0,
                    "rel": float(bad_ids + bad_emb), "maxrel": 0.0, "cos": 1.0, "ok": bad_ids == 0 and bad_emb == 0,
                    "note": f"{bad_ids} id rows, {bad_emb} emb rows differ"})
    return Cm.report(res, "1. n-gram ids + table gather (C++) vs goldens")


def check_ple(eng, pack, scale=2.0):
    c = golden.load("L01_decode1")
    L = eng.fn_layer(pack, 1)
    (p2, p1, t), = history_rows(c)
    _, emb = eng.ngram(pack, p2, p1, t)
    L.write("ple_emb", emb)
    L.write("ple_conv_state", c["ple.conv_state_in"].reshape(-1))
    L.set_state(c["gdn.conv_state_in"], c["gdn.ssm_state_in"])
    h = L.step(c["hc_in"])
    res = [Cm.check("hc_after_ple", L.read("hc_after_ple"), c["hc_after_ple"].reshape(-1), 1e-2),
           Cm.check("ple.keys|values (proj)", L.read("ple_kv")[10240:], c["ple.values"].reshape(-1), 5e-3),
           Cm.check("ple.conv_state_out", L.read("ple_conv_state"), c["ple.conv_state_out"].reshape(-1), 5e-3),
           Cm.check("attn_hc.block_in", L.read("attn_in"), c["attn_hc.block_in"].reshape(-1), 1e-2 * scale),
           Cm.check("gdn.out", L.read("attn_out"), c["gdn.out"].reshape(-1), 1e-2 * scale),
           Cm.check("hc_out", h, c["hc_out"].reshape(-1), 1e-2 * scale)]
    ok = Cm.report(res, f"2. L01_decode1: PLE + layer 1 step from hc_in (chain), gpu {L.gpu_ms:.3f} ms")
    L.free()
    return ok


def check_head(eng, pack):
    c = golden.load("HEAD_decode1")
    hm = "language_model.model.hyper_connection_mixer."
    T = lambda n: pack.info(n)  # noqa: E731
    h = eng.from_numpy(c["head.hc_in"].reshape(-1).astype(np.float32))
    mix = eng.alloc(320 * 4)
    inj = eng.alloc(8 * 4)
    part = eng.alloc(41 * 4 * 8 * 4)
    ctr = eng.alloc(64 * 4)
    dummy = eng.alloc(4 * 10240 * 2)
    x = eng.alloc(2560 * 4)
    lg = eng.alloc(VOCAB * 4)
    eng.run_kernel("fn_hc_pre_g64_r1", [h, T(hm + "hc_norm.weight"), T(hm + "input_mix_weight_down.weight"),
                                        T(hm + "input_mix_weight_down.scales"), T(hm + "input_mix_weight_down.biases"),
                                        dummy, mix, inj, part, ctr], (164, 1, 1), (256, 1, 1))
    eng.run_kernel("fn_hc_mix_g64_r1", [h, T(hm + "hc_norm.weight"), T(hm + "input_mix_weight_up.weight"),
                                        T(hm + "input_mix_weight_up.scales"), T(hm + "input_mix_weight_up.biases"),
                                        mix, inj, x], (320, 1, 1), (256, 1, 1))
    lh = "language_model.lm_head."
    ms = eng.run_kernel("fn_lm_head_g128_r1", [x, T(lh + "weight"), T(lh + "scales"), T(lh + "biases"), lg,
                                               np.uint32(VOCAB)], ((VOCAB + 7) // 8, 1, 1), (256, 1, 1))
    logits = lg.download(np.float32, (VOCAB,))
    ref = c["head.logits"].reshape(-1)
    res = [Cm.check("mixer.mix", mix.download(np.float32, (320,)), c["mixer.mix"].reshape(-1)),
           Cm.check("mixer.block_in", x.download(np.float32, (2560,)), c["mixer.block_in"].reshape(-1)),
           Cm.check("head.logits", logits, ref, 1e-2)]
    am, ar = int(np.argmax(logits)), int(np.argmax(ref))
    res.append({"name": f"argmax {am} vs {ar}", "kind": "exact", "tol": 0.0, "rel": float(am != ar), "maxrel": 0.0,
                "cos": 1.0, "ok": am == ar, "note": f"lm_head {ms:.2f} ms (one dispatch, cold)"})
    # sampler on these logits: greedy + logprob
    tok, lp = run_sampler(eng, lg, 0.0, 1.0, 1, 0, 1)
    lse = np.log(np.sum(np.exp(logits.astype(np.float64) - logits.max()))) + logits.max()
    res.append({"name": f"sampler greedy {tok} (logprob {lp:.4f} vs {logits[am] - lse:.4f})", "kind": "exact",
                "tol": 0.0, "rel": float(tok != am), "maxrel": 0.0, "cos": 1.0,
                "ok": tok == am and abs(lp - (logits[am] - lse)) < 1e-3, "note": ""})
    return Cm.report(res, "3. HEAD_decode1: mixer + lm_head + greedy sampler vs golden")


def run_sampler(eng, lg_buf, temp, top_p, top_k, seed, step, out=None):
    nb = (VOCAB + 1023) // 1024
    cand = eng.alloc(nb * 64 * 8)
    lse = eng.alloc(nb * 8)
    # SampleParams (kernels/fn_model.metal): temp top_p | top_k seed step nv mask_on | min_p presence frequency
    # repetition | bias_on pen_on cand_k lp_on  (all the lab-sampler knobs off; tests/server/sampler_check.py has them)
    sp = eng.from_numpy(np.concatenate([np.array([temp, top_p], np.float32).view(np.uint32),
                                        np.array([top_k, seed, step, VOCAB, 0], np.uint32),
                                        np.array([0, 0, 0, 1], np.float32).view(np.uint32),
                                        np.zeros(5, np.uint32)]))
    res = eng.alloc(16)
    tok = eng.alloc(16)
    top = eng.alloc(64 * 8)
    mask = eng.alloc(4)
    bias = eng.alloc(16)
    cnt = eng.alloc(16)
    eng.run_kernel("fn_sample_p1", [lg_buf, sp, cand, lse, mask, bias, cnt], (nb, 1, 1), (1024, 1, 1))
    eng.run_kernel("fn_sample_p2", [cand, lse, sp, res, tok, top, cnt], (1, 1, 1), (1024, 1, 1))
    r = res.download(np.uint32, (4,))
    return int(r[0]), float(r[1:2].view(np.float32)[0])


def check_sampler(eng):
    rng = np.random.default_rng(0)
    res = []
    logits = (rng.standard_normal(VOCAB) * 2).astype(np.float32)
    logits = np.round(logits * 8) / 8          # many ties
    m = logits.max()
    first = int(np.argmax(logits))
    logits[first + 777] = m                     # a tie with a later id
    lg = eng.from_numpy(logits)
    tok, _ = run_sampler(eng, lg, 0.0, 1.0, 20, 1, 1)
    res.append({"name": f"greedy with ties -> {tok} (numpy argmax {first})", "kind": "exact", "tol": 0.0,
                "rel": float(tok != first), "maxrel": 0.0, "cos": 1.0, "ok": tok == first, "note": ""})
    # top-k 20, top-p 0.95, temp 1 vs exact
    logits = (rng.standard_normal(VOCAB) * 0.5).astype(np.float32)
    top = rng.choice(VOCAB, 30, replace=False)
    logits[top] = np.linspace(8, 5, 30, dtype=np.float32)
    lg = eng.from_numpy(logits)
    order = np.argsort(-logits, kind="stable")[:20]
    p = np.exp(logits[order] - logits[order[0]])
    keep = int(np.searchsorted(np.cumsum(p), 0.95 * p.sum()) + 1)
    q = p[:keep] / p[:keep].sum()
    N = 3000
    cnt = {}
    for s in range(N):
        t, _ = run_sampler(eng, lg, 1.0, 0.95, 20, 1234, s + 1)
        cnt[t] = cnt.get(t, 0) + 1
    outside = sum(v for k, v in cnt.items() if k not in set(order[:keep].tolist()))
    f = np.array([cnt.get(int(i), 0) for i in order[:keep]]) / N
    dev = float(np.max(np.abs(f - q) / np.sqrt(q * (1 - q) / N + 1e-12)))
    res.append({"name": f"temp 1 / top-k 20 / top-p 0.95: {keep} kept, max z {dev:.2f}, outside {outside}",
                "kind": "exact", "tol": 0.0, "rel": 0.0, "maxrel": 0.0, "cos": 1.0, "ok": outside == 0 and dev < 4.5,
                "note": f"{N} draws"})
    return Cm.report(res, "4. sampler kernels on synthetic logits")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=PACK_DEFAULT)
    ap.add_argument("--only", default="ngram,ple,head,sampler")
    a = ap.parse_args()
    parts = a.only.split(",")
    eng = E.Engine()
    eng.set_ref_rounding(True)  # the goldens are the MLX reference's (the library default is plain f32)
    pack = eng.pack_open(a.pack)
    ok = True
    if "ngram" in parts:
        ok &= check_ngram(eng, pack)
    if "ple" in parts:
        ok &= check_ple(eng, pack)
    if "head" in parts:
        ok &= check_head(eng, pack)
    if "sampler" in parts:
        ok &= check_sampler(eng)
    pack.close()
    print("test_fn_forward:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
