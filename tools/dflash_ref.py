"""DFlash2 block drafter for Flash-Next: MLX reference + the step-A feasibility loop (lab-dflash; docs/mtp.md 13).

Drafter: incoai/Qwen3.8-Flash-Next-DFlash2 @ a22a1744e85c (Apache-2.0; ~/.cache/ijah/drafters/Qwen3.8-Flash-Next-DFlash2,
data only). The math follows z-lab/dflash `dflash/model.py` (DFlash2DraftModel; read, not run) and Splash's DFlash2 port
(splash-m5 runtime/model/DFlashDraft.cpp, runtime/metal/kernels/decode/draft.metal, sampling.metal draft_select_*):

  features  f_j = [mean_s H_l(j)[s] for l in 5, 15, 25, 35, 45]   (completed layer outputs, 4 HC streams averaged)
  context   c_j = hidden_norm(fc(f_j));  per layer K_j = rope(k_norm(k_proj c_j), j), V_j = v_proj c_j  (no input norm)
  block     rows x = embed([anchor, mask x (B - 1)]) at positions T .. T + B - 1 (anchor = the last emitted token)
  layer     h += conv1(o_proj(attn(conv0(n)))) ; h += conv1'(down(silu(gate(c)) * up(c))), c = conv0'(n'), n = rmsnorm
            conv_s(x)[r] = sum_{t in 0, 1} x[r - t] * (base[s][t] + dyn[r][s][t][group]), dyn = kernel_projection(n)
            attention: non-causal over context + block keys, |q - k| < 2048 (sliding window), GQA 32 / 8, theta 1e4
  readout   hn = norm(h) rows 1 .. B - 1 -> logits = target lm_head(hn); top-16 candidates (unary = raw logits)
  selector  s = hidden_projection(hn); path from the anchor: score_i = unary_i + <pred_cb[prev] * s_p, succ_cb[cand_i]>,
            greedy argmax or (temperature T) sample from q = softmax(score / T) over the 16 -- q is the proposal.

Step A (host-driven rounds through ne_fn_dfl_verify / ne_fn_dfl_commit; the engine verifies, this file drafts):

    NEED_GB=85 LOCK_OWNER=dflash LOCK_WAIT=600 tools/with_lock.sh "dflash step A (~84 GB)" HH:MM -- \\
        $PY tools/dflash_ref.py --n 512 [--blocks 4,8] [--only code_lru] [--mtp 2,3] [--json OUT]
"""
import argparse
import ctypes as C
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refenv  # noqa: E402

os.environ.setdefault("NE_FN_ROUND", "f32")
refenv.add_paths()
import numpy as np  # noqa: E402

DRAFTER = Path(os.environ.get("NE_DFL_PATH", Path.home() / ".cache/ijah/drafters/Qwen3.8-Flash-Next-DFlash2"))
LAYERS = (5, 15, 25, 35, 45)


# ------------------------------------------------------------------------------------------------ the drafter (MLX)
class DFlash2:
    def __init__(self, ddir=DRAFTER, pack=None, dtype="float32"):
        import mlx.core as mx
        self.mx = mx
        cfg = json.loads((Path(ddir) / "config.json").read_text())
        dc = cfg["dflash_config"]
        assert tuple(dc["target_layer_ids"]) == LAYERS, dc
        self.cfg, self.dc = cfg, dc
        self.mask = dc["mask_token_id"]
        self.K = dc["selector_top_k"]
        self.gs = dc["conv_group_size"]
        self.window = cfg["sliding_window"]
        self.theta = cfg["rope_parameters"]["rope_theta"]
        self.eps = cfg["rms_norm_eps"]
        self.nh, self.nkv, self.hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.dt = getattr(mx, dtype)
        w = mx.load(str(Path(ddir) / "model.safetensors"))
        self.w = {k: v.astype(self.dt) if not k.startswith("candidate_selector.") or "projection" in k else v
                  for k, v in w.items()}
        # the target's embed_tokens / lm_head (MLX affine in the pack)
        pk = Path(pack)
        e = mx.load(str(pk / "model-embed.safetensors"))
        h = mx.load(str(pk / "model-lm-head.safetensors"))
        self.emb = [e[f"language_model.model.embed_tokens.{x}"] for x in ("weight", "scales", "biases")]
        self.head = [h[f"language_model.lm_head.{x}"] for x in ("weight", "scales", "biases")]
        H = cfg["hidden_size"]
        self.eg, self.eb = H // self.emb[1].shape[1], self.emb[0].shape[1] * 32 // H
        self.hg, self.hb = H // self.head[1].shape[1], self.head[0].shape[1] * 32 // H
        self.L = cfg["num_hidden_layers"]
        self.reset()

    def reset(self):
        self.ck = [None] * self.L  # context keys [8][C][128] (roped), values [8][C][128]
        self.cv = [None] * self.L
        self.c0 = 0                # position of the first cached context row
        self.T = 0                 # positions [c0, T) cached

    def rms(self, x, w):
        mx = self.mx
        return (x * mx.rsqrt(mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True) + self.eps)).astype(x.dtype) * w

    def lin(self, x, name):
        return x @ self.w[name + ".weight"].T

    def embed(self, ids):
        mx = self.mx
        ids = mx.array(np.asarray(ids, np.int32))
        q, s, b = (t[ids] for t in self.emb)
        return mx.dequantize(q, s, b, group_size=self.eg, bits=self.eb).astype(self.dt)

    def logits(self, x):
        mx = self.mx
        q, s, b = self.head
        return mx.quantized_matmul(x.astype(mx.bfloat16), q, s, b, transpose=True, group_size=self.hg, bits=self.hb)

    def rope(self, x, offset):  # x [heads][L][128], positions offset .. offset + L - 1 (rotate-half form)
        return self.mx.fast.rope(x, self.hd, traditional=False, base=self.theta, scale=1.0, offset=offset)

    def add_context(self, feat):
        """feat f32 [n][12800]: the target features of positions [T, T + n) (committed rows) -> context K / V"""
        mx = self.mx
        n = feat.shape[0]
        c = self.rms(self.lin(mx.array(feat).astype(self.dt), "fc"), self.w["hidden_norm.weight"])
        for i in range(self.L):
            p = f"layers.{i}.self_attn."
            k = self.lin(c, p + "k_proj").reshape(n, self.nkv, self.hd)
            k = self.rope(self.rms(k, self.w[p + "k_norm.weight"]).transpose(1, 0, 2), self.T)
            v = self.lin(c, p + "v_proj").reshape(n, self.nkv, self.hd).transpose(1, 0, 2)
            self.ck[i] = k if self.ck[i] is None else mx.concatenate([self.ck[i], k], axis=1)
            self.cv[i] = v if self.cv[i] is None else mx.concatenate([self.cv[i], v], axis=1)
        self.T += n
        # keep what a block query can still see: keys k > T - window (queries at >= T)
        drop = max(0, (self.T - self.c0) - (self.window - 1))
        if drop:
            self.ck = [x[:, drop:] for x in self.ck]
            self.cv = [x[:, drop:] for x in self.cv]
            self.c0 += drop
        mx.eval(self.ck, self.cv)

    def conv(self, x, dyn, base):
        """x [B][H], dyn [B][2 taps][G], base [2 taps][H]: out[r] = sum_t x[r - t] * (base[t] + dyn[r][t][group])"""
        mx = self.mx
        B, H = x.shape
        G = H // self.gs
        out = None
        for t in range(2):
            xs = x if t == 0 else mx.concatenate([mx.zeros((1, H), x.dtype), x[:-1]], axis=0)
            k = base[t][None, :] + mx.repeat(dyn[:, t, :], self.gs, axis=1)  # [B][H]
            out = xs * k if out is None else out + xs * k
        return out

    def forward(self, anchor, B):
        """block rows [anchor, mask x (B - 1)] at positions T .. T + B - 1 -> normed hidden of rows 1 .. B - 1"""
        mx = self.mx
        H = self.cfg["hidden_size"]
        G = H // self.gs
        h = self.embed([anchor] + [self.mask] * (B - 1))
        T = self.T
        kpos = mx.concatenate([mx.arange(self.c0, T), mx.arange(T, T + B)])
        qpos = mx.arange(T, T + B)
        d = qpos[:, None] - kpos[None, :]
        mask = mx.where((d < self.window) & (-d < self.window), 0.0, -mx.inf).astype(self.dt)
        rep = self.nh // self.nkv
        for i in range(self.L):
            p = f"layers.{i}."
            a = p + "self_attn."
            n = self.rms(h, self.w[p + "input_layernorm.weight"])
            dyn = self.lin(n, p + "attention_conv.kernel_projection").reshape(B, 2, 2, G)
            base = self.w[p + "attention_conv.base_kernel"]
            x = self.conv(n, dyn[:, 0], base[0])
            q = self.lin(x, a + "q_proj").reshape(B, self.nh, self.hd)
            q = self.rope(self.rms(q, self.w[a + "q_norm.weight"]).transpose(1, 0, 2), T)
            k = self.lin(x, a + "k_proj").reshape(B, self.nkv, self.hd)
            k = self.rope(self.rms(k, self.w[a + "k_norm.weight"]).transpose(1, 0, 2), T)
            v = self.lin(x, a + "v_proj").reshape(B, self.nkv, self.hd).transpose(1, 0, 2)
            if self.ck[i] is not None:
                k = mx.concatenate([self.ck[i], k], axis=1)
                v = mx.concatenate([self.cv[i], v], axis=1)
            o = mx.fast.scaled_dot_product_attention(q[None], k[None], v[None], scale=self.hd ** -0.5,
                                                     mask=mask[None, None])[0]
            o = self.lin(o.transpose(1, 0, 2).reshape(B, self.nh * self.hd), a + "o_proj")
            h = h + self.conv(o, dyn[:, 1], base[1])
            n = self.rms(h, self.w[p + "post_attention_layernorm.weight"])
            dyn = self.lin(n, p + "mlp_conv.kernel_projection").reshape(B, 2, 2, G)
            base = self.w[p + "mlp_conv.base_kernel"]
            x = self.conv(n, dyn[:, 0], base[0])
            m = self.lin(mx.sigmoid(g := self.lin(x, p + "mlp.gate_proj")) * g * self.lin(x, p + "mlp.up_proj"),
                         p + "mlp.down_proj")
            h = h + self.conv(m, dyn[:, 1], base[1])
        return self.rms(h, self.w["norm.weight"])[1:]

    def propose(self, anchor, B, temp, rng, qtemp=None):
        """-> drafts [B - 1], proposals [(ids [16], q [16])] (q None when greedy), the candidates' raw logits"""
        mx = self.mx
        hn = self.forward(anchor, B)
        lg = self.logits(hn).astype(mx.float32)                     # [B - 1][V]
        cand = mx.argpartition(-lg, self.K - 1, axis=-1)[:, :self.K]
        un = mx.take_along_axis(lg, cand, axis=-1)
        s = self.lin(hn, "candidate_selector.hidden_projection").astype(mx.float32)
        pc = self.w["candidate_selector.predecessor_codebook"]
        sc = self.w["candidate_selector.successor_codebook"]
        cs = sc[cand.reshape(-1)].astype(mx.float32).reshape(B - 1, self.K, -1)  # successor rows of every candidate
        mx.eval(cand, un, s, cs)
        cand, un, s, cs = np.array(cand), np.array(un), np.array(s), np.array(cs)
        prev = anchor
        drafts, props = [], []
        qt = temp if qtemp is None else qtemp
        for j in range(B - 1):
            pv = np.array(pc[prev].astype(mx.float32))
            sco = un[j] + cs[j] @ (pv * s[j])
            if temp > 0:
                z = (sco - sco.max()) / qt
                q = np.exp(z)
                q /= q.sum()
                i = int(np.searchsorted(np.cumsum(q), rng.random() * q.sum(), side="right"))
                i = min(i, self.K - 1)
                props.append((cand[j].copy(), q))
            else:
                i = int(np.argmax(sco))
                props.append(None)
            prev = int(cand[j][i])
            drafts.append(prev)
        return drafts, props


# ------------------------------------------------------------------------------------------------ accept (= mtp.mm)
def make_dist(ids, logits, temp, top_k, top_p):
    """the engine's make_dist over the K verify (id, logit) pairs: temperature -> top-k -> top-p (descending order)"""
    o = sorted(range(len(ids)), key=lambda i: (-logits[i], ids[i]))[:top_k]
    lg = np.array([logits[i] for i in o], np.float64)
    w = np.exp((lg - lg.max()) / temp)
    c = np.cumsum(w / w.sum())
    n = int(np.searchsorted(c, top_p) + 1) if top_p < 1 else len(o)
    n = min(n, len(o))
    p = w[:n] / w[:n].sum()
    return [ids[o[i]] for i in range(n)], p


def dprob(d, ids, p):
    for i, x in enumerate(ids):
        if x == d:
            return float(p[i])
    return 0.0


def accept_sampled(drafts, P, Q, rng):
    """exact speculative sampling (mtp.mm accept_sampled_p): P[i] = (ids, p) of verify row i, Q[i] = (ids, q)"""
    out = []
    for i, d in enumerate(drafts):
        pd = dprob(d, *P[i])
        qd = dprob(d, *Q[i]) if Q[i] is not None else 1.0
        if rng.random() < min(1.0, pd / max(qd, 1e-12)):
            out.append(d)
            continue
        ids, p = P[i]
        r = np.array([max(0.0, p[j] - (dprob(ids[j], *Q[i]) if Q[i] is not None else float(ids[j] == d)))
                      for j in range(len(ids))])
        if r.sum() <= 0:
            r = np.array(p)
        r /= r.sum()
        out.append(ids[min(int(np.searchsorted(np.cumsum(r), rng.random(), side="right")), len(ids) - 1)])
        return len(out) - 1, out
    ids, p = P[len(drafts)]
    out.append(ids[min(int(np.searchsorted(np.cumsum(p), rng.random(), side="right")), len(ids) - 1)])
    return len(drafts), out


def topk_pairs(tk, R, K=20):
    a = tk.reshape(R, K, 2)
    ids = a[:, :, 0].copy().view(np.uint32).astype(np.int64)
    return ids, a[:, :, 1]


# ------------------------------------------------------------------------------------------------ step A loop
class Host:
    def __init__(self, eng, mtp):
        lib = eng.lib
        for f, at in (("ne_fn_dfl_capture", [C.c_void_p, C.c_int]), ("ne_fn_dfl_reset", [C.c_void_p]),
                      ("ne_fn_dfl_verify", [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_void_p, C.c_void_p]),
                      ("ne_fn_dfl_commit", [C.c_void_p, C.c_void_p, C.c_int])):
            getattr(lib, f).argtypes = at
            getattr(lib, f).restype = C.c_int
        self.eng, self.lib, self.p = eng, lib, C.c_void_p(mtp.ptr)
        self.tk = np.zeros(8 * 40, np.float32)
        self.ft = np.zeros(8 * 12800, np.float32)
        self.ms = C.c_double(0)

    def capture(self, on):
        self.eng._ok(self.lib.ne_fn_dfl_capture(self.p, on), "dfl_capture")

    def reset(self):
        self.eng._ok(self.lib.ne_fn_dfl_reset(self.p), "dfl_reset")

    def verify(self, toks):
        t = np.ascontiguousarray(toks, np.int32)
        R = len(t)
        self.eng._ok(self.lib.ne_fn_dfl_verify(self.p, t.ctypes.data, R, self.tk.ctypes.data, self.ft.ctypes.data,
                                               C.byref(self.ms)), "dfl_verify")
        ids, lg = topk_pairs(self.tk[:R * 40], R)
        return ids, lg, self.ft[:R * 12800].reshape(R, 12800).copy(), self.ms.value

    def commit(self, toks, c):
        t = np.ascontiguousarray(toks, np.int32)
        self.eng._ok(self.lib.ne_fn_dfl_commit(self.p, t.ctypes.data, c), "dfl_commit")


def run_dflash(host, dr, prompt, n_new, B, sp, seed, maxR, ndraft=None):
    """prompt through R-row verifies (features of every position), then DFlash2 rounds; returns stats"""
    temp, top_k, top_p = sp
    rng = np.random.default_rng(seed)
    host.reset()
    dr.reset()
    n = len(prompt)
    i = 0
    last = None
    while i < n:
        r = min(maxR, n - i)
        if n - i - r == 1:
            r -= 1
        if r < 2:  # a 1-token prompt tail: verify it with a dummy row (only row 0 is committed)
            ids, lg, ft, _ = host.verify([prompt[i], prompt[i]])
            host.commit(prompt[i:i + 1], 1)
            dr.add_context(ft[:1])
            last = (ids[0], lg[0])
            i += 1
            continue
        ids, lg, ft, _ = host.verify(prompt[i:i + r])
        host.commit(prompt[i:i + r], r)
        dr.add_context(ft)
        last = (ids[r - 1], lg[r - 1])
        i += r
    if temp <= 0:
        t1 = int(last[0][int(np.argmax(last[1]))])
    else:
        pid, pp = make_dist(list(last[0]), list(last[1]), temp, top_k, top_p)
        t1 = pid[min(int(np.searchsorted(np.cumsum(pp), rng.random(), side="right")), len(pid) - 1)]
    out = [t1]
    nd = B - 1 if ndraft is None else ndraft
    rounds, hist, vms, dms, log = 0, {}, [], [], []
    acc_pos = np.zeros(8)
    reach_pos = np.zeros(8)
    while len(out) < n_new:
        t0 = time.perf_counter()
        drafts, props = dr.propose(t1, B, temp, rng)
        dms.append((time.perf_counter() - t0) * 1e3)
        drafts, props = drafts[:nd], props[:nd]
        toks = [t1] + drafts
        ids, lg, ft, ms = host.verify(toks)
        vms.append(ms)
        if temp <= 0:  # greedy: accept while d_j = argmax of row j - 1
            am = [int(ids[j][int(np.argmax(lg[j]))]) for j in range(len(toks))]
            a = 0
            while a < len(drafts) and drafts[a] == am[a]:
                a += 1
            em = drafts[:a] + [am[a]]
        else:
            P = [make_dist(list(ids[j]), list(lg[j]), temp, top_k, top_p) for j in range(len(toks))]
            a, em = accept_sampled(drafts, P, props, rng)
        for j in range(min(a + 1, nd)):
            reach_pos[j] += 1
            if j < a:
                acc_pos[j] += 1
        c = a + 1
        if temp > 0:  # per round: q(d_j) of each draft, the proposal's top q, accepted, verify ms (gate simulation)
            log.append(([round(dprob(d, *props[j]), 4) for j, d in enumerate(drafts)],
                        [round(float(props[j][1].max()), 4) for j in range(len(drafts))], a, round(ms, 3)))
        host.commit(toks, c)
        dr.add_context(ft[:c])
        out.extend(em)
        t1 = em[-1]
        rounds += 1
        hist[a] = hist.get(a, 0) + 1
    return dict(tokens=len(out) - 1, rounds=rounds, tok_round=(len(out) - 1) / rounds, hist=hist,
                accept=[round(acc_pos[j] / reach_pos[j], 3) if reach_pos[j] else None for j in range(nd)],
                verify_ms=statistics.median(vms), draft_ms_mlx=statistics.median(dms), out=out, log=log)


def main():
    import engine as E
    from tokenizers import Tokenizer
    from bench_mtp import Mtp, accept_table, chat
    from bench_draft import H2H
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=(refenv.ROOT / "PACK_READY").read_text().strip())
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--blocks", default="4:3,8:3", help="B:k list: drafter block size B, k <= B - 1 drafts verified")
    ap.add_argument("--mtp", default="2,3", help="native MTP depths to run on the same prompts / seeds (paired)")
    ap.add_argument("--only", default="")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(Path(a.pack) / "tokenizer.json"))
    prompts = {k: (v, 500 + 10 * i) for i, (k, v) in enumerate(H2H.items())}
    if a.only:
        prompts = {k: v for k, v in prompts.items() if k in a.only.split(",")}
    enc = {k: tok.encode(chat(a.pack, q), add_special_tokens=False).ids for k, (q, _) in prompts.items()}
    tmax = (max(len(e) for e in enc.values()) + a.n + 256 + 63) // 64 * 64
    eng = E.Engine()
    pack = eng.pack_open(a.pack)
    t0 = time.time()
    m = eng.fn_model(pack, tmax)
    mtp = Mtp(eng, m, 5, 4)  # R <= 6 (fn_moe_router_r7 is not instantiated)
    host = Host(eng, mtp)
    dr = DFlash2(pack=a.pack)
    print(f"load {time.time() - t0:.1f} s, tmax {tmax}", flush=True)
    sp = (0.0, 1, 1.0) if a.greedy else (1.0, 20, 0.95)
    eng.lib.ne_fn_mtp_set_copy.argtypes = [C.c_void_p, C.c_int, C.c_int]
    eng._ok(eng.lib.ne_fn_mtp_set_copy(C.c_void_p(mtp.ptr), 16, 4), "set_copy")  # copy rounds <= 4 drafts (as max_depth 4)
    rows = []
    for rep in range(a.reps):
        for name, (q, seed) in prompts.items():
            for bk in a.blocks.split(","):
                B, _, k = bk.partition(":")
                B, k = int(B), int(k or int(B) - 1)
                host.capture(1)
                r = run_dflash(host, dr, enc[name], a.n, B, sp, seed + rep, 5, k)
                r.update(kind=f"dflash_b{B}_k{k}", prompt=name, rep=rep)
                rows.append(r)
                print(f"  {r['kind']:12s} {name:13s}: {r['tok_round']:.2f} tok/round, accept {r['accept']}, verify "
                      f"{r['verify_ms']:.2f} ms, drafter (MLX) {r['draft_ms_mlx']:.1f} ms", flush=True)
            host.capture(0)
            for d in [int(x) for x in a.mtp.split(",") if x]:
                m.set_sampler(*((0.0, 1.0, 1, 0) if a.greedy else (1.0, 0.95, 20, seed + rep)))
                out, s = mtp.generate(enc[name], a.n, d, seed=seed + rep)
                cond, tpr = accept_table(s["hist"])
                r = dict(kind=f"mtp_d{d}", prompt=name, rep=rep, tok_round=tpr, accept=cond[:d],
                         tps=s["tokens"] / (s["decode_ms"] / 1e3), verify_ms=s["verify_gpu_ms"],
                         draft_ms=s["draft_gpu_ms"], out=out)
                rows.append(r)
                print(f"  {r['kind']:12s} {name:13s}: {tpr:.2f} tok/round, accept {cond[:d]}, {r['tps']:.1f} tok/s, "
                      f"draft {r['draft_ms']:.2f} + verify {r['verify_ms']:.2f} ms", flush=True)
    print("\n== tokens per round, mean over prompts")
    for kind in dict.fromkeys(r["kind"] for r in rows):
        v = [r["tok_round"] for r in rows if r["kind"] == kind]
        print(f"  {kind:12s}: {statistics.fmean(v):.3f}  ({', '.join(f'{x:.2f}' for x in v)})")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    pack.close()


if __name__ == "__main__":
    main()
