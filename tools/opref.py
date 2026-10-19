"""Numpy op references for the Flash-Next layer math (lab-ref). Each function is the math of one kernel, in f32,
optionally rounding to bf16 where the reference stores bf16 (`bf=True`), so a kernel can be tested in isolation:
feed it a golden's inputs, compare with the function (or the golden's output) via tools/compare.py.
Every function is cross-checked against the MLX reference by tools/check_opref.py (errors in docs/reference.md).

Weights come straight from the pack through PackNP (numpy memmap of the safetensors shards):
    P = PackNP();  W = P.linear("language_model.model.layers.0.linear_attn.in_proj_qkv")   # dense f32 [out, in]
    We = P.expert(0, "gate", 17)                                                           # EXL3 -> f32 [in, out]
"""
import json
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refenv  # noqa: E402

refenv.add_paths()
import numpy as np  # noqa: E402

EPS = 1e-6
D = 2560
HC = 4


# ------------------------------------------------------------------ dtype helpers
def bf16(x):
    """Round f32 -> bf16 (nearest even), returned as f32."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    r = ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) & np.uint32(0xFFFF0000))
    return r.view(np.float32).reshape(np.shape(x))


def rb(x, bf=True):
    return bf16(x) if bf else np.asarray(x, np.float32)


def sigmoid(x):
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float32)))


def silu(x):
    x = np.asarray(x, np.float32)
    return x * sigmoid(x)


def softplus(x):
    x = np.asarray(x, np.float32)
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def sigmoid_bf(x):
    """sigmoid on bf16 as an uncompiled MLX op: 1 / (1 + exp(-x)), each step rounded (bit-exact on inject_w and
    the PLE gate). Fused/compiled reference sites (mix_streams, beta) sit between this and the f32 sigmoid."""
    x = np.asarray(x, np.float32)
    with np.errstate(over="ignore"):
        return bf16(1.0 / bf16(1.0 + bf16(np.exp(-x))))


def silu_bf(x):
    """silu as MLX evaluates it on bf16: x * bf16(sigmoid(x)), rounded (matches the goldens to ~2e-4)."""
    x = np.asarray(x, np.float32)
    return bf16(x * bf16(sigmoid(x)))


def softplus_bf(x):
    """softplus on bf16 as MLX: max(x, 0) + log1p(exp(-|x|)) with each step rounded (bit-exact on the goldens)."""
    x = np.asarray(x, np.float32)
    return bf16(np.maximum(x, 0) + bf16(np.log1p(bf16(np.exp(-np.abs(x))))))


# ------------------------------------------------------------------ pack access (numpy)
class PackNP:
    """Read tensors of the pack without MLX: bf16/f16 -> f32, U32/U16/I64 raw. Memmapped, nothing resident."""

    def __init__(self, path=None):
        self.path = Path(path or refenv.PACK)
        self.config = json.loads((self.path / "config.json").read_text())
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        self._h = {}
        self.eq = self.config.get("expert_quant", {})

    def _hdr(self, fname):
        if fname not in self._h:
            with open(self.path / fname, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._h[fname] = (json.loads(f.read(n)), 8 + n)
        return self._h[fname]

    def spec(self, key):
        h, _ = self._hdr(self.index[key])
        return h[key]

    def raw(self, key):
        fname = self.index[key]
        h, start = self._hdr(fname)
        s = h[key]
        dt = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "U32": np.uint32, "U16": np.uint16,
              "I64": np.int64, "I32": np.int32, "U8": np.uint8}[s["dtype"]]
        a, b = s["data_offsets"]
        return np.memmap(self.path / fname, dtype=dt, mode="r", offset=start + a, shape=tuple(s["shape"])), s["dtype"]

    def tensor(self, key):
        v, dt = self.raw(key)
        if dt == "BF16":
            return (np.asarray(v).astype(np.uint32) << 16).view(np.float32)
        if dt == "F16":
            return np.asarray(v).astype(np.float32)
        return np.asarray(v)

    def has(self, key):
        return key in self.index

    def linear(self, prefix):
        """Dense f32 W [out, in] of a Linear: 8-bit affine (prefix.weight/.scales/.biases), EXL3 (T6 pack:
        prefix.trellis/.suh/.svh, decoded by tools/convert_trunk_enc.decode; cached) or plain bf16."""
        if self.has(prefix + ".trellis"):
            if not hasattr(self, "_t6"):
                self._t6 = {}
            if prefix not in self._t6:
                import convert_trunk_enc as TE
                tr, _ = self.raw(prefix + ".trellis")
                W = TE.decode(np.asarray(tr), self.tensor(prefix + ".suh"), self.tensor(prefix + ".svh"))
                self._t6[prefix] = np.ascontiguousarray(W.T.astype(np.float32))
            return self._t6[prefix]
        if self.has(prefix + ".scales"):
            wq, _ = self.raw(prefix + ".weight")
            return affine_dequant(np.asarray(wq), self.tensor(prefix + ".scales"), self.tensor(prefix + ".biases"))
        return self.tensor(prefix + ".weight")

    def linear_rows(self, prefix, rows):
        """Rows of an 8-bit QuantizedEmbedding / Linear (embedding lookup)."""
        wq, _ = self.raw(prefix + ".weight")
        sc, _ = self.raw(prefix + ".scales")
        bi, _ = self.raw(prefix + ".biases")
        rows = np.asarray(rows).reshape(-1)
        f = lambda m: (np.asarray(m[rows]).astype(np.uint32) << 16).view(np.float32)  # noqa: E731
        return bf16(affine_dequant(np.asarray(wq[rows]), f(sc), f(bi)))   # MLX dequantizes in bf16

    def expert(self, layer, proj, e, mtp=False):
        """EXL3 routed expert projection -> dense f32 W [in, out] (y = x @ W), via exl3_codec (Sushi's layout)."""
        import exl3_codec as X
        pre = f"language_model.model.layers.{layer}.mlp.switch_mlp.{proj}_proj."
        tr, _ = self.raw(pre + "trellis")
        suh = self.tensor_row(pre + "suh", e)
        svh = self.tensor_row(pre + "svh", e)
        t = np.asarray(tr[e])
        inner = X.decode_inner(t, t.shape[-1], self.eq.get("codebook", "mcg"), int(self.eq.get("window", 15)))
        return X.reconstruct(inner, suh, svh).astype(np.float32)

    def expert_parts(self, layer, proj, e):
        """(inner W' [in, out] f32, suh [in], svh [out]) of one EXL3 expert projection."""
        import exl3_codec as X
        pre = f"language_model.model.layers.{layer}.mlp.switch_mlp.{proj}_proj."
        tr, _ = self.raw(pre + "trellis")
        t = np.asarray(tr[e])
        inner = X.decode_inner(t, t.shape[-1], self.eq.get("codebook", "mcg"), int(self.eq.get("window", 15)))
        return inner.astype(np.float32), self.tensor_row(pre + "suh", e), self.tensor_row(pre + "svh", e)

    def tensor_row(self, key, i):
        v, dt = self.raw(key)
        r = np.asarray(v[i])
        return r.astype(np.float32) if dt == "F16" else ((r.astype(np.uint32) << 16).view(np.float32) if dt == "BF16" else r)

    def norm(self, key):
        """A norm weight as the reference multiplies by it (ones-centred pack value, used as stored)."""
        return self.tensor(key).astype(np.float32)


def affine_dequant(wq, scales, biases, bits=8):
    """MLX affine: wq U32 [out, in*bits/32] (codes little-endian within each word), scales/biases [out, in/gs]."""
    assert bits == 8
    q = np.ascontiguousarray(wq).view(np.uint8).astype(np.float32)          # [out, in]
    out, n_in = q.shape
    ng = scales.shape[-1]
    gs = n_in // ng
    q = q.reshape(out, ng, gs)
    return (q * scales.astype(np.float32)[..., None] + biases.astype(np.float32)[..., None]).reshape(out, n_in)


def linear_big(x, P, prefix, chunk=16384, bf=True):
    """x @ W^T for a large 8-bit matrix (lm_head [248320, 2560]) dequantized in row chunks (bounded RAM)."""
    wq, _ = P.raw(prefix + ".weight")
    sc, _ = P.raw(prefix + ".scales")
    bi, _ = P.raw(prefix + ".biases")
    f = lambda m: (np.asarray(m).astype(np.uint32) << 16).view(np.float32)  # noqa: E731
    x = np.asarray(x, np.float32)
    out = np.empty((*x.shape[:-1], wq.shape[0]), np.float32)
    for r0 in range(0, wq.shape[0], chunk):
        W = affine_dequant(np.asarray(wq[r0:r0 + chunk]), f(sc[r0:r0 + chunk]), f(bi[r0:r0 + chunk]))
        out[..., r0:r0 + chunk] = x @ W.T
    return rb(out, bf)


def linear(x, W, bf=True):
    """y = x @ W^T (W [out, in]), f32 accumulation."""
    return rb(np.asarray(x, np.float32) @ W.T, bf)


# ------------------------------------------------------------------ norms
def rmsnorm(x, w, eps=EPS, group=None, bf=True):
    """x * rsqrt(mean(x^2) + eps) * w in f32 (w = the pack weight as stored); group = per-group size (HC: 2560)."""
    x = np.asarray(x, np.float32)
    s = x.shape
    if group:
        x = x.reshape(*s[:-1], -1, group)
        w = np.asarray(w, np.float32).reshape(-1, group)
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w
    return rb(y.reshape(s), bf)


def gated_rmsnorm(y, z, w, eps=EPS, bf=True):
    """GDN output norm per v-head: rms_norm(y, w) (bf16) * sigmoid(z)."""
    n = rmsnorm(y, w, eps, bf=bf)
    return rb(n * sigmoid(z), bf)


# ------------------------------------------------------------------ hyper-connections
def hc_read(h, P, prefix, bf=True, inject=True):
    """HC read of `prefix` (e.g. 'language_model.model.layers.0.attn_hyper_connection'). Returns a dict with the
    golden names' suffixes: normed, down, mix, up, block_in, inject_logits, inject_w."""
    r = {}
    r["normed"] = rmsnorm(h, P.norm(prefix + ".hc_norm.weight"), group=D, bf=bf)
    r["down"] = linear(r["normed"], P.linear(prefix + ".input_mix_weight_down"), bf)
    r["mix"] = silu_bf(rb(r["down"] / HC, bf)) if bf else silu(r["down"] / HC)
    r["up"] = linear(r["mix"], P.linear(prefix + ".input_mix_weight_up"), bf)
    s = r["up"].shape[:-1]
    m = sigmoid(r["up"]).reshape(*s, HC, D)
    m = rb(m, bf) * r["normed"].reshape(*s, HC, D)
    r["block_in"] = rb(np.mean(rb(m, bf), axis=-2), bf)
    if inject:
        r["inject_logits"] = linear(r["normed"], P.linear(prefix + ".block_inject_weight"), bf)
        r["inject_w"] = rb(2.0 * (sigmoid_bf if bf else sigmoid)(rb(r["inject_logits"] / HC, bf)), bf)
    return r


def hc_write(h, branch, inject_w, bf=True):
    """h_s + inject_w[s] * branch for each of the 4 streams."""
    s = h.shape[:-1]
    inj = rb(np.asarray(branch, np.float32)[..., None, :] * np.asarray(inject_w, np.float32)[..., :, None], bf)
    return rb(h + inj.reshape(*s, HC * D), bf)


# ------------------------------------------------------------------ convolutions
def causal_conv(state, x, w, dilation=1, bf=True):
    """Depthwise causal conv over concat(state, x) along time. state [B, (K-1)*dil, C], x [B, S, C],
    w [C, K] (pack conv1d.weight[:, :, 0]). Returns (out [B, S, C] before the activation, new state)."""
    w = np.asarray(w, np.float32)
    if w.ndim == 3:
        w = w[:, :, 0]
    K = w.shape[1]
    inp = np.concatenate([np.asarray(state, np.float32), np.asarray(x, np.float32)], axis=1)
    S = x.shape[1]
    out = np.zeros(x.shape, np.float32)
    for j in range(K):
        out += inp[:, j * dilation:j * dilation + S] * w[None, None, :, j]
    return rb(out, bf), inp[:, -state.shape[1]:]


# ------------------------------------------------------------------ GDN
def gdn_split(conv_out, hk=16, hv=48, dk=128, dv=128):
    s = conv_out.shape[:-1]
    q = conv_out[..., :hk * dk].reshape(*s, hk, dk)
    k = conv_out[..., hk * dk:2 * hk * dk].reshape(*s, hk, dk)
    v = conv_out[..., 2 * hk * dk:].reshape(*s, hv, dv)
    return q, k, v


def gdn_qk_norm(q, k, bf=True):
    """L2 normalisation per head (eps inside the sum), q also scaled by Dk^-0.5."""
    q = np.asarray(q, np.float32)
    k = np.asarray(k, np.float32)
    qn = q / np.sqrt(np.sum(q * q, -1, keepdims=True) + 1e-6)
    kn = k / np.sqrt(np.sum(k * k, -1, keepdims=True) + 1e-6)
    return rb(rb(qn, bf) * q.shape[-1] ** -0.5, bf), rb(kn, bf)


def gdn_gates(a, b, A_log, dt_bias, bf=True):
    """g = exp(-exp(A_log) * softplus(a + dt_bias)) (f32), beta = sigmoid(b) (bf16)."""
    z = rb(np.asarray(a, np.float32) + dt_bias, bf)
    sp = softplus_bf(z) if bf else softplus(z)
    g = np.exp(-np.exp(np.asarray(A_log, np.float32)) * sp).astype(np.float32)
    return g, rb(sigmoid(b), bf)


def gdn_recurrence(state, q, k, v, g, beta, bf=True, rows=False):
    """state [B, Hv, Dv, Dk] f32; q, k [B, S, Hk, Dk]; v [B, S, Hv, Dv]; g, beta [B, S, Hv].
    v-head h uses k-head h // (Hv / Hk). Returns (y [B, S, Hv, Dv], new state)."""
    st = np.array(state, np.float32, copy=True)
    B, S, Hk, Dk = q.shape
    Hv = v.shape[2]
    rep = Hv // Hk
    qh = np.repeat(np.asarray(q, np.float32), rep, axis=2)
    kh = np.repeat(np.asarray(k, np.float32), rep, axis=2)
    y = np.zeros(v.shape, np.float32)
    per_row = []
    for t in range(S):
        st *= g[:, t, :, None, None]
        kv = np.einsum("bhvk,bhk->bhv", st, kh[:, t])
        delta = (v[:, t] - kv) * beta[:, t, :, None]
        st += delta[..., :, None] * kh[:, t][..., None, :]
        y[:, t] = np.einsum("bhvk,bhk->bhv", st, qh[:, t])
        if rows:
            per_row.append(st.copy())
    if rows:
        return rb(y, bf), st, np.stack(per_row, 1)
    return rb(y, bf), st


# ------------------------------------------------------------------ rope / attention
def inv_freq(rot_dim=64, theta=1e7):
    return (1.0 / (theta ** (np.arange(0, rot_dim, 2, dtype=np.float32) / rot_dim))).astype(np.float32)


def rope(x, pos, rot_dim=64, theta=1e7, bf=True):
    """Partial NeoX rope on the first rot_dim dims of the last axis; x [..., S, Dh], pos [S]. Half-split pairs."""
    x = np.asarray(x, np.float32)
    ang = np.asarray(pos, np.float32)[:, None] * inv_freq(rot_dim, theta)[None]       # [S, rot/2]
    c, s = np.cos(ang), np.sin(ang)
    h = rot_dim // 2
    x1, x2 = x[..., :h], x[..., h:rot_dim]
    out = x.copy()
    out[..., :h] = x1 * c - x2 * s
    out[..., h:rot_dim] = x2 * c + x1 * s
    return rb(out, bf)


def attention(q, K, V, q_pos, allowed=None, scale=None, bf=True):
    """q [Hq, S, Dh], K/V [Hkv, T, Dh] (K roped); query row t at absolute position q_pos[t] attends keys
    0..q_pos[t] (causal), or the boolean allowed [S, T] if given. GQA: q head h -> kv head h // (Hq/Hkv).
    Returns [S, Hq*Dh] (heads concatenated, as attn.core)."""
    Hq, S, Dh = q.shape
    Hkv, T, _ = K.shape
    scale = scale or Dh ** -0.5
    if allowed is None:
        allowed = np.arange(T)[None, :] <= np.asarray(q_pos)[:, None]
    rep = Hq // Hkv
    Kh = np.repeat(np.asarray(K, np.float32), rep, 0)
    Vh = np.repeat(np.asarray(V, np.float32), rep, 0)
    sc = np.einsum("hsd,htd->hst", np.asarray(q, np.float32), Kh) * scale
    sc = np.where(allowed[None], sc, -np.inf)
    sc -= sc.max(-1, keepdims=True)
    p = np.exp(sc)
    p /= p.sum(-1, keepdims=True)
    o = np.einsum("hst,htd->hsd", p, Vh)
    return rb(o.transpose(1, 0, 2).reshape(S, Hq * Dh), bf)


def qsa_block_keys(raw_keys, k_ln_w, ratio=4, rot_dim=64, theta=1e7, bf=True):
    """raw_keys [T, 128] -> pooled block keys [T//ratio, 128]: rope(k_layernorm(bf16(mean_f32(4 keys))), pos 4b)."""
    nb = raw_keys.shape[0] // ratio
    m = rb(np.asarray(raw_keys[:nb * ratio], np.float32).reshape(nb, ratio, -1).mean(1), bf)
    n = rmsnorm(m, k_ln_w, bf=bf)
    return rope(n, np.arange(nb) * ratio, rot_dim, theta, bf)


def qsa_scores(iq, block_keys):
    """iq [4, S, 128] (normed + roped indexer queries), block_keys [nb, 128] -> [S, nb] f32."""
    sc = np.einsum("hsd,bd->hsb", np.asarray(iq, np.float32), np.asarray(block_keys, np.float32))
    return np.maximum(sc, 0).sum(0) / math.sqrt(iq.shape[-1])


def qsa_select(scores, T, budget_blocks=512, ratio=4):
    """Per row t (absolute end T + t + 1): the set of selected block ids (top budget among complete blocks) and
    the allowed-token mask [S, T+S] the attention uses (selected blocks + tail when sparse, else causal)."""
    S, nb = scores.shape
    key_len = T + S
    sets, allowed = [], np.zeros((S, key_len), bool)
    for t in range(S):
        end = T + t + 1
        comp = end // ratio
        sc = np.where(np.arange(nb) < comp, scores[t], -np.inf)
        order = np.argsort(-sc, kind="stable")[:budget_blocks]
        sel = set(int(b) for b in order if b < comp)
        sets.append(sel)
        if comp > budget_blocks:
            for b in sel:
                allowed[t, b * ratio:(b + 1) * ratio] = True
            allowed[t, comp * ratio:end] = True
        else:
            allowed[t, :end] = True
    return sets, allowed


# ------------------------------------------------------------------ MoE
def moe_route(x, Wg, top_k=10, bf=True):
    """Router: logits = x @ Wg^T (bf16), p = softmax (f32 -> bf16), top-k by p, weights renormalised (bf16).
    Returns logits, probs, ids [.., k] (descending p), weights [.., k]."""
    logits = linear(x, Wg, bf)
    z = logits - logits.max(-1, keepdims=True)
    p = np.exp(z)
    p = rb(p / p.sum(-1, keepdims=True), bf)
    ids = np.argsort(-p, axis=-1, kind="stable")[..., :top_k]
    w = np.take_along_axis(p, ids, -1)
    w = rb(w / rb(w.sum(-1, keepdims=True), bf), bf)
    return logits, p, ids, w


def had128(x):
    """x @ blockdiag(H128) along the last axis (H normalised, symmetric)."""
    import exl3_codec as X
    s = x.shape
    return (np.asarray(x, np.float64).reshape(*s[:-1], s[-1] // 128, 128) @ X.H128).reshape(s)


def exl3_inner_out(x, inner, suh):
    """The EXL3 GEMV before the output transform: ((x * suh) H) W'   (then y = (that H) * svh)."""
    return (had128(np.asarray(x, np.float64) * suh) @ inner).astype(np.float32)


def exl3_finish(inner_out, svh):
    return (had128(inner_out) * svh).astype(np.float32)


def swiglu_expert(x, Wg, Wu, Wd):
    """down(silu(x @ Wg) * (x @ Wu)) with W [in, out] (EXL3 orientation), f32 throughout."""
    return (silu(x @ Wg) * (x @ Wu)) @ Wd


def moe_routed(x, ids, w, P, layer, cache=None, bf=True):
    """sum_k w_k * expert_{ids_k}(x) per row, experts decoded from the pack (dense f32). x [N, D], ids/w [N, k]."""
    cache = {} if cache is None else cache
    x = np.asarray(x, np.float32).reshape(-1, x.shape[-1])
    ids = np.asarray(ids).reshape(x.shape[0], -1)
    w = np.asarray(w, np.float32).reshape(x.shape[0], -1)
    y = np.zeros_like(x)
    for r in range(x.shape[0]):
        for e, we in zip(ids[r], w[r]):
            e = int(e)
            if (layer, e) not in cache:
                cache[(layer, e)] = tuple(P.expert(layer, p, e) for p in ("gate", "up", "down"))
            y[r] += we * swiglu_expert(x[r], *cache[(layer, e)])
    return rb(y, bf)


def shared_expert(x, P, prefix, bf=True):
    """prefix = '...layers.N.mlp'. Returns dict gate_proj, up_proj, act, out, scale (as the goldens)."""
    r = {}
    r["shared_gate_proj"] = linear(x, P.linear(prefix + ".shared_expert.gate_proj"), bf)
    r["shared_up_proj"] = linear(x, P.linear(prefix + ".shared_expert.up_proj"), bf)
    r["shared_act"] = rb((silu_bf(r["shared_gate_proj"]) if bf else silu(r["shared_gate_proj"])) * r["shared_up_proj"], bf)
    r["shared_out"] = linear(r["shared_act"], P.linear(prefix + ".shared_expert.down_proj"), bf)
    r["shared_scale"] = rb(sigmoid(linear(x, P.linear(prefix + ".shared_expert_gate"), bf)), bf)
    return r


# ------------------------------------------------------------------ PLE
def ngram_ids(history, multipliers, sizes, offsets, eos, ngram=3, heads=8):
    """history [B, 2 + S] int64 (2 previous tokens, EOS-filled at start, then this step's ids) -> ids [B, S, 16]."""
    h = np.asarray(history, np.int64)
    B, Lh = h.shape
    pos = np.arange(Lh)
    sh = [h]
    for s in range(1, ngram):
        out = np.full_like(h, eos)
        for b in range(B):
            last_eos = -1
            for p in range(Lh):
                seg = last_eos + 1                     # segment start from EOS strictly before p
                if p - s >= 0 and p - seg >= s:
                    out[b, p] = h[b, p - s]
                if h[b, p] == eos:
                    last_eos = p
        sh.append(out)
    m = np.asarray(multipliers, np.int64)
    sizes = np.asarray(sizes, np.int64)
    offsets = np.asarray(offsets, np.int64)
    blocks = []
    with np.errstate(over="ignore"):
        for n in range(2, ngram + 1):
            mixed = sh[0] * m[0]
            for j in range(1, n):
                mixed = np.bitwise_xor(mixed, sh[j] * m[j])
            st = (n - 2) * heads
            blocks.append(mixed[..., None] % sizes[None, None, st:st + heads] + offsets[None, None, st:st + heads])
    del pos
    return np.concatenate(blocks, -1)[:, -(Lh - (ngram - 1)):]


class NgramTable:
    def __init__(self, P):
        path = P.path / P.config["ngram_table"]["file"]
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        w = hdr["weight"]
        a, _ = w["data_offsets"]
        self.mm = np.memmap(path, dtype=np.uint16, mode="r", offset=8 + n + a, shape=tuple(w["shape"]))

    def rows(self, ids):
        ids = np.asarray(ids)
        r = np.asarray(self.mm[ids.reshape(-1)])
        return (r.astype(np.uint32) << 16).view(np.float32).reshape(*ids.shape[:-1], -1)


def ple_block(h, emb, conv_state, P, prefix, bf=True):
    """prefix = '...layers.1.ple'. h [B,S,10240], emb [B,S,2560] (gathered rows). Returns dict as the goldens."""
    r = {}
    s = h.shape[:-1]
    r["keys"] = rmsnorm(linear(emb, P.linear(prefix + ".key_proj"), bf), P.norm(prefix + ".norm_key.weight"),
                        group=D, bf=bf).reshape(*s, HC, D)
    r["values"] = linear(emb, P.linear(prefix + ".value_proj"), bf)
    r["queries"] = rmsnorm(h, P.norm(prefix + ".norm_query.weight"), group=D, bf=bf).reshape(*s, HC, D)
    g = rb(np.sum(r["keys"] * r["queries"], -1, keepdims=True, dtype=np.float32), bf)
    g = rb(g / math.sqrt(D), bf)
    r["gate"] = rb(np.sign(g) * np.sqrt(np.maximum(np.abs(g), 1e-6)), bf)
    r["gated_values"] = rb((sigmoid_bf if bf else sigmoid)(r["gate"]) * r["values"][..., None, :], bf).reshape(*s, HC * D)
    r["normed"] = rmsnorm(r["gated_values"], P.norm(prefix + ".norm_conv.weight"), group=D, bf=bf)
    w = P.tensor(prefix + ".conv1d.weight")
    co, r["conv_state_out"] = causal_conv(conv_state, r["normed"], w, dilation=3, bf=bf)
    r["conv_out"] = silu_bf(co) if bf else silu(co)
    r["out"] = rb(r["gated_values"] + r["conv_out"], bf)
    return r
