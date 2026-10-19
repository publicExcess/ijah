"""Sub-block goldens for single Flash-Next decoder layers, from the MLX reference on the same pack (lab-ref).

Reference = fn_exl3 (mlx-vlm 0.7.7 qwen4_exp, routed experts on the EXL3 kernels) over the K5 pack
(IJAH_GOLDEN_PACK: the K5 pack, K5 experts + Sushi 8-bit trunk). Only the layers needed are built
(0..max(--layers)), plus the 8-bit embedding.

Each layer is run through an INSTRUMENTED copy of Qwen4ExpDecoderLayer.__call__ that calls the very same
modules / compiled helpers in the same order and records every intermediate; the same step is then run through
the stock layer on a cache rebuilt from the saved state, and the outputs AND caches must agree bit for bit
(so the goldens are exactly the reference, and the saved state is enough to resume).

Scenario: a real prompt of --prompt-len tokens is prefilled (case `prefill`), then from that state
  decode1  : 1 row, the next prompt token, position T0
  verify4  : 4 rows (MTP-verify shape), the next 4 prompt tokens, positions T0..T0+3
both resuming from the SAME post-prefill state. Optional --long N: an N-token prefill (chunks of 2048, as the
model) so the QSA indexer is active (N > 2048), then decode1 / verify4 on the attention layers only.

Output: tests/golden/data/L{LL}_{case}/{name}.npy + manifest.json (see docs/reference.md). Floats are saved as
float32 (bf16/f16 upcasts are exact); manifest records each tensor's reference dtype.

Run (holding the machine lock):
  .venv/bin/python -I -B tools/gen_goldens.py
      [--layers 0,1,3] [--prompt-len 48] [--long 2304]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refenv  # noqa: E402

refenv.add_paths(mlx=True)
import fn_exl3 as F  # noqa: E402
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402
from mlx_vlm.models.cache import ArraysCache  # noqa: E402
from mlx_vlm.models.qwen3_5 import language as Q35  # noqa: E402
from mlx_vlm.models.qwen3_5.gated_delta import _compute_g_beta, gated_delta_update_with_states  # noqa: E402

L = F.L
PROMPT = ("The Apple M5 Max has a unified memory architecture: the CPU and the GPU share one pool of fast memory, "
          "so a large language model can be memory-mapped once and read by Metal kernels without copies. Decoding "
          "one token is bound by memory bandwidth, because every active weight must be read once per step. A "
          "mixture-of-experts model only reads the experts its router selects, which is why a 35B model with ten "
          "active experts out of 512 can decode quickly on a laptop. Write a short explanation of why that is.")
LONG_TEXT = refenv.mlx_vlm_file("models/qwen4_exp/language.py")

# ------------------------------------------------------------------ tensor metadata (name -> desc, pack weights)
# {L} is replaced by the layer prefix "language_model.model.layers.N."
META = {
    "hc_in": ("decoder-layer input: 4 hyper-connection streams [.., 4*2560] (stream s = cols s*2560..)", []),
    "embed": ("token embedding (8-bit g64 QuantizedEmbedding), before the x4 tile", ["language_model.model.embed_tokens"]),
    "ple.ngram_ids": ("n-gram row ids into ngram_table.bin, [.., 16 heads] (heads 0-7 bigram, 8-15 trigram)", ["{L}ple.ple_embedding.*"]),
    "ple.token_history_in": ("PLE cache[3]: last 2 token ids before this step (EOS-filled at start)", []),
    "ple.token_history_out": ("PLE cache[3] after the step", []),
    "ple.embeddings": ("gathered n-gram rows [.., 16*160=2560] (bf16 table rows, weight_scale 1)", ["ngram_table.bin"]),
    "ple.keys": ("norm_key(key_proj(emb)) [.., 4, 2560]", ["{L}ple.key_proj", "{L}ple.norm_key.weight"]),
    "ple.values": ("value_proj(emb) [.., 2560]", ["{L}ple.value_proj"]),
    "ple.queries": ("norm_query(hc_in) [.., 4, 2560]", ["{L}ple.norm_query.weight"]),
    "ple.gate": ("signed-sqrt gate per stream [.., 4, 1]", []),
    "ple.gated_values": ("sigmoid(gate) * values, [.., 4*2560]", []),
    "ple.normed": ("norm_conv(gated_values) [.., 4*2560]", ["{L}ple.norm_conv.weight"]),
    "ple.conv_state_in": ("PLE cache[2]: last 9 normed rows (dilated conv state) [1, 9, 10240]", []),
    "ple.conv_state_out": ("PLE cache[2] after the step", []),
    "ple.conv_out": ("silu(depthwise dilated conv(normed)) [.., 10240]", ["{L}ple.conv1d.weight"]),
    "ple.out": ("PLE output gated_values + conv_out, added to hc_in", []),
    "hc_after_ple": ("hc_in + ple.out (the attention HC read input on a PLE layer)", []),
    "attn_hc.normed": ("hc_norm(h): RMSNorm per stream (groups of 2560) [.., 10240]", ["{L}attn_hyper_connection.hc_norm.weight"]),
    "attn_hc.down": ("input_mix_weight_down(normed) [.., 320]", ["{L}attn_hyper_connection.input_mix_weight_down"]),
    "attn_hc.mix": ("silu(down / 4) [.., 320]", []),
    "attn_hc.up": ("input_mix_weight_up(mix) [.., 10240]", ["{L}attn_hyper_connection.input_mix_weight_up"]),
    "attn_hc.block_in": ("HC read output = mean_s(sigmoid(up_s) * normed_s) [.., 2560]: the attention/GDN input", []),
    "attn_hc.inject_logits": ("block_inject_weight(normed) [.., 4]", ["{L}attn_hyper_connection.block_inject_weight.weight"]),
    "attn_hc.inject_w": ("2 * sigmoid(inject_logits / 4) [.., 4]", []),
    "attn_hc.out": ("HC write: h + branch[.., None, :] * inject_w[.., :, None] [.., 10240]", []),
    "gdn.qkv": ("in_proj_qkv(x) [.., 10240] = q 2048 | k 2048 | v 6144", ["{L}linear_attn.in_proj_qkv"]),
    "gdn.z": ("in_proj_z(x) [.., 6144] (48 v-heads x 128)", ["{L}linear_attn.in_proj_z"]),
    "gdn.a": ("in_proj_a(x) [.., 48]", ["{L}linear_attn.in_proj_a"]),
    "gdn.b": ("in_proj_b(x) [.., 48]", ["{L}linear_attn.in_proj_b"]),
    "gdn.conv_state_in": ("GDN cache[0]: last 3 qkv rows [1, 3, 10240]", []),
    "gdn.conv_state_out": ("GDN cache[0] after the step", []),
    "gdn.conv_out": ("silu(depthwise causal conv K=4 over [state; qkv]) [.., 10240]", ["{L}linear_attn.conv1d.weight"]),
    "gdn.q": ("L2-normalised q * 128^-0.5 [.., 16, 128]", []),
    "gdn.k": ("L2-normalised k [.., 16, 128]", []),
    "gdn.v": ("v [.., 48, 128] (slice of conv_out)", []),
    "gdn.g": ("decay exp(-exp(A_log) * softplus(a + dt_bias)) [.., 48] f32", ["{L}linear_attn.A_log", "{L}linear_attn.dt_bias"]),
    "gdn.beta": ("sigmoid(b) [.., 48]", []),
    "gdn.ssm_state_in": ("GDN cache[1]: recurrent state [1, 48, 128(v), 128(k)] f32", []),
    "gdn.ssm_state_out": ("GDN cache[1] after the step", []),
    "gdn.ssm_state_rows": ("state after each row [1, S, 48, 128, 128] f32 (mlx-vlm's speculative with-states kernel; "
                           "row S-1 = ssm_state_out up to kernel rounding)", []),
    "gdn.core_out": ("gated-delta output y [.., 48, 128]", []),
    "gdn.gated_norm": ("rms_norm(y) * norm.weight * sigmoid(z) per v-head [.., 48, 128]", ["{L}linear_attn.norm.weight"]),
    "gdn.out": ("out_proj(gated_norm) [.., 2560]: the branch", ["{L}linear_attn.out_proj"]),
    "attn.idx_qk": ("indexer.index_qk_proj(x) [.., 640] = 4 q-heads x 128 | 1 k x 128", ["{L}self_attn.indexer.index_qk_proj"]),
    "attn.idx_q": ("indexer query: q_layernorm then rope (partial 64 of 128) [1, 4, S, 128]", ["{L}self_attn.indexer.q_layernorm.weight"]),
    "attn.idx_raw_keys": ("indexer raw keys of this step [1, S, 128] (appended to index_keys)", []),
    "attn.index_keys_in": ("cache: raw indexer keys of all past tokens [1, T, 128]", []),
    "attn.index_keys_out": ("cache: raw indexer keys after the step [1, T+S, 128]", []),
    "attn.index_block_keys_in": ("cache: pooled block keys k_layernorm(mean of 4 raw keys) + rope at block start [1, 1, nb, 128]", ["{L}self_attn.indexer.k_layernorm.weight"]),
    "attn.index_block_keys_out": ("cache: pooled block keys after the step", []),
    "attn.idx_scores": ("block scores sum_h relu(q_h . K_b) / sqrt(128) [1, S, nb] f32 (before the causal -inf mask)", []),
    "attn.selected_blocks": ("indexer selection [1, S, 512] block ids (-1 = none); dense causal unless > 512 complete blocks", []),
    "attn.q_proj": ("q_proj(x) [.., 12288] = per head [256 q | 256 gate] x 24", ["{L}self_attn.q_proj"]),
    "attn.k_proj": ("k_proj(x) [.., 512] (2 kv heads x 256)", ["{L}self_attn.k_proj"]),
    "attn.v_proj": ("v_proj(x) [.., 512]", ["{L}self_attn.v_proj"]),
    "attn.q_normed": ("q_norm(q) per head, before rope [.., 24, 256]", ["{L}self_attn.q_norm.weight"]),
    "attn.k_normed": ("k_norm(k) per head, before rope [.., 2, 256]", ["{L}self_attn.k_norm.weight"]),
    "attn.q_rope": ("queries after rope [1, 24, S, 256]", []),
    "attn.k_rope": ("keys of this step after rope [1, 2, S, 256]", []),
    "attn.k_cache_in": ("KV cache keys (roped) before the step [1, 2, T, 256]", []),
    "attn.v_cache_in": ("KV cache values before the step [1, 2, T, 256]", []),
    "attn.k_cache_out": ("KV cache keys after the step [1, 2, T+S, 256]", []),
    "attn.v_cache_out": ("KV cache values after the step [1, 2, T+S, 256]", []),
    "attn.core": ("softmax(q k^T * 256^-0.5 + causal) v, heads concatenated [.., 6144]", []),
    "attn.gate": ("output gate logits (second half of each q_proj head) [.., 6144]", []),
    "attn.gated": ("core * sigmoid(gate) [.., 6144]", []),
    "attn.out": ("o_proj(gated) [.., 2560]: the branch", ["{L}self_attn.o_proj"]),
    "mlp_hc.normed": ("hc_norm(h) [.., 10240]", ["{L}mlp_hyper_connection.hc_norm.weight"]),
    "mlp_hc.down": ("input_mix_weight_down(normed) [.., 320]", ["{L}mlp_hyper_connection.input_mix_weight_down"]),
    "mlp_hc.mix": ("silu(down / 4)", []),
    "mlp_hc.up": ("input_mix_weight_up(mix) [.., 10240]", ["{L}mlp_hyper_connection.input_mix_weight_up"]),
    "mlp_hc.block_in": ("HC read output: the MoE input [.., 2560]", []),
    "mlp_hc.inject_logits": ("block_inject_weight(normed) [.., 4]", ["{L}mlp_hyper_connection.block_inject_weight.weight"]),
    "mlp_hc.inject_w": ("2 * sigmoid(inject_logits / 4)", []),
    "moe.router_logits": ("gate(x) [.., 512] (bf16 Linear)", ["{L}mlp.gate.weight"]),
    "moe.router_probs": ("softmax(router_logits, precise) [.., 512]", []),
    "moe.topk_ids": ("top-10 expert ids [.., 10] (argpartition order: a SET, order is not meaningful)", []),
    "moe.topk_weights": ("renormalised top-10 probs [.., 10], same order as topk_ids", []),
    "moe.routed_out": ("sum_k w_k * expert_k(x) on the EXL3 kernels [.., 2560]", ["{L}mlp.switch_mlp.{gate,up,down}_proj.{trellis,suh,svh}"]),
    "moe.shared_gate_proj": ("shared_expert.gate_proj(x) [.., 640]", ["{L}mlp.shared_expert.gate_proj"]),
    "moe.shared_up_proj": ("shared_expert.up_proj(x) [.., 640]", ["{L}mlp.shared_expert.up_proj"]),
    "moe.shared_act": ("silu(gate) * up [.., 640]", []),
    "moe.shared_out": ("shared_expert(x) = down_proj(act) [.., 2560]", ["{L}mlp.shared_expert.down_proj"]),
    "moe.shared_scale": ("sigmoid(shared_expert_gate(x)) [.., 1]", ["{L}mlp.shared_expert_gate.weight"]),
    "moe.out": ("routed_out + shared_scale * shared_out [.., 2560]: the branch", []),
    "hc_out": ("decoder-layer output streams [.., 10240] (= next layer's hc_in)", []),
    "head.hc_in": ("head input streams [.., 10240] (here: layer 3's hc_out, a synthetic stand-in for layer 47's)", []),
    "mixer.normed": ("final mixer hc_norm(h)", ["language_model.model.hyper_connection_mixer.hc_norm.weight"]),
    "mixer.down": ("final mixer input_mix_weight_down(normed) [.., 320]", ["language_model.model.hyper_connection_mixer.input_mix_weight_down"]),
    "mixer.mix": ("silu(down / 4)", []),
    "mixer.up": ("final mixer input_mix_weight_up(mix) [.., 10240]", ["language_model.model.hyper_connection_mixer.input_mix_weight_up"]),
    "mixer.block_in": ("final hidden = mean_s(sigmoid(up_s) * normed_s) [.., 2560] (no inject)", []),
    "head.logits": ("lm_head(final hidden) [.., 248320] (8-bit)", ["language_model.lm_head"]),
}


class Rec:
    def __init__(self, on=True):
        self.on = on
        self.items = {}

    def __call__(self, name, a):
        if self.on and a is not None:
            assert name not in self.items, name
            self.items[name] = a
        return a


def np_of(a):
    """mx array -> (numpy array to save, reference dtype string). Floats -> f32 (exact for bf16/f16)."""
    dt = str(a.dtype).replace("mlx.core.", "")
    if a.dtype in (mx.bfloat16, mx.float16, mx.float32):
        return np.array(a.astype(mx.float32)), dt
    if a.dtype == mx.bool_:
        return np.array(a).astype(np.uint8), dt
    return np.array(a), dt


def pos3(start, n):
    """Text position ids exactly as LanguageModel builds them for the model: [3, 1, n] (all MRoPE axes equal)."""
    return mx.broadcast_to(mx.arange(start, start + n)[None, None], (3, 1, n))


# ------------------------------------------------------------------ instrumented sub-blocks (same calls, same order)
def hc_read(hc, h, rec, p):
    act, mix_streams, inj_gate = L._qwen4_hyper_pointwise(hc.hc_count, hc.hidden_size)
    normed = rec(p + ".normed", hc.hc_norm(h))
    down = rec(p + ".down", hc.input_mix_weight_down(normed))
    mix = rec(p + ".mix", act(down))
    up = rec(p + ".up", hc.input_mix_weight_up(mix))
    mixed = rec(p + ".block_in", mix_streams(up, normed))
    inj = rec(p + ".inject_logits", L._row_invariant_quantized_projection(hc.block_inject_weight, normed))
    injw = rec(p + ".inject_w", inj_gate(inj))
    return mixed, injw


def hc_write(h, branch, injw):
    injection = branch[..., None, :] * injw[..., None]
    return h + injection.reshape(*h.shape)


def ple_fwd(ple, h, ids, cache, mask, rec):
    e = ple.ple_embedding
    prev = cache[3] if cache[3] is not None else mx.full((ids.shape[0], e.context_len), e.eos_token_id, dtype=mx.int64)
    rec("ple.token_history_in", prev)
    hist = mx.concatenate([prev, ids.astype(mx.int64)], axis=-1)
    rec("ple.ngram_ids", F.ngram_row_ids(e, hist)[:, -ids.shape[1]:])
    emb = rec("ple.embeddings", e(ids, cache))
    rec("ple.token_history_out", cache[3])
    keys = rec("ple.keys", ple.norm_key(ple.key_proj(emb)).reshape(*h.shape[:-1], ple.hc_count, ple.hidden_size))
    values = rec("ple.values", ple.value_proj(emb))
    queries = rec("ple.queries", ple.norm_query(h).reshape(*h.shape[:-1], ple.hc_count, ple.hidden_size))
    import math
    gate = mx.sum(keys * queries, axis=-1, keepdims=True) / math.sqrt(ple.hidden_size)
    gate = rec("ple.gate", mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)))
    gated = rec("ple.gated_values", (mx.sigmoid(gate) * values[..., None, :]).reshape(*h.shape))
    normed = rec("ple.normed", ple.norm_conv(gated))
    assert mask is None or not (isinstance(mask, mx.array) and mask.ndim == 2)
    rec("ple.conv_state_in", cache[2] if cache[2] is not None
        else mx.zeros((h.shape[0], ple.short_conv_state_len, h.shape[-1]), dtype=h.dtype))
    conv = rec("ple.conv_out", ple._short_conv(normed, cache))
    rec("ple.conv_state_out", cache[2])
    return rec("ple.out", gated + conv)


def gdn_fwd(m, x, cache, mask, rec):
    B, S, _ = x.shape
    qkv = rec("gdn.qkv", m.in_proj_qkv(x))
    z = rec("gdn.z", m.in_proj_z(x))
    b, a = m._project_gates(x)
    rec("gdn.a", a)
    rec("gdn.b", b)
    z = z.reshape(B, S, -1, m.head_v_dim)
    if cache[0] is not None and cache[0].shape[0] == B:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros((B, m.conv_kernel_size - 1, m.conv_dim), dtype=x.dtype)
    rec("gdn.conv_state_in", conv_state)
    assert mask is None, "goldens are unpadded B=1"
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    cache.update_window(0, conv_input, m.conv_kernel_size - 1, lengths=cache.lengths)
    rec("gdn.conv_state_out", cache[0])
    if S == 1 and conv_input.shape[1] == m.conv_kernel_size and m.conv1d.weight.dtype in (mx.bfloat16, mx.float16):
        conv_out = nn.silu(m._causal_conv1d_decode(conv_input))
    else:
        conv_out = nn.silu(m.conv1d(conv_input))
    rec("gdn.conv_out", conv_out)
    q, k, v = [t.reshape(B, S, h, d) for t, h, d in zip(
        mx.split(conv_out, [m.key_dim, 2 * m.key_dim], -1),
        [m.num_k_heads, m.num_k_heads, m.num_v_heads], [m.head_k_dim, m.head_k_dim, m.head_v_dim])]
    q, k = m._normalize_qk(q, k)
    rec("gdn.q", q)
    rec("gdn.k", k)
    rec("gdn.v", v)
    g, beta = _compute_g_beta(m.A_log, a, b, m.dt_bias)
    rec("gdn.g", g)
    rec("gdn.beta", beta)
    st = cache[1] if cache[1] is not None else mx.zeros((B, m.num_v_heads, m.head_v_dim, m.head_k_dim), mx.float32)
    rec("gdn.ssm_state_in", st)
    out, _ = Q35.gated_delta_update(q, k, v, a, b, m.A_log, m.dt_bias, mask=mask, use_kernel=not m.training, cache=cache)
    rec("gdn.core_out", out)
    rec("gdn.ssm_state_out", cache[1])
    if 1 < S <= 16:   # MTP verify: the state after each row (what a rejected draft rolls back to), side computation
        rec("gdn.ssm_state_rows", gated_delta_update_with_states(q, k, v, a, b, m.A_log, m.dt_bias, state=st,
                                                                 state_steps=S)[2])
    if hasattr(cache, "advance"):
        cache.advance(S)
        Q35._qwen3_5_advance_lengths_info(cache, S)
    out = rec("gdn.gated_norm", m.norm(out, z))
    return rec("gdn.out", m.out_proj(out.reshape(B, S, -1)))


def attn_fwd(m, x, cache, mask, pos, rec):
    ix = m.indexer
    B, S, _ = x.shape
    qk = rec("attn.idx_qk", ix.index_qk_proj(x))
    qk4 = qk.reshape(B, S, ix.n_heads + ix.kv_heads, ix.head_dim)
    iq = ix._apply_rope(ix.q_layernorm(qk4[:, :, :ix.n_heads]).transpose(0, 2, 1, 3), pos)
    rec("attn.idx_q", iq)
    rec("attn.idx_raw_keys", qk4[:, :, ix.n_heads:].squeeze(2))
    rec("attn.index_keys_in", cache.index_keys)
    rec("attn.index_block_keys_in", cache.index_block_keys)
    kv = cache.state
    rec("attn.k_cache_in", kv[0])
    rec("attn.v_cache_in", kv[1])
    selection = ix.select_from_projected(qk, cache, pos)
    rec("attn.index_keys_out", cache.index_keys)
    rec("attn.index_block_keys_out", cache.index_block_keys)
    bk = cache.index_block_keys
    if bk is not None and bk.shape[2] > 0:
        sc = iq.astype(mx.float32) @ bk.astype(mx.float32).transpose(0, 1, 3, 2)
        rec("attn.idx_scores", mx.sum(mx.maximum(sc, 0), axis=1) / (ix.head_dim ** 0.5))
    if selection is not None:
        rec("attn.selected_blocks", selection.selected_blocks)

    q_out, keys, values = m.q_proj(x), m.k_proj(x), m.v_proj(x)
    rec("attn.q_proj", q_out)
    rec("attn.k_proj", keys)
    rec("attn.v_proj", values)
    qs, _g = mx.split(q_out.reshape(B, S, m.num_attention_heads, -1), 2, axis=-1)
    rec("attn.q_normed", m.q_norm(qs))
    rec("attn.k_normed", m.k_norm(keys.reshape(B, S, m.num_key_value_heads, -1)))

    standard = mask is None or (isinstance(mask, str) and mask == "causal")
    if selection is None or (isinstance(mask, str) and not standard):
        path = "dense"
        queries, k_all, v_all, gate, mask2 = m._prepare_projected_qkv(q_out, keys, values, cache, pos, None, mask)
        assert not (isinstance(mask2, str) and mask2 == "left_padded_decode")
        output = Q35.scaled_dot_product_attention(queries, k_all, v_all, cache=cache, scale=m.scale, mask=mask2)
    else:
        sparse_candidate = selection.zero_padding and standard
        qsa_mask = None if sparse_candidate else ix.build_mask(selection)
        if qsa_mask is not None:
            assert standard
            mask = qsa_mask
        path = "qsa"
        queries, k_all, v_all, gate, mask2 = m._prepare_projected_qkv(
            q_out, keys, values, cache, pos, None, None if sparse_candidate else mask)
        output = L.dispatch_qsa_attention(
            queries, k_all, v_all, selection.selected_blocks, selection.query_ends, cache=cache, scale=m.scale,
            block_size=ix.compress_ratio, causal=standard, mask=mask, mask_factory=lambda: ix.build_mask(selection))
    rec("attn.q_rope", queries)
    rec("attn.k_rope", k_all[..., -S:, :])
    kv = cache.state
    rec("attn.k_cache_out", kv[0])
    rec("attn.v_cache_out", kv[1])
    output = output.transpose(0, 2, 1, 3).reshape(B, S, -1)
    rec("attn.core", output)
    rec("attn.gate", gate)
    gated = rec("attn.gated", output * mx.sigmoid(gate))
    return rec("attn.out", m.o_proj(gated)), path


def moe_fwd(m, x, rec):
    logits = rec("moe.router_logits", m.gate(x))
    gates = rec("moe.router_probs", mx.softmax(logits, axis=-1, precise=True))
    k = m.top_k
    inds = mx.stop_gradient(mx.argpartition(gates, kth=-k, axis=-1)[..., -k:])
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if m.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    i2, s2 = m.route(x)
    assert np.array_equal(np.array(i2), np.array(inds)) and np.array_equal(np.array(s2.astype(mx.float32)),
                                                                           np.array(scores.astype(mx.float32)))
    rec("moe.topk_ids", inds.astype(mx.int32))
    rec("moe.topk_weights", scores)
    y = rec("moe.routed_out", m.routed(x, inds, scores))
    se = m.shared_expert
    gp = rec("moe.shared_gate_proj", se.gate_proj(x))
    up = rec("moe.shared_up_proj", se.up_proj(x))
    act = rec("moe.shared_act", nn.silu(gp) * up)
    so = rec("moe.shared_out", se.down_proj(act))
    sc = rec("moe.shared_scale", m._shared_expert_scale(x))
    return rec("moe.out", y + sc * so)


def head_fwd(mixer, lm_head, h, rec):
    """Final hyper-connection mixer (HC read without inject) + lm_head, checked against the stock modules."""
    rec("head.hc_in", h)
    act, mix_streams, _ = L._qwen4_hyper_pointwise(mixer.hc_count, mixer.hidden_size)
    normed = rec("mixer.normed", mixer.hc_norm(h))
    mix = rec("mixer.mix", act(rec("mixer.down", mixer.input_mix_weight_down(normed))))
    hm = rec("mixer.block_in", mix_streams(rec("mixer.up", mixer.input_mix_weight_up(mix)), normed))
    logits = rec("head.logits", lm_head(hm))
    ref = lm_head(mixer(h))
    mx.eval(logits, ref)
    assert same(logits, ref), "instrumented head != stock mixer + lm_head"
    return logits


def build_head(pack, cfg):
    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.hyper_connection_mixer = L.Qwen4ExpGatedResidual(cfg, use_combine=False)

    class Head(nn.Module):       # same paths as the pack under "language_model."
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    hd = Head()
    F._quantize_like_pack(hd, pack, "language_model.")
    F.load_module(hd.model.hyper_connection_mixer, pack, "language_model.model.hyper_connection_mixer.")
    F.load_module(hd.lm_head, pack, "language_model.lm_head.")
    mx.eval(hd.parameters())
    assert isinstance(hd.lm_head, nn.QuantizedLinear)
    pack.drop_files()
    return hd.model.hyper_connection_mixer, hd.lm_head


def layer_fwd(layer, h, ids, mask, cache, pos, rec):
    rec("hc_in", h)
    path = "gdn" if layer.is_linear else None
    if "ple" in layer:
        h = rec("hc_after_ple", h + ple_fwd(layer.ple, h, ids, cache, mask, rec))
    mixed, injw = hc_read(layer.attn_hyper_connection, h, rec, "attn_hc")
    if layer.is_linear:
        branch = gdn_fwd(layer.linear_attn, mixed, cache, mask, rec)
    else:
        branch, path = attn_fwd(layer.self_attn, mixed, cache, mask, pos, rec)
    h = rec("attn_hc.out", hc_write(h, branch, injw))
    mixed, injw = hc_read(layer.mlp_hyper_connection, h, rec, "mlp_hc")
    h = rec("hc_out", hc_write(h, moe_fwd(layer.mlp, mixed, rec), injw))
    return h, path


# ------------------------------------------------------------------ caches: snapshot (numpy) / rebuild
def fresh_cache(layer):
    return ArraysCache(size=4 if "ple" in layer else 2) if layer.is_linear else L.QSAKVCache()


def snapshot(layer, c):
    out = {}
    if layer.is_linear:
        names = ["gdn.conv_state", "gdn.ssm_state"] + (["ple.conv_state", "ple.token_history"] if "ple" in layer else [])
        for i, n in enumerate(names):
            out[n] = c[i]
    else:
        k, v, ik, ipos, ibk, ratio = c.state
        out.update({"attn.k_cache": k, "attn.v_cache": v, "attn.index_keys": ik, "attn.index_position_ids": ipos,
                    "attn.index_block_keys": ibk})
        out["_ratio"] = ratio
    return out


def restore(layer, snap):
    """Rebuild a cache from a snapshot through numpy (proves the saved state is enough to resume)."""
    def rt(a):
        if a is None or not isinstance(a, mx.array):
            return a
        arr, dt = np_of(a)
        return mx.array(arr).astype(getattr(mx, dt))
    c = fresh_cache(layer)
    if layer.is_linear:
        for i, n in enumerate(["gdn.conv_state", "gdn.ssm_state", "ple.conv_state", "ple.token_history"][:len(c.cache)]):
            c[i] = rt(snap[n])
    else:
        c.state = tuple(rt(snap[n]) for n in ("attn.k_cache", "attn.v_cache", "attn.index_keys",
                                               "attn.index_position_ids", "attn.index_block_keys")) + (snap["_ratio"],)
    return c


def masks(layer, h, cache):
    return L._create_qwen3_5_ssm_mask(h, cache) if layer.is_linear else L._create_qwen4_exp_attention_mask(h, cache)


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a, b).item())


def run_step(layer, lid, h, ids, cache_snap, start, rec):
    """Instrumented step on a rebuilt cache, checked bit-exact against the stock layer on another rebuilt cache."""
    pos = pos3(start, h.shape[1])
    c1 = restore(layer, cache_snap) if cache_snap is not None else fresh_cache(layer)
    c2 = restore(layer, cache_snap) if cache_snap is not None else fresh_cache(layer)
    mask = masks(layer, h, c1)
    out, path = layer_fwd(layer, h, ids, mask, c1, pos, rec)
    mx.eval(out, *rec.items.values())
    ref = layer(h, ids, mask=masks(layer, h, c2), cache=c2, position_ids=pos)
    mx.eval(ref)
    s1, s2 = snapshot(layer, c1), snapshot(layer, c2)
    ok = same(out, ref) and all(same(s1[k], s2[k]) if k != "_ratio" else s1[k] == s2[k] for k in s1)
    if not ok:
        d = float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))).item())
        raise AssertionError(f"layer {lid}: instrumented != stock (max |d| {d}); caches equal: "
                             f"{ {k: (same(s1[k], s2[k]) if k != '_ratio' else s1[k] == s2[k]) for k in s1} }")
    return out, s1, path, mask


def save_case(outdir, name, lid, layer, rec, meta, pack):
    d = outdir / name
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.npy"):
        f.unlink()
    pre = f"language_model.model.layers.{lid}."
    tensors = {}
    total = 0
    for n, a in rec.items.items():
        arr, dt = np_of(a)
        np.save(d / f"{n}.npy", arr)
        total += arr.nbytes
        desc, w = META.get(n, ("", []))
        tensors[n] = {"file": f"{n}.npy", "shape": list(arr.shape), "saved_dtype": str(arr.dtype), "ref_dtype": dt,
                      "desc": desc, "pack_weights": [x.replace("{L}", pre) for x in w]}
    m = dict(meta)
    m.update({"case": name, "layer": lid, "layer_type": layer.layer_type if layer is not None else "head",
              "has_ple": layer is not None and "ple" in layer,
              "pack": str(pack.path), "bytes": total, "tensors": tensors})
    (d / "manifest.json").write_text(json.dumps(m, indent=1))
    return total


# ------------------------------------------------------------------ main
def build(pack, upto):
    t0 = time.time()
    cfg, layers = F.build_layers(pack, list(range(upto + 1)))
    for ly in layers.values():
        ly.eval()     # a layer built outside the model is in training mode: mlx-vlm's GDN would skip its kernel

    class Emb(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
    emb = Emb()
    F._quantize_like_pack(emb, pack, "language_model.model.")
    F.load_module(emb.embed_tokens, pack, "language_model.model.embed_tokens.")
    mx.eval(emb.parameters())
    pack.drop_files()
    print(f"built layers 0..{upto} + embed in {time.time() - t0:.1f} s, active {mx.get_active_memory() / 2**30:.1f} GiB",
          flush=True)
    return cfg, layers, emb.embed_tokens


def tokens(n_needed, text):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(refenv.PACK / "tokenizer.json"))
    ids = tok.encode(text, add_special_tokens=False).ids
    assert len(ids) >= n_needed, (len(ids), n_needed)
    return ids[:n_needed]


def scenario(layers, embed, cfg, ids_all, T0, save_layers, outdir, pack, tag, prefill_chunk=2048, save_prefill=True,
             head=None):
    """Prefill ids_all[:T0], then decode1 (ids[T0]) and verify4 (ids[T0:T0+4]) from the same state."""
    order = sorted(layers)
    hc = cfg.hc_count
    summary = []

    def embed_tile(ids):
        e = embed(mx.array([ids]))
        return e, mx.tile(e, (1, 1, hc))

    # prefill (chunked as the model does)
    snaps = {}
    caches_snap = {lid: None for lid in order}
    for c0 in range(0, T0, prefill_chunk):
        chunk = ids_all[c0:min(T0, c0 + prefill_chunk)]
        e, h = embed_tile(chunk)
        ids = mx.array([chunk])
        for lid in order:
            rec = Rec(on=save_prefill and lid in save_layers and T0 <= prefill_chunk)
            if rec.on and lid == order[0]:
                rec("embed", e)
            h, caches_snap[lid], path, mask = run_step(layers[lid], lid, h, ids, caches_snap[lid], c0, rec)
            if rec.on:
                meta = {"step": "prefill", "rows": len(chunk), "position_start": c0, "token_ids": chunk,
                        "cache_len_in": 0, "attn_path": path, "mask": repr(mask)[:40]}
                nb = save_case(outdir, f"L{lid:02d}_{tag}prefill", lid, layers[lid], rec, meta, pack)
                summary.append((f"L{lid:02d}_{tag}prefill", nb))
    snaps = caches_snap
    for case, n in (("decode1", 1), ("verify4", 4)):
        chunk = ids_all[T0:T0 + n]
        e, h = embed_tile(chunk)
        ids = mx.array([chunk])
        for lid in order:
            rec = Rec(on=lid in save_layers)
            if rec.on and lid == order[0]:
                rec("embed", e)
            h, _, path, mask = run_step(layers[lid], lid, h, ids, snaps[lid], T0, rec)
            if rec.on:
                meta = {"step": case, "rows": n, "position_start": T0, "token_ids": chunk, "cache_len_in": T0,
                        "attn_path": path, "mask": repr(mask)[:40], "prompt_token_ids": ids_all[:T0] if T0 <= 256 else None}
                nb = save_case(outdir, f"L{lid:02d}_{tag}{case}", lid, layers[lid], rec, meta, pack)
                summary.append((f"L{lid:02d}_{tag}{case}", nb))
                print(f"  L{lid:02d} {tag}{case}: {len(rec.items)} tensors, {nb / 2**20:.1f} MiB, path {path}", flush=True)
        if head is not None:
            rec = Rec()
            head_fwd(*head, h, rec)
            meta = {"step": case, "rows": n, "position_start": T0, "token_ids": chunk, "cache_len_in": T0,
                    "attn_path": None, "mask": None, "head_input": f"hc_out of layer {order[-1]} (synthetic)"}
            nb = save_case(outdir, f"HEAD_{tag}{case}", -1, None, rec, meta, pack)
            summary.append((f"HEAD_{tag}{case}", nb))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,1,3", help="layers to save (all layers up to the max are run)")
    ap.add_argument("--prompt-len", type=int, default=48)
    ap.add_argument("--long", type=int, default=0, help="also a long-context scenario of N tokens (QSA active > 2048)")
    ap.add_argument("--out", default=str(refenv.GOLDEN))
    ap.add_argument("--head", action="store_true", help="also final mixer + lm_head cases (HEAD_decode1/verify4)")
    a = ap.parse_args()
    save = [int(x) for x in a.layers.split(",")]
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    pack = F.Pack(refenv.PACK)
    cfg, layers, embed = build(pack, max(save))
    summary = []
    T0 = a.prompt_len
    ids = tokens(T0 + 4, PROMPT)
    head = build_head(pack, cfg) if a.head else None
    summary += scenario(layers, embed, cfg, ids, T0, save, outdir, pack, "", head=head)
    print(f"short scenario done, peak {mx.get_peak_memory() / 2**30:.1f} GiB", flush=True)
    if a.long:
        lids = tokens(a.long + 4, LONG_TEXT.read_text())
        attn_layers = [x for x in save if not layers[x].is_linear]
        summary += scenario(layers, embed, cfg, lids, a.long, attn_layers, outdir, pack, "long_", save_prefill=False)
        print(f"long scenario done, peak {mx.get_peak_memory() / 2**30:.1f} GiB", flush=True)
    (outdir / "INDEX.json").write_text(json.dumps({"cases": [c for c, _ in summary], "pack": str(pack.path),
                                                   "generator": "tools/gen_goldens.py", "args": vars(a)}, indent=1))
    print("cases:", ", ".join(f"{c} ({nb / 2**20:.1f} MiB)" for c, nb in summary))


if __name__ == "__main__":
    main()
