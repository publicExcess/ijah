"""Cross-check every op reference (tools/opref.py) against the MLX reference goldens (numpy only, CPU).

Two modes per case:
  isolated  each op fed the golden's own inputs (what a kernel test does) -> rel L2 vs the golden output
  chained   the whole layer from hc_in alone through op references (no golden intermediates): compounding error
    $PY tools/check_opref.py [CASE ...] [--no-experts] [--summary-json f]
Exit 0 if every isolated check passes compare.py's tolerance for its kind.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare as C  # noqa: E402
import golden  # noqa: E402
import numpy as np  # noqa: E402
import opref as O  # noqa: E402

EXPERT_CACHE = {}


def hc_part(c, P, pre, which, h, out, label):
    r = O.hc_read(h, P, pre + which)
    p = "attn_hc" if which.startswith("attn") else "mlp_hc"
    out(f"{label}rmsnorm grouped (hc_norm)", p + ".normed", r["normed"])
    # isolated ops on golden inputs
    out(f"{label}affine8 g64 (hc down)", p + ".down", O.linear(c[p + ".normed"], P.linear(pre + which + ".input_mix_weight_down")))
    out(f"{label}silu(d/4)", p + ".mix", O.silu_bf(O.rb(c[p + ".down"] / 4)))
    out(f"{label}affine8 g64 (hc up)", p + ".up", O.linear(c[p + ".mix"], P.linear(pre + which + ".input_mix_weight_up")))
    s = c[p + ".up"].shape[:-1]
    m = O.rb(O.rb(O.sigmoid(c[p + ".up"])).reshape(*s, 4, 2560) * c[p + ".normed"].reshape(*s, 4, 2560))
    out(f"{label}hc mix_streams", p + ".block_in", O.rb(np.mean(m, -2)))
    out(f"{label}bf16 linear (inject)", p + ".inject_logits",
        O.linear(c[p + ".normed"], P.linear(pre + which + ".block_inject_weight")))
    out(f"{label}2*sigmoid(p/4)", p + ".inject_w", O.rb(2 * O.sigmoid_bf(O.rb(c[p + ".inject_logits"] / 4))))
    out(f"{label}hc read (whole, from h)", p + ".block_in#whole", r["block_in"])


def gdn_part(c, P, pre, out):
    g = pre + "linear_attn"
    x = c["attn_hc.block_in"]
    out("affine8 (in_proj_qkv)", "gdn.qkv", O.linear(x, P.linear(g + ".in_proj_qkv")))
    out("affine8 (in_proj_z)", "gdn.z", O.linear(x, P.linear(g + ".in_proj_z")))
    out("affine8 (in_proj_a)", "gdn.a", O.linear(x, P.linear(g + ".in_proj_a")))
    out("affine8 (in_proj_b)", "gdn.b", O.linear(x, P.linear(g + ".in_proj_b")))
    co, st = O.causal_conv(c["gdn.conv_state_in"], c["gdn.qkv"], P.tensor(g + ".conv1d.weight"))
    out("causal conv K4 + silu", "gdn.conv_out", O.silu_bf(co))
    out("conv state update", "gdn.conv_state_out", st)
    q, k, v = O.gdn_split(c["gdn.conv_out"])
    qn, kn = O.gdn_qk_norm(q, k)
    out("GDN q L2 norm * Dk^-0.5", "gdn.q", qn)
    out("GDN k L2 norm", "gdn.k", kn)
    out("GDN v split", "gdn.v", v)
    gg, beta = O.gdn_gates(c["gdn.a"], c["gdn.b"], P.tensor(g + ".A_log"), P.tensor(g + ".dt_bias"))
    out("GDN decay g", "gdn.g", gg)
    out("GDN beta", "gdn.beta", beta)
    y, s2, rows = O.gdn_recurrence(c["gdn.ssm_state_in"], c["gdn.q"], c["gdn.k"], c["gdn.v"], c["gdn.g"],
                                   c["gdn.beta"], rows=True)
    out("GDN recurrence y", "gdn.core_out", y)
    out("GDN recurrence state", "gdn.ssm_state_out", s2)
    if "gdn.ssm_state_rows" in c:
        out("GDN per-row states (MTP verify)", "gdn.ssm_state_rows", rows)
        out("with-states kernel last row vs state_out", "gdn.ssm_state_out#rows", c["gdn.ssm_state_rows"][:, -1],
            c["gdn.ssm_state_out"])
    z = c["gdn.z"].reshape(c["gdn.core_out"].shape)
    out("gated rmsnorm", "gdn.gated_norm", O.gated_rmsnorm(c["gdn.core_out"], z, P.norm(g + ".norm.weight")))
    out("affine8 (out_proj)", "gdn.out", O.linear(c["gdn.gated_norm"].reshape(*z.shape[:2], -1), P.linear(g + ".out_proj")))


def attn_part(c, P, pre, out):
    a = pre + "self_attn"
    x = c["attn_hc.block_in"]
    pos = c.positions
    T = c.m["cache_len_in"]
    S = c.m["rows"]
    qk = O.linear(x, P.linear(a + ".indexer.index_qk_proj"))
    out("affine8 (indexer qk)", "attn.idx_qk", qk)
    g4 = c["attn.idx_qk"].reshape(1, S, 5, 128)
    iq = O.rope(O.rmsnorm(g4[:, :, :4], P.norm(a + ".indexer.q_layernorm.weight")).transpose(0, 2, 1, 3), pos)
    out("indexer q norm + rope", "attn.idx_q", iq)
    out("indexer raw keys", "attn.idx_raw_keys", g4[:, :, 4])
    if "attn.index_block_keys_out" in c:
        ik = c["attn.index_keys_out"][0]
        bk = O.qsa_block_keys(ik, P.norm(a + ".indexer.k_layernorm.weight"))
        nb = c["attn.index_block_keys_out"].shape[2]
        out("QSA block keys (pool+norm+rope)", "attn.index_block_keys_out", bk[None, None, :nb])
        sc = O.qsa_scores(c["attn.idx_q"][0], c["attn.index_block_keys_out"][0, 0])
        out("QSA block scores", "attn.idx_scores", sc[None])
    allowed = None
    if "attn.selected_blocks" in c:
        sets, allowed = O.qsa_select(c["attn.idx_scores"][0], T)
        sb = np.full((1, S, 512), -1, np.int32)
        for t, sset in enumerate(sets):
            ids = sorted(sset)
            sb[0, t, :len(ids)] = ids
        out("QSA top-512 selection", "attn.selected_blocks", sb)
    out("affine8 (q_proj)", "attn.q_proj", O.linear(x, P.linear(a + ".q_proj")))
    out("affine8 (k_proj)", "attn.k_proj", O.linear(x, P.linear(a + ".k_proj")))
    out("affine8 (v_proj)", "attn.v_proj", O.linear(x, P.linear(a + ".v_proj")))
    qs = c["attn.q_proj"].reshape(1, S, 24, 512)[..., :256]
    out("q_norm per head", "attn.q_normed", O.rmsnorm(qs, P.norm(a + ".q_norm.weight")))
    out("k_norm per head", "attn.k_normed", O.rmsnorm(c["attn.k_proj"].reshape(1, S, 2, 256), P.norm(a + ".k_norm.weight")))
    out("rope partial 64/256 (q)", "attn.q_rope", O.rope(c["attn.q_normed"].transpose(0, 2, 1, 3), pos))
    out("rope partial 64/256 (k)", "attn.k_rope", O.rope(c["attn.k_normed"].transpose(0, 2, 1, 3), pos))
    out("KV append", "attn.k_cache_out", np.concatenate([c["attn.k_cache_in"], c["attn.k_rope"]], 2)
        if "attn.k_cache_in" in c else c["attn.k_rope"])
    core = O.attention(c["attn.q_rope"][0], c["attn.k_cache_out"][0], c["attn.v_cache_out"][0], pos, allowed)
    out("attention (sdpa / QSA mask)" if allowed is not None else "attention (sdpa causal)", "attn.core", core[None])
    gate = c["attn.q_proj"].reshape(1, S, 24, 512)[..., 256:].reshape(1, S, -1)
    out("output gate split", "attn.gate", gate)
    out("core * sigmoid(gate)", "attn.gated", O.rb(c["attn.core"] * O.rb(O.sigmoid(c["attn.gate"]))))
    out("affine8 (o_proj)", "attn.out", O.linear(c["attn.gated"], P.linear(a + ".o_proj")))


def moe_part(c, P, pre, out, experts=True, res_append=None):
    m = pre + "mlp"
    x = c["mlp_hc.block_in"]
    lg, p, ids, w = O.moe_route(x, P.linear(m + ".gate"))
    out("router logits (bf16 linear)", "moe.router_logits", lg)
    out("router softmax", "moe.router_probs", O.moe_route(x, P.linear(m + ".gate"))[1])
    _, _, ids2, w2 = route_from_probs(c["moe.router_probs"])
    r = C.check_topk("moe.topk_ids", ids2, c["moe.topk_ids"], c["moe.router_probs"])
    r.update(op="router top-10 (set, tie-aware)")
    res_append(r)
    order = [np.argsort(np.argsort(r)) for r in []]
    del order
    out("router renorm weights (sorted)", "moe.topk_weights#sorted", np.sort(w2, -1), np.sort(c["moe.topk_weights"], -1))
    if experts:
        y = O.moe_routed(x, c["moe.topk_ids"], c["moe.topk_weights"], P, c.layer, EXPERT_CACHE)
        out("EXL3 routed experts (dense f32)", "moe.routed_out", y.reshape(x.shape))
    r = O.shared_expert(x, P, m)
    for k in ("shared_gate_proj", "shared_up_proj"):
        out(f"affine8 ({k})", "moe." + k, r[k])
    out("swiglu (shared)", "moe.shared_act", O.rb(O.silu_bf(c["moe.shared_gate_proj"]) * c["moe.shared_up_proj"]))
    out("affine8 (shared down)", "moe.shared_out", O.linear(c["moe.shared_act"], P.linear(m + ".shared_expert.down_proj")))
    out("sigmoid(shared gate)", "moe.shared_scale", r["shared_scale"])
    out("moe combine", "moe.out", O.rb(c["moe.routed_out"] + O.rb(c["moe.shared_scale"] * c["moe.shared_out"])))


def route_from_probs(p, k=10):
    ids = np.argsort(-p, axis=-1, kind="stable")[..., :k]
    w = np.take_along_axis(p, ids, -1)
    return None, p, ids, O.rb(w / O.rb(w.sum(-1, keepdims=True)))


def ple_part(c, P, pre, out):
    pl = pre + "ple"
    e = pl + ".ple_embedding"
    hist = np.concatenate([c["ple.token_history_in"], np.array([c.m["token_ids"]], np.int64)], -1)
    ids = O.ngram_ids(hist, P.tensor(e + ".layer_multipliers"), P.tensor(e + ".ngram_heads_vocab_sizes"),
                      P.tensor(e + ".ngram_heads_offsets"), P.config["text_config"]["eos_token_id"])
    out("PLE n-gram ids", "ple.ngram_ids", ids)
    out("PLE token history", "ple.token_history_out", hist[:, -2:])
    out("PLE table gather", "ple.embeddings", NG[0].rows(c["ple.ngram_ids"]))
    r = O.ple_block(c["hc_in"], c["ple.embeddings"], c["ple.conv_state_in"], P, pl)
    for k in ("keys", "values", "queries", "gate", "gated_values", "normed", "conv_out", "conv_state_out", "out"):
        out(f"PLE {k} (chained in block)", "ple." + k, r[k])
    co, _ = O.causal_conv(c["ple.conv_state_in"], c["ple.normed"], P.tensor(pl + ".conv1d.weight"), dilation=3)
    out("PLE dilated conv + silu (isolated)", "ple.conv_out#iso", O.silu_bf(co), c["ple.conv_out"])
    out("PLE add", "hc_after_ple", O.rb(c["hc_in"] + c["ple.out"]))


NG = []


def chained_layer(c, P, pre):
    """Whole layer from hc_in using only op references (own routing). Returns hc_out."""
    h = c["hc_in"]
    S = c.m["rows"]
    pos = c.positions
    if c.m["has_ple"]:
        hist = np.concatenate([c["ple.token_history_in"], np.array([c.m["token_ids"]], np.int64)], -1)
        e = pre + "ple.ple_embedding"
        ids = O.ngram_ids(hist, P.tensor(e + ".layer_multipliers"), P.tensor(e + ".ngram_heads_vocab_sizes"),
                          P.tensor(e + ".ngram_heads_offsets"), P.config["text_config"]["eos_token_id"])
        h = O.rb(h + O.ple_block(h, NG[0].rows(ids), c["ple.conv_state_in"], P, pre + "ple")["out"])
    r = O.hc_read(h, P, pre + "attn_hyper_connection")
    x = r["block_in"]
    if c.m["layer_type"] == "linear_attention":
        g = pre + "linear_attn"
        qkv = O.linear(x, P.linear(g + ".in_proj_qkv"))
        z = O.linear(x, P.linear(g + ".in_proj_z")).reshape(1, S, 48, 128)
        a = O.linear(x, P.linear(g + ".in_proj_a"))
        b = O.linear(x, P.linear(g + ".in_proj_b"))
        co, _ = O.causal_conv(c["gdn.conv_state_in"], qkv, P.tensor(g + ".conv1d.weight"))
        q, k, v = O.gdn_split(O.silu_bf(co))
        q, k = O.gdn_qk_norm(q, k)
        gg, beta = O.gdn_gates(a, b, P.tensor(g + ".A_log"), P.tensor(g + ".dt_bias"))
        y, _ = O.gdn_recurrence(c["gdn.ssm_state_in"], q, k, v, gg, beta)
        br = O.linear(O.gated_rmsnorm(y, z, P.norm(g + ".norm.weight")).reshape(1, S, -1), P.linear(g + ".out_proj"))
    else:
        a = pre + "self_attn"
        qp = O.linear(x, P.linear(a + ".q_proj")).reshape(1, S, 24, 512)
        q = O.rope(O.rmsnorm(qp[..., :256], P.norm(a + ".q_norm.weight")).transpose(0, 2, 1, 3), pos)
        kk = O.rope(O.rmsnorm(O.linear(x, P.linear(a + ".k_proj")).reshape(1, S, 2, 256), P.norm(a + ".k_norm.weight"))
                    .transpose(0, 2, 1, 3), pos)
        vv = O.linear(x, P.linear(a + ".v_proj")).reshape(1, S, 2, 256).transpose(0, 2, 1, 3)
        K = np.concatenate([c["attn.k_cache_in"], kk], 2) if "attn.k_cache_in" in c else kk
        V = np.concatenate([c["attn.v_cache_in"], vv], 2) if "attn.v_cache_in" in c else vv
        allowed = None
        if "attn.selected_blocks" in c:
            T = c.m["cache_len_in"]
            qk = O.linear(x, P.linear(a + ".indexer.index_qk_proj")).reshape(1, S, 5, 128)
            iq = O.rope(O.rmsnorm(qk[:, :, :4], P.norm(a + ".indexer.q_layernorm.weight")).transpose(0, 2, 1, 3), pos)
            raw = np.concatenate([c["attn.index_keys_in"][0], qk[0, :, 4]], 0) if "attn.index_keys_in" in c else qk[0, :, 4]
            bk = O.qsa_block_keys(raw, P.norm(a + ".indexer.k_layernorm.weight"))
            _, allowed = O.qsa_select(O.qsa_scores(iq[0], bk), T)
        core = O.attention(q[0], K[0], V[0], pos, allowed)[None]
        br = O.linear(O.rb(core * O.rb(O.sigmoid(qp[..., 256:].reshape(1, S, -1)))), P.linear(a + ".o_proj"))
    h = O.hc_write(h, br, r["inject_w"])
    r = O.hc_read(h, P, pre + "mlp_hyper_connection")
    x = r["block_in"]
    m = pre + "mlp"
    _, _, ids, w = O.moe_route(x, P.linear(m + ".gate"))
    y = O.moe_routed(x, ids, w, P, c.layer, EXPERT_CACHE).reshape(x.shape)
    sh = O.shared_expert(x, P, m)
    br = O.rb(y + O.rb(sh["shared_scale"] * sh["shared_out"]))
    return O.hc_write(h, br, r["inject_w"]), ids


def run_case(name, P, experts=True, chained=True):
    c = golden.load(name)
    res = []

    def out(op, key, got, ref=None):
        gname = key.split("#")[0]
        if ref is None:
            ref = c[gname]
        r = C.check(gname, got, ref)
        r["op"] = op
        r["name"] = key
        res.append(r)

    if c.m["layer_type"] == "head":
        mx_ = "language_model.model.hyper_connection_mixer"
        r = O.hc_read(c["head.hc_in"], P, mx_, inject=False)
        out("final mixer rmsnorm grouped", "mixer.normed", r["normed"])
        out("final mixer (whole, from h)", "mixer.block_in", r["block_in"])
        out("affine8 g128 lm_head (isolated)", "head.logits", O.linear_big(c["mixer.block_in"], P, "language_model.lm_head"))
        lg = c["head.logits"].reshape(-1, c["head.logits"].shape[-1])
        mine = O.linear_big(r["block_in"], P, "language_model.lm_head").reshape(lg.shape)
        top = [C.check("head.top1", np.argmax(mine, -1), np.argmax(lg, -1))]
        top[0].update(op="logits argmax (chained)", name="head.argmax", kind="exact")
        res.extend(top)
        return res
    pre = c.prefix

    if "embed" in c:
        out("affine8 embedding rows", "embed", P.linear_rows("language_model.model.embed_tokens", c.m["token_ids"])[None])
        out("x4 tile", "hc_in", np.tile(c["embed"], (1, 1, 4)))
    if c.m["has_ple"]:
        ple_part(c, P, pre, out)
    h = c["hc_after_ple"] if c.m["has_ple"] else c["hc_in"]
    hc_part(c, P, pre, "attn_hyper_connection", h, out, "")
    if c.m["layer_type"] == "linear_attention":
        gdn_part(c, P, pre, out)
        br = c["gdn.out"]
    else:
        attn_part(c, P, pre, out)
        br = c["attn.out"]
    out("hc write (attn)", "attn_hc.out", O.hc_write(h, br, c["attn_hc.inject_w"]))
    hc_part(c, P, pre, "mlp_hyper_connection", c["attn_hc.out"], out, "mlp ")
    moe_part(c, P, pre, out, experts, res.append)
    out("hc write (mlp)", "hc_out", O.hc_write(c["attn_hc.out"], c["moe.out"], c["mlp_hc.inject_w"]))
    if chained and experts:
        hc, ids = chained_layer(c, P, pre)
        r = C.check("hc_out", hc, c["hc_out"], tol=2e-2)
        r.update(op="CHAINED whole layer from hc_in", name="hc_out#chained")
        res.append(r)
        r = C.check_topk("moe.topk_ids", ids, c["moe.topk_ids"], c["moe.router_probs"])
        r.update(op="CHAINED routing (own top-10)", name="moe.topk_ids#chained", ok=True)   # informative only
        res.append(r)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--no-experts", action="store_true")
    ap.add_argument("--summary-json")
    a = ap.parse_args()
    P = O.PackNP()
    NG.append(O.NgramTable(P))
    cases = a.cases or [x for x in golden.cases() if not x.endswith("prefill")]
    allres = {}
    ok = True
    for cn in cases:
        t0 = time.time()
        res = run_case(cn, P, experts=not a.no_experts)
        allres[cn] = res
        print(f"\n{cn} ({time.time() - t0:.1f} s)")
        print(f"  {'op':38s} {'tensor':28s} {'rel L2':>9s} {'max/max':>9s} {'cosine':>10s}  ok")
        for r in res:
            ok &= r["ok"]
            print(f"  {r['op']:38s} {r['name']:28s} {r['rel']:9.2e} {r['maxrel']:9.2e} {r['cos']:10.7f}  "
                  f"{'PASS' if r['ok'] else 'FAIL'} {r.get('note', '')}")
    # summary: worst rel per op over cases
    worst = {}
    for cn, res in allres.items():
        for r in res:
            k = r["op"]
            if k not in worst or r["rel"] > worst[k][0]:
                worst[k] = (r["rel"], r["cos"], cn, r["ok"])
    print("\nSUMMARY (worst over cases): op | rel L2 | cosine | case")
    for k, (rel, cos, cn, okk) in worst.items():
        print(f"  {k:38s} {rel:9.2e} {cos:10.7f} {cn} {'' if okk else 'FAIL'}")
    if a.summary_json:
        Path(a.summary_json).write_text(json.dumps({k: {"rel": v[0], "cos": v[1], "case": v[2], "ok": v[3]}
                                                    for k, v in worst.items()}, indent=1))
    print("ALL PASS" if ok else "SOME FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
