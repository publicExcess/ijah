"""Engine Flash-Next layer (models/flash_next, via tools/engine.py) vs the goldens, sub-block by sub-block (lab-ref).

    $PY tests/test_fn_layer.py [CASE ...=L00_decode1] [--f32] [--pack DIR]   (e.g. L00_verify4 L01_decode1 L03_verify4)
1. chain: set the golden cache state, one ne_fn_layer_step from hc_in, then every intermediate the engine exposes
   (ne_fn_layer_read) and the new state vs the golden tensor it corresponds to (MAP below).
2. stage: each sub-block alone (ne_fn_layer_run_stage) with its inputs overwritten by the golden's, so an error
   points at one sub-block instead of accumulating.
Exit 0 = all compared tensors within compare.py's tolerance for their kind.
The engine-name -> golden mapping follows models/flash_next/layer.h (lab-layer); update MAP when that changes.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import compare as Cm  # noqa: E402
import engine as E  # noqa: E402
import golden  # noqa: E402
import numpy as np  # noqa: E402
import opref as O  # noqa: E402

# engine read name -> (golden name, slice of the engine vector, transform of the golden)
MAP = {
    "attn_mix": ("attn_hc.mix", None),
    "attn_inj": ("attn_hc.inject_w", slice(0, 4)),
    "attn_in": ("attn_hc.block_in", None),
    "qkv": ("gdn.conv_out", None),
    "z": ("gdn.z", None),
    "gb:g": ("gdn.g", slice(0, 48)),
    "gb:beta": ("gdn.beta", slice(48, 96)),
    "core": ("gdn.gated_norm", None),
    "attn_out": ("gdn.out", None),
    "h_mid": ("attn_hc.out", None),
    "mlp_mix": ("mlp_hc.mix", None),
    "mlp_inj": ("mlp_hc.inject_w", slice(0, 4)),
    "mlp_in": ("mlp_hc.block_in", None),
    "logits": ("moe.router_logits", slice(0, 512)),
    "shared_gu:gate": ("moe.shared_gate_proj", slice(0, 640)),
    "shared_gu:up": ("moe.shared_up_proj", slice(640, 1280)),
    "shared_down": ("moe.shared_out", None),
    "mlp_out": ("moe.out", None),
}
# stage -> engine names it reads (inputs, set from goldens) and writes (outputs, compared)
STAGES = [
    (0, "AttnHC", ["h_in"], ["attn_mix", "attn_inj", "attn_in"]),
    (1, "GdnIn", ["attn_in", "conv_state"], ["qkv", "z", "gb:g", "gb:beta"]),
    (2, "GdnStep", ["qkv", "gb", "z", "ssm_state"], ["core"]),
    (3, "GdnOut", ["core", "h_in", "attn_inj"], ["attn_out", "h_mid"]),
    (4, "MlpHC", ["h_mid"], ["mlp_mix", "mlp_inj", "mlp_in"]),
    (5, "Moe", ["mlp_in", "h_mid", "mlp_inj"], ["logits", "route", "shared_gu:gate", "shared_gu:up", "shared_down",
                                                 "mlp_out", "h_out"]),
]


def eng_value(L, key):
    name, _, part = key.partition(":")
    v = L.read(name)
    if key in MAP and MAP[key][1] is not None:
        v = v[MAP[key][1]]
    return v


# attention layers (layer.h: GdnIn/GdnStep/GdnOut = attn_proj / attn_core+merge / o_proj + HC write)
MAP_ATTN = {
    "attn_mix": ("attn_hc.mix", None),
    "attn_inj": ("attn_hc.inject_w", slice(0, 4)),
    "attn_in": ("attn_hc.block_in", None),
    "attn_q": ("attn.q_proj", None),
    "attn_iqk": ("attn.idx_qk", None),
    "core": ("attn.gated", None),
    "attn_out": ("attn.out", None),
    "h_mid": ("attn_hc.out", None),
    "mlp_mix": ("mlp_hc.mix", None),
    "mlp_inj": ("mlp_hc.inject_w", slice(0, 4)),
    "mlp_in": ("mlp_hc.block_in", None),
    "logits": ("moe.router_logits", slice(0, 512)),
    "shared_down": ("moe.shared_out", None),
    "mlp_out": ("moe.out", None),
}
# engine vectors of these names have a per-row stride larger than the golden's row (only a prefix is compared)
ROW = {"attn_inj": 8, "mlp_inj": 8, "gb:g": 96, "gb:beta": 96, "logits": 513, "shared_gu:gate": 1280,
       "shared_gu:up": 1280}


def same_route_rows(L, c):
    """rows whose engine top-10 set equals the golden's (a row that differs only by an exact bf16 tie routes to a
    different expert: its MoE output legitimately differs and is left out of the MoE comparisons)"""
    R = c.m["rows"]
    ids = L.read("route").reshape(R, 20)[:, :10].copy().view(np.uint32).astype(np.int64)
    gi = c["moe.topk_ids"].reshape(R, 10)
    return [r for r in range(R) if set(ids[r].tolist()) == set(gi[r].tolist())]


MOE_ROUTED = ("mlp_out", "h_out")  # depend on which experts were chosen


def compare_key(L, c, key, scale=1.0, M=None, rows=None):
    M = M or MAP
    R = c.m["rows"]
    if key == "route":
        v = L.read("route").reshape(R, 20)
        ids = v[:, :10].copy().view(np.uint32).astype(np.int64)
        gi = c["moe.topk_ids"].reshape(R, 10)
        gw = c["moe.topk_weights"].reshape(R, 10)
        keep = same_route_rows(L, c)
        out = [Cm.check_topk("moe.topk_ids", ids, gi, c["moe.router_probs"].reshape(R, -1))]
        if keep:
            w = np.concatenate([sort_by(v[r, 10:20], ids[r]) for r in keep])
            wr = np.concatenate([sort_by(gw[r], gi[r]) for r in keep])
            out.append(dict(Cm.check("moe.topk_weights", w, wr, 5e-3 * scale, scale ** 2),
                            name=f"moe.topk_weights (by id, {len(keep)}/{R} rows)"))
        return out
    if key == "h_out":
        return [Cm.check("hc_out", L.read("h_out"), c["hc_out"].reshape(-1))]
    g, sl = M[key]
    v = L.read(key.partition(":")[0]).reshape(R, -1)
    ref = c[g].reshape(R, -1)
    if sl is not None:
        v = v[:, sl]
    sel = list(range(R)) if rows is None or key not in MOE_ROUTED else rows
    if not sel:
        return []
    r = Cm.check(g, v[sel].reshape(-1), ref[sel].reshape(-1), Cm.kind_of(g)[1] * scale, scale ** 2)
    r["name"] = f"{key} ~ {g}" + (f" ({len(sel)}/{R} rows: same experts)" if len(sel) < R else "")
    return [r]


def sort_by(vals, ids):
    o = np.argsort(np.asarray(ids))
    return np.asarray(vals, np.float32)[o]


def ple_inputs(eng, pack, L, c):
    """PLE layer: the gathered n-gram rows (C++ ids + gather, = the goldens' exactly) and the PLE conv state"""
    hist = [int(x) for x in c["ple.token_history_in"].reshape(-1)]
    rows = []
    for t in c.m["token_ids"]:
        rows.append(eng.ngram(pack, hist[-2], hist[-1], int(t))[1])
        hist.append(int(t))
    L.write("ple_emb", np.concatenate(rows))
    L.write("ple_conv_state", c["ple.conv_state_in"].reshape(-1))


def chain_any(eng, pack, c, scale):
    """chain mode for any case: GDN or attention, 1-4 rows, PLE"""
    attn = c.m["layer_type"] != "linear_attention"
    L = eng.fn_layer(pack, c.layer, c.m["rows"])
    if attn:
        T = c.m["cache_len_in"]
        L.set_kv(c["attn.k_cache_in"].reshape(2, T, 256), c["attn.v_cache_in"].reshape(2, T, 256),
                 c["attn.index_keys_in"].reshape(T, 128), T)
    else:
        L.set_state(c["gdn.conv_state_in"], c["gdn.ssm_state_in"])
    if c.m["has_ple"]:
        ple_inputs(eng, pack, L, c)
    h = L.step(c["hc_in"])
    keep = same_route_rows(L, c)
    R = c.m["rows"]
    out_scale = scale
    # compounding intermediates downstream of several sub-blocks get 1.5x the chain scale; a QSA-sparse case is
    # compared with the engine's dense attention (lab-attn's QSA path is not wired yet): 2x on everything after it
    mid_scale = scale * 1.5
    if c.m.get("attn_path") == "qsa" and os.environ.get("NE_FN_ATTN", "qsa") == "dense":
        out_scale, mid_scale = scale * 2, scale * 2
    hr = Cm.check("hc_out", h.reshape(R, -1)[keep].reshape(-1), c["hc_out"].reshape(R, -1)[keep].reshape(-1),
                  Cm.kind_of("hc_out")[1] * out_scale, out_scale ** 2) if keep else None
    res = [dict(hr, name=f"hc_out ({len(keep)}/{R} rows: same experts)")] if keep else []
    if c.m["has_ple"]:
        res.append(Cm.check("hc_after_ple", L.read("hc_after_ple"), c["hc_after_ple"].reshape(-1), 1e-2))
        res.append(Cm.check("ple.conv_state_out", L.read("ple_conv_state"), c["ple.conv_state_out"].reshape(-1)))
    M = MAP_ATTN if attn else MAP
    for k in M:
        res += compare_key(L, c, k, mid_scale, M, keep)
    rt = compare_key(L, c, "route", mid_scale)
    if (c.m.get("attn_path") == "qsa" and c.m["cache_len_in"] // 4 > 512 and not rt[0]["ok"]
            and os.environ.get("NE_FN_ATTN", "qsa") == "dense"):
        # sparse in the reference, dense in the engine: a different MoE input may pick a different expert
        rt[0] = dict(rt[0], ok=True, note=rt[0]["note"] + " (allowed: engine dense vs reference QSA-sparse)")
    res += rt
    if attn:
        T, S = c.m["cache_len_in"], c.m["rows"]
        k, v, ik = L.get_kv(T + S)
        res.append(Cm.check("attn.k_cache_out", k, c["attn.k_cache_out"].reshape(2, T + S, 256)))
        res.append(Cm.check("attn.v_cache_out", v, c["attn.v_cache_out"].reshape(2, T + S, 256)))
        res.append(Cm.check("attn.index_keys_out", ik, c["attn.index_keys_out"].reshape(T + S, 128)))
    else:
        conv, ssm = L.get_state()
        res.append(Cm.check("gdn.conv_state_out", conv, c["gdn.conv_state_out"].reshape(-1), 5e-3 * scale))
        res.append(Cm.check("gdn.ssm_state_out", ssm, c["gdn.ssm_state_out"].reshape(-1), 5e-3 * scale,
                            scale ** 2 * 10))
    path = c.m.get("attn_path", "")
    dense = os.environ.get("NE_FN_ATTN", "qsa") == "dense"
    note = f" (engine {'dense' if dense else 'QSA'}, reference QSA)" if path == "qsa" and attn else ""
    ok = Cm.report(res, f"{c.m['case']}: engine layer step from hc_in (chain, {c.m['rows']} rows){note}, "
                        f"gpu {L.gpu_ms:.3f} ms")
    L.free()
    return ok


def golden_input(L, c, name):
    """The engine buffer `name` filled from the golden (partial buffers keep the engine's own extra entries)."""
    if name == "h_in":
        return c["hc_in"].reshape(-1)
    if name == "conv_state":
        return c["gdn.conv_state_in"].reshape(-1)
    if name == "ssm_state":
        return c["gdn.ssm_state_in"].reshape(-1)
    if name == "gb":
        return np.concatenate([c["gdn.g"].reshape(-1), c["gdn.beta"].reshape(-1)])
    if name in ("attn_inj", "mlp_inj"):
        v = L.read(name)
        v[:4] = c[MAP[name][0]].reshape(-1)
        return v
    return c[MAP[name][0]].reshape(-1)


def kernel_checks(L, c):
    """3. per kernel, isolated through the engine's OWN inputs: run the chain step, read each stage's input buffers
    back, recompute that stage with the f32 op references (bf=False, same pack weights), compare its outputs.
    Independent of run_stage isolation and of upstream error; this is rung 1 of the ladder (kernel vs numpy)."""
    P = O.PackNP()
    pre = c.prefix
    L.set_state(c["gdn.conv_state_in"], c["gdn.ssm_state_in"])
    h_in = c["hc_in"].reshape(1, 1, -1).astype(np.float32)
    h_out = L.step(h_in)
    r = lambda n: L.read(n)  # noqa: E731
    res = []

    def chk(label, gname, got, ref, tol=None):
        x = Cm.check(gname, got, ref, tol)
        x["name"] = label
        res.append(x)

    # AttnHC
    hc = O.hc_read(h_in, P, pre + "attn_hyper_connection", bf=False)
    chk("AttnHC mix", "attn_hc.mix", r("attn_mix"), hc["mix"])
    chk("AttnHC inject_w", "attn_hc.inject_w", r("attn_inj")[:4], hc["inject_w"])
    chk("AttnHC block_in", "attn_hc.block_in", r("attn_in"), hc["block_in"])
    # GdnIn (engine attn_in + golden conv state)
    g = pre + "linear_attn"
    x = r("attn_in").reshape(1, 1, -1)
    qkv = O.linear(x, P.linear(g + ".in_proj_qkv"), bf=False)
    co, _ = O.causal_conv(c["gdn.conv_state_in"], qkv, P.tensor(g + ".conv1d.weight"), bf=False)
    chk("GdnIn qkv (conv+silu)", "gdn.conv_out", r("qkv"), O.silu(co))
    chk("GdnIn z", "gdn.z", r("z"), O.linear(x, P.linear(g + ".in_proj_z"), bf=False))
    a = O.linear(x, P.linear(g + ".in_proj_a"), bf=False)
    b = O.linear(x, P.linear(g + ".in_proj_b"), bf=False)
    gg, beta = O.gdn_gates(a, b, P.tensor(g + ".A_log"), P.tensor(g + ".dt_bias"), bf=False)
    chk("GdnIn g", "gdn.g", r("gb")[:48], gg)
    chk("GdnIn beta", "gdn.beta", r("gb")[48:96], beta)
    # GdnStep (engine qkv, gb, z + golden state in)
    q, k, v = O.gdn_split(r("qkv").reshape(1, 1, -1))
    q, k = O.gdn_qk_norm(q, k, bf=False)   # mode 3 runs on a layer created with rounding OFF (f32 kernels)
    gb = r("gb")
    y, st = O.gdn_recurrence(c["gdn.ssm_state_in"], q, k, v, gb[:48].reshape(1, 1, 48), gb[48:96].reshape(1, 1, 48),
                             bf=False)
    core = O.gated_rmsnorm(y, r("z").reshape(1, 1, 48, 128), P.norm(g + ".norm.weight"), bf=False)
    chk("GdnStep core (gated norm)", "gdn.gated_norm", r("core"), core)
    chk("GdnStep ssm state", "gdn.ssm_state_out", L.get_state()[1], st, 1e-5)
    # GdnOut (engine core, attn_inj)
    br = O.linear(r("core").reshape(1, 1, -1), P.linear(g + ".out_proj"), bf=False)
    chk("GdnOut out_proj", "gdn.out", r("attn_out"), br)
    chk("GdnOut h_mid", "attn_hc.out", r("h_mid"), O.hc_write(h_in, r("attn_out"), r("attn_inj")[:4], bf=False))
    # MlpHC (engine h_mid)
    hm = r("h_mid").reshape(1, 1, -1)
    hc = O.hc_read(hm, P, pre + "mlp_hyper_connection", bf=False)
    chk("MlpHC mix", "mlp_hc.mix", r("mlp_mix"), hc["mix"])
    chk("MlpHC inject_w", "mlp_hc.inject_w", r("mlp_inj")[:4], hc["inject_w"])
    chk("MlpHC block_in", "mlp_hc.block_in", r("mlp_in"), hc["block_in"])
    # Moe (engine mlp_in, route)
    m = pre + "mlp"
    xm = r("mlp_in").reshape(1, -1)
    lg = O.linear(xm, P.linear(m + ".gate"), bf=False)
    lgs = O.linear(xm, P.linear(m + ".shared_expert_gate"), bf=False)
    eng_lg = r("logits")
    chk("Moe router logits", "moe.router_logits", eng_lg[:512], lg[0])
    chk("Moe shared gate logit", "moe.inject_logits", eng_lg[512:513], lgs[0])
    rt = r("route")
    ids = rt[:10].view(np.uint32).astype(np.int64)
    p = np.exp(lg[0] - lg[0].max())
    p /= p.sum()
    ref_ids = np.argsort(-p, kind="stable")[:10]
    res.append(dict(Cm.check_topk("moe.topk_ids", ids[None], ref_ids[None], p[None]), name="Moe top-10 (own logits)"))
    w = p[ids] / p[ids].sum()
    chk("Moe top-10 weights", "moe.topk_weights", rt[10:20], w)
    sgp = O.linear(xm, P.linear(m + ".shared_expert.gate_proj"), bf=False)
    sup = O.linear(xm, P.linear(m + ".shared_expert.up_proj"), bf=False)
    chk("Moe shared gate_proj", "moe.shared_gate_proj", r("shared_gu")[:640], sgp[0])
    chk("Moe shared up_proj", "moe.shared_up_proj", r("shared_gu")[640:], sup[0])
    gu = r("experts_gu").reshape(10, 2, 640)
    y = np.zeros(2560, np.float64)
    for kk, e in enumerate(ids):
        parts = {pj: P.expert_parts(c.layer, pj, int(e)) for pj in ("gate", "up", "down")}
        for j, pj in enumerate(("gate", "up")):
            inner, suh, _ = parts[pj]
            chk(f"Moe EXL3 {pj} inner, slot {kk} (e{e})", "moe.routed_out", gu[kk, j], O.exl3_inner_out(xm[0], inner, suh))
        gate_f = O.exl3_finish(gu[kk, 0], parts["gate"][2])
        up_f = O.exl3_finish(gu[kk, 1], parts["up"][2])
        mid = O.silu(gate_f) * up_f
        inner, suh, svh = parts["down"]
        y += rt[10 + kk] * O.exl3_finish(O.exl3_inner_out(mid, inner, suh), svh)
    # keep only a summary of the 20 per-slot rows
    slots = [x for x in res if x["name"].startswith("Moe EXL3")]
    res = [x for x in res if not x["name"].startswith("Moe EXL3")]
    worst = max(slots, key=lambda x: x["rel"])
    res.append(dict(worst, name=f"Moe EXL3 gate/up GEMV (worst of 20: {worst['name'][9:]})"))
    so = O.linear(O.silu(sgp) * sup, P.linear(m + ".shared_expert.down_proj"), bf=False)
    chk("Moe shared down", "moe.shared_out", r("shared_down"), so[0])
    full = y + O.sigmoid(lgs[0, 0]) * r("shared_down")
    chk("Moe out (routed + shared, from engine gu)", "moe.out", r("mlp_out"), full)
    chk("Moe h_out", "hc_out", h_out, O.hc_write(hm, r("mlp_out").reshape(1, 1, -1), r("mlp_inj")[:4], bf=False))
    return Cm.report(res, "3. kernels in isolation: engine's own inputs -> f32 op reference (same pack weights)")


ROUND = True


def main():
    global ROUND
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", default=["L00_decode1"],
                    help="cases; stage + kernel checks run for 1-row GDN cases without PLE, chain for every case")
    ap.add_argument("--pack", default=None)
    ap.add_argument("--f32", action="store_true", help="engine without the reference rounding points")
    ap.add_argument("--chain-scale", type=float, default=2.0,
                    help="tolerance multiplier in chain mode (errors of earlier sub-blocks propagate); the cosine "
                         "slack is scaled by its square")
    a = ap.parse_args()
    ROUND = not a.f32
    eng = E.Engine()
    eng.set_ref_rounding(ROUND)
    pack = eng.pack_open(a.pack)
    ok = True
    for case in a.cases:
        c = golden.load(case)
        simple = c.m["rows"] == 1 and c.m["layer_type"] == "linear_attention" and not c.m["has_ple"]
        ok &= chain_any(eng, pack, c, a.chain_scale)
        if simple:
            ok &= stages_and_kernels(eng, pack, c)
    pack.close()
    print("test_fn_layer:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def stages_and_kernels(eng, pack, c):
    L = eng.fn_layer(pack, c.layer)
    ok = True
    LK = None

    # 2. stages in isolation
    for sid, sname, ins, outs in STAGES:
        L.set_state(c["gdn.conv_state_in"], c["gdn.ssm_state_in"])
        L.step(c["hc_in"])                       # fill every buffer with the engine's own values first
        L.set_state(c["gdn.conv_state_in"], c["gdn.ssm_state_in"])
        for n in ins:
            L.write(n, golden_input(L, c, n))
        L.run_stage(sid)
        res = []
        for n in ins:   # the stage must not have modified its golden inputs
            if n not in ("attn_inj", "mlp_inj", "gb", "conv_state", "ssm_state"):
                try:
                    kept = np.array_equal(L.read(n), golden_input(L, c, n))
                except KeyError:
                    continue
                res.append({"name": f"input {n} unchanged", "kind": "exact", "tol": 0.0, "rel": 0.0 if kept else 1.0,
                            "maxrel": 0.0, "cos": 1.0, "ok": kept,
                            "note": "" if kept else "run_stage overwrote its input: not isolated"})
        for k in outs:
            res += compare_key(L, c, k)
        ok &= Cm.report(res, f"stage {sid} {sname} (inputs from the golden: {', '.join(ins)})")
    L.free()
    # 3. kernels vs f32 op references: on a layer created with the reference rounding OFF (with it on, the GDN q/k
    # norm etc. are bf16 ops by design and an f32 reference at 1e-5 cannot match)
    eng.set_ref_rounding(False)
    LK = eng.fn_layer(pack, c.layer)
    eng.set_ref_rounding(ROUND)
    ok &= kernel_checks(LK, c)
    LK.free()
    return ok


if __name__ == "__main__":
    main()
