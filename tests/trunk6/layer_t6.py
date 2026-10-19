"""lab-t6wire: the layer graph on a T6 pack (EXL3 trunk), every t6 stage vs the f32 op references ON THE SAME T6 WEIGHTS.

    PY=.venv/bin/python
    LOCK_OWNER=lab-t6wire tools/with_lock.sh "t6 layer check" HH:MM -- \\
        $PY -I -B tests/trunk6/layer_t6.py --pack ~/.cache/ijah/packs/Qwen3.8-Flash-Next-EXL3-Ijah [CASE ...] [--round]

Golden layer cases can't validate a T6 layer (its weights differ from the reference's 8-bit trunk); they only supply
realistic inputs (hc_in, states, KV caches, token history). Method = test_fn_layer.py's kernel checks generalised to
GDN / attention / PLE layers and 1..4 rows: one engine step from the golden hc_in, then every stage that
models/flash_next/layer_t6.mm records is recomputed from the ENGINE's own input buffers (read back) with
tools/opref.py (PackNP decodes the EXL3 Linears, tools/convert_trunk_enc.decode) and compared. Engine with the
reference rounding OFF (f32) against bf=False references by default: the expected error is the t6 kernels' f16 input
preparation (~3e-4 rel, docs/trunk6.md 2); --round runs the rounding-ON engine against the bf16 references (bf16
flips: ~1e-2). Stages left to the q8 graph's kernels (GDN step, attention core, routed experts) are not re-checked.
Rows 1 / 2 -> t6 SIMD path, 4 -> NAX path (t6::plan).
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import refenv  # noqa: E402
import compare as Cm  # noqa: E402
import engine as E  # noqa: E402
import golden  # noqa: E402
import numpy as np  # noqa: E402
import opref as O  # noqa: E402

TOL = {False: 2e-3, True: 2.5e-2}


def rel(a, b):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def run(eng, pack, P, case, rows_override, bf, experts):
    c = golden.load(case)
    R = c.m["rows"]
    attn = c.m["layer_type"] != "linear_attention"
    pre = c.prefix
    L = eng.fn_layer(pack, c.layer, R)
    if attn:
        T = c.m["cache_len_in"]
        L.set_kv(c["attn.k_cache_in"].reshape(2, T, 256), c["attn.v_cache_in"].reshape(2, T, 256),
                 c["attn.index_keys_in"].reshape(T, 128), T)
    else:
        L.set_state(c["gdn.conv_state_in"], c["gdn.ssm_state_in"])
    if c.m["has_ple"]:
        hist = [int(x) for x in c["ple.token_history_in"].reshape(-1)]
        emb = []
        for t in c.m["token_ids"]:
            emb.append(eng.ngram(pack, hist[-2], hist[-1], int(t))[1])
            hist.append(int(t))
        emb = np.concatenate(emb).reshape(1, R, 2560)
        L.write("ple_emb", emb)
        L.write("ple_conv_state", c["ple.conv_state_in"].reshape(-1))
    h0 = c["hc_in"].reshape(1, R, -1).astype(np.float32)
    h_out = L.step(h0).reshape(1, R, -1)
    rd = lambda n, w=None: L.read(n).reshape(1, R, -1) if w is None else L.read(n).reshape(1, R, w)  # noqa: E731
    res = []

    def chk(label, got, ref):
        r = rel(got, ref)
        res.append((label, r, r <= TOL[bf]))

    lin = lambda x, p: O.linear(x, P.linear(p), bf=bf)  # noqa: E731
    h = h0
    if c.m["has_ple"]:
        pl = pre + "ple"
        kv = rd("ple_kv")
        chk("PLE key_proj (K8)", kv[..., :10240], lin(emb, pl + ".key_proj"))
        chk("PLE value_proj (K8)", kv[..., 10240:], lin(emb, pl + ".value_proj"))
        h = rd("hc_after_ple")
    # attention HC read (t6 down K6 / up K8 + epilogues)
    hc = O.hc_read(h, P, pre + "attn_hyper_connection", bf=bf)
    chk("AttnHC mix (down K6 + silu)", rd("attn_mix"), hc["mix"])
    chk("AttnHC inject_w", rd("attn_inj", 8)[..., :4], hc["inject_w"])
    chk("AttnHC 1/rms", rd("attn_inj", 8)[..., 4:], 1 / np.sqrt(np.mean(h.reshape(1, R, 4, 2560) ** 2, -1) + 1e-6))
    chk("AttnHC block_in (up K8 + mix)", rd("attn_in"), hc["block_in"])
    x = rd("attn_in")
    if not attn:
        g = pre + "linear_attn"
        qkv = lin(x, g + ".in_proj_qkv")
        co, st = O.causal_conv(c["gdn.conv_state_in"], qkv, P.tensor(g + ".conv1d.weight"), bf=bf)
        chk("GdnIn qkv (K6 + conv + silu)", rd("qkv"), O.silu_bf(co) if bf else O.silu(co))
        chk("GdnIn conv state", L.get_state()[0], st)
        chk("GdnIn z (K6)", rd("z"), lin(x, g + ".in_proj_z"))
        gg, beta = O.gdn_gates(lin(x, g + ".in_proj_a"), lin(x, g + ".in_proj_b"), P.tensor(g + ".A_log"),
                               P.tensor(g + ".dt_bias"), bf=bf)
        chk("GdnIn g (a K6)", rd("gb", 96)[..., :48], gg)
        chk("GdnIn beta (b K6)", rd("gb", 96)[..., 48:], beta)
        op = g + ".out_proj"
    else:
        a = pre + "self_attn"
        chk("Attn q_proj (K6)", rd("attn_q"), O.linear(x, P.linear(a + ".q_proj"), bf=False))
        kvv = rd("attn_kv", 1024)
        chk("Attn k_proj (K6)", kvv[..., :512], O.linear(x, P.linear(a + ".k_proj"), bf=False))
        chk("Attn v_proj (K6)", kvv[..., 512:], O.linear(x, P.linear(a + ".v_proj"), bf=False))
        chk("Attn index_qk (K6)", rd("attn_iqk"), O.linear(x, P.linear(a + ".indexer.index_qk_proj"), bf=False))
        op = a + ".o_proj"
    br = lin(rd("core"), op)
    chk("Out proj (K6)", rd("attn_out"), br)
    chk("Out HC write h_mid", rd("h_mid"), O.hc_write(h, rd("attn_out"), rd("attn_inj", 8)[..., :4], bf=bf))
    hm = rd("h_mid")
    hc = O.hc_read(hm, P, pre + "mlp_hyper_connection", bf=bf)
    chk("MlpHC mix", rd("mlp_mix"), hc["mix"])
    chk("MlpHC inject_w", rd("mlp_inj", 8)[..., :4], hc["inject_w"])
    chk("MlpHC block_in", rd("mlp_in"), hc["block_in"])
    xm = rd("mlp_in")
    m = pre + "mlp"
    lg = O.linear(xm, P.linear(m + ".gate"), bf=bf)
    chk("Moe router logits (bf16, unchanged)", rd("logits", 513)[..., :512], lg)
    sh = O.shared_expert(xm, P, m, bf=bf)
    gu = rd("shared_gu", 1280)
    chk("Moe shared gate_proj (K6)", gu[..., :640], sh["shared_gate_proj"])
    chk("Moe shared up_proj (K6)", gu[..., 640:], sh["shared_up_proj"])
    act = O.rb((O.silu_bf(gu[..., :640]) if bf else O.silu(gu[..., :640])) * gu[..., 640:], bf)
    chk("Moe shared down_proj (K6, engine act)", rd("shared_down"), lin(act, m + ".shared_expert.down_proj"))
    if experts:  # whole MoE output from the engine's route (routed experts decoded in numpy) + gated shared
        rt = rd("route", 20)[0]
        ids = rt[:, :10].copy().view(np.uint32).astype(np.int64)
        w = rt[:, 10:20]
        y = O.moe_routed(xm[0], ids, w, P, c.layer, {}, bf=bf)
        sg = O.rb(O.sigmoid(rd("logits", 513)[0, :, 512:513]), bf)
        full = O.rb(y + O.rb(sg * rd("shared_down")[0], bf), bf)
        chk("Moe out (routed + gated shared)", rd("mlp_out")[0], full)
        chk("h_out (HC write)", h_out, O.hc_write(hm, rd("mlp_out"), rd("mlp_inj", 8)[..., :4], bf=bf))
    L.free()
    ok = all(r[2] for r in res)
    print(f"\n{case} (layer {c.layer}, {R} rows, {'attention' if attn else 'GDN'}{', PLE' if c.m['has_ple'] else ''}, "
          f"rounding {'ON vs bf16 refs' if bf else 'OFF vs f32 refs'}; tol {TOL[bf]:.0e})")
    for label, r, k in res:
        print(f"  {label:40s} rel L2 {r:9.2e}  {'PASS' if k else 'FAIL'}")
    return ok, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", default=["L00_decode1", "L00_verify4", "L01_decode1", "L01_verify4",
                                                 "L03_decode1", "L03_verify4"])
    ap.add_argument("--pack", default=refenv.ijah_paths.pack())
    ap.add_argument("--round", action="store_true", help="engine rounding ON vs the bf16 op references")
    ap.add_argument("--no-experts", action="store_true")
    a = ap.parse_args()
    os.environ["NE_PACK"] = a.pack
    eng = E.Engine()
    eng.set_ref_rounding(a.round)
    pack = eng.pack_open(a.pack)
    P = O.PackNP(a.pack)
    ok = True
    worst = {}
    for case in a.cases:
        k, res = run(eng, pack, P, case, None, a.round, not a.no_experts)
        ok &= k
        for label, r, _ in res:
            worst[label] = max(worst.get(label, 0.0), r)
    print("\nworst over cases:")
    for label, r in worst.items():
        print(f"  {label:40s} {r:9.2e}")
    pack.close()
    print("layer_t6:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
