"""MTP-head goldens (lab-mtp): the Flash-Next MTP draft head alone, from the MLX reference on the same pack.

Reference = mlx-vlm 0.7.7 `speculative/drafters/qwen4_exp_mtp` (Qwen4ExpMTPDraftModel + AutoregressiveMTPDraftModel's
draft lifecycle) with the head's routed experts on the EXL3 kernels (fn_exl3, as lab-ref's gen_goldens.py), loaded
from `language_model.mtp.*` of the pack, plus the target's 8-bit embedding and lm_head (shared, not dedicated:
config mtp_use_dedicated_embeddings = False). Native rope (fn_exl3 convention).

The head (per row r, at query position r of the head's own KV cache):
  e   = embed(token)                                   (target embedding, 8-bit)
  pe  = fc_embedding(rmsnorm(e) * w_e)                 (8-bit g64 2560 -> 2560)
  hn  = rmsnorm over all 10240 of hidden * w_h         (ONE global norm over the 4 streams)
  ph_s = fc_hidden(hn_s)  for each of the 4 streams    (same 2560 -> 2560 weight)
  h_in = concat_s(pe + ph_s)                           -> one QSA attention + MoE decoder layer (its own HC mixers)
  out  = hc_out (pre-mixer streams: the next draft step's hidden) ; logits = lm_head(mixer(hc_out))
Inputs it needs: the target's PRE-mixer hidden streams [.., 10240] and the next token. Here (synthetic, as lab-ref's
HEAD goldens): layer 3's real hc_out on the real prompt (L03_prefill / L03_verify4 goldens) stands in for layer 47's.

Scenario (greedy, the drafter's own lifecycle):
  MTP_prefill  48 rows: tokens ids[1..48] (shifted; ids[48] = the target's bonus token), hidden H[0..47], pos 0..47
  MTP_draft1   1 row:   token = argmax of the prefill's last row (the seed draft), hidden = prefill hc_out[-1], pos 48
  MTP_draft2/3 1 row:   token = argmax of the previous draft, hidden = its hc_out, pos 49 / 50 (chained, depth 3)
  MTP_rows4    4 rows:  from the post-prefill cache, tokens ids[49..52] with hidden H[48..51] (an accept-style
               multi-row head forward, as accept_verified_tokens does with the verify rows' hidden)
Every case is run through an instrumented copy (gen_goldens' layer_fwd + the fuse and the mixer) and checked
bit-exact against the stock drafter `_forward_hidden` on a cache rebuilt from the saved state.

Output: tests/golden/data/MTP_<case>/ (same format as docs/reference.md section 2; extra tensors below).
Run (machine lock; ~4 GB):  tools/with_lock.sh "mtp goldens" HH:MM -- $PY tools/mtp_goldens.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refenv  # noqa: E402

refenv.add_paths(mlx=True)
import gen_goldens as GG  # noqa: E402
import fn_exl3 as F  # noqa: E402
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402
from mlx_vlm.models.qwen3_5.language import _create_qwen3_5_attention_mask  # noqa: E402
from mlx_vlm.speculative.drafters.qwen4_exp_mtp.config import Qwen4ExpMTPConfig  # noqa: E402
from mlx_vlm.speculative.drafters.qwen4_exp_mtp.qwen4_exp_mtp import Qwen4ExpMTPDraftModel  # noqa: E402

L = F.L
PRE = "language_model.mtp."
MTP_META = {
    "mtp.token_ids": ("head input tokens of this step [1, S] (row r: the token after the hidden's position)", []),
    "mtp.hidden_in": ("target pre-mixer hidden streams [1, S, 10240] (synthetic: layer 3 hc_out)", []),
    "mtp.embed": ("embed(token) [1, S, 2560] (target 8-bit embedding)", ["language_model.model.embed_tokens"]),
    "mtp.emb_normed": ("pre_fc_norm_embedding(embed) [1, S, 2560]", [PRE + "pre_fc_norm_embedding.weight"]),
    "mtp.fc_embedding": ("fc_embedding(emb_normed) [1, S, 2560]", [PRE + "fc_embedding"]),
    "mtp.hidden_normed": ("pre_fc_norm_hidden(hidden) [1, S, 10240]: ONE rmsnorm over all 4 streams",
                          [PRE + "pre_fc_norm_hidden.weight"]),
    "mtp.fc_hidden": ("fc_hidden per stream [1, S, 4, 2560]", [PRE + "fc_hidden"]),
    "mtp.argmax": ("argmax of head.logits per row [1, S] (the greedy draft)", []),
    "mixer.normed": ("MTP mixer hc_norm(hc_out)", [PRE + "hyper_connection_mixer.hc_norm.weight"]),
    "mixer.down": ("MTP mixer down", [PRE + "hyper_connection_mixer.input_mix_weight_down"]),
    "mixer.up": ("MTP mixer up", [PRE + "hyper_connection_mixer.input_mix_weight_up"]),
    "mixer.block_in": ("MTP mixer output = the lm_head input [1, S, 2560]", []),
    "head.logits": ("lm_head(mixer.block_in) [1, S, 248320] (target lm_head, 8-bit g128)", ["language_model.lm_head"]),
}


def build(pack):
    F._patch_classes(pack)
    cfg = F.text_config(pack)
    mtp = Qwen4ExpMTPDraftModel(Qwen4ExpMTPConfig(text_config=cfg))
    F._quantize_like_pack(mtp, pack, PRE)
    F.fit_exl3_banks(mtp.layers[0], pack, PRE + "layers.0.")
    F.load_module(mtp, pack, PRE)
    mtp.eval()
    mx.eval(mtp.parameters())

    class Emb(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
    emb = Emb()
    F._quantize_like_pack(emb, pack, "language_model.model.")
    F.load_module(emb.embed_tokens, pack, "language_model.model.embed_tokens.")
    mx.eval(emb.parameters())
    _, lm_head = GG.build_head(pack, cfg)
    pack.drop_files()
    return cfg, mtp, emb.embed_tokens, lm_head


def mtp_step(mtp, embed, lm_head, tokens, hidden, snap, start, rec):
    """Instrumented head step on a rebuilt cache, checked bit-exact against the stock drafter."""
    layer = mtp.layers[0]
    S = len(tokens)
    toks = mx.array([tokens], dtype=mx.int32)
    rec("mtp.token_ids", toks)
    rec("mtp.hidden_in", hidden)
    c1 = GG.restore(layer, snap) if snap is not None else GG.fresh_cache(layer)
    c2 = GG.restore(layer, snap) if snap is not None else GG.fresh_cache(layer)
    mtp._next_position = start
    pos = mtp._position_ids(length=S)
    e = rec("mtp.embed", embed(toks))
    pe = rec("mtp.fc_embedding", mtp.fc_embedding(rec("mtp.emb_normed", mtp.pre_fc_norm_embedding(e))))
    hn = rec("mtp.hidden_normed", mtp.pre_fc_norm_hidden(hidden))
    ph = rec("mtp.fc_hidden", mtp.fc_hidden(hn.reshape(*hidden.shape[:-1], 4, -1)))
    fused = (pe[..., None, :] + ph).reshape(hidden.shape)
    mask = _create_qwen3_5_attention_mask(fused, c1)
    out, path = GG.layer_fwd(layer, fused, toks, mask, c1, pos, rec)
    logits = GG.head_fwd(mtp.hyper_connection_mixer, lm_head, out, rec)
    rec("mtp.argmax", mx.argmax(logits, axis=-1).astype(mx.int32))
    mx.eval(out, *rec.items.values())
    # stock: the drafter's own fuse + layer + mixer (its _forward_hidden), on another rebuilt cache
    mtp._next_position = start
    lh, pre = mtp._forward_hidden(e, hidden, toks, [c2])
    mx.eval(lh, pre)
    s1, s2 = GG.snapshot(layer, c1), GG.snapshot(layer, c2)
    ok = GG.same(out, pre) and GG.same(lh, rec.items["mixer.block_in"]) and all(
        GG.same(s1[k], s2[k]) if k != "_ratio" else s1[k] == s2[k] for k in s1)
    assert ok, "instrumented MTP step != stock drafter"
    return out, s1, path, np.array(rec.items["mtp.argmax"])[0]


def save(outdir, name, rec, meta, pack, layer):
    d = outdir / name
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.npy"):
        f.unlink()
    tensors, total = {}, 0
    for n, a in rec.items.items():
        arr, dt = GG.np_of(a)
        np.save(d / f"{n}.npy", arr)
        total += arr.nbytes
        desc, w = MTP_META.get(n) or GG.META.get(n, ("", []))
        tensors[n] = {"file": f"{n}.npy", "shape": list(arr.shape), "saved_dtype": str(arr.dtype), "ref_dtype": dt,
                      "desc": desc, "pack_weights": [x.replace("{L}", PRE + "layers.0.") for x in w]}
    m = dict(meta)
    m.update({"case": name, "layer": "mtp", "layer_type": layer.layer_type, "has_ple": False, "pack": str(pack.path),
              "bytes": total, "tensors": tensors})
    (d / "manifest.json").write_text(json.dumps(m, indent=1))
    return total


def main():
    t0 = time.time()
    outdir = Path(refenv.GOLDEN)
    pack = F.Pack(refenv.PACK)
    cfg, mtp, embed, lm_head = build(pack)
    print(f"built MTP head + embed + lm_head in {time.time() - t0:.1f} s, active "
          f"{mx.get_active_memory() / 2**30:.1f} GiB", flush=True)
    gp = np.load(outdir / "L03_prefill/hc_out.npy")      # [1, 48, 10240]
    gv = np.load(outdir / "L03_verify4/hc_out.npy")      # [1, 4, 10240]
    H = mx.array(np.concatenate([gp, gv], axis=1)).astype(mx.bfloat16)
    T0 = gp.shape[1]
    ids = GG.tokens(T0 + 5, GG.PROMPT)
    m0 = json.loads((outdir / "L03_prefill/manifest.json").read_text())
    assert m0["token_ids"] == ids[:T0], "prompt tokens differ from the layer goldens"
    layer = mtp.layers[0]
    summary = []
    # prefill: shifted tokens, bonus = ids[T0]
    rec = GG.Rec()
    out, snap, path, am = mtp_step(mtp, embed, lm_head, ids[1:T0 + 1], H[:, :T0], None, 0, rec)
    meta = {"step": "prefill", "rows": T0, "position_start": 0, "cache_len_in": 0, "attn_path": path,
            "token_ids": ids[1:T0 + 1]}
    summary.append(("MTP_prefill", save(outdir, "MTP_prefill", rec, meta, pack, layer)))
    post_prefill = snap
    tok, hid, pos = int(am[-1]), out[:, -1:], T0
    drafts = []
    for d in (1, 2, 3):
        rec = GG.Rec()
        out, snap, path, am = mtp_step(mtp, embed, lm_head, [tok], hid, snap, pos, rec)
        meta = {"step": f"draft{d}", "rows": 1, "position_start": pos, "cache_len_in": pos, "attn_path": path,
                "token_ids": [tok], "draft_out": int(am[0])}
        summary.append((f"MTP_draft{d}", save(outdir, f"MTP_draft{d}", rec, meta, pack, layer)))
        drafts.append(tok)
        tok, hid, pos = int(am[0]), out, pos + 1
    rec = GG.Rec()
    _, _, path, am = mtp_step(mtp, embed, lm_head, ids[T0 + 1:T0 + 5], H[:, T0:T0 + 4], post_prefill, T0, rec)
    meta = {"step": "rows4", "rows": 4, "position_start": T0, "cache_len_in": T0, "attn_path": path,
            "token_ids": ids[T0 + 1:T0 + 5]}
    summary.append(("MTP_rows4", save(outdir, "MTP_rows4", rec, meta, pack, layer)))
    print(f"drafts (greedy chain from the prefill seed): {drafts + [tok]}; prompt continues {ids[T0 + 1:T0 + 5]}")
    print("cases:", ", ".join(f"{c} ({nb / 2**20:.1f} MiB)" for c, nb in summary),
          f"; peak {mx.get_peak_memory() / 2**30:.1f} GiB, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
