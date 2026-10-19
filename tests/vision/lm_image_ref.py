"""lab-vision whole-model reference on an image prompt (fn_exl3 = mlx-vlm qwen4_exp + EXL3 experts on the K5 pack, MLX
vision tower bf16): the reference's own image features, multimodal positions (get_rope_index) and masked scatter,
one prefill call, then greedy decode. Saves what tests/vision/lm_image_engine.py compares against.

  tools/with_lock.sh "vision LM ref (MLX whole model ~80 GB)" HH:MM -- \
      $PY -I -B tests/vision/lm_image_ref.py IMAGE "QUESTION" OUT_DIR [--pack K5] [--n 16]

OUT_DIR: tokens.npy (prompt tokens), keys.json, grid.npy, feats.npy (bf16 bits [tok][2560]), positions.npy
([3][n] reference), logits.npy (bf16 bits [1 + n][V]: the last prompt position, then each greedy decode position),
greedy.npy (the n greedy tokens), meta.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "tests/vision"))
import refenv  # noqa: E402

refenv.add_paths(mlx=True)
import fn_exl3 as F  # noqa: E402
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

import chat  # noqa: E402
import vision as V  # noqa: E402
from vision_ref import load_tower  # noqa: E402


def request(image, question, think=False):
    return {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}},
                                                      {"type": "text", "text": question}]}],
            "chat_template_kwargs": {"enable_thinking": think}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("question")
    ap.add_argument("out")
    ap.add_argument("--pack", default=refenv.ijah_paths.golden_pack())
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(Path(a.pack) / "tokenizer.json"))
    tpl = chat.Template(a.pack, tok)
    srcs = []
    text, _ = tpl.render(request(a.image, a.question), srcs)
    ids = tpl.encode(text)
    im = V.VisionImage(V.load_image(srcs[0])[0])
    keys, toks, grids = V.expand_prompt(ids, [im])
    tower = load_tower(Path(a.pack), False)
    feats, _ = tower(mx.array(im.pixel_values).astype(mx.bfloat16), mx.array([im.grid]))  # as get_input_embeddings
    mx.eval(feats)
    del tower
    np.save(out / "feats.npy", np.array(feats.astype(mx.bfloat16).view(mx.uint16)))
    t0 = time.time()
    model, _ = F.build_model(F.Pack(a.pack))
    print(f"model loaded in {time.time() - t0:.0f} s", flush=True)
    x = mx.array([toks])
    emb = model.model.embed_tokens(x)
    idx = np.nonzero(np.array(toks) == V.IMAGE_TOKEN)[0]
    emb[0, mx.array(idx)] = feats.astype(emb.dtype)
    pos, rd = model.get_rope_index(x, mx.array([im.grid]), None, None)
    cache = model.make_cache()
    t0 = time.time()
    lg = model(x, inputs_embeds=emb, cache=cache, position_ids=pos, rope_deltas=rd).logits[0, -1]
    mx.eval(lg)
    t_pre = time.time() - t0
    logits = [np.array(lg.astype(mx.bfloat16).view(mx.uint16))]
    greedy = []
    for _ in range(a.n):
        greedy.append(int(np.argmax(np.array(lg.astype(mx.float32)))))
        lg = model(mx.array([[greedy[-1]]]), cache=cache).logits[0, -1]
        mx.eval(lg)
        logits.append(np.array(lg.astype(mx.bfloat16).view(mx.uint16)))
    # control: the same prompt with 1-D (text) positions, teacher-forced on the same greedy tokens -> how much the
    # multimodal positions move the logits (the scale the engine's position handling is judged on)
    cache = model.make_cache()
    tp = mx.broadcast_to(mx.arange(len(toks))[None, None], (3, 1, len(toks)))
    lg = model(x, inputs_embeds=emb, cache=cache, position_ids=tp, rope_deltas=mx.zeros((1, 1), mx.int32)).logits[0, -1]
    ctl = [np.array(lg.astype(mx.bfloat16).view(mx.uint16))]
    for t in greedy[:-1]:
        lg = model(mx.array([[t]]), cache=cache).logits[0, -1]
        mx.eval(lg)
        ctl.append(np.array(lg.astype(mx.bfloat16).view(mx.uint16)))
    np.save(out / "logits_textpos.npy", np.stack(ctl))
    np.save(out / "tokens.npy", np.array(toks, np.int32))
    np.save(out / "grid.npy", np.array(im.grid, np.int64))
    np.save(out / "positions.npy", np.array(pos)[:, 0, :].astype(np.int32))
    np.save(out / "logits.npy", np.stack(logits))
    np.save(out / "greedy.npy", np.array(greedy, np.int32))
    meta = {"pack": a.pack, "image": a.image, "question": a.question, "prompt_tokens": len(toks),
            "image_tokens": im.n_tokens, "grid": list(im.grid), "rope_delta": int(np.array(rd).ravel()[0]),
            "prefill_s": t_pre, "completion": tok.decode(greedy, skip_special_tokens=False)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
