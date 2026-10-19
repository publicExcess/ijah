"""lab-vision CPU unit tests: OpenAI image parts -> template placeholders -> expanded prompt keys, positions, and the
session's prefix reuse with images (mock backend). No engine, no GPU.

    $PY tests/vision/test_units.py        ($PY = .venv/bin/python -I -B)
"""
import base64
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "tools"))
import refenv  # noqa: E402
import ijah_paths  # noqa: E402

refenv.add_paths()
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

import backend as B  # noqa: E402
import chat  # noqa: E402
import vision as V  # noqa: E402

PACK = ijah_paths.pack_or_skip("tests/vision/test_units.py", need=("tokenizer.json", "chat_template.jinja"))
TOK = Tokenizer.from_file(str(Path(PACK) / "tokenizer.json"))
FAILS = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        FAILS.append(what)


def data_url(img):
    b = io.BytesIO()
    img.save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def prompt_for(tpl, req):
    srcs = []
    text, _ = tpl.render(req, srcs)
    ids = tpl.encode(text)
    ims = [V.VisionImage(V.load_image(s)[0]) for s in srcs]
    keys, toks, grids = V.expand_prompt(ids, ims)
    return text, ids, keys, toks, grids, ims


def main():
    tpl = chat.Template(PACK, TOK)
    rng = np.random.default_rng(0)
    red = Image.fromarray(np.full((300, 400, 3), (200, 30, 30), np.uint8))
    noise = Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))
    tmp = Path("/tmp") if not Path(sys.argv[-1]).is_dir() else Path(sys.argv[-1])
    fpath = tmp / "ne_vision_unit.png"
    noise.save(fpath)
    os.environ["NE_VISION_ROOTS"] = os.pathsep.join([str(Path.home()), str(tmp)])   # local paths: allowlisted directories
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Describe "},
                                         {"type": "image_url", "image_url": {"url": data_url(red)}},
                                         {"type": "text", "text": " and "},
                                         {"type": "image_url", "image_url": {"url": str(fpath)}}]}]
    text, ids, keys, toks, grids, ims = prompt_for(tpl, {"messages": msgs})
    check(text.count("<|vision_start|><|image_pad|><|vision_end|>") == 2, "template renders two image placeholders")
    check("Describe <|vision_start|>" in text and "<|vision_end|> and <|vision_start|>" in text,
          "text and images keep their order")
    check(ids.count(V.IMAGE_TOKEN) == 2, "one <|image_pad|> token per image before expansion")
    n_img = sum(im.n_tokens for im in ims)
    check(len(keys) == len(ids) - 2 + n_img and sum(V.is_key(k) for k in keys) == n_img,
          f"expanded prompt: {n_img} image tokens ({ims[0].grid}, {ims[1].grid})")
    check(ims[0].grid == (1, 18, 24), f"smart resize 400x300 -> grid {ims[0].grid} (min_pixels 65536)")
    check(V.keys_to_tokens(keys) == toks, "keys map back to tokens")
    check(ims[0].digest != ims[1].digest and len(set(keys)) == len(set(ids)) - 1 + n_img,
          "image keys are distinct per image and token")
    pos, delta = V.positions(toks, grids)
    i0 = toks.index(V.IMAGE_TOKEN)
    check(pos.shape == (len(toks), 3) and (pos[:i0] == np.arange(i0)[:, None]).all(), "text before the image: 1-D positions")
    g = grids[0]
    blk = pos[i0:i0 + V.n_tokens(g)]
    check(blk[:, 0].max() == i0 and blk[:, 1].max() == i0 + g[1] // 2 - 1 and blk[:, 2].max() == i0 + g[2] // 2 - 1,
          "image block: t / h / w offsets from the image start")
    check(delta == int(pos.max()) + 1 - len(toks) and delta < 0, f"rope delta {delta}")
    # refusals
    try:
        tpl.render({"messages": msgs})
        check(False, "image parts refused without an image list")
    except chat.RequestError:
        check(True, "image parts refused without an image list")
    try:
        V.load_image("https://example.com/x.png")
        check(False, "remote URLs refused")
    except V.ImageError:
        check(True, "remote URLs refused")
    try:
        tpl.render({"messages": [{"role": "system", "content": [{"type": "image_url", "image_url": {"url": "x"}}]},
                                 {"role": "user", "content": "hi"}]}, [])
        check(False, "system images refused")
    except chat.RequestError:
        check(True, "system images refused")
    # text-only requests render exactly as before (content parts joined)
    t1, _ = tpl.render({"messages": [{"role": "user", "content": [{"type": "text", "text": "a"}, "b"]}]}, [])
    t2, _ = tpl.render({"messages": [{"role": "user", "content": "ab"}]})
    check(t1 == t2, "text-only content parts unchanged")
    # session reuse: same image -> the prefix is reused; a different image of the same size -> not past it
    mb = B.MockBackend(TOK)
    s = B.Session(mb, 32768, mtp_policy="never", log=lambda *a: None)
    mb.prepare(keys, {im.d40: im for im in ims})
    s.generate(keys, 8, (0, 1, 1, 0), False, lambda t: None, lambda: False, tpl.im_end)
    msgs2 = msgs + [{"role": "assistant", "content": "Hello from the mock engine."},
                    {"role": "user", "content": "more IMAGETEST"}]
    _, _, keys2, _, _, ims2 = prompt_for(tpl, {"messages": msgs2})
    k, src = s.plan(keys2)
    check(k > i0 + n_img and src is not None, f"second turn reuses {k} tokens incl. both images")
    other = Image.fromarray(np.full((300, 400, 3), (30, 200, 30), np.uint8))
    msgs3 = [dict(msgs[0], content=[msgs[0]["content"][0], {"type": "image_url", "image_url": {"url": data_url(other)}}]
                  + msgs[0]["content"][2:])]
    _, _, keys3, _, _, _ = prompt_for(tpl, {"messages": msgs3})
    k3, _ = s.plan(keys3)
    check(k3 <= i0, f"a different image of the same size is not reused (reuse {k3} <= {i0})")
    out = []
    st = s.generate(keys2, 64, (0, 1, 1, 0), False, out.extend, lambda: False, tpl.im_end)
    check(f"I see {n_img} image tokens" in TOK.decode(out), "mock sees the image tokens")
    check(st["cached_tokens"] == k, "cached tokens reported")
    # another conversation (a different image) in between: the first one is parked (prefix cache) and comes back
    s.generate(keys3, 8, (0, 1, 1, 0), False, lambda t: None, lambda: False, tpl.im_end)
    k4, src4 = s.plan(keys2 + [V.IMAGE_TOKEN - 1])
    check(k4 > i0 + n_img, f"after another image conversation the first is reused from the cache ({k4} tokens, {src4})")
    print("all passed" if not FAILS else f"{len(FAILS)} FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
