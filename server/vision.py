"""Image input for the Ijah server (lab-vision): image loading (OpenAI image_url data: URLs, file paths),
the Qwen3-VL image preprocessing, the multimodal (MRoPE) positions of a prompt, and the prefix keys of image tokens.

Pure CPU (numpy + PIL, requirements.txt). The preprocessing is a line-by-line port of mlx-vlm's numpy
Qwen3VLImageProcessor (mlx_vlm/models/qwen3_vl/processing_qwen3_vl.py, mlx-vlm 0.7.7; MIT, Copyright (c) 2025 Prince Canuma; THIRD_PARTY_NOTICES.md): smart resize to
multiples of 32, PIL bicubic resize of the uint8 RGB image, x / 255, (x - 0.5) / 0.5, the frame duplicated along T
(temporal_patch_size 2), patches in merge-block order. positions() ports Qwen3_5LanguageModel.get_rope_index (same
file family, qwen3_5/language.py). Checks: tests/vision/test_vision_ref.py (bit-exact pixel values and positions vs
mlx-vlm).
"""
import base64
import hashlib
import io
import math
import os
from pathlib import Path

import numpy as np

PATCH, TPS, MERGE = 16, 2, 2
FACTOR = PATCH * MERGE
IMAGE_TOKEN, VIDEO_TOKEN, VISION_START, VISION_END = 248056, 248057, 248053, 248054
# Qwen3.8's processor bounds (preprocessor_config.json of the source checkpoint: shortest_edge 65536, longest_edge
# 16777216; the EXL3 packs do not ship the file). The engine caps the area like Sushi (ENGINE_MAX_PIXELS 1536^2:
# ~9.2K patches, ~2.3K tokens): NE_VISION_MAX_PIXELS overrides.
MIN_PIXELS = 65536
CKPT_MAX_PIXELS = 16777216
MAX_PIXELS = int(os.environ.get("NE_VISION_MAX_PIXELS", 1536 * 1536))
MEAN = STD = (0.5, 0.5, 0.5)
# image prefix keys: (1 << 62) | (40-bit digest << 20) | token index (< 2^20 tokens per image)
KEY_BASE = 1 << 62


class ImageError(ValueError):
    """-> HTTP 400"""


def _pil():
    try:
        from PIL import Image
    except ImportError as e:  # pragma: no cover
        raise ImageError(f"image input needs Pillow: {e}")
    return Image


def local_roots():
    """directories local image paths may be read from: $NE_VISION_ROOTS (os.pathsep separated; empty = no local files
    at all), default the home directory. Read at each call (tests set it)."""
    env = os.environ.get("NE_VISION_ROOTS")
    if env is None:
        return [Path.home().resolve()]
    return [Path(r).expanduser().resolve() for r in env.split(os.pathsep) if r.strip()]


def max_input_pixels():
    """the largest image (width x height) decoded; the processor resizes to MAX_PIXELS afterwards, but a decompression
    bomb is decoded at full size first. $NE_VISION_MAX_INPUT_PIXELS, default 40 megapixels."""
    return int(os.environ.get("NE_VISION_MAX_INPUT_PIXELS", 40_000_000))


def load_image(src, max_bytes=64 << 20):
    """data: URL (base64), file:// URL or local path -> (PIL RGB image, raw bytes). Remote URLs are refused (the
    server is loopback-only and does not fetch). Local paths must resolve (symlinks followed) to a file under one of
    local_roots(); an image over max_input_pixels() is refused before it is decoded."""
    Image = _pil()
    if not isinstance(src, str) or not src:
        raise ImageError("image_url.url must be a non-empty string")
    if src.startswith("data:"):
        head, sep, data = src.partition(",")
        if not sep or ";base64" not in head:
            raise ImageError("data: URLs must be base64 encoded")
        try:
            raw = base64.b64decode(data, validate=False)
        except (ValueError, TypeError) as e:
            raise ImageError(f"bad base64 image data: {e}")
    elif src.startswith(("http://", "https://")):
        raise ImageError("remote image URLs are not fetched: send a data: URL or a local file path")
    else:
        path = Path(src[len("file://"):] if src.startswith("file://") else src).expanduser()
        roots = local_roots()
        try:
            real = path.resolve()
        except (OSError, RuntimeError):
            raise ImageError(f"image file not found: {src}")
        if not any(real == r or r in real.parents for r in roots):
            raise ImageError("local image paths are only read under " + (", ".join(str(r) for r in roots) or "(no directory: "
                             "NE_VISION_ROOTS is empty)") + "; send a data: URL instead")
        path = real
        if not path.is_file():
            raise ImageError(f"image file not found: {src}")
        if path.stat().st_size > max_bytes:
            raise ImageError(f"image file larger than {max_bytes >> 20} MB")
        raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise ImageError(f"image larger than {max_bytes >> 20} MB")
    try:
        img = Image.open(io.BytesIO(raw))   # (the header only)
        cap = max_input_pixels()
        if img.size[0] * img.size[1] > cap:
            raise ImageError(f"image is {img.size[0]}x{img.size[1]} = {img.size[0] * img.size[1] / 1e6:.0f} megapixels; "
                             f"the limit is {cap / 1e6:.0f} (NE_VISION_MAX_INPUT_PIXELS)")
        img.load()
    except ImageError:
        raise
    except Exception as e:  # PIL raises many types
        raise ImageError(f"cannot decode image: {e}")
    return img, raw


def smart_resize(height, width, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """= mlx-vlm _smart_resize_image (HF qwen2_vl smart_resize; Python round = half to even)"""
    if max(height, width) / min(height, width) > 200:
        raise ImageError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def to_chw_uint8(img):
    """= mlx-vlm _to_numpy_image for a PIL image: RGB, (C, H, W) uint8"""
    arr = np.array(img.convert("RGB"))
    return np.transpose(arr, (2, 0, 1))


def preprocess(img, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """PIL image -> (pixel_values f32 [gh*gw][3*2*16*16] in the reference's (C, T, P, P) patch order, (1, gh, gw))"""
    Image = _pil()
    chw = to_chw_uint8(img)
    C, H, W = chw.shape
    rh, rw = smart_resize(H, W, min_pixels=min_pixels, max_pixels=max_pixels)
    pil = Image.fromarray(np.transpose(chw, (1, 2, 0)))
    pil = pil.resize((rw, rh), resample=Image.BICUBIC)
    frame = np.transpose(np.array(pil), (2, 0, 1))
    x = frame.astype(np.float32)
    x = x * (1 / 255.0)
    mean = np.array(MEAN, dtype=np.float32)[:, None, None]
    std = np.array(STD, dtype=np.float32)[:, None, None]
    x = (x - mean) / std
    patches = np.repeat(x[None, None, ...], TPS, axis=1)
    gh, gw = rh // PATCH, rw // PATCH
    patches = patches.reshape(1, 1, TPS, C, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    patches = patches.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    return patches.reshape(gh * gw, C * TPS * PATCH * PATCH), (1, gh, gw)


def engine_patches(pixel_values):
    """reference (C, T, P, P) patch rows -> the engine's (T, P, P, C) order (= the MLX Conv3d weight layout), f32"""
    n = pixel_values.shape[0]
    return np.ascontiguousarray(
        pixel_values.reshape(n, 3, TPS, PATCH, PATCH).transpose(0, 2, 3, 4, 1).reshape(n, -1), dtype=np.float32)


def n_tokens(grid):
    t, h, w = grid
    return t * h * w // (MERGE * MERGE)


def positions(tokens, grids):
    """MRoPE positions of a prompt (= get_rope_index for one unpadded row, images only): tokens = ids (image tokens
    = IMAGE_TOKEN, one run of n_tokens(grid) per image, in order), grids = [(t, h, w)] of the images.
    -> (pos int32 [n][3] (t, h, w), delta) with positions of later rows r = r + delta on all three axes."""
    n = len(tokens)
    if not grids:
        return np.repeat(np.arange(n, dtype=np.int32)[:, None], 3, axis=1), 0
    out = np.zeros((n, 3), dtype=np.int64)
    toks = list(tokens)
    st, nxt = 0, 0
    for (t, h, w) in grids:
        try:
            ed = toks.index(IMAGE_TOKEN, st)
        except ValueError:
            raise ImageError("fewer image token runs than images")
        lt, lh, lw = t, h // MERGE, w // MERGE
        tl = ed - st
        out[st:ed] = (nxt + np.arange(tl))[:, None]
        ti, hi, wi = np.meshgrid(np.arange(lt), np.arange(lh), np.arange(lw), indexing="ij")
        blk = np.stack([ti.ravel(), hi.ravel(), wi.ravel()], axis=1) + tl + nxt
        k = lt * lh * lw
        if ed + k > n or any(x != IMAGE_TOKEN for x in toks[ed:ed + k]):
            raise ImageError("image token run shorter than the image's token count")
        out[ed:ed + k] = blk
        nxt = int(blk.max()) + 1
        st = ed + k
    if st < n:
        out[st:] = (nxt + np.arange(n - st))[:, None]
    delta = int(out.max()) + 1 - n
    return out.astype(np.int32), delta


class VisionImage:
    """one preprocessed image: digest (sha256 of the pixel values + grid), grid, engine patch rows"""

    def __init__(self, img, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
        pv, grid = preprocess(img, min_pixels, max_pixels)
        self.pixel_values = pv
        self.grid = grid
        self.n_tokens = n_tokens(grid)
        h = hashlib.sha256(pv.tobytes())
        h.update(np.asarray(grid, np.int64).tobytes())
        self.digest = h.hexdigest()
        self.d40 = int(self.digest[:10], 16)

    def keys(self):
        """the prefix keys of this image's tokens (distinct per image content and token index)"""
        base = KEY_BASE | (self.d40 << 20)
        return [base | j for j in range(self.n_tokens)]


def is_key(k):
    return k >= KEY_BASE


def key_index(k):
    """-> (40-bit digest, token index)"""
    return (k >> 20) & ((1 << 40) - 1), k & ((1 << 20) - 1)


def expand_prompt(ids, images):
    """token ids with one IMAGE_TOKEN per image (the template's placeholder) -> (key list, token list, grids): each
    placeholder becomes the image's n_tokens keys / IMAGE_TOKEN ids"""
    keys, toks, grids = [], [], []
    it = iter(images)
    prev = None
    for i, t in enumerate(ids):
        if t == IMAGE_TOKEN:
            # only the template's placeholder counts: <|vision_start|><|image_pad|><|vision_end|> (literal text in a
            # message is not tokenized as these ids: chat.Template.render_ids)
            if prev != VISION_START or i + 1 >= len(ids) or ids[i + 1] != VISION_END:
                raise ImageError("an image placeholder outside <|vision_start|> ... <|vision_end|>")
            try:
                im = next(it)
            except StopIteration:
                raise ImageError("more image placeholders than images")
            keys += im.keys()
            toks += [IMAGE_TOKEN] * im.n_tokens
            grids.append(im.grid)
        else:
            keys.append(t)
            toks.append(t)
        prev = t
    if next(it, None) is not None:
        raise ImageError("more images than image placeholders")
    return keys, toks, grids


def keys_to_tokens(keys):
    return [IMAGE_TOKEN if k >= KEY_BASE else k for k in keys]
