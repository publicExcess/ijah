"""Engine backends + the prefix-reusing, multi-slot session of the Ijah OpenAI server (lab-server,
lab-srvcache).

Backend primitives (RealBackend over tools/engine.py's ctypes binding, MockBackend for CPU-only tests), one backend
per slot (= one model instance: own KV buffer, recurrent state, sampler; the pack's weights shared):
    reset(), feed(tokens) -> sample after the last (state advanced by all of them), step(token) -> next sample,
    state_save() -> (buf, meta) | None / state_load(buf, meta) (the recurrent state, ~113 MB), kv_save(a, b) ->
    (buf, cap) / kv_load(a, b, buf, cap) (KV rows), set_sampler(...), mtp(prompt, ...) (MTP token loop; always starts
    from an empty state and leaves the state unusable), T (tokens in the caches).

Session: the slots + the PrefixCache (server/prefix_cache.py). Each slot has a resident conversation (its KV rows'
tokens + checkpoints) and `live` (the recurrent state equals all of its tokens). A request reuses the longest of: a
slot's live state, or any conversation's checkpoint whose tokens are a prefix of the prompt (resident anywhere, parked
in RAM or on disk: its KV rows are copied into the slot); GDN states cannot be truncated, so only checkpoint
positions are reusable. run() is a generator (one yield per prefill piece / decode token): the worker interleaves the
requests of different slots token by token.
"""
import ctypes as C
import json
import os
import sys
import threading
import traceback
import zlib
from collections import OrderedDict
import time

import numpy as np

from prefix_cache import Conv, PrefixCache, Snap, lcp
import vision as V

# chunked prefill pieces: 4096-token chunks (= kChunk, model.h) (+1 for the head step ne_fn_model_prefill runs on each
# piece's last token), then the rest as ONE piece (lab-ttft, docs/server.md 3e): every piece is a whole pass over the
# weights (~105 ms at 16-64 rows, the trunk's dense W' decode + every expert the rows touch), so the old binary split
# (2049 ... 17: 150 tokens = 129 + 17 + 4) cost 2-5 passes per turn; lab-prefill records one chunk length once
# (~10-30 ms, LRU of NE_PF_MAX_RECS). Below 17 tokens the engine's own step path runs. NE_SRV_PIECES=binary: the old split.
PIECES = [4097] + [(1 << j) + 1 for j in range(11, 3, -1)]
# ne_fn_mtp_feed chunks all n tokens of a piece (no head step): 4096 = kChunk (then the rest, < 16 tokens: steps)
MTP_PIECES = [4096] + [1 << j for j in range(11, 3, -1)]
BINARY_PIECES = os.environ.get("NE_SRV_PIECES", "") == "binary"
# lab-pfpath: NE_SRV_TRACE=1 logs a timestamped line at each step of a request's path (HTTP parse, template, queue,
# restore, every prefill piece, checkpoint, first token, first SSE write) to stderr; no behaviour change
TRACE = os.environ.get("NE_SRV_TRACE", "") == "1"
# lab-pfpath: pass the next piece's tokens to the engine (its n-gram rows read in the current piece's GPU shadow);
# NE_SRV_AHEAD=0: off (A/B)
AHEAD = os.environ.get("NE_SRV_AHEAD", "1") != "0"
# lab-pfpath: a request's first segment starts with its short piece (pieces(rest_first)); NE_SRV_REST_FIRST=0: last
REST_FIRST = os.environ.get("NE_SRV_REST_FIRST", "1") != "0"
# lab-snap: shared-prefix snapshots (docs/server.md 3f). NE_SHARED_SNAPS=0: off (= the earlier behaviour);
# NE_SHARED_MIN: a capture must save at least this many tokens over the best stored checkpoint; NE_SHARED_ALIGN /
# NE_SHARED_MARGIN: a learned boundary = (common prefix - margin) rounded down to a multiple of align; NE_SHARED_MAX_POS:
# no shared prefix longer than this (KV rows ~28 KB per token); NE_SHARED_RAM / NE_SHARED_DISK: snapshots kept per tier
SHARED = os.environ.get("NE_SHARED_SNAPS", "1") != "0"
SHARED_MIN = int(os.environ.get("NE_SHARED_MIN", "512"))
SHARED_ALIGN = max(1, int(os.environ.get("NE_SHARED_ALIGN", "64")))
SHARED_MARGIN = int(os.environ.get("NE_SHARED_MARGIN", "8"))
SHARED_MAX_POS = int(os.environ.get("NE_SHARED_MAX_POS", "32768"))
SHARED_RAM = int(os.environ.get("NE_SHARED_RAM", "8"))
SHARED_DISK = int(os.environ.get("NE_SHARED_DISK", "32"))


def trace(msg):
    if TRACE:
        print(f"trace {time.perf_counter():.4f} [{threading.current_thread().name}] {msg}", file=sys.stderr, flush=True)


def pieces(n, cap=None, table=PIECES, rest_first=False):
    """prefill pieces of n tokens: chunks of the largest table size (<= cap: another request is decoding, shorter
    stalls), then the rest in one piece (BINARY_PIECES: binary pieces of the table's sizes). rest_first (lab-pfpath:
    a request's first piece): the rest (>= 16 tokens) goes first -- the first chunk's n-gram rows are read before
    the GPU starts (~50 ms per 1K tokens of new text), every later chunk's during the previous one -- the same
    number of pieces"""
    ps = [p for p in table if cap is None or p <= cap] or table[-1:]
    out = []
    while n >= ps[0]:
        out.append(ps[0])
        n -= ps[0]
    if BINARY_PIECES:
        for p in ps[1:]:
            if n >= p:
                out.append(p)
                n -= p
    if n:
        out.append(n)
        if rest_first and n >= 16 and len(out) > 1 and not BINARY_PIECES:
            out = [n] + out[:-1]
    return out


class Rates:
    """online throughput estimates (EMA) for the MTP / step path choice"""

    def __init__(self):
        self.v = {"pf": 1600.0, "pf_mtp": 1000.0, "dec": 68.0, "dec_mtp": 95.0, "out": 1500.0}

    def get(self, k):
        return self.v[k]

    def obs(self, k, x, w=0.3):
        if x and x > 0 and np.isfinite(x):
            self.v[k] = (1 - w) * self.v[k] + w * x


# ------------------------------------------------------------------------------------------------- backends
class RealBackend:
    STOPS = (248046, 248044)  # <|im_end|>, <|endoftext|>
    VOCAB = 248320            # lm_head rows = the sampler's mask is (VOCAB + 31) // 32 words

    def __init__(self, root, pack_dir, ctx, mtp_depth=2, lib=None, shim=None, log=print, share=None, engine=None,
                 state_from=None):
        """share: another RealBackend whose engine / pack this one reuses (a further slot: own model instance =
        own KV / GDN state / scratch, the pack's weights shared). engine: another RealBackend whose engine opens this
        backend's pack (a further model in the same process: tensors in files both packs list -- same inode -- are
        the same buffers, core/pack.mm; own trunk + KV + scratch + MTP head state). state_from: another backend
        (another model's slot) whose KV + recurrent state buffers this model uses (ne_fn_model_create_shared; the two
        are used one at a time: Session hands the buffers over, Phys)"""
        import sys
        sys.path.insert(0, str(root / "tools"))
        os.environ.setdefault("NE_FN_ROUND", "f32")  # generation default (docs/forward.md section 2)
        if lib:
            os.environ["NE_LIB"] = str(lib)
        import engine as E
        self.E = E
        self.log = log
        self.vroot = share.vroot if share is not None else self  # lab-vision: one vision tower + feature cache
        if share is not None:
            self.eng, self.pack = share.eng, share.pack
        elif engine is not None:
            self.eng = engine.eng
            self.pack = self.eng.pack_open(pack_dir)
        else:
            self.eng = E.Engine(lib)
            missing = [s for s in ("ne_fn_model_create", "ne_fn_model_step", "ne_fn_model_prefill")
                       if s in self.eng.missing]
            if missing:
                raise E.EngineError(f"libengine lacks {missing}")
            self.pack = self.eng.pack_open(pack_dir)
        t0 = time.time()
        if state_from is not None:
            f = self.eng.fn("ne_fn_model_create_shared", C.c_void_p, [C.c_void_p, C.c_int, C.c_void_p])
            ptr = f(C.c_void_p(self.pack.ptr), ctx, C.c_void_p(state_from.m.ptr))
            if not ptr:
                raise E.EngineError(f"ne_fn_model_create_shared: {self.eng.last_error()}")
            self.m = E.FnModel.__new__(E.FnModel)
            self.m.eng, self.m.pack, self.m.ptr, self.m.tmax = self.eng, self.pack, ptr, ctx
        else:
            self.m = self.eng.fn_model(self.pack, ctx)
        self.ctx = ctx
        self.load_s = time.time() - t0
        self._nx, self._lp, self._am, self._ms = C.c_int(-1), C.c_float(0), C.c_int(-1), C.c_double(0)
        self.mtp_ptr = None
        self._masked = False
        self._extra = None   # lab-sampler: the running request's extra sampling parameters (set_extra)
        self._seen = None
        self.mtp_depth = mtp_depth
        if mtp_depth > 0:
            L = self.eng.lib
            L.ne_fn_mtp_create.restype = C.c_void_p
            L.ne_fn_mtp_create.argtypes = [C.c_void_p, C.c_int, C.c_int]
            L.ne_fn_mtp_generate.restype = C.c_int
            L.ne_fn_mtp_generate.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_uint64,
                                             C.c_void_p, C.c_void_p, C.c_void_p]
            self.mtp_ptr = L.ne_fn_mtp_create(C.c_void_p(self.m.ptr), max(4, mtp_depth), 4)
            if not self.mtp_ptr:
                raise E.EngineError(f"ne_fn_mtp_create: {self.eng.last_error()}")
        # MTP from the current state (lab-mtpcont, docs/mtp.md 10.9): feeds / steps keep the MTP head in step
        self.mtp_cont = bool(self.mtp_ptr) and self.eng.has("ne_fn_mtp_generate_from")
        self.mtp_abortable = bool(self.mtp_ptr) and self.eng.has("ne_fn_mtp_generate_cb")
        vp, ip = C.c_void_p, C.c_int
        if self.mtp_cont:
            self._CB = C.CFUNCTYPE(C.c_int, C.c_void_p, C.POINTER(C.c_int), C.c_int)
            self._f = {k: self.eng.fn(k, r, a) for k, (r, a) in {
                "ne_fn_mtp_feed": (ip, [vp, vp, ip]),
                **({"ne_fn_mtp_feed_ahead": (ip, [vp, vp, ip, vp, ip])} if self.eng.has("ne_fn_mtp_feed_ahead") else {}),
                "ne_fn_mtp_step": (ip, [vp, ip, C.POINTER(ip), vp]),
                "ne_fn_mtp_head_ok": (ip, [vp]),
                "ne_fn_mtp_generate_from": (ip, [vp, vp, ip, ip, ip, ip, C.c_uint64, vp, vp, vp, self._CB, vp]),
                # lab-draft: the copy rule's token history after a state import (docs/mtp.md 11)
                **({"ne_fn_mtp_context_len": (ip, [vp]), "ne_fn_mtp_set_context": (ip, [vp, vp, ip])}
                   if self.eng.has("ne_fn_mtp_set_context") else {}),
            }.items()}
        self.shim = None
        # lab-pfpath: the idle keep-warm (ping(); docs/prefill.md "server path")
        self._ping = self.eng.fn("ne_fn_gpu_ping", ip, [vp, vp]) if self.eng.has("ne_fn_gpu_ping") else None
        self._mtouch = self.eng.fn("ne_fn_mtp_touch", ip, [vp]) if self.mtp_ptr and self.eng.has("ne_fn_mtp_touch") \
            else None
        # the MTP head's KV rows + bookkeeping travel with the conversation (lab-srvcache, mtp_loop.mm)
        self.head_io = self.mtp_cont and all(self.eng.has(f) for f in (
            "ne_fn_mtp_kv_bytes", "ne_fn_mtp_kv_io", "ne_fn_mtp_state_bytes", "ne_fn_mtp_state_io"))
        if self.head_io:
            self._h = {k: self.eng.fn(k, r, a) for k, (r, a) in {
                "ne_fn_mtp_kv_bytes": (C.c_int64, [vp, ip]),
                "ne_fn_mtp_kv_io": (ip, [vp, ip, ip, ip, vp, ip]),
                "ne_fn_mtp_state_bytes": (C.c_int64, [vp]),
                "ne_fn_mtp_state_io": (ip, [vp, ip, vp, C.c_int64, vp]),
            }.items()}
        self.kvio = all(self.eng.has(f) for f in ("ne_fn_kv_bytes", "ne_fn_kv_io", "ne_fn_state_bytes",
                                                   "ne_fn_state_io"))
        if self.kvio:  # lab-srvcache: models/flash_next/kvio.mm (state + KV rows <-> host buffers)
            L = self.eng.lib
            L.ne_fn_kv_bytes.restype = C.c_int64
            L.ne_fn_kv_bytes.argtypes = [C.c_void_p, C.c_int]
            L.ne_fn_kv_io.restype = C.c_int
            L.ne_fn_kv_io.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_int]
            L.ne_fn_state_bytes.restype = C.c_int64
            L.ne_fn_state_bytes.argtypes = [C.c_void_p]
            L.ne_fn_state_io.restype = C.c_int
            L.ne_fn_state_io.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_int64, C.c_void_p]
            self.snap_bytes = int(self._h["ne_fn_mtp_state_bytes"](C.c_void_p(self.mtp_ptr)) if self.head_io else
                                  L.ne_fn_state_bytes(C.c_void_p(self.m.ptr)))
            self.kv_row_bytes = int(L.ne_fn_kv_bytes(C.c_void_p(self.m.ptr), 4096)) / 4096
        else:
            log("library without ne_fn_kv_io / ne_fn_state_io (models/flash_next/kvio.mm): prefix cache = live "
                "state only")
        if self.mtp_cont:
            log(f"MTP continues from the current state; the head {'travels with' if self.head_io else 'is NOT in'} "
                "parked conversations")
        self.reset()

    def share_prefill(self, src):
        """this model instance's chunked prefill uses src's chunk buffers (~3.5 GB at a 262K window; the engine thread
        runs one prefill at a time): ne_fn_model_share_prefill. -> True when shared"""
        if not self.eng.has("ne_fn_model_share_prefill"):
            return False
        f = self.eng.fn("ne_fn_model_share_prefill", C.c_int, [C.c_void_p, C.c_void_p])
        rc = f(C.c_void_p(self.m.ptr), C.c_void_p(src.m.ptr))
        if rc < 0:
            raise self.E.EngineError(f"ne_fn_model_share_prefill: {self.eng.last_error()}")
        return rc == 0

    @property
    def T(self):
        return self.m.info()["T"]

    def cache_layout(self):
        """what the bytes of a checkpoint / KV segment look like (part of the SSD tier's key)"""
        if not self.kvio:
            return {"kind": "native", "kvio": False}
        L = self.eng.lib
        d = {"kind": "native", "snap_bytes": self.snap_bytes, "meta": 6, "head_io": self.head_io,
             "kv_bytes_4096": int(L.ne_fn_kv_bytes(C.c_void_p(self.m.ptr), 4096))}
        if self.head_io:
            d["head_kv_bytes_4096"] = int(self._h["ne_fn_mtp_kv_bytes"](C.c_void_p(self.mtp_ptr), 4096))
        return d

    def reset(self):
        self.m.reset()

    def set_sampler(self, temp, top_p, top_k, seed):
        self.m.set_sampler(temp, top_p, top_k, seed)

    # -- lab-vision: image features (vision tower, LRU by image digest) and multimodal positions
    def _vision_fns(self):
        if getattr(self, "_vfn", None) is None:
            L = self.eng.lib
            sm = self.eng.fn("ne_fn_model_set_mrope", C.c_int, [C.c_void_p, C.c_void_p, C.c_int, C.c_int])
            si = self.eng.fn("ne_fn_model_set_image_rows", C.c_int, [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p])
            self._vfn = (sm, si)
            self.mm = False
            R = self.vroot
            if not hasattr(R, "vis_cache"):  # shared by the slots
                R.vis_cache = OrderedDict()  # d40 -> (grid, bf16 u16 [n][2560])
                R.vis = None
        return self._vfn

    def prepare(self, prompt, images):
        """before a request: the features of its images (encoded once per image content) and the positions of the
        whole prompt (keys: vision.VisionImage.keys for image tokens)"""
        sm, _ = self._vision_fns()
        R = self.vroot
        grids = []
        # (lab-ttft: a text prompt skips the per-token scan, ~5 ms per 160K tokens on the engine thread)
        for i, k in enumerate(prompt if prompt and max(prompt) >= V.KEY_BASE else ()):
            if V.is_key(k):
                d, j = V.key_index(k)
                if j == 0:
                    if d not in R.vis_cache:
                        im = images.get(d)
                        if im is None:
                            raise self.E.EngineError("image features missing for an image in the prompt")
                        if R.vis is None:
                            t0 = time.time()
                            R.vis = self.eng.vision(self.pack)
                            self.log(f"vision tower loaded ({time.time() - t0:.1f} s)")
                        t0 = time.time()
                        f = R.vis.encode(V.engine_patches(im.pixel_values), [im.grid])
                        self.log(f"image {im.grid[1]}x{im.grid[2]} patches -> {im.n_tokens} tokens: tower "
                                 f"{R.vis.gpu_ms:.0f} ms GPU, {(time.time() - t0) * 1e3:.0f} ms wall")
                        R.vis_cache[d] = (im.grid, f)
                        while sum(v[1].shape[0] for v in R.vis_cache.values()) > 65536 and len(R.vis_cache) > 1:
                            R.vis_cache.popitem(last=False)
                    R.vis_cache.move_to_end(d)
                    grids.append(R.vis_cache[d][0])
        if grids:
            pos, delta = V.positions(V.keys_to_tokens(prompt), grids)
            pos = np.ascontiguousarray(pos, np.int32)
            rc = sm(C.c_void_p(self.m.ptr), pos.ctypes.data, len(prompt), delta)
            self.mm = True
        else:
            rc = sm(C.c_void_p(self.m.ptr), None, 0, 0)
            if rc == 0 and self.mm:  # lab-mtpimg: the last image feed's rows (the next feed of a text-only request
                rc = self._vfn[1](C.c_void_p(self.m.ptr), None, 0, None)  # would not clear them: mm is off)
            self.mm = False
        if rc != 0:
            raise self.E.EngineError(f"set_mrope: {self.eng.last_error()}")

    def _image_rows(self, toks):
        """the image rows of a feed at the current position -> engine (k = 0 clears)"""
        _, si = self._vision_fns()
        R = self.vroot
        T0 = self.T
        rows, feats = [], []
        for i, k in enumerate(toks):
            if V.is_key(k):
                d, j = V.key_index(k)
                rows.append(T0 + i)
                feats.append(R.vis_cache[d][1][j])
        if rows:
            r = np.ascontiguousarray(rows, np.int32)
            f = np.ascontiguousarray(np.stack(feats), np.uint16)
            rc = si(C.c_void_p(self.m.ptr), r.ctypes.data, len(rows), f.ctypes.data)
        else:
            rc = si(C.c_void_p(self.m.ptr), None, 0, None)
        if rc != 0:
            raise self.E.EngineError(f"set_image_rows: {self.eng.last_error()}")

    @property
    def can_mask(self):
        return self.eng.has("ne_fn_model_set_mask")

    def set_mask(self, bits):
        """allowed-token bits for the next samples (grammar-constrained requests); None = no mask. Only calls the
        library when the state changes, so unconstrained requests never touch it."""
        if bits is None and not self._masked:
            return
        self.m.set_mask(bits)
        self._masked = bits is not None

    # -- lab-sampler: logit_bias, penalties, min_p, logprobs, seeded RNG (docs/server.md section 11)
    @property
    def can_extra(self):
        return all(self.eng.has(f) for f in ("ne_fn_model_set_sampler_ex", "ne_fn_model_set_logit_bias",
                                             "ne_fn_model_set_seen", "ne_fn_model_set_rng_step", "ne_fn_model_top"))

    def set_extra(self, ex, prompt):
        """the request's extra sampling parameters (sampling.Extras.engine_extra) for the following samples; cleared
        by clear_extra() at the request's end. prompt: its tokens (the repetition penalty covers them)."""
        self._extra = ex
        self._seen = None
        if ex["repetition"] != 1.0:
            self._seen = np.unique(np.asarray([t for t in prompt if not V.is_key(t)], np.int32))
        self.m.set_logit_bias(ex["bias"])
        self.arm()

    def arm(self):
        """zero the generated-token counts, mark the prompt as seen and restart the RNG counter: the next sample is
        this request's first (the discarded samples of a plain prefill's non-final pieces must not count)"""
        ex = self._extra
        if ex is None:
            return
        self.m.set_sampler_ex(ex["min_p"], ex["presence"], ex["frequency"], ex["repetition"], ex["logprob_k"])
        if self._seen is not None:
            self.m.set_seen(self._seen)
        if ex["rng0"]:
            self.m.set_rng_step(0)

    def clear_extra(self):
        if self._extra is None:
            return
        self._extra = self._seen = None
        self.m.set_sampler_ex(0.0, 0.0, 0.0, 1.0, 0)
        self.m.set_logit_bias(None)

    def last_top(self):
        """the last sample's candidates (ids, log-probabilities), needs logprob_k > 0"""
        return self.m.top(64)

    def feed(self, toks, sample=True, ahead=None):
        """prefill toks at the current position (chunked pieces); returns the sample after the last (None when not
        sample: then every token is chunked, no head step). ahead (lab-pfpath): the next feed's tokens (its first
        piece), whose n-gram rows the engine reads while this feed's last chunk runs on the GPU"""
        if any(V.is_key(k) for k in toks) or getattr(self, "mm", False):  # lab-vision: image rows at this position
            self._image_rows(toks)
            toks = V.keys_to_tokens(toks)
            ahead = None
        if self.mtp_cont and self.head_ok():  # the MTP head kept in step (a stale one: plain prefill below)
            body = toks[:-1] if sample else toks
            fa = self._f.get("ne_fn_mtp_feed_ahead")
            if ahead is not None and (sample or not AHEAD):
                ahead = None
            elif ahead and max(ahead) >= V.KEY_BASE:  # (image keys: the feed gathers their token ids)
                ahead = V.keys_to_tokens(ahead)
            i = 0
            ps = pieces(len(body), table=MTP_PIECES)
            for j, p in enumerate(ps):
                a = np.ascontiguousarray(body[i:i + p], np.int32)
                nxt = body[i + p:i + p + ps[j + 1]] if j + 1 < len(ps) else ahead
                if fa is not None and nxt is not None and len(nxt) >= 16:
                    b = np.ascontiguousarray(nxt, np.int32)
                    rc = fa(C.c_void_p(self.mtp_ptr), a.ctypes.data, len(a), b.ctypes.data, len(b))
                else:
                    rc = self._f["ne_fn_mtp_feed"](C.c_void_p(self.mtp_ptr), a.ctypes.data, len(a))
                if rc != 0:
                    raise self.E.EngineError(f"ne_fn_mtp_feed: {self.eng.last_error()}")
                i += p
            if sample:
                self.arm()
            return self.step(toks[-1]) if sample else None
        nx = -1
        i = 0
        ps = pieces(len(toks))
        for j, p in enumerate(ps):
            if sample and j == len(ps) - 1:  # (every piece samples after its last token; only this one is kept)
                self.arm()
            nx, _, _ = self.m.prefill(toks[i:i + p])
            i += p
        return nx if sample else None

    def step(self, tok):
        rc = -1
        if self.mtp_cont:  # refuses (state untouched) when the head is stale: then the plain step
            if getattr(self, "_mstep", None) is None:  # (per-token call: bound once)
                self._mstep, self._mp, self._nxr = self._f["ne_fn_mtp_step"], C.c_void_p(self.mtp_ptr), C.byref(self._nx)
            rc = self._mstep(self._mp, tok, self._nxr, None)
            if rc != 0 and self.head_ok():
                raise self.E.EngineError(f"ne_fn_mtp_step: {self.eng.last_error()}")
        if rc != 0:
            rc = self.eng.lib.ne_fn_model_step(self.m.ptr, tok, 1, None, C.byref(self._nx), C.byref(self._lp),
                                    C.byref(self._am), C.byref(self._ms))
        if rc != 0:
            raise self.E.EngineError(f"step: {self.eng.last_error()}")
        return self._nx.value

    def head_ok(self):
        return self.mtp_cont and self._f["ne_fn_mtp_head_ok"](C.c_void_p(self.mtp_ptr)) == 1

    def ping(self):
        """lab-pfpath: keep the GPU and this model's recordings warm while the engine thread idles (a command buffer
        after ~1 s without GPU work waits ~0.6 s, a recording's first replay ~0.1 s more for its residency):
        one trivial command buffer + requestResidency of the model's and the MTP loop's recordings"""
        if self._ping is not None:
            self._ping(C.c_void_p(self.m.ptr), None)
        if self._mtouch is not None:
            self._mtouch(C.c_void_p(self.mtp_ptr))

    # -- prefix cache I/O (lab-srvcache): numpy buffers the cache owns, accounts and may write to disk
    META_DTYPE = np.int64

    def state_save(self):
        """-> (buf uint8 [snap_bytes], meta int64 [6] = T, hist0, hist1, head in step, head KV rows Lh, pending) or
        None; with head_io the MTP head's bookkeeping is part of it"""
        if not self.kvio:
            return None
        buf = self._state_buf()
        meta = np.zeros(6, np.int64)
        if self.head_io:
            rc = self._h["ne_fn_mtp_state_io"](C.c_void_p(self.mtp_ptr), 0, buf.ctypes.data, buf.nbytes,
                                               meta.ctypes.data)
        else:
            rc = self.eng.lib.ne_fn_state_io(C.c_void_p(self.m.ptr), 0, buf.ctypes.data, buf.nbytes, meta.ctypes.data)
        if rc:
            raise self.E.EngineError(f"state export: {self.eng.last_error()}")
        return buf, meta

    def _state_buf(self):
        """a checkpoint buffer (snap_bytes): one this backend handed out before that nothing references any more (no
        checkpoint, fork or SSD write holds it: refcount = the pool + loop variable + getrefcount's argument), else a new
        one -- a fresh 113 MB buffer page-faults in for 3-25 ms on the engine thread, on every turn's critical path
        (lab-ttft); the pool keeps at most 3 (<= 0.36 GB outside the cache budget)"""
        pool = self.__dict__.setdefault("_spool", [])
        for b in pool:
            if sys.getrefcount(b) <= 3:
                return b
        b = np.empty(self.snap_bytes, np.uint8)
        pool.append(b)
        if len(pool) > 3:
            pool.pop(0)
        return b

    def state_load(self, buf, meta):
        if self.head_io:
            rc = self._h["ne_fn_mtp_state_io"](C.c_void_p(self.mtp_ptr), 1, buf.ctypes.data, buf.nbytes,
                                               meta.ctypes.data)
        else:
            rc = self.eng.lib.ne_fn_state_io(C.c_void_p(self.m.ptr), 1, buf.ctypes.data, buf.nbytes, meta.ctypes.data)
        if rc:
            raise self.E.EngineError(f"state import: {self.eng.last_error()}")

    @staticmethod
    def snap_head(meta):
        """head KV rows a checkpoint needs (Lh), or None when its head was not in step"""
        return int(meta[4]) if len(meta) > 4 and meta[3] else None

    def kv_save(self, a, b, ha=0, hb=0):
        """KV rows [a, b) (+ the MTP head's rows [ha, hb)) -> (buf, cap, hcap)"""
        cap, hcap = b - a, (hb - ha if self.head_io else 0)
        nm = int(self.eng.lib.ne_fn_kv_bytes(C.c_void_p(self.m.ptr), cap))
        nh = int(self._h["ne_fn_mtp_kv_bytes"](C.c_void_p(self.mtp_ptr), hcap)) if hcap else 0
        buf = np.empty(nm + nh, np.uint8)
        if self.eng.lib.ne_fn_kv_io(C.c_void_p(self.m.ptr), 0, a, b, buf.ctypes.data, cap):
            raise self.E.EngineError(f"ne_fn_kv_io: {self.eng.last_error()}")
        if hcap and self._h["ne_fn_mtp_kv_io"](C.c_void_p(self.mtp_ptr), 0, ha, hb, buf.ctypes.data + nm, hcap):
            raise self.E.EngineError(f"ne_fn_mtp_kv_io: {self.eng.last_error()}")
        return buf, cap, hcap

    def kv_load(self, a, b, buf, cap, ha=0, hb=0, hcap=0):
        if self.eng.lib.ne_fn_kv_io(C.c_void_p(self.m.ptr), 1, a, b, buf.ctypes.data, cap):
            raise self.E.EngineError(f"ne_fn_kv_io: {self.eng.last_error()}")
        if hcap and hb > ha:
            nm = int(self.eng.lib.ne_fn_kv_bytes(C.c_void_p(self.m.ptr), cap))
            if self._h["ne_fn_mtp_kv_io"](C.c_void_p(self.mtp_ptr), 1, ha, hb, buf.ctypes.data + nm, hcap):
                raise self.E.EngineError(f"ne_fn_mtp_kv_io: {self.eng.last_error()}")

    MTP_KEYS = ("prefill_ms", "head_prefill_ms", "decode_ms", "tokens", "rounds", "draft_gpu_ms", "verify_gpu_ms",
                "gather_us", "host_us", "max_depth")

    def mtp_from(self, toks, max_new, depth, seed, on_tokens, cancelled, preempt=None, ctx=None):
        """MTP token loop continuing from the current state over the new tokens toks (>= 1); streams through the
        per-round callback (nonzero = stop after the round: cancel, a stop token, or preempt(): another request needs
        the engine thread; that round's tokens are streamed and kept). Leaves state = fed + out[:-1].
        ctx (optional): the state's tokens, given to the loop's copy rule when it does not know them (after a parked
        conversation's import; lab-draft). Returns (tokens, stats)."""
        if ctx and "ne_fn_mtp_set_context" in self._f \
                and self._f["ne_fn_mtp_context_len"](C.c_void_p(self.mtp_ptr)) < len(ctx):
            c = np.ascontiguousarray(V.keys_to_tokens(ctx) if any(V.is_key(k) for k in ctx) else ctx, np.int32)
            self._f["ne_fn_mtp_set_context"](C.c_void_p(self.mtp_ptr), c.ctypes.data, len(c))
        if any(V.is_key(k) for k in toks) or getattr(self, "mm", False):  # lab-vision
            self._image_rows(toks)
            toks = V.keys_to_tokens(toks)
        p = np.ascontiguousarray(toks, np.int32)
        out = np.full(max_new, -1, np.int32)
        st = np.zeros(10, np.float64)

        def cb(_, t, n):
            if cancelled():
                return 1
            t = t[:n]
            on_tokens(t)
            return 1 if any(x in self.STOPS for x in t) or (preempt is not None and preempt()) else 0

        ccb = self._CB(cb)
        k = self._f["ne_fn_mtp_generate_from"](C.c_void_p(self.mtp_ptr), p.ctypes.data, len(p), max_new,
                                               self.STOPS[0], depth, seed & 0xFFFFFFFFFFFFFFFF, out.ctypes.data,
                                               st.ctypes.data, None, ccb, None)
        if k < 0:
            raise self.E.EngineError(f"ne_fn_mtp_generate_from: {self.eng.last_error()}")
        return out[:k].tolist(), dict(zip(self.MTP_KEYS, st.tolist()))

    def mtp(self, prompt, max_new, depth, seed, on_tokens, cancelled):
        """MTP token loop over the whole prompt (resets the model). The C call has no callback: it runs in a thread
        and the tokens it writes into `out` are streamed as they appear (out is pre-filled with -1). It cannot be
        aborted (cancelled() only stops the streaming). Returns (tokens, stats)."""
        p = np.ascontiguousarray(V.keys_to_tokens(prompt), np.int32)
        out = np.full(max_new, -1, np.int32)
        st = np.zeros(10, np.float64)
        keys = ("prefill_ms", "head_prefill_ms", "decode_ms", "tokens", "rounds", "draft_gpu_ms", "verify_gpu_ms",
                "gather_us", "host_us", "max_depth")
        if self.eng.has("ne_fn_mtp_generate_cb"):  # requested in docs/kernels.md: per-round callback, abortable
            CB = C.CFUNCTYPE(C.c_int, C.c_void_p, C.POINTER(C.c_int), C.c_int)
            f = self.eng.fn("ne_fn_mtp_generate_cb", C.c_int,
                            [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_uint64, C.c_void_p,
                             C.c_void_p, C.c_void_p, CB, C.c_void_p])

            def cb(_, toks, n):
                if cancelled():
                    return 1
                t = toks[:n]
                on_tokens(t)
                return 1 if any(x in self.STOPS for x in t) else 0

            ccb = CB(cb)
            k = f(C.c_void_p(self.mtp_ptr), p.ctypes.data, len(p), max_new, self.STOPS[0], depth,
                  seed & 0xFFFFFFFFFFFFFFFF, out.ctypes.data, st.ctypes.data, None, ccb, None)
            if k < 0:
                raise self.E.EngineError(f"ne_fn_mtp_generate_cb: {self.eng.last_error()}")
            return out[:k].tolist(), dict(zip(keys, st.tolist()))
        rc = [None]

        def run():
            rc[0] = self.eng.lib.ne_fn_mtp_generate(C.c_void_p(self.mtp_ptr), p.ctypes.data, len(p), max_new,
                                                    self.STOPS[0], depth, seed & 0xFFFFFFFFFFFFFFFF, out.ctypes.data,
                                                    st.ctypes.data, None)

        th = threading.Thread(target=run, daemon=True)
        th.start()
        k = 0
        while True:
            alive = th.is_alive()
            j = k
            while j < max_new and out[j] >= 0:
                j += 1
            if j > k and not cancelled():
                on_tokens(out[k:j].tolist())
            k = j
            if not alive:
                break
            time.sleep(0.003)
        th.join()
        if rc[0] is None or rc[0] < 0:
            raise self.E.EngineError(f"ne_fn_mtp_generate: {self.eng.last_error()}")
        toks = out[:rc[0]].tolist()
        if len(toks) > k and not cancelled():
            on_tokens(toks[k:])
        return toks, dict(zip(keys, st.tolist()))


_TL = threading.local()


def precise_sleep(d):
    """sleep d seconds without macOS timer coalescing (time.sleep(0.014) takes ~18-19 ms here, which made the mock's
    per-token server overhead look like ~3-5 ms): a kqueue timer with NOTE_CRITICAL wakes on time (14.02 ms); the GIL is
    released while waiting, as during the real engine's ctypes calls"""
    if d <= 0:
        return
    try:
        import select
        if not hasattr(_TL, "kq"):
            _TL.kq = select.kqueue()
        ev = select.kevent(1, select.KQ_FILTER_TIMER, select.KQ_EV_ADD | select.KQ_EV_ONESHOT, 0x2 | 0x20,
                           max(1, int(d * 1e6)))   # NOTE_USECONDS | NOTE_CRITICAL
        _TL.kq.control([ev], 1, None)
    except (ImportError, AttributeError, OSError):
        time.sleep(d)


def _shared(k):
    return property(lambda self: self._s[k], lambda self, v: self._s.__setitem__(k, v))


class MockBackend:
    """CPU stand-in with the same state semantics (for tests): the 'model' answers from a script keyed by markers
    in the prompt text (of the recurrent state's tokens), one token per step after `delay` seconds. It keeps the
    recurrent state (fed), the KV rows (rows: token per position) and, for MTP-continuation engines (cont), the MTP
    head's KV rows (hrows: row j = the pair ending in token j + 1, valid for j < T - 1 while the head is in step) apart,
    and asserts on every feed / step / MTP run that the rows the model would attend over hold the state's tokens:
    any reuse bookkeeping error (a KV copy missed, a stale checkpoint) fails loudly."""
    STOPS = (248046, 248044)
    VOCAB = 248320
    fed, rows, dirty = _shared("fed"), _shared("rows"), _shared("dirty")

    def __init__(self, tokenizer, ctx=32768, delay=0.0, prefill_rate=None, mtp_depth=2, snapshots=True, cont=True,
                 name=None, share_state=None):
        self.name = name      # served model id (several models in one server: WHOAMI answers with it)
        # share_state: another mock whose recurrent state + KV rows this one uses (--share-state; the head is own)
        self._s = share_state._s if share_state is not None else {"fed": [], "rows": [], "dirty": False}
        self.mtp_cont = cont  # cont: the current engine (MTP from the current state); False: the older library
        self.mtp_abortable = True
        self.head_io = cont
        self.head = True      # the MTP head is in step with `fed` (cont engines keep it through feed / step)
        self.tok = tokenizer
        self.ctx = ctx
        self.delay = delay
        self.prefill_rate = prefill_rate
        self.mtp_depth = mtp_depth
        self.snapshots = snapshots
        self.fed = []        # the recurrent state (every token fed since the last reset / restore)
        self.rows = []       # KV rows: token per position (what attention would read)
        self.hrows = []      # the MTP head's KV rows
        self.dirty = False
        self.reply = []
        self.pos = 0
        self.base = None     # len(fed) when self.reply was scripted (None: the state moved elsewhere)
        self.stats = {"fed_tokens": 0, "restores": 0, "resets": 0, "mtp_runs": 0, "kv_saved": 0, "kv_loaded": 0,
                      "head_restores": 0, "mask_sets": 0, "masked_samples": 0}
        self.mask = None          # allowed-token bits (int32 words) or None
        self.extra = None         # lab-sampler: the running request's extra sampling parameters
        self.armed = 0            # arm() calls (a request arms once per kept first sample)
        self.can_extra = True
        self._top = None
        self.rest = None          # masked mode: the script text not yet said
        self.can_mask = True
        self.load_s = 0.0
        self.snap_bytes = 4 * 64
        self.kvio = snapshots

    def _rows_ok(self):
        """the KV rows attention reads ([0, T)) must hold the tokens the recurrent state saw (and the head's)"""
        assert self.rows[:len(self.fed)] == self.fed, \
            f"KV rows != state tokens at {len(self.fed)} (lcp {sum(1 for _ in zip(self.rows, self.fed))})"
        if self.head_ok():
            L = max(0, len(self.fed) - 1)
            assert self.hrows[:L] == self.fed[1:L + 1], f"head KV rows != state tokens (T {len(self.fed)})"

    def _write_rows(self, toks):
        T = len(self.fed)
        self.rows[T:T + len(toks)] = list(toks)
        if self.head_ok():
            f = self.fed + list(toks)
            a = max(0, T - 1)
            self.hrows[a:len(f) - 1] = f[a + 1:]

    @property
    def T(self):
        return len(self.fed)

    def cache_layout(self):
        return {"kind": "mock", "snap_bytes": self.snap_bytes, "head_io": self.head_io, "kvio": self.kvio}

    def reset(self):
        self.fed, self.dirty, self.head = [], False, True
        self.base = None
        self.stats["resets"] += 1

    def head_ok(self):
        return self.mtp_cont and self.head and not self.dirty

    def set_sampler(self, *a):
        pass

    def prepare(self, prompt, images):
        self.images = len(images)

    def set_mask(self, bits):
        if bits is not None:
            self.stats["mask_sets"] += 1
        self.mask = None if bits is None else np.array(bits, dtype=np.int32).reshape(-1).copy()

    def _allowed(self, t):
        return self.mask is None or bool((int(self.mask[t >> 5]) >> (t & 31)) & 1)

    # -- lab-sampler: the mock honours bans (logit_bias <= -100: the scripted token is skipped) and fakes the
    # candidate list (the sampled token first, then its successors), so the server plumbing is testable on the CPU
    def set_extra(self, ex, prompt):
        self.extra = ex
        self.stats["extra_sets"] = self.stats.get("extra_sets", 0) + 1
        self.arm()

    def arm(self):
        if self.extra is not None:
            self.armed += 1

    def clear_extra(self):
        if self.extra is not None:
            self.stats["extra_clears"] = self.stats.get("extra_clears", 0) + 1
        self.extra = None
        self._top = None

    def last_top(self):
        assert self.extra is not None and self.extra["logprob_k"] > 0, "last_top without logprob_k"
        return self._top

    def _fake_top(self, t):
        k = max(self.extra["logprob_k"], 1) if self.extra else 0
        ids = [(t + i) % self.VOCAB for i in range(k)]
        self._top = (ids, [-0.05 * (i + 1) ** 2 for i in range(k)])

    def _script(self, prompt):
        r = self._script0(prompt)
        if self.extra and self.extra["bias"]:   # logit_bias <= -100: the banned tokens are never said (any path)
            r = [t for t in r if self.extra["bias"].get(t, 0) > -100 or t in self.STOPS]
        return r

    def _script0(self, prompt):
        text = self.tok.decode(V.keys_to_tokens(prompt), skip_special_tokens=False)
        last = text[text.rfind("<|im_start|>user"):] if "<|im_start|>user" in text else text
        think = text.endswith("<think>\n")
        r = "Let me think about this.\n</think>\n\n" if think else ""
        if "<tool_response>" in last and "TOOLTEST" in text:
            r += "The weather in Paris is 21 C and sunny."
        elif "TOOLTEST" in last:
            r += ("I will call the tool.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
                  "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>")
        elif "CUTTEST" in last:   # a call cut off mid-parameter, then EOS (lab-fixsrv)
            r += ("I will call the tool.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
                  "<parameter=days>\n3")
        elif "IMAGETEST" in last:
            r += f"I see {sum(1 for k in prompt if V.is_key(k))} image tokens."
        elif "JSONTEST" in last:   # prose first: a model under a response_format mask has to skip to the document
            r += 'Sure, here is the person you asked for: ' + json.dumps({
                "name": "Ann Lee", "age": 31, "email": "ann@example.org",
                "address": {"street": "1 Main St", "city": "Lyon", "geo": {"lat": 45.7, "lon": 4.8}},
                "tags": ["a", "b"], "active": True}) + ' Hope that helps!'
        elif "FORCETEST" in last:
            r += ("I will look it up.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
                  "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>")
        elif "STATETEST" in last:   # a digest of every token in the recurrent state: = a fresh prefill's iff reuse was exact
            r += f"State {len(prompt)} {zlib.crc32(np.asarray(prompt, np.int64).tobytes()):08x}."
        elif "WHOAMI" in last and self.name:
            r += f"I am {self.name}."
        elif "STOPTEST" in last:
            r += "one two three STOP_HERE four five"
        elif "LONGTEST" in last:
            r += " ".join(["word"] * 400)
        else:
            r += "Hello from the mock engine."
        return self.tok.encode(r, add_special_tokens=False).ids + [self.STOPS[0]]

    def feed(self, toks, sample=True, ahead=None):
        assert not self.dirty, "feed on a dirty state"
        assert len(self.fed) + len(toks) <= self.ctx
        self._rows_ok()
        if self.prefill_rate:
            precise_sleep(len(toks) / self.prefill_rate)
        self._write_rows(toks)
        self.fed = self.fed + list(toks)
        self.stats["fed_tokens"] += len(toks)
        self.reply, self.pos, self.rest, self.base = self._script(self.fed), 0, None, len(self.fed)
        return self._next() if sample else None

    def _next(self):
        t = self._next0()
        if self.extra and self.extra["logprob_k"]:
            self._fake_top(t)
        return t

    def _next0(self):
        if self.mask is not None:
            # a model that follows the mask: it says its script with the longest allowed token that fits the rest of
            # the text and drops a character it cannot say; the stop token when the script is spent
            self.stats["masked_samples"] += 1
            if self.rest is None:
                self.rest = self.tok.decode(self.reply[self.pos:], skip_special_tokens=False)
            while self.rest:
                for n in range(min(len(self.rest), 14), 0, -1):
                    ids = self.tok.encode(self.rest[:n], add_special_tokens=False).ids
                    if len(ids) == 1 and self._allowed(ids[0]):
                        self.rest = self.rest[n:]
                        return ids[0]
                self.rest = self.rest[1:]
            if self._allowed(self.STOPS[0]):
                return self.STOPS[0]
            return int(np.flatnonzero(np.unpackbits(self.mask.view(np.uint8), bitorder="little"))[0])
        t = self.reply[min(self.pos, len(self.reply) - 1)]
        self.pos += 1
        if self.extra and self.extra["bias"]:
            while self.extra["bias"].get(t, 0) <= -100 and t not in self.STOPS and self.pos < len(self.reply):
                t = self.reply[self.pos]
                self.pos += 1
        return t

    def step(self, tok):
        assert not self.dirty
        if self.delay:
            precise_sleep(self.delay)
        self._write_rows([tok])
        self.fed.append(tok)
        return self._next()

    META_DTYPE = np.int64

    def state_save(self):
        if not self.snapshots or self.dirty:
            return None
        h = self.head_ok()
        return (np.asarray(self.fed, np.int64).view(np.uint8).copy(),  # int64: image prefix keys (lab-vision)
                np.asarray([len(self.fed), 0, 0, int(h), max(0, len(self.fed) - 1) if h else 0, int(h)], np.int64))

    def state_load(self, buf, meta):
        self.fed, self.dirty = buf.view(np.int64).tolist(), False
        assert len(self.fed) == int(meta[0])
        self.base = None
        self.head = bool(meta[3]) and self.mtp_cont
        self.stats["restores"] += 1
        self.stats["head_restores"] += int(self.head)
        self._rows_ok()  # the rows must already be loaded (KV first, then the state)

    @staticmethod
    def snap_head(meta):
        return int(meta[4]) if meta[3] else None

    def kv_save(self, a, b, ha=0, hb=0):
        assert b <= len(self.rows) and hb <= len(self.hrows)
        self.stats["kv_saved"] += b - a
        hcap = hb - ha if self.head_io else 0
        buf = np.full(b - a + hcap + 3, -7, np.int64)  # (int64: image prefix keys) cap = b - a rows (+ the padding the real layout has)
        buf[:b - a] = self.rows[a:b]
        buf[b - a:b - a + hcap] = self.hrows[ha:ha + hcap]
        return buf.view(np.uint8), b - a, hcap

    def kv_load(self, a, b, buf, cap, ha=0, hb=0, hcap=0):
        assert b - a <= cap and hb - ha <= hcap
        self.stats["kv_loaded"] += b - a
        v = buf.view(np.int64)
        for rows, x, y, off in ((self.rows, a, b, 0), (self.hrows, ha, hb, cap)):
            if y > x:
                if len(rows) < x:
                    rows += [-1] * (x - len(rows))
                rows[x:y] = v[off:off + y - x].tolist()

    def mtp(self, prompt, max_new, depth, seed, on_tokens, cancelled):
        assert self.mask is None, "MTP does not apply the token mask"
        self.stats["mtp_runs"] += 1
        self.fed, self.dirty, self.head = [], False, False
        self.rows = list(prompt)
        if self.prefill_rate:
            precise_sleep(len(prompt) / self.prefill_rate)
        reply = self._script(list(prompt))
        out = []
        for i in range(0, len(reply), 3):  # rounds of up to 3 tokens
            if len(out) >= max_new:
                break
            r = reply[i:i + 3][:max_new - len(out)]
            if self.delay:
                precise_sleep(self.delay * 1.5)
            out += r
            if not cancelled():
                on_tokens(r)
        self.fed = list(prompt) + out[:-1]
        self.rows = list(prompt) + out  # the verify writes rejected drafts' rows past the committed length
        self.dirty = True
        return out, {"prefill_ms": 0.0, "decode_ms": 0.0, "tokens": len(out) - 1}


    def mtp_from(self, toks, max_new, depth, seed, on_tokens, cancelled, preempt=None, ctx=None):
        assert self.mtp_cont and self.head_ok(), "mtp_from without the head in step"
        assert self.mask is None, "MTP does not apply the token mask"
        assert toks and len(self.fed) + len(toks) <= self.ctx
        self._rows_ok()
        self.stats["mtp_runs"] += 1
        self.stats["fed_tokens"] += len(toks)
        if self.prefill_rate:
            precise_sleep(len(toks) / self.prefill_rate)
        # mid-reply (the step path's pending token, after a preempted MTP run or steps): continue the same script
        mid = self.base is not None and self.pos > 0 and len(self.fed) == self.base + self.pos - 1 \
            and list(toks) == [self.reply[min(self.pos - 1, len(self.reply) - 1)]]
        self._write_rows(toks)
        self.fed = self.fed + list(toks)
        if mid:
            reply, m = self.reply[self.pos:], self.pos
        else:
            self.reply, self.base, m = self._script(self.fed), len(self.fed), 0
            reply = self.reply
        out = []
        for i in range(0, len(reply), 3):  # rounds of up to 3 tokens; the callback can stop after a round
            if len(out) >= max_new:
                break
            r = reply[i:i + 3][:max_new - len(out)]
            if self.delay:
                precise_sleep(self.delay * 1.5)
            if cancelled():
                break
            out += r
            on_tokens(r)
            if any(x in self.STOPS for x in r) or (preempt is not None and preempt()):
                break
        self.pos, self.rest = m + len(out), None  # a step(out[-1]) continues the script
        self._write_rows(out[:-1])
        self.fed = self.fed + out[:-1]
        T = len(self.fed)
        self.rows[T:T + 2] = [-5, -5]   # the verify wrote rejected drafts' rows past the committed length
        self.hrows[T - 1:T + 1] = [-5, -5]
        return out, {"prefill_ms": 0.0, "decode_ms": 0.0, "tokens": len(out) - 1}


# -------------------------------------------------------------------------------------------------- session
class Phys:
    """the state buffers (KV + recurrent state) of a slot; with --share-state one Phys serves the same slot of several
    models (one weight set each): one owner at a time, busy while any of them runs"""

    def __init__(self):
        self.owner = None
        self.busy = False


class Slot:
    """one model instance: its backend, the conversation resident in its KV buffer, whether the recurrent state
    equals all of that conversation's tokens (live), and whether a request runs on it (busy)"""

    def __init__(self, i, backend, phys=None):
        self.i, self.b = i, backend
        self.conv = None
        self.live = True
        self.phys = phys or Phys()
        self.sess = None

    @property
    def busy(self):
        return self.phys.busy

    @busy.setter
    def busy(self, v):
        self.phys.busy = v

    @property
    def ctx(self):
        return self.b.ctx


class Session:
    IM_START = 248045
    IM_END = 248046

    def __init__(self, backends, ctx, max_ckpts=4, mtp_policy="auto", sys_ckpt_min=256, log=print,
                 cache_ram=8 << 30, disk_dir=None, disk_bytes=0, pf_cap=0, phys=None, ident=None, shared=None,
                 markers=None):
        """pf_cap: largest prefill piece while another request runs (0 = the canonical pieces always: a request's
        output then never depends on what else runs; smaller pieces round differently, docs/server.md 3c).
        disk_dir / disk_bytes: the SSD tier (kept across restarts); ident: what its entries are keyed by
        (prefix_cache.identity: engine build, pack, model id, layout).
        shared (lab-snap, docs/server.md 3f): shared-prefix snapshots on (default: NE_SHARED_SNAPS); markers: token
        sequences whose end inside the first message is a structural boundary (the template's tools block end)"""
        self.pf_cap = pf_cap or None
        if not isinstance(backends, (list, tuple)):
            backends = [backends]
        self.slots = [Slot(i, b, phys[i] if phys else None) for i, b in enumerate(backends)]
        for sl in self.slots:
            sl.sess = self
        self.b = backends[0]
        self.ctx = ctx
        self.cache = PrefixCache(cache_ram, max_ckpts, disk_dir, disk_bytes, log, ident, SHARED_RAM, SHARED_DISK)
        self.shared = SHARED if shared is None else bool(shared)
        self.markers = [list(m) for m in (markers or []) if m]
        for sl in self.slots:
            sl.conv = self.cache.new(sl.i)
        self.mtp_policy = mtp_policy if getattr(self.b, "mtp_depth", 0) > 0 else "never"
        self.sys_ckpt_min = sys_ckpt_min
        self.rates = Rates()
        self.log = log

    # -- slot 0 views (single-slot callers and tests)
    @property
    def kv(self):
        return self.slots[0].conv.toks

    @property
    def live(self):
        return self.slots[0].live

    # -- planning
    def candidates(self, prompt, reuse=True):
        if not reuse:
            return []
        return self.cache.candidates(prompt, len(prompt), {s.i: s.live for s in self.slots})

    def pick(self, prompt, reuse=True):
        """-> (slot, candidate | None) for a new request, or (None, None) when no idle slot can hold the prompt.
        Prefers the longest reusable prefix, then the slot that already holds this conversation's prefix (its own
        earlier turn: more than the shared system prompt), else the slot whose resident conversation is smallest
        (empty first; then least recently used; then the lowest slot). A shared system checkpoint resident next to
        another conversation is copied to a free slot rather than displacing that conversation (a subagent's request
        must not push the main conversation out of its slot)."""
        n = len(prompt)
        idle = [s for s in self.slots if not s.busy and n + 1 <= s.ctx]
        if not idle:
            return None, None
        cands = self.candidates(prompt, reuse)
        sysb = self.system_boundary(prompt, self.IM_END) if self.IM_END is not None else 0
        best, bkey = (idle[0], None), None
        for s in idle:
            for c in [None] + cands:
                if c is not None and c[2] == "live" and c[1].slot != s.i:
                    continue
                p = c[0] if c else 0
                here = c is not None and isinstance(c[1], Conv) and c[1].slot == s.i
                own = here and (p > sysb or c[2] == "live")  # this conversation's own earlier state
                key = (p, own, -(0 if own else len(s.conv.toks)), c is not None and c[2] == "live", -s.conv.last, -s.i)
                if bkey is None or key > bkey:
                    best, bkey = (s, c), key
        return best

    def plan(self, prompt):
        """-> (k reusable tokens, source) on the slot pick() would take (logs / tests)"""
        sl, c = self.pick(prompt)
        if c is None:
            return 0, None
        return c[0], ("live" if c[2] == "live" else f"{c[2]}:c{c[1].id}")

    def choose_mtp(self, n, k, max_new, can_mtp):
        if not can_mtp or self.mtp_policy == "never":
            return False
        if self.mtp_policy == "always":
            return True
        r = self.rates
        e = min(max_new, r.get("out"))
        if getattr(self.b, "mtp_cont", False):
            # MTP continues from the reused state and leaves a reusable one: both paths prefill the same n - k tokens
            # (the feed keeps the MTP head in step either way), so the decode rates decide
            t_mtp = (n - k) / r.get("pf") + e / r.get("dec_mtp")
        else:
            # older library: full prefill now and nothing reusable for the next turn (the MTP loop keeps no
            # checkpoint), so add the next turn's re-prefill of this prompt on the step path
            t_mtp = n / r.get("pf_mtp") + e / r.get("dec_mtp") + n / r.get("pf")
        t_step = (n - k) / r.get("pf") + e / r.get("dec")
        return t_mtp < t_step

    def last_start(self, prompt):
        """position of the last <|im_start|> (the generation prompt), or 0"""
        for i in range(len(prompt) - 1, max(-1, len(prompt) - 64), -1):
            if prompt[i] == self.IM_START:
                return i
        return 0

    def system_boundary(self, prompt, im_end):
        """tokens of the system message (up to and incl. its <|im_end|>), or 0"""
        if len(prompt) < self.sys_ckpt_min + 2:
            return 0
        try:
            i = prompt.index(im_end, 0, min(len(prompt), 200000))
        except ValueError:
            return 0
        return i + 1 if i + 1 >= self.sys_ckpt_min else 0

    # -- lab-snap: shared-prefix snapshots (docs/server.md 3f)
    def struct_boundary(self, prompt, im_end):
        """the end of the last marker (the template's tools block: '</IMPORTANT>') inside the first message, or 0"""
        if not self.markers:
            return 0
        try:
            lim = prompt.index(im_end, 0, min(len(prompt), 200000))
        except ValueError:
            return 0
        for m in self.markers:
            w = len(m)
            for i in range(lim - w, -1, -1):
                if prompt[i] == m[0] and prompt[i:i + w] == m:
                    return i + w
        return 0

    def shared_marks(self, prompt, k, end, sysb, im_end, L, owner, src, last=None):
        """positions in (k, end) where this prefill takes a shared-prefix snapshot: the structural boundary (the
        tools block's end, when the system text behind it is long) and the learned one (the longest prefix this prompt
        shares with another stored sequence, minus a margin, rounded down to SHARED_ALIGN) when either saves >=
        SHARED_MIN tokens over the checkpoint the request restored (k). Not when the stored sequence covers the prompt up
        to its generation prompt (last): the same first turn again (a retry with reuse off), not a shared prefix"""
        if not self.shared or not getattr(self.b, "kvio", True):
            return []
        out = []
        tb = self.struct_boundary(prompt, im_end)
        if tb and tb - k >= SHARED_MIN and tb < end and tb <= SHARED_MAX_POS and (not sysb or sysb - tb >= SHARED_MIN):
            out.append(tb)
        if owner is not None and owner is not src and L > k and (not last or L < last):
            x = (L - SHARED_MARGIN) // SHARED_ALIGN * SHARED_ALIGN
            if x - k >= SHARED_MIN and x >= self.sys_ckpt_min and x < end and x <= SHARED_MAX_POS \
                    and all(abs(x - y) >= SHARED_MIN for y in out):
                out.append(x)
        return sorted(out)

    def _finish_shared(self, sl, N, pending):
        """after the prefill passed them: each captured position x becomes a shared-prefix snapshot (a Conv with
        shared=True: toks [0, x), the checkpoint, host copies of KV rows [0, x) + the MTP head's rows the checkpoint
        needs; a later capture's rows are copied from the earlier one's end), written through to the SSD tier; the
        slot's conversation shares the copied rows (parking it later copies only the rows past them). -> count"""
        made, prev = 0, None
        if N.segs:  # host copies of the conversation's first rows (the shared snapshot it was restored from): reused
            prev = (N.saved_n, N.head_n, [list(sg) for sg in N.segs])
        for x, snap in sorted(pending, key=lambda t: t[0]):
            if len(N.toks) < x:
                break
            toks = list(N.toks[:x])
            if self.cache.find_shared(toks) is not None:
                continue
            lh = sl.b.snap_head(snap.meta) or 0
            if prev is not None and not (prev[0] < x and prev[1] <= lh):
                prev = None  # (the copies reach past x: copy rows [0, x) afresh)
            a, ha, segs = (prev[0], prev[1], list(prev[2])) if prev else (0, 0, [])
            buf, cap, hcap = sl.b.kv_save(a, x, ha, max(ha, lh))
            segs.append([a, x, buf, cap, ha, ha + hcap, hcap])
            S = Conv()
            S.shared, S.slot, S.toks, S.snaps, S.segs = True, None, toks, {x: snap}, segs
            self.cache.add(S)
            self.cache.stats["shared_made"] += 1
            if self.cache.disk is not None:
                try:
                    self.cache.persist(S, None)
                except Exception as e:  # noqa: BLE001 (the snapshot stays in RAM)
                    self.log(f"prefix cache: SSD write of a shared prefix failed: {e!r}")
            self.log(f"prefix cache: shared prefix c{S.id} at {x} tokens (slot {sl.i})")
            prev = (x, ha + hcap, segs)
            made += 1
        if made and prev is not None and N.saved_n <= prev[0] <= len(N.toks):
            N.segs = [list(sg) for sg in prev[2]]
        return made

    def _finish_shared_safe(self, sl, N, pending):
        """_finish_shared after the reply: a failure (e.g. MemoryError copying the rows) is logged, never the reply's"""
        try:
            return self._finish_shared(sl, N, pending)
        except Exception as e:   # noqa: BLE001
            self.log(f"slot {sl.i}: making a shared-prefix snapshot failed: {e!r}")
            traceback.print_exc(file=sys.stderr)
            return 0

    # -- conversation moves
    def _park(self, sl):
        """the slot's resident conversation leaves its KV buffer: keep it (host copy of the rows up to its newest
        checkpoint, only those not copied yet) if it has checkpoints, else forget it"""
        R = sl.conv
        sl.conv = None
        if R is None:
            return
        if not R.snaps:
            self.cache.drop(R)
            return
        t0 = time.perf_counter()
        m = max(R.snaps)
        R.truncate(m)
        a, ha = R.saved_n, R.head_n
        # the MTP head's rows the checkpoints need (rows below a checkpoint's Lh are never rewritten later)
        hb = max([ha] + [h for h in (sl.b.snap_head(x.meta) for x in R.snaps.values()) if h is not None])
        if a < m or ha < hb:
            buf, cap, hcap = sl.b.kv_save(a, m, ha, hb)
            R.segs.append([a, m, buf, cap, ha, ha + hcap, hcap])
        R.slot = None
        self.cache.stats["parks"] += 1
        self.cache.stats["park_ms"] += (time.perf_counter() - t0) * 1e3

    def _prepare(self, sl, cand):
        """make the slot's state = the candidate's prefix; -> reused tokens"""
        if cand is None:
            self._park(sl)
            sl.conv = self.cache.new(sl.i)
            sl.b.reset()
            sl.live = True
            return 0
        p, c, kind = cand
        R = sl.conv
        if kind == "disk":
            c = self.cache.disk.take(c, sl.b.META_DTYPE)
            if c is None:  # damaged on disk (checksum, size): dropped; this request prefills cold
                return self._prepare(sl, None)
            self.cache.add(c)
            self.cache.stats["disk_hits"] += 1
            kind = "snap"
        if c is R:
            if kind != "live":
                if any(q > p for q in R.snaps):  # later checkpoints would be overwritten: keep them parked
                    self._park(sl)
                    N = self.cache.add(R.fork(p))
                    N.slot, sl.conv = sl.i, N
                else:
                    R.truncate(p)
                s = sl.conv.snaps[p]
                sl.b.state_load(s.buf, s.meta)
        else:
            t0 = time.perf_counter()
            snap = c.snaps[p]
            lh = sl.b.snap_head(snap.meta)  # the MTP head's KV rows this checkpoint needs (None: head stale)
            if c.slot is not None:  # resident in another slot (rows [0, p) are stable while it runs)
                buf, cap, hcap = self.slots[c.slot].b.kv_save(0, p, 0, lh or 0)
                segs = [[0, p, buf, cap, 0, hcap, hcap]]
            else:
                segs = c.segs
            self._park(sl)
            head_to = 0
            for a, b, buf, cap, ha, hb, hcap in segs:
                if a < p:
                    hb2 = min(hb, lh or 0)
                    sl.b.kv_load(a, min(b, p), buf, cap, ha, max(ha, hb2), hcap)
                    head_to = max(head_to, hb2)
            meta = snap.meta
            if lh is not None and head_to < lh:  # head rows missing: restore the model only (MTP restarts cold)
                meta = meta.copy()
                meta[3] = 0
            if c.slot is None and not any(q > p for q in c.snaps) and not c.shared:
                c.truncate(p)
                N = c
            else:  # (a shared-prefix snapshot is never adopted: the request continues on a fork of it)
                N = self.cache.add(c.fork(p))
            if c.shared:
                c.touch()
                self.cache.stats["shared_hits"] += 1
            N.slot, sl.conv = sl.i, N
            sl.b.state_load(snap.buf, meta)
            self.cache.stats["loads"] += 1
            self.cache.stats["load_ms"] += (time.perf_counter() - t0) * 1e3
        sl.live = True
        sl.conv.touch()
        return p

    def _persist(self, sl):
        """write the slot's resident conversation through to the SSD tier (only the rows past its last write and
        the new checkpoints; the writer thread lands them); -> ms on the engine thread"""
        N = sl.conv
        if self.cache.disk is None or N is None or not N.snaps or N.slot != sl.i:
            return 0.0
        t0 = time.perf_counter()
        # the MTP head's rows the checkpoints need (as _park)
        hb = max([0] + [h for h in (sl.b.snap_head(x.meta) for x in N.snaps.values()) if h is not None])

        def src(a, m, ha):
            if a >= m:
                return None
            return sl.b.kv_save(a, m, ha, max(ha, hb))

        self.cache.persist(N, src)
        return (time.perf_counter() - t0) * 1e3

    def _add_ckpt(self, sl):
        N = sl.conv
        pos = len(N.toks)
        if pos in N.snaps:
            return
        r = sl.b.state_save()
        if r is None:
            return
        N.snaps[pos] = Snap(pos, *r)
        N.thin(self.cache.max_snaps)

    def _own(self, sl):
        """--share-state: make sl the owner of its state buffers. The previous owner (the same slot of another model)
        parks its resident conversation first (host copy of its KV rows up to its newest checkpoint; its checkpoints
        stay in its own prefix cache) and keeps an empty one: its next request restores or resets."""
        ph = sl.phys
        prev = ph.owner
        if prev is not None and prev is not sl:
            t0 = time.perf_counter()
            S = prev.sess
            S._park(prev)
            prev.conv = S.cache.new(prev.i)
            prev.live = True
            self.cache.stats["handovers"] = self.cache.stats.get("handovers", 0) + 1
            self.cache.stats["handover_ms"] = self.cache.stats.get("handover_ms", 0.0) + (time.perf_counter() - t0) * 1e3
        ph.owner = sl

    def _enforce(self):
        """the cache budget after a finished reply; a failure here is logged, never the reply's"""
        try:
            self.cache.enforce()
        except Exception as e:   # noqa: BLE001
            self.log(f"prefix cache: enforcing the budget failed after a finished reply: {e!r}")
            traceback.print_exc(file=sys.stderr)

    def fail(self, sl):
        """after an engine error: forget the slot's conversation and reset it"""
        if sl.conv is not None:
            self.cache.drop(sl.conv)
        sl.conv = self.cache.new(sl.i)
        sl.live = True
        sl.busy = False
        try:
            sl.b.reset()
        except Exception:
            pass

    # -- generation
    def generate(self, prompt, max_new, sampler, use_mtp, on_tokens, cancelled, im_end, depth=2, reuse=True,
                 constraint=None, extra=None):
        """one completion, synchronously, on the slot pick() chooses; returns the stats dict"""
        sl, cand = self.pick(prompt, reuse)
        g = self.run(sl, cand, prompt, max_new, sampler, use_mtp, on_tokens, cancelled, im_end, depth,
                     constraint=constraint, extra=extra)
        while True:
            try:
                next(g)
            except StopIteration as e:
                return e.value

    def run(self, sl, cand, prompt, max_new, sampler, use_mtp, on_tokens, cancelled, im_end, depth=2,
            others=lambda: False, constraint=None, contended=lambda: False, extra=None, capture=True):
        """-> generator: one completion on slot sl from candidate cand (pick()); yields after every prefill piece and
        decode token (the worker interleaves slots); returns the stats dict. others(): another request is running
        (prefill pieces are then capped at pf_cap tokens, if set, so its decode stalls less). The slot is busy from this call
        on (a later pick() skips it). constraint (server/grammar.py; step path only): .start() -> the allowed-token
        bits of the first sampled token, .advance(token) -> those of the next; the mask is the slot's model's own
        (ne_fn_model_set_mask is per model instance), set before every sample of this request and cleared at its end.
        use_mtp: this request may decode with MTP. On MTP-continuation engines it does so whenever contended() is
        false (no other request runs, or waits with a free slot for it): an MTP run yields at its next round boundary
        once contended() turns true and the reply continues on the step path (token by token, interleaved), then
        resumes MTP when contended() is false again. The state is fed + out[:-1] at every switch, so the output is
        the one the request gives alone. Older libraries: MTP from an empty state, decided here.
        extra (sampling.Extras.engine_extra; step path only): logit_bias / penalties / min_p / logprobs / seeded RNG for
        this request, cleared at its end. With logprobs the sampled token's candidates travel with it:
        on_tokens([token], (ids, logprobs)).
        capture (lab-snap): this request may take shared-prefix snapshots (request extension ne_shared false: not;
        A/B tests of the cold path)"""
        sl.busy = True
        return self._run(sl, cand, prompt, max_new, sampler, use_mtp, on_tokens, cancelled, im_end, depth, others,
                         constraint, contended, extra, capture)

    def _run(self, sl, cand, prompt, max_new, sampler, use_mtp, on_tokens, cancelled, im_end, depth, others,
             constraint, contended, extra=None, capture=True):
        n = len(prompt)
        stats = {"prompt_tokens": n, "cached_tokens": 0, "path": "mtp" if use_mtp else "step", "slot": sl.i}
        sl.busy = True
        want_lp = bool(extra and extra.get("logprob_k"))
        try:
            self._own(sl)
            sl.b.set_sampler(*sampler)
            if depth > 2 and (sampler[0] < 0.01 or sampler[2] == 1):
                # lab-draft: greedy requests have no draft proposals for the gated verify (docs/mtp.md 11) and a fixed
                # depth 3 loses to 2 there (111 vs 103 tok/s): they draft 2 deep
                depth = 2
            if extra is not None:
                # the MTP loop applies logit_bias / penalties / min_p (exactly: the verify rows get the transform) but
                # gives no logprobs, and only the continuing MTP (mtp_cont) keeps the counts of a run it resumes
                assert not use_mtp or (getattr(sl.b, "mtp_cont", False) and not extra.get("logprob_k")), \
                    "this MTP path cannot apply logprobs (or knobs on the older library)"
                sl.b.set_extra(extra, prompt)
            t0 = time.perf_counter()
            if constraint is not None:
                assert not use_mtp, "the MTP loop has no token mask"
            cont = use_mtp and getattr(sl.b, "mtp_cont", False)
            if use_mtp and not cont:  # older library: the MTP loop starts from an empty state
                self._park(sl)
                N = sl.conv = self.cache.new(sl.i)
                toks, st = sl.b.mtp(prompt, max_new, depth, sampler[3], on_tokens, cancelled)
                N.toks = list(prompt) + toks[:-1]
                sl.live = False
                stats.update(completion_tokens=len(toks), prefill_ms=st.get("prefill_ms", 0.0),
                             decode_ms=st.get("decode_ms", 0.0), wall_ms=(time.perf_counter() - t0) * 1e3)
                if st.get("prefill_ms"):
                    self.rates.obs("pf_mtp", n / (st["prefill_ms"] / 1e3))
                if st.get("decode_ms") and st.get("tokens", 0) > 32:
                    self.rates.obs("dec_mtp", st["tokens"] / (st["decode_ms"] / 1e3))
                self.rates.obs("out", len(toks))
                self._enforce()
                return stats
            stats["cache_src"] = "none" if cand is None else {"live": "live", "disk": "ssd"}.get(cand[2], "ram")
            trace(f"run start: prompt {n}")
            # lab-snap: the longest prefix shared with any stored sequence, before _prepare parks / truncates anything
            # (only when a capture can follow: the restored prefix leaves >= SHARED_MIN tokens below SHARED_MAX_POS --
            # a long conversation's later turn compares nothing more)
            k0 = cand[0] if cand is not None else 0
            L, owner = self.cache.longest_match(prompt, min(n - 1, SHARED_MAX_POS + SHARED_MARGIN)) \
                if self.shared and capture and n > SHARED_MIN and k0 + SHARED_MIN <= SHARED_MAX_POS else (0, None)
            src = cand[1] if cand is not None else None
            stats["shared_src"] = bool(src is not None and getattr(src, "shared", False))
            k = self._prepare(sl, cand)
            trace(f"prepared: reused {k}, {(time.perf_counter() - t0) * 1e3:.1f} ms")
            if cand is not None and k == 0:
                stats["cache_src"] = "none"  # an SSD entry rejected (checksum): cold
                stats["shared_src"] = False
            N = sl.conv
            if cont and not sl.b.head_ok():  # e.g. a checkpoint whose head was stale: keep the reuse, decode 1-row
                cont = False
            stats["cached_tokens"] = k
            stats["restore_ms"] = (time.perf_counter() - t0) * 1e3
            if constraint is not None:  # the sample at the end of the prefill is the first token
                sl.b.set_mask(constraint.start())
            # checkpoints on the way: the end of the system prompt (shared by conversations) and the start of the
            # generation prompt (the last <|im_start|>: the next turn re-renders this turn's '<think>\n' + reasoning,
            # and '\n' merges with what follows, so the end of the prompt itself is rarely a prefix of the next one)
            sysb, last = self.system_boundary(prompt, im_end), self.last_start(prompt)
            if last == sysb + 1:  # (lab-pfpath) the first message is the only one: the generation prompt's checkpoint
                sysb = 0          # one '\n' later serves every reuse of that one (a 1-token piece + a checkpoint less)
            t_snap = 0.0
            nx = -1
            # MTP from the state: the loop takes the last prompt token itself (contended at the start: the step
            # path's prefill; the decode switches to MTP once the engine is free)
            first_mtp = cont and not contended()
            end = n - 1 if first_mtp else n
            cmarks = {x for x in (sysb, last) if k < x < n}
            # lab-snap: shared-prefix captures (docs/server.md 3f); taken at piece boundaries, made after the reply
            smarks = set(self.shared_marks(prompt, k, end, sysb, im_end, L, owner, src, last)) if capture else set()
            # shared boundaries supersede the end of the system message: with one below it (captured now: the tools
            # block's end or a learned one; or the shared snapshot this request restored) the system checkpoint is
            # not taken -- the prefill keeps master's number of passes (each ~0.1-0.2 s on a cold prompt) and a
            # session with the same system text gets a learned boundary at most SHARED_ALIGN + margin before it
            if sysb in cmarks and any(x <= sysb for x in smarks | ({k} if stats["shared_src"] else set())):
                cmarks.discard(sysb)
            pending = []
            marks = sorted(cmarks | smarks)
            aborted = False
            segs = [y for y in marks if y < end] + [end]

            def piece_at(a, cap, table):  # (lab-pfpath) the piece a feed starting at a would take, or None
                for y in segs:
                    if y > a:
                        return prompt[a:a + pieces(y - a, cap, table)[0]]
                return None
            for si, x in enumerate(segs):
                i = k
                table = MTP_PIECES if sl.b.head_ok() else PIECES  # (an MTP feed chunks every token of a piece)
                cap = self.pf_cap if others() else None
                for p in pieces(x - k, cap, table, rest_first=si == 0 and REST_FIRST):
                    if cancelled():   # the client went away (or a stop): don't prefill the rest of a long prompt
                        aborted = True
                        break
                    sample = i + p == n
                    t2 = time.perf_counter()
                    # the next piece's tokens: the engine reads their n-gram rows during this piece's last chunk
                    r = sl.b.feed(prompt[i:i + p], sample=sample, ahead=None if sample else piece_at(i + p, cap, table))
                    trace(f"feed {p} at {i} (sample {sample}): {(time.perf_counter() - t2) * 1e3:.1f} ms")
                    if sample:
                        nx = r
                    i += p
                    N.toks = list(prompt[:i])
                    if i < n:
                        yield
                if aborted:
                    break
                if x < end:
                    t1 = time.perf_counter()
                    if x in cmarks:
                        self._add_ckpt(sl)
                    if x in smarks:
                        sn = N.snaps.get(x)
                        if sn is None:
                            r = sl.b.state_save()
                            sn = Snap(x, *r) if r is not None else None
                        if sn is not None:
                            pending.append((x, sn))
                    t_snap += time.perf_counter() - t1
                    trace(f"checkpoint at {x}{' (shared prefix)' if x in smarks else ''}: "
                          f"{(time.perf_counter() - t1) * 1e3:.1f} ms")
                k = x
            sl.live = True
            if aborted:   # cancelled between prefill pieces: the state holds exactly N.toks (a valid prefix, kept)
                if pending:
                    stats["shared_made"] = self._finish_shared_safe(sl, N, pending)
                N.touch()
                self._enforce()
                stats.update(completion_tokens=0, prefill_ms=(time.perf_counter() - t0) * 1e3, decode_ms=0.0,
                             snapshot_ms=t_snap * 1e3, wall_ms=(time.perf_counter() - t0) * 1e3, cancelled="prefill")
                return stats
            if first_mtp and contended():  # another request arrived during the prefill: its last token, sampled
                nx = sl.b.feed(prompt[n - 1:], sample=True)
                N.toks = list(prompt)
                first_mtp = False
            t_dec = time.perf_counter()
            prefill_ms = (t_dec - t0) * 1e3
            out = []
            own = 0.0  # this request's own decode time (the wall time includes other slots' interleaved steps)
            n_mtp = n_step = 0
            mtp_dec_ms = 0.0
            stops = sl.b.STOPS

            def mtp_seg(new):  # state = N.toks; feeds new (>= 1 token), then MTP rounds until done / contended
                t3 = time.perf_counter()
                toks, st = sl.b.mtp_from(new, max_new - len(out), depth, sampler[3], on_tokens, cancelled,
                                         preempt=contended, ctx=N.toks)
                N.toks = N.toks + list(new) + toks[:-1]  # rows past it (rejected drafts) are stale: never trusted
                out.extend(toks)
                if st.get("decode_ms") and st.get("tokens", 0) > 32:
                    self.rates.obs("dec_mtp", st["tokens"] / (st["decode_ms"] / 1e3))
                return toks, st, time.perf_counter() - t3

            if first_mtp:  # MTP rounds from the reused state
                # (set_extra armed at the request's start and no sample has been drawn since: the feeds before an MTP run
                # draw none, so the counts are still zero)
                toks, st, dt = mtp_seg(prompt[n - 1:])
                prefill_ms += st.get("prefill_ms", 0.0)
                own += dt - st.get("prefill_ms", 0.0) / 1e3
                n_mtp += len(toks)
                mtp_dec_ms += st.get("decode_ms", 0.0)
            else:
                out.append(nx)
                n_step += 1
                trace(f"first token (step path), prefill {prefill_ms:.1f} ms")
                if want_lp:
                    on_tokens([nx], sl.b.last_top())
                else:
                    on_tokens([nx])
            if n - stats["cached_tokens"] >= 512:
                self.rates.obs("pf", (n - stats["cached_tokens"]) / max(1e-3, prefill_ms / 1e3 - t_snap))
            if out:
                yield
            while out and len(out) < max_new and not cancelled():
                nx = out[-1]
                if nx in stops or len(N.toks) + 1 > sl.ctx:
                    break
                if cont and constraint is None and not contended() and sl.b.head_ok():
                    toks, st, dt = mtp_seg([nx])  # the pending token + MTP rounds (resumed after the step path)
                    own += dt
                    n_mtp += len(toks)
                    mtp_dec_ms += st.get("decode_ms", 0.0)
                    stats["mtp_resumes"] = stats.get("mtp_resumes", 0) + 1
                    if not toks:
                        break
                    yield
                    continue
                N.toks.append(nx)
                if constraint is not None:
                    sl.b.set_mask(constraint.advance(nx))
                t3 = time.perf_counter()
                nx = sl.b.step(nx)
                dt = time.perf_counter() - t3
                own += dt
                out.append(nx)
                n_step += 1
                if want_lp:
                    on_tokens([nx], sl.b.last_top())
                else:
                    on_tokens([nx])
                yield
            N.touch()
            if n_mtp and n_step:
                stats["path"] = "mtp+step"
                stats["mtp_tokens"] = n_mtp
            else:
                stats["path"] = "mtp" if n_mtp else "step"
            if n_mtp and not n_step:
                dec_ms = mtp_dec_ms
            else:
                dec_ms = (time.perf_counter() - t_dec) * 1e3
                if n_step > 32 and not n_mtp and own > 0:
                    self.rates.obs("dec", (n_step - 1) / own)
            stats["decode_own_ms"] = own * 1e3
            if n_step > 1 and not n_mtp and not others():
                # the engine thread's own time between step calls (scheduler, callbacks, GIL waits), per token
                stats["host_us_per_token"] = max(0.0, dec_ms - own * 1e3) * 1e3 / (n_step - 1)
            self.rates.obs("out", len(out))
            if pending:
                stats["shared_made"] = self._finish_shared_safe(sl, N, pending)
            if self.cache.disk is not None:
                try:
                    stats["persist_ms"] = self._persist(sl)
                except Exception as e:   # noqa: BLE001  (e.g. MemoryError copying the KV rows out)
                    # the reply is complete: a failed write-through must not turn it into an error; the mirror
                    # catches up at the next turn (its record is only replaced after a whole write)
                    self.log(f"slot {sl.i}: SSD write-through failed after a finished reply: {e!r}")
                    traceback.print_exc(file=sys.stderr)
            self._enforce()
            stats.update(completion_tokens=len(out), prefill_ms=prefill_ms, decode_ms=dec_ms, snapshot_ms=t_snap * 1e3,
                         wall_ms=(time.perf_counter() - t0) * 1e3)
            return stats
        finally:
            if constraint is not None:  # the mask must not leak into the slot's next request
                try:
                    sl.b.set_mask(None)
                except Exception as e:  # noqa: BLE001
                    self.log(f"clearing the token mask of slot {sl.i} failed: {e}")
            if extra is not None:  # nor bias / penalties / the counts
                try:
                    sl.b.clear_extra()
                except Exception as e:  # noqa: BLE001
                    self.log(f"clearing the extra sampling parameters of slot {sl.i} failed: {e}")
            sl.busy = False
