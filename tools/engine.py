"""ctypes binding of include/engine.h (build/libengine.dylib) + the Flash-Next layer entry points lab-layer exports
(models/flash_next/layer.h). Owner: lab-ref. numpy only; no MLX.

    import engine as E
    eng = E.Engine()                                  # loads build/libengine.dylib (NE_LIB overrides)
    eng.device()                                      # dict: name, has_nax, has_metal4, backend, ...
    b = eng.from_numpy(np.arange(16, dtype=np.float32)); b.numpy(np.float32, (16,)); b.free()
    p = eng.pack_open("~/.cache/ijah/packs/Qwen3.8-Flash-Next-EXL3-Ijah")
    t = p.info("language_model.model.layers.0.linear_attn.A_log")   # TensorDesc: dtype, shape, host (np view), buf
    ms = eng.run_kernel("my_kernel", [b, (b2, 0), np.uint32(n)], grid=(g, 1, 1), tg=(256, 1, 1),
                        fconsts=[(0, "u32", 2560)], reps=1)
    L = eng.fn_layer(p, 0); L.set_state(conv, ssm); h_out = L.step(h_in); L.read("qkv")
    eng.fn("ne_other_entry", ctypes.c_int, [ctypes.c_void_p, ...])(...)   # any other NE_API symbol
"""
import ctypes as C
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refenv  # noqa: E402

refenv.add_paths()
import numpy as np  # noqa: E402

LIB = Path(os.environ.get("NE_LIB", refenv.ROOT / "build/libengine.dylib"))
MAX_DIMS = 8
DTYPES = {0: "F32", 1: "F16", 2: "BF16", 3: "U8", 4: "U16", 5: "U32", 6: "I32", 7: "I64", 8: "I8", 9: "I16",
          10: "U64", 11: "F64", 12: "BOOL"}
NP = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "U8": np.uint8, "U16": np.uint16, "U32": np.uint32,
      "I32": np.int32, "I64": np.int64, "I8": np.int8, "I16": np.int16, "U64": np.uint64, "F64": np.float64,
      "BOOL": np.uint8}
FC_TYPES = {"u32": 0, "i32": 1, "f32": 2, "bool": 3, "u16": 4, "i16": 5}


class DeviceInfo(C.Structure):
    _fields_ = [("name", C.c_char * 64), ("has_nax", C.c_int), ("has_metal4", C.c_int),
                ("max_buffer_length", C.c_uint64), ("recommended_working_set", C.c_uint64),
                ("max_threadgroup_memory", C.c_uint64), ("backend", C.c_char * 16)]


class TensorDescC(C.Structure):
    _fields_ = [("dtype", C.c_int), ("ndim", C.c_int), ("shape", C.c_int64 * MAX_DIMS), ("nbytes", C.c_uint64),
                ("file", C.c_char_p), ("file_offset", C.c_uint64), ("host", C.c_void_p), ("buf", C.c_void_p),
                ("buf_offset", C.c_uint64)]


class FConst(C.Structure):
    _fields_ = [("index", C.c_uint32), ("type", C.c_uint32), ("bits", C.c_uint32)]


class ArgC(C.Structure):
    _fields_ = [("buf", C.c_void_p), ("offset", C.c_uint64), ("bytes", C.c_void_p), ("nbytes", C.c_uint32)]


class EngineError(RuntimeError):
    pass


def bf16_to_f32(u16):
    return (np.asarray(u16, np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x):
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) >> 16).astype(np.uint16)


class Buf:
    """An ne_buf handle. owned=False for pack-owned views (never freed)."""

    def __init__(self, eng, ptr, owned=True):
        self.eng, self.ptr, self.owned = eng, ptr, owned

    @property
    def size(self):
        return self.eng.lib.ne_buf_size(self.ptr)

    def contents(self):
        return self.eng.lib.ne_buf_contents(self.ptr)

    def gpu_address(self):
        return self.eng.lib.ne_buf_gpu_address(self.ptr)

    def upload(self, arr, offset=0):
        a = np.ascontiguousarray(arr)
        self.eng._ok(self.eng.lib.ne_upload(self.ptr, offset, a.ctypes.data, a.nbytes), "ne_upload")

    def download(self, dtype=np.float32, shape=None, offset=0):
        dtype = np.dtype(dtype)
        n = (self.size - offset) // dtype.itemsize if shape is None else int(np.prod(shape))
        out = np.empty(n, dtype)
        self.eng._ok(self.eng.lib.ne_download(self.ptr, offset, out.ctypes.data, out.nbytes), "ne_download")
        return out.reshape(shape) if shape is not None else out

    def numpy(self, dtype=np.float32, shape=None, offset=0):
        """Zero-copy CPU view of the shared buffer (valid while the buffer lives; sync with the GPU yourself)."""
        dtype = np.dtype(dtype)
        n = (self.size - offset) // dtype.itemsize if shape is None else int(np.prod(shape))
        raw = (C.c_char * (n * dtype.itemsize)).from_address(self.contents() + offset)
        a = np.frombuffer(raw, dtype, n)
        return a.reshape(shape) if shape is not None else a

    def free(self):
        if self.ptr and self.owned:
            self.eng.lib.ne_free(self.ptr)
        self.ptr = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


class TensorDesc:
    def __init__(self, eng, name, d):
        self.name = name
        self.dtype = DTYPES.get(d.dtype, str(d.dtype))
        self.shape = tuple(d.shape[i] for i in range(d.ndim))
        self.nbytes = d.nbytes
        self.file = d.file.decode() if d.file else None
        self.file_offset = d.file_offset
        self._host = d.host
        self.buf = Buf(eng, d.buf, owned=False) if d.buf else None
        self.buf_offset = d.buf_offset

    @property
    def host(self):
        """Read-only numpy view of the mmapped bytes (BF16 as uint16)."""
        dt = np.dtype(NP[self.dtype])
        raw = (C.c_char * self.nbytes).from_address(self._host)
        a = np.frombuffer(raw, dt, self.nbytes // dt.itemsize).reshape(self.shape)
        a.flags.writeable = False
        return a

    def f32(self):
        h = self.host
        return bf16_to_f32(h) if self.dtype == "BF16" else h.astype(np.float32)

    def arg(self):
        """(buf, offset) for run_kernel."""
        return (self.buf, self.buf_offset)

    def __repr__(self):
        return f"TensorDesc({self.name}, {self.dtype}, {self.shape})"


class Pack:
    def __init__(self, eng, path):
        self.eng = eng
        self.path = str(Path(path).expanduser())
        self.ptr = eng.lib.ne_pack_open(self.path.encode())
        if not self.ptr:
            raise EngineError(f"ne_pack_open({self.path}): {eng.last_error()}")

    def config(self):
        return json.loads(self.eng.lib.ne_pack_config_json(self.ptr).decode())

    def names(self):
        n = self.eng.lib.ne_pack_num_tensors(self.ptr)
        return [self.eng.lib.ne_pack_tensor_name(self.ptr, i).decode() for i in range(n)]

    def info(self, name):
        d = TensorDescC()
        if self.eng.lib.ne_tensor_info(self.ptr, name.encode(), C.byref(d)) != 0:
            raise KeyError(f"{name}: {self.eng.last_error()}")
        return TensorDesc(self.eng, name, d)

    def view_rows(self, name, row0, nrows):
        off = C.c_uint64(0)
        p = self.eng.lib.ne_pack_view_rows(self.ptr, name.encode(), row0, nrows, C.byref(off))
        if not p:
            raise EngineError(self.eng.last_error())
        return Buf(self.eng, p), off.value

    def close(self):
        if self.ptr:
            self.eng.lib.ne_pack_close(self.ptr)
        self.ptr = None


class FnLayer:
    """models/flash_next/layer.h: one Flash-Next decoder layer (rows = 1, 2 or 4 tokens per step), f32 host I/O."""
    HCD = 4 * 2560
    CONV = 3 * 10240
    SSM = 48 * 128 * 128

    def __init__(self, eng, pack, layer, rows=1):
        self.eng = eng
        self.ptr = eng.lib.ne_fn_layer_create_rows(pack.ptr, layer, rows)
        if not self.ptr:
            raise EngineError(f"ne_fn_layer_create_rows(L{layer}, {rows}): {eng.last_error()}")
        self.layer = layer
        self.rows = rows
        self.HCD = 4 * 2560 * rows

    def set_kv(self, k, v, ikeys, T):
        """attention layers: k, v f32 [2, T, 256] (keys roped), ikeys f32 [T, 128]; sets pos = T"""
        a = [np.ascontiguousarray(x, np.float32) for x in (k, v, ikeys)]
        self.eng._ok(self.eng.lib.ne_fn_layer_set_kv(self.ptr, a[0].ctypes.data, a[1].ctypes.data, a[2].ctypes.data,
                                                     int(T)), "set_kv")

    def get_kv(self, T):
        k = np.empty((2, T, 256), np.float32)
        v = np.empty((2, T, 256), np.float32)
        ik = np.empty((T, 128), np.float32)
        n = C.c_int(0)
        self.eng._ok(self.eng.lib.ne_fn_layer_get_kv(self.ptr, k.ctypes.data, v.ctypes.data, ik.ctypes.data,
                                                     C.byref(n)), "get_kv")
        assert n.value == T, (n.value, T)
        return k, v, ik

    def _f(self, a, n):
        a = np.ascontiguousarray(a, np.float32).reshape(-1)
        assert a.size == n, (a.size, n)
        return a

    def set_state(self, conv=None, ssm=None):
        c = self._f(conv, self.CONV) if conv is not None else None
        s = self._f(ssm, self.SSM) if ssm is not None else None
        self.eng._ok(self.eng.lib.ne_fn_layer_set_state(self.ptr, c.ctypes.data if c is not None else None,
                                                        s.ctypes.data if s is not None else None), "set_state")

    def get_state(self):
        c = np.empty(self.CONV, np.float32)
        s = np.empty(self.SSM, np.float32)
        self.eng._ok(self.eng.lib.ne_fn_layer_get_state(self.ptr, c.ctypes.data, s.ctypes.data), "get_state")
        return c, s

    def step(self, h_in=None):
        h = self._f(h_in, self.HCD) if h_in is not None else None
        out = np.empty(self.HCD, np.float32)
        ms = C.c_double(0)
        self.eng._ok(self.eng.lib.ne_fn_layer_step(self.ptr, h.ctypes.data if h is not None else None,
                                                   out.ctypes.data, C.byref(ms)), "step")
        self.gpu_ms = ms.value
        return out

    def run_stage(self, stage):
        self.eng._ok(self.eng.lib.ne_fn_layer_run_stage(self.ptr, int(stage)), f"run_stage({stage})")

    def size(self, name):
        n = self.eng.lib.ne_fn_layer_read(self.ptr, name.encode(), None, 0)
        if n < 0:
            raise KeyError(f"{name}: {self.eng.last_error()}")
        return n

    def read(self, name):
        n = self.size(name)
        out = np.empty(n, np.float32)
        self.eng.lib.ne_fn_layer_read(self.ptr, name.encode(), out.ctypes.data, n)
        return out

    def write(self, name, arr):
        a = np.ascontiguousarray(arr, np.float32).reshape(-1)
        self.eng._ok(self.eng.lib.ne_fn_layer_write(self.ptr, name.encode(), a.ctypes.data, a.size), f"write({name})")

    def time(self, reps1=8, reps2=40, flush_mb=0):
        us, fl = C.c_double(0), C.c_double(0)
        self.eng._ok(self.eng.lib.ne_fn_layer_time(self.ptr, reps1, reps2, flush_mb, C.byref(us), C.byref(fl)), "time")
        return us.value, fl.value

    def free(self):
        if self.ptr:
            self.eng.lib.ne_fn_layer_free(self.ptr)
        self.ptr = None


class Engine:
    def __init__(self, lib=None):
        path = Path(lib or LIB)
        if not path.exists():
            raise EngineError(f"{path} not built (run make)")
        self.lib = C.CDLL(str(path))
        L = self.lib
        sig = {
            "ne_abi_version": (C.c_int, []), "ne_last_error": (C.c_char_p, []),
            "ne_dtype_name": (C.c_char_p, [C.c_int]), "ne_dtype_size": (C.c_int, [C.c_int]),
            "ne_device_query": (C.c_int, [C.POINTER(DeviceInfo)]),
            "ne_alloc": (C.c_void_p, [C.c_uint64]), "ne_free": (None, [C.c_void_p]),
            "ne_buf_size": (C.c_uint64, [C.c_void_p]), "ne_buf_contents": (C.c_void_p, [C.c_void_p]),
            "ne_buf_gpu_address": (C.c_uint64, [C.c_void_p]),
            "ne_upload": (C.c_int, [C.c_void_p, C.c_uint64, C.c_void_p, C.c_uint64]),
            "ne_download": (C.c_int, [C.c_void_p, C.c_uint64, C.c_void_p, C.c_uint64]),
            "ne_pack_open": (C.c_void_p, [C.c_char_p]), "ne_pack_close": (None, [C.c_void_p]),
            "ne_pack_config_json": (C.c_char_p, [C.c_void_p]), "ne_pack_num_tensors": (C.c_int64, [C.c_void_p]),
            "ne_pack_tensor_name": (C.c_char_p, [C.c_void_p, C.c_int64]),
            "ne_tensor_info": (C.c_int, [C.c_void_p, C.c_char_p, C.POINTER(TensorDescC)]),
            "ne_pack_view_rows": (C.c_void_p, [C.c_void_p, C.c_char_p, C.c_int64, C.c_int64, C.POINTER(C.c_uint64)]),
            "ne_kernel_exists": (C.c_int, [C.c_char_p]),
            "ne_run_kernel": (C.c_int, [C.c_char_p, C.POINTER(FConst), C.c_int, C.POINTER(ArgC), C.c_int,
                                        C.POINTER(C.c_uint32 * 3), C.POINTER(C.c_uint32 * 3), C.c_uint32, C.c_int,
                                        C.c_char_p, C.POINTER(C.c_double)]),
            # models/flash_next/layer.h (lab-layer); bound only if exported
            "ne_fn_layer_create": (C.c_void_p, [C.c_void_p, C.c_int]), "ne_fn_layer_free": (None, [C.c_void_p]),
            "ne_fn_layer_set_state": (C.c_int, [C.c_void_p, C.c_void_p, C.c_void_p]),
            "ne_fn_layer_get_state": (C.c_int, [C.c_void_p, C.c_void_p, C.c_void_p]),
            "ne_fn_layer_step": (C.c_int, [C.c_void_p, C.c_void_p, C.c_void_p, C.POINTER(C.c_double)]),
            "ne_fn_layer_run_stage": (C.c_int, [C.c_void_p, C.c_int]),
            "ne_fn_layer_write": (C.c_int, [C.c_void_p, C.c_char_p, C.c_void_p, C.c_int]),
            "ne_fn_layer_read": (C.c_int, [C.c_void_p, C.c_char_p, C.c_void_p, C.c_int]),
            "ne_fn_layer_time": (C.c_int, [C.c_void_p, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_double),
                                           C.POINTER(C.c_double)]),
            "ne_fn_layer_create_rows": (C.c_void_p, [C.c_void_p, C.c_int, C.c_int]),
            "ne_fn_layer_set_kv": (C.c_int, [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int]),
            "ne_fn_layer_get_kv": (C.c_int, [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.POINTER(C.c_int)]),
            "ne_fn_set_ref_rounding": (C.c_int, [C.c_int]),
            # models/flash_next/model.h (lab-forward)
            "ne_fn_model_create": (C.c_void_p, [C.c_void_p, C.c_int]), "ne_fn_model_free": (None, [C.c_void_p]),
            "ne_fn_model_reset": (C.c_int, [C.c_void_p]),
            "ne_fn_model_set_sampler": (C.c_int, [C.c_void_p, C.c_float, C.c_float, C.c_int, C.c_uint32]),
            "ne_fn_model_set_mask": (C.c_int, [C.c_void_p, C.c_void_p]),   # lab-api: allowed-token bits / NULL
            # lab-sampler: min_p / penalties / logit_bias / RNG counter / candidate list (model.h)
            "ne_fn_model_set_sampler_ex": (C.c_int, [C.c_void_p, C.c_float, C.c_float, C.c_float, C.c_float, C.c_int]),
            "ne_fn_model_set_seen": (C.c_int, [C.c_void_p, C.c_void_p, C.c_int]),
            "ne_fn_model_set_logit_bias": (C.c_int, [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int]),
            "ne_fn_model_set_rng_step": (C.c_int, [C.c_void_p, C.c_uint32]),
            "ne_fn_model_top": (C.c_int, [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int]),
            "ne_fn_model_step": (C.c_int, [C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.POINTER(C.c_int),
                                           C.POINTER(C.c_float), C.POINTER(C.c_int), C.POINTER(C.c_double)]),
            "ne_fn_model_generate": (C.c_int, [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_void_p,
                                               C.c_void_p, C.c_void_p]),
            "ne_fn_model_time": (C.c_int, [C.c_void_p, C.c_int, C.c_int, C.POINTER(C.c_double),
                                           C.POINTER(C.c_double), C.POINTER(C.c_double)]),
            "ne_fn_model_info": (C.c_int, [C.c_void_p, C.POINTER(C.c_int), C.POINTER(C.c_int), C.POINTER(C.c_int)]),
            "ne_fn_model_run_head": (C.c_int, [C.c_void_p, C.c_void_p, C.c_void_p, C.POINTER(C.c_int),
                                               C.POINTER(C.c_float)]),
            "ne_fn_ngram": (C.c_int, [C.c_void_p, C.c_int64, C.c_int64, C.c_int64, C.c_void_p, C.c_void_p]),
            "ne_fn_model_prefill": (C.c_int, [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.POINTER(C.c_int),
                                              C.POINTER(C.c_double)]),
        }
        self.missing = []
        for name, (res, args) in sig.items():
            try:
                f = getattr(L, name)
            except AttributeError:
                self.missing.append(name)
                continue
            f.restype, f.argtypes = res, args

    def has(self, name):
        try:
            getattr(self.lib, name)
            return True
        except AttributeError:
            return False

    def fn(self, name, restype, argtypes):
        """Any other exported NE_API function."""
        f = getattr(self.lib, name)
        f.restype, f.argtypes = restype, argtypes
        return f

    def last_error(self):
        e = self.lib.ne_last_error()
        return e.decode() if e else ""

    def _ok(self, rc, what):
        if rc != 0:
            raise EngineError(f"{what}: {self.last_error()}")

    def abi_version(self):
        return self.lib.ne_abi_version()

    def device(self):
        d = DeviceInfo()
        self._ok(self.lib.ne_device_query(C.byref(d)), "ne_device_query")
        return {"name": d.name.decode(), "has_nax": bool(d.has_nax), "has_metal4": bool(d.has_metal4),
                "max_buffer_length": d.max_buffer_length, "recommended_working_set": d.recommended_working_set,
                "max_threadgroup_memory": d.max_threadgroup_memory, "backend": d.backend.decode()}

    def alloc(self, nbytes):
        p = self.lib.ne_alloc(int(nbytes))
        if not p:
            raise EngineError(f"ne_alloc({nbytes}): {self.last_error()}")
        return Buf(self, p)

    def from_numpy(self, arr):
        a = np.ascontiguousarray(arr)
        b = self.alloc(max(a.nbytes, 1))
        b.upload(a)
        return b

    def pack_open(self, path=None):
        return Pack(self, path or refenv.PACK)

    def kernel_exists(self, fn):
        return self.lib.ne_kernel_exists(fn.encode()) == 1

    def run_kernel(self, fn, args, grid, tg, fconsts=(), tg_mem=0, reps=1, backend=None):
        """args: Buf | (Buf, offset) | TensorDesc | bytes | numpy scalar/array (inline bytes). fconsts: (index, type,
        value) with type in u32/i32/f32/bool/u16/i16. Returns GPU ms of the whole command buffer."""
        keep = []
        ca = (ArgC * max(len(args), 1))()
        for i, a in enumerate(args):
            if isinstance(a, TensorDesc):
                a = a.arg()
            if isinstance(a, Buf):
                ca[i] = ArgC(a.ptr, 0, None, 0)
            elif isinstance(a, tuple) and isinstance(a[0], Buf):
                ca[i] = ArgC(a[0].ptr, int(a[1]), None, 0)
            else:
                raw = a if isinstance(a, (bytes, bytearray)) else np.ascontiguousarray(a).tobytes()
                cb = C.create_string_buffer(bytes(raw), len(raw))
                keep.append(cb)
                ca[i] = ArgC(None, 0, C.cast(cb, C.c_void_p), len(raw))
        fc = (FConst * max(len(fconsts), 1))()
        for i, (idx, ty, val) in enumerate(fconsts):
            t = FC_TYPES[ty]
            bits = int(np.float32(val).view(np.uint32)) if ty == "f32" else int(val) & 0xFFFFFFFF
            fc[i] = FConst(idx, t, bits)
        g = (C.c_uint32 * 3)(*grid)
        t3 = (C.c_uint32 * 3)(*tg)
        ms = C.c_double(0)
        rc = self.lib.ne_run_kernel(fn.encode(), fc, len(fconsts), ca, len(args), C.byref(g), C.byref(t3), tg_mem,
                                    reps, backend.encode() if backend else None, C.byref(ms))
        self._ok(rc, f"ne_run_kernel({fn})")
        return ms.value

    def fn_layer(self, pack, layer, rows=1):
        if "ne_fn_layer_create" in self.missing:
            raise EngineError("libengine.dylib has no ne_fn_layer_* (models/flash_next not built in)")
        return FnLayer(self, pack, layer, rows)

    def set_ref_rounding(self, on):
        """reference bf16 rounding points on/off for layers / models created afterwards"""
        self._ok(self.lib.ne_fn_set_ref_rounding(1 if on else 0), "set_ref_rounding")

    def vision(self, pack):
        return Vision(self, pack)

    def fn_model(self, pack, tmax=8192):
        if "ne_fn_model_create" in self.missing:
            raise EngineError("libengine.dylib has no ne_fn_model_*")
        return FnModel(self, pack, tmax)

    def ngram(self, pack, p2, p1, t, gather=True):
        """C++ n-gram ids (int64 [16]) of token t after (p2, p1) and the gathered row f32 [2560] (or None)"""
        ids = np.empty(16, np.int64)
        emb = np.empty(2560, np.float32) if gather else None
        self._ok(self.lib.ne_fn_ngram(pack.ptr, int(p2), int(p1), int(t), ids.ctypes.data,
                                      emb.ctypes.data if gather else None), "ngram")
        return ids, emb


class FnModel:
    """models/flash_next/model.h: the whole Flash-Next model, one recorded decode step per token."""
    VOCAB = 248320

    def __init__(self, eng, pack, tmax=8192):
        self.eng = eng
        self.pack = pack
        self.ptr = eng.lib.ne_fn_model_create(pack.ptr, int(tmax))
        if not self.ptr:
            raise EngineError(f"ne_fn_model_create: {eng.last_error()}")
        self.tmax = tmax

    def reset(self):
        self.eng._ok(self.eng.lib.ne_fn_model_reset(self.ptr), "reset")

    def set_sampler(self, temp=0.0, top_p=1.0, top_k=1, seed=0):
        self.eng._ok(self.eng.lib.ne_fn_model_set_sampler(self.ptr, float(temp), float(top_p), int(top_k),
                                                          int(seed) & 0xFFFFFFFF), "set_sampler")

    def set_mask(self, bits=None):
        """allowed-token mask for the sampler (uint32/int32 words, bit i = token i; (VOCAB + 31) // 32 of them); None clears"""
        if bits is None:
            self.eng._ok(self.eng.lib.ne_fn_model_set_mask(self.ptr, None), "set_mask")
            return
        a = np.ascontiguousarray(bits)
        assert a.dtype.itemsize == 4 and a.size == (self.VOCAB + 31) // 32, (a.dtype, a.size)
        self.eng._ok(self.eng.lib.ne_fn_model_set_mask(self.ptr, a.ctypes.data), "set_mask")

    # -- lab-sampler (docs/server.md section 11)
    def set_sampler_ex(self, min_p=0.0, presence=0.0, frequency=0.0, repetition=1.0, logprob_k=0):
        """min_p, penalties (zeroes the generated-token counts and the prompt flags), candidate count for top()"""
        self.eng._ok(self.eng.lib.ne_fn_model_set_sampler_ex(self.ptr, float(min_p), float(presence), float(frequency),
                                                             float(repetition), int(logprob_k)), "set_sampler_ex")

    def set_seen(self, ids):
        """repetition penalty: the prompt's token ids (after set_sampler_ex)"""
        a = np.ascontiguousarray(ids, np.int32)
        self.eng._ok(self.eng.lib.ne_fn_model_set_seen(self.ptr, a.ctypes.data, len(a)), "set_seen")

    def set_logit_bias(self, bias=None):
        """{token id: bias}: added to the logits (<= -100: banned); None / {} clears"""
        if not bias:
            self.eng._ok(self.eng.lib.ne_fn_model_set_logit_bias(self.ptr, None, None, 0), "set_logit_bias")
            return
        ids = np.ascontiguousarray(list(bias.keys()), np.int32)
        v = np.ascontiguousarray(list(bias.values()), np.float32)
        self.eng._ok(self.eng.lib.ne_fn_model_set_logit_bias(self.ptr, ids.ctypes.data, v.ctypes.data, len(ids)),
                     "set_logit_bias")

    def set_rng_step(self, step=0):
        self.eng._ok(self.eng.lib.ne_fn_model_set_rng_step(self.ptr, int(step) & 0xFFFFFFFF), "set_rng_step")

    def top(self, k=64):
        """the last sample's candidates (needs logprob_k > 0): ([token ids], [log-probabilities])"""
        ids = np.empty(k, np.int32)
        lp = np.empty(k, np.float32)
        n = self.eng.lib.ne_fn_model_top(self.ptr, ids.ctypes.data, lp.ctypes.data, int(k))
        if n < 0:
            raise EngineError(f"top: {self.eng.last_error()}")
        return ids[:n].tolist(), lp[:n].tolist()

    def step(self, token, head=True, logits=False):
        """one token at the current position -> (next token, its logprob, argmax token, logits or None)"""
        lg = np.empty(self.VOCAB, np.float32) if (logits and head) else None
        nx, am, lp, ms = C.c_int(-1), C.c_int(-1), C.c_float(0), C.c_double(0)
        self.eng._ok(self.eng.lib.ne_fn_model_step(self.ptr, int(token), 1 if head else 0,
                                                   lg.ctypes.data if lg is not None else None, C.byref(nx),
                                                   C.byref(lp), C.byref(am), C.byref(ms)), "step")
        self.gpu_ms = ms.value
        return nx.value, lp.value, am.value, lg

    def prefill(self, prompt, logits=False):
        """prompt ids (4-row chunks + 1-row steps) -> (next token, logits of the last position or None, wall ms)"""
        p = np.ascontiguousarray(prompt, np.int32)
        lg = np.empty(self.VOCAB, np.float32) if logits else None
        nx, ms = C.c_int(-1), C.c_double(0)
        self.eng._ok(self.eng.lib.ne_fn_model_prefill(self.ptr, p.ctypes.data, len(p),
                                                      lg.ctypes.data if lg is not None else None, C.byref(nx),
                                                      C.byref(ms)), "prefill")
        return nx.value, lg, ms.value

    def generate(self, prompt, max_new=64, stop=-1):
        """prompt ids -> (generated ids, logprobs, stats dict)"""
        p = np.ascontiguousarray(prompt, np.int32)
        out = np.empty(max_new, np.int32)
        lp = np.empty(max_new, np.float32)
        st = np.zeros(6, np.float64)
        n = self.eng.lib.ne_fn_model_generate(self.ptr, p.ctypes.data, len(p), int(max_new), int(stop),
                                              out.ctypes.data, lp.ctypes.data, st.ctypes.data)
        if n < 0:
            raise EngineError(f"generate: {self.eng.last_error()}")
        keys = ("prefill_ms", "decode_ms", "decode_steps", "gpu_ms_mean", "gather_us_mean", "dispatches")
        return out[:n].tolist(), lp[:n].tolist(), dict(zip(keys, st.tolist()))

    def time(self, T, reps=20):
        med, mn, wall = C.c_double(0), C.c_double(0), C.c_double(0)
        self.eng._ok(self.eng.lib.ne_fn_model_time(self.ptr, int(T), int(reps), C.byref(med), C.byref(mn),
                                                   C.byref(wall)), "time")
        return {"T": T, "gpu_ms_median": med.value, "gpu_ms_min": mn.value, "wall_ms_median": wall.value}

    def info(self):
        a, b, t = C.c_int(0), C.c_int(0), C.c_int(0)
        self.eng._ok(self.eng.lib.ne_fn_model_info(self.ptr, C.byref(a), C.byref(b), C.byref(t)), "info")
        return {"dispatches_full": a.value, "dispatches_body": b.value, "T": t.value}

    def run_head(self, h):
        h = np.ascontiguousarray(h, np.float32).reshape(-1)
        assert h.size == 4 * 2560
        lg = np.empty(self.VOCAB, np.float32)
        nx, lp = C.c_int(-1), C.c_float(0)
        self.eng._ok(self.eng.lib.ne_fn_model_run_head(self.ptr, h.ctypes.data, lg.ctypes.data, C.byref(nx),
                                                       C.byref(lp)), "run_head")
        return lg, nx.value, lp.value

    def free(self):
        if self.ptr:
            self.eng.lib.ne_fn_model_free(self.ptr)
            self.ptr = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


class Vision:
    """lab-vision: the pack's vision tower (models/flash_next/vision.h)"""

    def __init__(self, eng, pack):
        self.eng = eng
        L = eng.lib
        L.ne_vision_create.restype, L.ne_vision_create.argtypes = C.c_void_p, [C.c_void_p]
        L.ne_vision_free.restype, L.ne_vision_free.argtypes = None, [C.c_void_p]
        L.ne_vision_encode.restype = C.c_int
        L.ne_vision_encode.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_int, C.c_void_p, C.c_void_p,
                                       C.POINTER(C.c_double)]
        self.ptr = L.ne_vision_create(pack.ptr)
        if not self.ptr:
            raise EngineError(f"ne_vision_create: {eng.last_error()}")
        self.gpu_ms = 0.0

    def encode(self, patches, grids, hidden=False):
        """patches f32 [N][1536] (engine order), grids [(1, h, w)] -> bf16 bits u16 [N/4][2560] (, hidden [N][1152])"""
        x = np.ascontiguousarray(patches, np.float32)
        g = np.ascontiguousarray(np.asarray(grids, np.int32).reshape(-1, 3))
        out = np.zeros((x.shape[0] // 4, 2560), np.uint16)
        hid = np.zeros((x.shape[0], 1152), np.uint16) if hidden else None
        ms = C.c_double(0)
        rc = self.eng.lib.ne_vision_encode(self.ptr, x.ctypes.data, x.shape[0], g.ctypes.data, g.shape[0],
                                           out.ctypes.data, hid.ctypes.data if hidden else None, C.byref(ms))
        self.eng._ok(rc, "vision encode")
        self.gpu_ms = ms.value
        return (out, hid) if hidden else out

    def free(self):
        if self.ptr:
            self.eng.lib.ne_vision_free(self.ptr)
            self.ptr = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


if __name__ == "__main__":
    e = Engine()
    print("abi", e.abi_version(), e.device())
    print("missing symbols:", e.missing or "none")
    b = e.from_numpy(np.arange(8, dtype=np.float32))
    assert np.array_equal(b.download(np.float32, (8,)), np.arange(8, dtype=np.float32))
    assert np.array_equal(b.numpy(np.float32, (8,)), np.arange(8, dtype=np.float32))
    b.free()
    p = e.pack_open()
    t = p.info("language_model.model.layers.0.linear_attn.A_log")
    print(t, t.f32()[:4], "buf", bool(t.buf), "tensors", len(p.names()))
    p.close()
    print("engine.py self-check OK")
