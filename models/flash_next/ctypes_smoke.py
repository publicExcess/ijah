# lab-layer: ctypes smoke test of ne_fn_layer_* vs the L00_decode1 golden. Run from the repo root:
#   .venv/bin/python -I -B models/flash_next/ctypes_smoke.py
import ctypes as C, sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "tools"))
import refenv; refenv.add_paths(mlx=False)
import numpy as np, golden, engine
E = engine.Engine()
lib = E.lib
pk = engine.Pack(E, refenv.ijah_paths.golden_pack())
lib.ne_fn_layer_create.restype = C.c_void_p
lib.ne_fn_layer_create.argtypes = [C.c_void_p, C.c_int]
fp = C.POINTER(C.c_float)
for n in ("ne_fn_layer_set_state", "ne_fn_layer_get_state"): getattr(lib, n).argtypes = [C.c_void_p, fp, fp]
lib.ne_fn_layer_step.argtypes = [C.c_void_p, fp, fp, C.POINTER(C.c_double)]
lib.ne_fn_layer_read.argtypes = [C.c_void_p, C.c_char_p, fp, C.c_int]
handle = pk.ptr
l = lib.ne_fn_layer_create(handle, 0)
assert l, lib.ne_last_error()
c = golden.load("L00_decode1")
f = lambda a: np.ascontiguousarray(a, np.float32).ravel()
p = lambda a: a.ctypes.data_as(fp)
conv, ssm, h = f(c["gdn.conv_state_in"]), f(c["gdn.ssm_state_in"]), f(c["hc_in"])
assert lib.ne_fn_layer_set_state(l, p(conv), p(ssm)) == 0
out = np.zeros(10240, np.float32); ms = C.c_double()
assert lib.ne_fn_layer_step(l, p(h), p(out), C.byref(ms)) == 0
g = f(c["hc_out"])
print("ctypes h_out rel L2 %.3e, gpu %.3f ms" % (np.linalg.norm(out - g) / np.linalg.norm(g), ms.value))
# determinism: same state + input again -> bit-identical
assert lib.ne_fn_layer_set_state(l, p(conv), p(ssm)) == 0
out2 = np.zeros_like(out); lib.ne_fn_layer_step(l, p(h), p(out2), None)
print("bit-identical rerun:", np.array_equal(out, out2))
