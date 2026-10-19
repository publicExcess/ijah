# lab-attn: QSA attention host code (kernels/attn/qsa.mm) linked into libengine.dylib; kernels/attn/qsa.metal is
# compiled into the metallib by the core Makefile. Included from kernels/kernels.mk. Tests: tests/attn/.
ENGINE_SRCS += kernels/attn/qsa.mm kernels/attn/qsa_chunk.mm
-include tests/attn/attn_tests.mk
