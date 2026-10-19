# lab-layer kernels: every kernels/*.metal is compiled into build/engine.metallib by the core Makefile (no entry
# needed). Host-side dispatch wrappers live with the layer graph (models/flash_next/layer.mm) for now.
ENGINE_SRCS += kernels/fn_rec.mm  # lab-speed: recording helpers for the v2 kernels (kernels/fn_rec.h)
ENGINE_SRCS += kernels/trunk6/trunk6.mm  # lab-trunk6: EXL3 / bf16 trunk Linears (docs/trunk6.md)
-include tests/trunk6/trunk6.mk
-include kernels/nax/nax.mk
-include kernels/attn/attn.mk
-include kernels/prefill/prefill.mk
# lab-speed: per-kernel decode-step profile (leave-one-out in context), build/fn-prof PACK [reps T rounds layers]
TEST_BINS += $(BUILD)/fn-prof
$(BUILD)/fn-prof: $(BUILD)/obj/kernels/bench/fn_prof.mm.o $(BUILD)/libengine.dylib
	$(CXX) $< $(LINK_ENGINE) -o $@
-include $(BUILD)/obj/kernels/bench/fn_prof.mm.d
