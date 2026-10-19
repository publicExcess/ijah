# lab-prefill: chunked prefill (kernels/prefill/*.metal go into the metallib automatically; models/flash_next/prefill.mm
# into libengine) and its checks (tests/prefill). Included from kernels/kernels.mk.
ENGINE_SRCS += models/flash_next/prefill.mm
TEST_BINS += $(BUILD)/pf-check
$(BUILD)/pf-check: $(BUILD)/obj/tests/prefill/pf_check.mm.o $(LIB)
	$(CXX) $< $(LINK_ENGINE) -o $@
-include $(BUILD)/obj/tests/prefill/pf_check.mm.d
