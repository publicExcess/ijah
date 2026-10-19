# lab-trunk6: build/t6-check (tests/trunk6/t6_check.mm). kernels/trunk6/trunk6.metal goes into the metallib on its own.
# Until kernels/kernels.mk includes this file (request in docs/kernels.md):
#   make -f Makefile -f tests/trunk6/trunk6.mk BUILD=build-t6 build-t6/t6-check
# trunk6.mm is linked into t6-check directly unless libengine already has it (ENGINE_SRCS).
T6_OBJS := $(BUILD)/obj/tests/trunk6/t6_check.mm.o
ifeq ($(filter kernels/trunk6/trunk6.mm,$(ENGINE_SRCS)),)
T6_OBJS += $(BUILD)/obj/kernels/trunk6/trunk6.mm.o
endif
TEST_BINS += $(BUILD)/t6-check
$(BUILD)/t6-check: $(T6_OBJS) $(LIB)
	$(CXX) $(T6_OBJS) $(LINK_ENGINE) -o $@
-include $(BUILD)/obj/tests/trunk6/t6_check.mm.d $(BUILD)/obj/kernels/trunk6/trunk6.mm.d
