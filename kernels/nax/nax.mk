# lab-mtp: NAX multi-row kernels (kernels/nax/*.metal go into the metallib automatically), the MTP head
# (models/flash_next/mtp.*) and their checks (tests/mtp). Included from kernels/kernels.mk.
ENGINE_SRCS += kernels/nax/nax.mm models/flash_next/mtp.mm
TEST_BINS += $(BUILD)/mtp-nax-check
$(BUILD)/mtp-nax-check: $(BUILD)/obj/tests/mtp/nax_check.mm.o $(LIB)
	$(CXX) $< $(LINK_ENGINE) -o $@
-include $(BUILD)/obj/tests/mtp/nax_check.mm.d
TEST_BINS += $(BUILD)/mtp-accept-test $(BUILD)/mtp-head-check
$(BUILD)/mtp-accept-test: $(BUILD)/obj/tests/mtp/accept_test.mm.o $(LIB)
	$(CXX) $< $(LINK_ENGINE) -o $@
-include $(BUILD)/obj/tests/mtp/accept_test.mm.d
$(BUILD)/mtp-head-check: $(BUILD)/obj/tests/mtp/mtp_check.mm.o $(LIB)
	$(CXX) $< $(LINK_ENGINE) -o $@
-include $(BUILD)/obj/tests/mtp/mtp_check.mm.d
TEST_CMDS += $(BUILD)/mtp-accept-test
TEST_BINS += $(BUILD)/mtp-accept-eval
$(BUILD)/mtp-accept-eval: $(BUILD)/obj/tests/mtp/accept_eval.mm.o $(LIB)
	$(CXX) $< $(LINK_ENGINE) -o $@
-include $(BUILD)/obj/tests/mtp/accept_eval.mm.d
