# Flash-Next layer graph (lab-layer): linked into libengine.dylib; build/fn-layer-check compares one layer step with
# lab-ref's goldens:  build/fn-layer-check PACK tests/golden/data/L00_decode1 [chain|stage|time|all] [layer]
# (This file is included before the Makefile defines `all` and LIB: keep `all` the default goal, spell the lib path.)
.DEFAULT_GOAL := all
ENGINE_SRCS += models/flash_next/layer.mm models/flash_next/model.mm models/flash_next/mtp_loop.mm
ENGINE_SRCS += models/flash_next/vision.mm  # lab-vision: the vision tower
ENGINE_SRCS += models/flash_next/layer_t6.mm  # lab-t6wire: EXL3 / bf16 trunk Linears in the layer graph
ENGINE_SRCS += models/flash_next/kvio.mm
ENGINE_SRCS += models/flash_next/dflash.mm  # lab-dflash: the DFlash2 block drafter (docs/mtp.md 13)  # lab-srvcache: KV rows / recurrent state <-> host buffers (server prefix cache)
TEST_BINS += $(BUILD)/fn-layer-check
$(BUILD)/fn-layer-check: $(BUILD)/obj/models/flash_next/fn_layer_check.mm.o $(BUILD)/libengine.dylib
	$(CXX) $< $(LINK_ENGINE) -o $@
-include $(BUILD)/obj/models/flash_next/fn_layer_check.mm.d
# lab-loop: MTP token loop kernels check (layer 1 PLE weights only): build/mtp-loop-check PACK
TEST_BINS += $(BUILD)/mtp-loop-check
$(BUILD)/mtp-loop-check: $(BUILD)/obj/tests/mtp/loop_check.mm.o $(BUILD)/libengine.dylib
	$(CXX) $< $(LINK_ENGINE) -o $@
-include $(BUILD)/obj/tests/mtp/loop_check.mm.d
