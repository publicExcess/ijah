# Ijah build. Other directories hook in through kernels/kernels.mk, models/*/model.mk and tests/tests.mk (included
# below) -- see docs/core.md "Build".
#
#   make            build/libengine.dylib, build/engine.metallib, build/engine-cli, build/core-test
#   make venv       .venv with requirements.txt (python3.11+; PYTHON=... picks the interpreter)
#   make test       core self-tests (+ TEST_CMDS from tests/tests.mk; pack-dependent parts skip without a pack)
#   make test-cpu   the CPU unit tests of the server, chat layer, sampling, vision preprocessing (no GPU, no model)
#   make bench      dispatch cost per recorder backend (takes the optional machine lock, IJAH_LOCK_SCRIPT)
#   make clean
#
# Requirements: an Apple M5 GPU (Metal 4 tensor units; M1-M4 are not supported), macOS 27, Xcode command-line tools
# with the Metal toolchain (xcodebuild -downloadComponent MetalToolchain).

BUILD     := build
CXX       := clang++
CXXSTD    := -std=c++20
OPT       ?= -O2
CXXFLAGS  := $(CXXSTD) -fobjc-arc $(OPT) -g -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
             -I. -Iinclude -MMD -MP $(EXTRA_CXXFLAGS)
FRAMEWORKS := -framework Metal -framework Foundation
# xcrun sometimes can't locate the downloaded Metal toolchain from sandboxed shells (17 Oct); fall back to the
# toolchain's own binary (its search path from xcodebuild) with the macOS SDK as sysroot.
# Evaluated once per make (simply expanded), and xcodebuild only when xcrun can't find the compiler.
ifndef METAL
METAL := $(shell xcrun -sdk macosx metal --version >/dev/null 2>&1 && echo "xcrun -sdk macosx metal")
ifeq ($(METAL),)
METAL_TC := $(shell xcodebuild -showComponent MetalToolchain 2>/dev/null | sed -n 's/^Toolchain Search Path: //p')
METAL := $(METAL_TC)/Metal.xctoolchain/usr/bin/metal -isysroot $(shell xcrun --show-sdk-path)
endif
endif
METAL_STD ?= metal4.0
# Python for the tests and tools: the project venv (make venv), or IJAH_PY
IJAH_PY ?= $(CURDIR)/.venv/bin/python
PYTHON  ?= python3
export IJAH_PY
# The engine needs the M5 GPU's tensor units (Metal 4 / NAX): say so at build time when the chip is older.
CHIP := $(shell sysctl -n machdep.cpu.brand_string 2>/dev/null)
ifneq ($(filter M1 M2 M3 M4,$(word 2,$(CHIP))),)
$(warning Ijah needs an Apple M5 GPU (Metal 4 tensor units); this machine is "$(CHIP)": the build may work, the engine will not run)
endif
METALFLAGS := -std=$(METAL_STD) -I. -Iinclude -Wall -MMD -MP $(EXTRA_METALFLAGS)

# Sources linked into libengine.dylib. Other directories append with `ENGINE_SRCS += path/file.mm` in their .mk.
ENGINE_SRCS := $(filter-out core/main_%,$(wildcard core/*.mm core/*.cpp))
# Every .metal under core/, kernels/ and models/ goes into the one metallib (no .mk entry needed).
METAL_SRCS := $(shell find core kernels models -name '*.metal' 2>/dev/null | sort)
# Extra tests (the .mk files append). After the core test, `make test` runs every TEST_CMDS entry in order -- each
# entry must be ONE word (a binary or script path; make splits on spaces, so wrap a command with arguments in a
# script) -- then makes every TEST_TARGETS target (phony rules defined in the .mk; any recipe).
# Defined before the includes (usable in their rules): BUILD LIB METALLIB LINK_ENGINE CXX CXXFLAGS.
TEST_BINS :=
TEST_CMDS :=
TEST_TARGETS :=

LIB      := $(BUILD)/libengine.dylib
METALLIB := $(BUILD)/engine.metallib
CLI      := $(BUILD)/engine-cli
CORETEST := $(BUILD)/core-test
# Link line for any executable in build/ that uses the engine (rpath finds libengine.dylib next to it).
LINK_ENGINE := -L$(BUILD) -lengine -Wl,-rpath,@loader_path $(FRAMEWORKS)
# Rules in the included .mk files must not become the default goal.
.DEFAULT_GOAL := all

-include kernels/kernels.mk
-include $(wildcard models/*/model.mk)
-include tests/tests.mk

objs_of = $(patsubst %,$(BUILD)/obj/%.o,$(1))
ENGINE_OBJS := $(call objs_of,$(ENGINE_SRCS))
AIRS := $(patsubst %.metal,$(BUILD)/air/%.air,$(METAL_SRCS))

.PHONY: all test bench clean venv test-cpu
all: $(LIB) $(METALLIB) $(CLI) $(CORETEST) $(TEST_BINS)

$(BUILD)/obj/%.mm.o: %.mm
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(BUILD)/obj/%.cpp.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(BUILD)/air/%.air: %.metal
	@mkdir -p $(dir $@)
	$(METAL) $(METALFLAGS) -c $< -o $@

$(LIB): $(ENGINE_OBJS)
	$(CXX) -dynamiclib -install_name @rpath/libengine.dylib $^ $(FRAMEWORKS) -o $@
$(METALLIB): $(AIRS)
	$(METAL) $^ -o $@
$(CLI): $(call objs_of,core/main_cli.mm) $(LIB)
	$(CXX) $< $(LINK_ENGINE) -o $@
$(CORETEST): $(call objs_of,core/main_test.mm) $(LIB)
	$(CXX) $< $(LINK_ENGINE) -o $@

test: all
	$(CORETEST)
	$(foreach c,$(TEST_CMDS),$(c) &&) true
ifneq ($(strip $(TEST_TARGETS)),)
	$(MAKE) $(TEST_TARGETS)
endif

bench: all
	$(CORETEST) --bench

venv: .venv/bin/python
.venv/bin/python: requirements.txt
	tools/setup.sh $(PYTHON)

# CPU unit tests (no GPU, no model; the ones that read the tokenizer skip without a pack, see README "Tests")
test-cpu:
	tests/cpu_tests.sh

clean:
	rm -rf $(BUILD)

-include $(ENGINE_OBJS:.o=.d) $(BUILD)/obj/core/main_cli.mm.d $(BUILD)/obj/core/main_test.mm.d $(AIRS:.air=.d)
# every other object's header dependencies too (TEST_BINS objects from the .mk files; lab-layer's request)
-include $(filter-out $(ENGINE_OBJS:.o=.d),$(shell find $(BUILD)/obj -name '*.d' 2>/dev/null))
