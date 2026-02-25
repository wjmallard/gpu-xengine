NVCC       := nvcc
SM_ARCH    ?= sm_86
NVCC_FLAGS := -arch=$(SM_ARCH) -O2 -lineinfo

SRC_DIR := src

SRCS := $(SRC_DIR)/test_cpu.cu \
        $(SRC_DIR)/fengine_sim.cu \
        $(SRC_DIR)/xengine_cpu.cu

HDRS := $(SRC_DIR)/common.h      \
        $(SRC_DIR)/fengine_sim.h  \
        $(SRC_DIR)/xengine_cpu.h

.PHONY: all clean

all: test_cpu

test_cpu: $(SRCS) $(HDRS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(SRCS)

clean:
	rm -f test_cpu
