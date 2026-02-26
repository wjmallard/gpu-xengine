NVCC       := nvcc
SM_ARCH    ?= sm_86
NVCC_FLAGS := -arch=$(SM_ARCH) -O2 -lineinfo -Iinc

vpath %.h  inc
vpath %.cu src

.PHONY: all clean

all: test_cpu

obj/%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

obj/fengine_sim.o: common.h fengine_sim.h
obj/xengine_cpu.o: common.h xengine_cpu.h
obj/test_cpu.o: common.h fengine_sim.h xengine_cpu.h

test_cpu: obj/test_cpu.o \
	obj/fengine_sim.o \
	obj/xengine_cpu.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

clean:
	rm -vf obj/*.o
	rm -vf test_cpu
