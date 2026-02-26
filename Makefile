NVCC       := nvcc
SM_ARCH    ?= sm_86
NVCC_FLAGS := -arch=$(SM_ARCH) -O2 -lineinfo -Iinc

vpath %.h  inc
vpath %.cu src

.PHONY: all clean

all: test_cpu test_gpu

obj/%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

obj/fengine_sim.o: common.h fengine_sim.h
obj/xengine_cpu.o: common.h xengine_cpu.h
obj/test_cpu.o: common.h fengine_sim.h xengine_cpu.h

obj/corner_turn.o: common.h corner_turn.h
obj/xengine_gpu.o: common.h xengine_gpu.h
obj/test_gpu.o: common.h fengine_sim.h xengine_cpu.h corner_turn.h xengine_gpu.h

test_cpu: obj/test_cpu.o \
	obj/fengine_sim.o \
	obj/xengine_cpu.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

test_gpu: obj/test_gpu.o \
	obj/fengine_sim.o \
	obj/xengine_cpu.o \
	obj/corner_turn.o \
	obj/xengine_gpu.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

clean:
	rm -vf obj/*.o
	rm -vf test_cpu
	rm -vf test_gpu
