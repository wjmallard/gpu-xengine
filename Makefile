NVCC       := nvcc
SM_ARCH    ?= sm_86
NVCC_FLAGS := -arch=$(SM_ARCH) -O2 -lineinfo -Iinc

vpath %.h  inc
vpath %.cu src

.PHONY: all clean

all: test_cpu test_gpu benchmark

# --- Compile ---

obj/%.o: %.cu %.h common.h
	@mkdir -p obj
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

obj/%.o: %.cu common.h
	@mkdir -p obj
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<


# --- Binary dependencies ---

obj/test_cpu.o: \
	fengine_sim.h \
	xengine_cpu.h

obj/test_gpu.o: \
	fengine_sim.h \
	xengine_cpu.h \
	corner_turn.h \
	xengine_gpu.h

obj/benchmark.o: \
	fengine_sim.h \
	corner_turn.h \
	xengine_gpu.h


# --- Link ---

test_cpu: \
	obj/test_cpu.o \
	obj/fengine_sim.o \
	obj/xengine_cpu.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

test_gpu: \
	obj/test_gpu.o \
	obj/fengine_sim.o \
	obj/xengine_cpu.o \
	obj/corner_turn.o \
	obj/xengine_gpu.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

benchmark: \
	obj/benchmark.o \
	obj/fengine_sim.o \
	obj/corner_turn.o \
	obj/xengine_gpu.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^


# --- Clean ---

clean:
	rm -vf obj/*.o
	rm -vf test_cpu
	rm -vf test_gpu
	rm -vf benchmark
