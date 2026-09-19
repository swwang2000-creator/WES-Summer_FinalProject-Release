CC ?= gcc
CPPFLAGS ?= -Iinclude
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Wpedantic

BUILD_DIR := build
CPU_TEST := $(BUILD_DIR)/cpu_tests
CPU_SOURCES := src/matrix.c src/cpu_reference.c tests/test_cpu.c
GPU_BENCH := $(BUILD_DIR)/gemm_bench
GPU_SOURCES := src/main.c src/formal_benchmark.c src/opencl_runtime.c src/benchmark.c src/matrix.c src/cpu_reference.c
OPENCL_LIBS ?= -lOpenCL

.PHONY: all cpu-test clean

# The GPU target includes the three correctness kernels and the M4 formal
# collection path. The board container remains the required build environment.
all: $(CPU_TEST) $(GPU_BENCH)

$(CPU_TEST): $(CPU_SOURCES) include/int4_project.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(CPU_SOURCES) -o $@

cpu-test: $(CPU_TEST)
	./$(CPU_TEST)

$(GPU_BENCH): $(GPU_SOURCES) include/int4_project.h include/opencl_runtime.h include/benchmark.h include/formal_benchmark.h kernels/gemm_all.cl kernels/gemm_int32.cl kernels/gemm_int8.cl kernels/gemm_int4.cl
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(GPU_SOURCES) -o $@ $(OPENCL_LIBS)

clean:
	rm -rf $(BUILD_DIR)
