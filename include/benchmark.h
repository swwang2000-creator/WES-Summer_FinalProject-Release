#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <stddef.h>
#include <stdint.h>

#include "opencl_runtime.h"

typedef struct {
    double kernel_ms;
    double device_path_ms;
} gpu_timing;

int int32_gpu_gemm(const ocl_runtime *runtime, cl_kernel kernel,
                   const int32_t *a, const int32_t *b, int32_t *c,
                   size_t n, gpu_timing *timing);

int int8_gpu_gemm(const ocl_runtime *runtime, cl_kernel kernel,
                  const int8_t *a, const int8_t *b, int32_t *c,
                  size_t n, gpu_timing *timing);

int int4_gpu_gemm(const ocl_runtime *runtime, cl_kernel kernel,
                  const uint8_t *a, const uint8_t *b, int32_t *c,
                  size_t n, gpu_timing *timing);

#endif
