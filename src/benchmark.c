#define _POSIX_C_SOURCE 200809L

#include "benchmark.h"

#include <stdio.h>
#include <stdint.h>
#include <time.h>

static int monotonic_ms(struct timespec *value)
{
    return clock_gettime(CLOCK_MONOTONIC, value) == 0 ? 0 : -1;
}

static double elapsed_ms(const struct timespec *start,
                         const struct timespec *end)
{
    time_t seconds = end->tv_sec - start->tv_sec;
    long nanoseconds = end->tv_nsec - start->tv_nsec;

    return (double)seconds * 1000.0 + (double)nanoseconds / 1000000.0;
}

static int gpu_gemm_buffers(const ocl_runtime *runtime, cl_kernel kernel,
                            const void *a, const void *b, int32_t *c,
                            size_t n, size_t input_bytes,
                            gpu_timing *timing)
{
    size_t elements;
    size_t output_bytes;
    size_t global_size[2];
    const size_t local_size[2] = {16u, 16u};
    cl_mem device_a = NULL;
    cl_mem device_b = NULL;
    cl_mem device_c = NULL;
    cl_event write_a_event = NULL;
    cl_event write_b_event = NULL;
    cl_event kernel_event = NULL;
    cl_event read_event = NULL;
    cl_event write_events[2];
    struct timespec host_start;
    struct timespec host_end;
    cl_int error;
    int n_arg;
    cl_ulong start_ns = 0;
    cl_ulong end_ns = 0;
    int result = -1;

    if (runtime == NULL || kernel == NULL || a == NULL || b == NULL ||
        c == NULL || timing == NULL || n == 0 || n % 16u != 0 ||
        n > SIZE_MAX / n) {
        return -1;
    }
    elements = n * n;
    if (elements > SIZE_MAX / sizeof(*c)) {
        return -1;
    }
    output_bytes = elements * sizeof(*c);
    global_size[0] = n;
    global_size[1] = n;
    device_a = clCreateBuffer(runtime->context, CL_MEM_READ_ONLY, input_bytes,
                              NULL, &error);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "A buffer creation failed: %s (%d)\n",
                ocl_error_name(error), error);
        goto cleanup;
    }
    device_b = clCreateBuffer(runtime->context, CL_MEM_READ_ONLY, input_bytes,
                              NULL, &error);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "B buffer creation failed: %s (%d)\n",
                ocl_error_name(error), error);
        goto cleanup;
    }
    device_c = clCreateBuffer(runtime->context, CL_MEM_WRITE_ONLY, output_bytes,
                              NULL, &error);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "C buffer creation failed: %s (%d)\n",
                ocl_error_name(error), error);
        goto cleanup;
    }
    if (monotonic_ms(&host_start) != 0) {
        fprintf(stderr, "Monotonic clock start failed\n");
        goto cleanup;
    }
    error = clEnqueueWriteBuffer(runtime->queue, device_a, CL_FALSE, 0,
                                 input_bytes, a, 0, NULL, &write_a_event);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "A upload failed: %s (%d)\n",
                ocl_error_name(error), error);
        goto cleanup;
    }
    error = clEnqueueWriteBuffer(runtime->queue, device_b, CL_FALSE, 0,
                                 input_bytes, b, 0, NULL, &write_b_event);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "B upload failed: %s (%d)\n",
                ocl_error_name(error), error);
        goto cleanup;
    }
    error = clSetKernelArg(kernel, 0, sizeof(device_a), &device_a);
    if (error != CL_SUCCESS) {
        goto cleanup;
    }
    error = clSetKernelArg(kernel, 1, sizeof(device_b), &device_b);
    if (error != CL_SUCCESS) {
        goto cleanup;
    }
    error = clSetKernelArg(kernel, 2, sizeof(device_c), &device_c);
    if (error != CL_SUCCESS) {
        goto cleanup;
    }
    n_arg = (int)n;
    error = clSetKernelArg(kernel, 3, sizeof(n_arg), &n_arg);
    if (error != CL_SUCCESS) {
        goto cleanup;
    }
    write_events[0] = write_a_event;
    write_events[1] = write_b_event;
    error = clEnqueueNDRangeKernel(runtime->queue, kernel, 2, NULL,
                                   global_size, local_size, 2, write_events,
                                   &kernel_event);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "Kernel enqueue failed: %s (%d)\n",
                ocl_error_name(error), error);
        goto cleanup;
    }
    error = clEnqueueReadBuffer(runtime->queue, device_c, CL_TRUE, 0,
                                output_bytes, c, 1, &kernel_event,
                                &read_event);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "C download failed: %s (%d)\n",
                ocl_error_name(error), error);
        goto cleanup;
    }
    if (monotonic_ms(&host_end) != 0) {
        fprintf(stderr, "Monotonic clock end failed\n");
        goto cleanup;
    }
    error = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START,
                                    sizeof(start_ns), &start_ns, NULL);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "Kernel start profiling failed: %s (%d)\n",
                ocl_error_name(error), error);
        goto cleanup;
    }
    error = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END,
                                    sizeof(end_ns), &end_ns, NULL);
    if (error != CL_SUCCESS || end_ns < start_ns) {
        fprintf(stderr, "Kernel end profiling failed: %s (%d)\n",
                ocl_error_name(error), error);
        goto cleanup;
    }
    timing->kernel_ms = (double)(end_ns - start_ns) / 1000000.0;
    timing->device_path_ms = elapsed_ms(&host_start, &host_end);
    if (timing->device_path_ms < 0.0) {
        fprintf(stderr, "Device-path timing was negative\n");
        goto cleanup;
    }
    result = 0;

cleanup:
    if (read_event != NULL) {
        (void)clReleaseEvent(read_event);
    }
    if (kernel_event != NULL) {
        (void)clReleaseEvent(kernel_event);
    }
    if (write_b_event != NULL) {
        (void)clReleaseEvent(write_b_event);
    }
    if (write_a_event != NULL) {
        (void)clReleaseEvent(write_a_event);
    }
    if (device_c != NULL) {
        (void)clReleaseMemObject(device_c);
    }
    if (device_b != NULL) {
        (void)clReleaseMemObject(device_b);
    }
    if (device_a != NULL) {
        (void)clReleaseMemObject(device_a);
    }
    return result;
}

int int32_gpu_gemm(const ocl_runtime *runtime, cl_kernel kernel,
                   const int32_t *a, const int32_t *b, int32_t *c,
                   size_t n, gpu_timing *timing)
{
    return gpu_gemm_buffers(runtime, kernel, a, b, c,
                            n, n * n * sizeof(*a), timing);
}

int int8_gpu_gemm(const ocl_runtime *runtime, cl_kernel kernel,
                  const int8_t *a, const int8_t *b, int32_t *c,
                  size_t n, gpu_timing *timing)
{
    return gpu_gemm_buffers(runtime, kernel, a, b, c,
                            n, n * n * sizeof(*a), timing);
}

int int4_gpu_gemm(const ocl_runtime *runtime, cl_kernel kernel,
                  const uint8_t *a, const uint8_t *b, int32_t *c,
                  size_t n, gpu_timing *timing)
{
    size_t elements;
    size_t packed_bytes;

    if (n == 0 || n > SIZE_MAX / n) {
        return -1;
    }
    elements = n * n;
    if (elements > SIZE_MAX - 1u) {
        return -1;
    }
    packed_bytes = (elements + 1u) / 2u;
    return gpu_gemm_buffers(runtime, kernel, a, b, c,
                            n, packed_bytes, timing);
}
