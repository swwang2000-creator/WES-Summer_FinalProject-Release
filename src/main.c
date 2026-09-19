#include "benchmark.h"
#include "formal_benchmark.h"
#include "int4_project.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    KERNEL_ALL,
    KERNEL_INT32,
    KERNEL_INT8,
    KERNEL_INT4
} kernel_selection;

static void usage(const char *program)
{
    printf("Usage: %s [--list-devices] [--check-only] [--size N]\n", program);
    printf("  --list-devices   enumerate all OpenCL platforms/devices\n");
    printf("  --check-only     run one correctness check (default mode)\n");
    printf("  --size N         aligned square size, default 128\n");
    printf("  --kernels K      int32, int8, int4, or all (default all)\n");
}

static int parse_size(const char *text, size_t *value)
{
    char *end = NULL;
    unsigned long parsed;

    if (text == NULL || value == NULL || text[0] == '\0') {
        return -1;
    }
    errno = 0;
    parsed = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0 ||
        parsed > (unsigned long)INT_MAX) {
        return -1;
    }
    *value = (size_t)parsed;
    return 0;
}

static int parse_kernel_selection(const char *text, kernel_selection *selection)
{
    if (text == NULL || selection == NULL) {
        return -1;
    }
    if (strcmp(text, "all") == 0) {
        *selection = KERNEL_ALL;
    } else if (strcmp(text, "int32") == 0) {
        *selection = KERNEL_INT32;
    } else if (strcmp(text, "int8") == 0) {
        *selection = KERNEL_INT8;
    } else if (strcmp(text, "int4") == 0) {
        *selection = KERNEL_INT4;
    } else {
        return -1;
    }
    return 0;
}

static int count_mismatches(const int32_t *expected, const int32_t *actual,
                            size_t count, size_t *first_index)
{
    size_t i;
    int mismatches = 0;

    *first_index = count;
    for (i = 0; i < count; ++i) {
        if (expected[i] != actual[i]) {
            if (mismatches == 0) {
                *first_index = i;
            }
            ++mismatches;
        }
    }
    return mismatches;
}

static int report_result(const char *name, const int32_t *expected,
                         const int32_t *actual, size_t elements, size_t n,
                         const gpu_timing *timing)
{
    size_t first_mismatch;
    int mismatches = count_mismatches(expected, actual, elements,
                                      &first_mismatch);
    if (mismatches != 0) {
        size_t row = first_mismatch / n;
        size_t col = first_mismatch % n;
        fprintf(stderr, "%s correctness FAILED: %d mismatches; first at "
                "(%zu,%zu), expected=%d actual=%d\n",
                name, mismatches, row, col, expected[first_mismatch],
                actual[first_mismatch]);
        return -1;
    }
    printf("%s correctness PASSED: N=%zu, mismatches=0, kernel_ms=%.6f, "
           "device_path_ms=%.6f\n", name, n, timing->kernel_ms,
           timing->device_path_ms);
    return 0;
}

int main(int argc, char **argv)
{
    size_t n = 128;
    int list_devices = 0;
    int check_only = 0;
    kernel_selection selection = KERNEL_ALL;
    int i;
    int32_t *a = NULL;
    int32_t *b = NULL;
    int8_t *a8 = NULL;
    int8_t *b8 = NULL;
    uint8_t *a4 = NULL;
    uint8_t *b4 = NULL;
    int32_t *expected = NULL;
    int32_t *actual = NULL;
    size_t elements;
    size_t bytes;
    size_t packed_bytes;
    gpu_timing timing;
    ocl_runtime runtime;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_int error;
    int result = EXIT_FAILURE;

    memset(&runtime, 0, sizeof(runtime));
    memset(&timing, 0, sizeof(timing));
    if (formal_benchmark_requested(argc, argv)) {
        return formal_benchmark_main(argc, argv);
    }
    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--list-devices") == 0) {
            list_devices = 1;
        } else if (strcmp(argv[i], "--check-only") == 0) {
            check_only = 1;
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            if (parse_size(argv[++i], &n) != 0) {
                fprintf(stderr, "Invalid --size value\n");
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--kernels") == 0 && i + 1 < argc) {
            if (parse_kernel_selection(argv[++i], &selection) != 0) {
                fprintf(stderr, "Invalid --kernels value\n");
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return EXIT_SUCCESS;
        } else {
            usage(argv[0]);
            return EXIT_FAILURE;
        }
    }
    if (list_devices) {
        return ocl_print_devices() == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (n % 16u != 0) {
        fprintf(stderr, "Size %zu is unsupported: N must be divisible by 16\n", n);
        return EXIT_FAILURE;
    }
    if (n > SIZE_MAX / n) {
        fprintf(stderr, "Size %zu overflows the element count\n", n);
        return EXIT_FAILURE;
    }
    elements = n * n;
    if (elements > SIZE_MAX / sizeof(*a)) {
        fprintf(stderr, "Size %zu overflows the allocation size\n", n);
        return EXIT_FAILURE;
    }
    bytes = elements * sizeof(*a);
    packed_bytes = (elements + 1u) / 2u;
    a = malloc(bytes);
    b = malloc(bytes);
    a8 = malloc(elements * sizeof(*a8));
    b8 = malloc(elements * sizeof(*b8));
    a4 = malloc(packed_bytes);
    b4 = malloc(packed_bytes);
    expected = malloc(bytes);
    actual = malloc(bytes);
    if (a == NULL || b == NULL || a8 == NULL || b8 == NULL ||
        a4 == NULL || b4 == NULL || expected == NULL || actual == NULL) {
        fprintf(stderr, "Host allocation failed for N=%zu\n", n);
        goto cleanup;
    }
    if (int4_generate_inputs(a, b, n, 237u) != 0 ||
        int4_convert_i8(a, a8, elements) != 0 ||
        int4_convert_i8(b, b8, elements) != 0 ||
        int4_pack_signed(a, a4, elements) != 0 ||
        int4_pack_signed(b, b4, elements) != 0 ||
        int4_gemm_reference(a, b, expected, n) != 0) {
        fprintf(stderr, "CPU input/reference preparation failed\n");
        goto cleanup;
    }
    if (ocl_init(&runtime) != 0) {
        goto cleanup;
    }
    if (ocl_build_program_from_file(&runtime, "kernels/gemm_all.cl",
                                    "-cl-std=CL1.2", &program) != 0) {
        goto cleanup;
    }
    if (selection == KERNEL_ALL || selection == KERNEL_INT32) {
        kernel = clCreateKernel(program, "gemm_int32_tiled", &error);
        if (error != CL_SUCCESS || kernel == NULL) {
            fprintf(stderr, "INT32 clCreateKernel failed: %s (%d)\n",
                    ocl_error_name(error), error);
            goto cleanup;
        }
        if (int32_gpu_gemm(&runtime, kernel, a, b, actual, n,
                           &timing) != 0 ||
            report_result("INT32", expected, actual, elements, n,
                          &timing) != 0) {
            goto cleanup;
        }
        (void)clReleaseKernel(kernel);
        kernel = NULL;
    }
    if (selection == KERNEL_ALL || selection == KERNEL_INT8) {
        kernel = clCreateKernel(program, "gemm_int8_tiled", &error);
        if (error != CL_SUCCESS || kernel == NULL) {
            fprintf(stderr, "INT8 clCreateKernel failed: %s (%d)\n",
                    ocl_error_name(error), error);
            goto cleanup;
        }
        if (int8_gpu_gemm(&runtime, kernel, a8, b8, actual, n,
                          &timing) != 0 ||
            report_result("INT8", expected, actual, elements, n,
                          &timing) != 0) {
            goto cleanup;
        }
        (void)clReleaseKernel(kernel);
        kernel = NULL;
    }
    if (selection == KERNEL_ALL || selection == KERNEL_INT4) {
        kernel = clCreateKernel(program, "gemm_int4_tiled", &error);
        if (error != CL_SUCCESS || kernel == NULL) {
            fprintf(stderr, "INT4 clCreateKernel failed: %s (%d)\n",
                    ocl_error_name(error), error);
            goto cleanup;
        }
        if (int4_gpu_gemm(&runtime, kernel, a4, b4, actual, n,
                          &timing) != 0 ||
            report_result("INT4", expected, actual, elements, n,
                          &timing) != 0) {
            goto cleanup;
        }
        (void)clReleaseKernel(kernel);
        kernel = NULL;
    }
    if (check_only) {
        printf("Mode: correctness check only; this is not an official benchmark run.\n");
    }
    result = EXIT_SUCCESS;

cleanup:
    if (kernel != NULL) {
        (void)clReleaseKernel(kernel);
    }
    if (program != NULL) {
        (void)clReleaseProgram(program);
    }
    ocl_release(&runtime);
    free(actual);
    free(expected);
    free(b4);
    free(a4);
    free(b8);
    free(a8);
    free(b);
    free(a);
    return result;
}
