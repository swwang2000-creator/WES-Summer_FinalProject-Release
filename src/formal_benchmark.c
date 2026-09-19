#define _POSIX_C_SOURCE 200809L

#include "formal_benchmark.h"

#include "benchmark.h"
#include "int4_project.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define FB_DEFAULT_SIZE 128u
#define FB_DEFAULT_SEED 237u
#define FB_DEFAULT_WARMUPS 5u
#define FB_DEFAULT_RUNS 30u
#define FB_MAX_SIZES 16u
#define FB_SOURCE_VERSION "m4-2026-09-18"

typedef enum {
    FB_SELECTION_ALL,
    FB_SELECTION_INT32,
    FB_SELECTION_INT8,
    FB_SELECTION_INT4
} fb_selection;

typedef enum {
    FB_KIND_INT32,
    FB_KIND_INT8,
    FB_KIND_INT4,
    FB_KIND_COUNT
} fb_kind;

typedef struct {
    size_t sizes[FB_MAX_SIZES];
    size_t size_count;
    uint32_t seed;
    size_t warmups;
    size_t runs;
    fb_selection selection;
    const char *output_dir;
    int list_devices;
    int check_only;
} fb_config;

typedef struct {
    size_t n;
    size_t elements;
    size_t bytes;
    size_t packed_bytes;
    int32_t *a;
    int32_t *b;
    int8_t *a8;
    int8_t *b8;
    uint8_t *a4;
    uint8_t *b4;
    int32_t *expected;
    int32_t *actual;
} fb_data;

typedef struct {
    FILE *raw;
    FILE *packing;
    FILE *log;
    char metadata_path[PATH_MAX];
} fb_files;

int formal_benchmark_requested(int argc, char **argv)
{
    int i;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--output") == 0) {
            return 1;
        }
    }
    return 0;
}

static void fb_usage(const char *program)
{
    printf("Usage: %s --output DIR [options]\n", program);
    printf("  --sizes LIST       comma-separated sizes (default 128)\n");
    printf("  --kernels K        int32, int8, int4, or all (default all)\n");
    printf("  --seed S           nonzero deterministic seed (default 237)\n");
    printf("  --warmup N         warm-ups per kernel/size (default 5)\n");
    printf("  --runs N           measured runs per kernel/size (default 30)\n");
    printf("  --list-devices     list devices and exit\n");
}

static int fb_parse_unsigned(const char *text, unsigned long long maximum,
                             unsigned long long *value)
{
    char *end = NULL;
    unsigned long long parsed;

    if (text == NULL || value == NULL || text[0] == '\0') {
        return -1;
    }
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed > maximum) {
        return -1;
    }
    *value = parsed;
    return 0;
}

static int fb_parse_size(const char *text, size_t *value)
{
    unsigned long long parsed;

    if (fb_parse_unsigned(text, (unsigned long long)INT_MAX, &parsed) != 0 ||
        parsed == 0) {
        return -1;
    }
    *value = (size_t)parsed;
    return 0;
}

static int fb_parse_sizes(const char *text, size_t *sizes, size_t *count)
{
    char *copy;
    char *token;
    char *save = NULL;
    size_t used = 0;

    if (text == NULL || sizes == NULL || count == NULL || text[0] == '\0') {
        return -1;
    }
    copy = malloc(strlen(text) + 1u);
    if (copy == NULL) {
        return -1;
    }
    strcpy(copy, text);
    token = strtok_r(copy, ",", &save);
    while (token != NULL) {
        if (used == FB_MAX_SIZES || fb_parse_size(token, &sizes[used]) != 0) {
            free(copy);
            return -1;
        }
        ++used;
        token = strtok_r(NULL, ",", &save);
    }
    free(copy);
    if (used == 0) {
        return -1;
    }
    *count = used;
    return 0;
}

static int fb_parse_selection(const char *text, fb_selection *selection)
{
    if (text == NULL || selection == NULL) {
        return -1;
    }
    if (strcmp(text, "all") == 0) {
        *selection = FB_SELECTION_ALL;
    } else if (strcmp(text, "int32") == 0) {
        *selection = FB_SELECTION_INT32;
    } else if (strcmp(text, "int8") == 0) {
        *selection = FB_SELECTION_INT8;
    } else if (strcmp(text, "int4") == 0) {
        *selection = FB_SELECTION_INT4;
    } else {
        return -1;
    }
    return 0;
}

static const char *fb_selection_name(fb_selection selection)
{
    switch (selection) {
    case FB_SELECTION_ALL: return "all";
    case FB_SELECTION_INT32: return "int32";
    case FB_SELECTION_INT8: return "int8";
    case FB_SELECTION_INT4: return "int4";
    default: return "unknown";
    }
}

static const char *fb_kind_name(fb_kind kind)
{
    switch (kind) {
    case FB_KIND_INT32: return "int32";
    case FB_KIND_INT8: return "int8";
    case FB_KIND_INT4: return "int4";
    default: return "unknown";
    }
}

static const char *fb_kernel_function(fb_kind kind)
{
    switch (kind) {
    case FB_KIND_INT32: return "gemm_int32_tiled";
    case FB_KIND_INT8: return "gemm_int8_tiled";
    case FB_KIND_INT4: return "gemm_int4_tiled";
    default: return "";
    }
}

static int fb_selected(fb_selection selection, fb_kind kind)
{
    return selection == FB_SELECTION_ALL ||
           (selection == FB_SELECTION_INT32 && kind == FB_KIND_INT32) ||
           (selection == FB_SELECTION_INT8 && kind == FB_KIND_INT8) ||
           (selection == FB_SELECTION_INT4 && kind == FB_KIND_INT4);
}

static int fb_now(struct timespec *value)
{
    return clock_gettime(CLOCK_MONOTONIC, value) == 0 ? 0 : -1;
}

static double fb_elapsed_ms(const struct timespec *start,
                            const struct timespec *end)
{
    time_t seconds = end->tv_sec - start->tv_sec;
    long nanoseconds = end->tv_nsec - start->tv_nsec;

    return (double)seconds * 1000.0 + (double)nanoseconds / 1000000.0;
}

static int fb_prepare(size_t n, uint32_t seed, fb_data *data,
                      double *packing_ms)
{
    struct timespec start;
    struct timespec end;

    if (data == NULL || packing_ms == NULL || n == 0 || n % 16u != 0 ||
        n > SIZE_MAX / n) {
        return -1;
    }
    memset(data, 0, sizeof(*data));
    data->n = n;
    data->elements = n * n;
    if (data->elements > SIZE_MAX / sizeof(*data->a) ||
        data->elements > SIZE_MAX - 1u) {
        return -1;
    }
    data->bytes = data->elements * sizeof(*data->a);
    data->packed_bytes = (data->elements + 1u) / 2u;
    data->a = malloc(data->bytes);
    data->b = malloc(data->bytes);
    data->a8 = malloc(data->elements * sizeof(*data->a8));
    data->b8 = malloc(data->elements * sizeof(*data->b8));
    data->a4 = malloc(data->packed_bytes);
    data->b4 = malloc(data->packed_bytes);
    data->expected = malloc(data->bytes);
    data->actual = malloc(data->bytes);
    if (data->a == NULL || data->b == NULL || data->a8 == NULL ||
        data->b8 == NULL || data->a4 == NULL || data->b4 == NULL ||
        data->expected == NULL || data->actual == NULL) {
        return -1;
    }
    if (int4_generate_inputs(data->a, data->b, n, seed) != 0 ||
        int4_convert_i8(data->a, data->a8, data->elements) != 0 ||
        int4_convert_i8(data->b, data->b8, data->elements) != 0 ||
        int4_gemm_reference(data->a, data->b, data->expected, n) != 0 ||
        fb_now(&start) != 0 ||
        int4_pack_signed(data->a, data->a4, data->elements) != 0 ||
        int4_pack_signed(data->b, data->b4, data->elements) != 0 ||
        fb_now(&end) != 0) {
        return -1;
    }
    *packing_ms = fb_elapsed_ms(&start, &end);
    return isfinite(*packing_ms) && *packing_ms >= 0.0 ? 0 : -1;
}

static void fb_free_data(fb_data *data)
{
    if (data == NULL) {
        return;
    }
    free(data->actual);
    free(data->expected);
    free(data->b4);
    free(data->a4);
    free(data->b8);
    free(data->a8);
    free(data->b);
    free(data->a);
    memset(data, 0, sizeof(*data));
}

static size_t fb_count_mismatches(const int32_t *expected, const int32_t *actual,
                                  size_t count, size_t *first)
{
    size_t i;
    size_t mismatches = 0;

    *first = count;
    for (i = 0; i < count; ++i) {
        if (expected[i] != actual[i]) {
            if (mismatches == 0) {
                *first = i;
            }
            ++mismatches;
        }
    }
    return mismatches;
}

static int fb_timing_valid(const gpu_timing *timing)
{
    return timing != NULL && isfinite(timing->kernel_ms) &&
           isfinite(timing->device_path_ms) && timing->kernel_ms > 0.0 &&
           timing->device_path_ms > 0.0;
}

static int fb_mkdirs(const char *path)
{
    char buffer[PATH_MAX];
    char *cursor;
    size_t length;

    if (path == NULL || path[0] == '\0') {
        return -1;
    }
    length = strlen(path);
    if (length >= sizeof(buffer)) {
        return -1;
    }
    strcpy(buffer, path);
    for (cursor = buffer + 1; *cursor != '\0'; ++cursor) {
        if (*cursor == '/') {
            *cursor = '\0';
            if (buffer[0] != '\0' && mkdir(buffer, 0775) != 0 &&
                errno != EEXIST) {
                return -1;
            }
            *cursor = '/';
        }
    }
    return mkdir(buffer, 0775) == 0 || errno == EEXIST ? 0 : -1;
}

static int fb_join(char *path, size_t path_size, const char *directory,
                   const char *name)
{
    int written = snprintf(path, path_size, "%s/%s", directory, name);
    return written >= 0 && (size_t)written < path_size ? 0 : -1;
}

static int fb_exists(const char *path)
{
    struct stat information;
    return stat(path, &information) == 0;
}

static int fb_open_outputs(const fb_config *config, fb_files *files)
{
    char path[PATH_MAX];

    memset(files, 0, sizeof(*files));
    if (fb_mkdirs(config->output_dir) != 0 ||
        fb_join(path, sizeof(path), config->output_dir, "raw.csv") != 0 ||
        fb_exists(path)) {
        fprintf(stderr, "Output directory unavailable or raw.csv exists: %s\n",
                config->output_dir);
        return -1;
    }
    files->raw = fopen(path, "wb");
    if (files->raw == NULL) goto failure;
    if (fb_join(path, sizeof(path), config->output_dir, "packing.csv") != 0 ||
        fb_exists(path)) goto failure;
    files->packing = fopen(path, "wb");
    if (files->packing == NULL) goto failure;
    if (fb_join(path, sizeof(path), config->output_dir, "run.log") != 0 ||
        fb_exists(path)) goto failure;
    files->log = fopen(path, "wb");
    if (files->log == NULL) goto failure;
    if (fb_join(files->metadata_path, sizeof(files->metadata_path),
                config->output_dir, "metadata.json") != 0 ||
        fb_exists(files->metadata_path)) goto failure;
    fprintf(files->raw,
            "n,kernel,trial,seed,kernel_ms,device_path_ms,input_bytes,"
            "output_bytes,mismatches\n");
    fprintf(files->packing, "n,seed,pack_a_b_ms,input_bytes\n");
    return 0;

failure:
    if (files->log != NULL) fclose(files->log);
    if (files->packing != NULL) fclose(files->packing);
    if (files->raw != NULL) fclose(files->raw);
    memset(files, 0, sizeof(*files));
    return -1;
}

static void fb_close_outputs(fb_files *files)
{
    if (files == NULL) return;
    if (files->log != NULL) fclose(files->log);
    if (files->packing != NULL) fclose(files->packing);
    if (files->raw != NULL) fclose(files->raw);
    memset(files, 0, sizeof(*files));
}

static void fb_json_string(FILE *file, const char *value)
{
    const unsigned char *cursor = (const unsigned char *)value;

    fputc('"', file);
    while (cursor != NULL && *cursor != '\0') {
        if (*cursor == '"' || *cursor == '\\') fputc('\\', file);
        if (*cursor == '\n') fputs("\\n", file);
        else fputc(*cursor, file);
        ++cursor;
    }
    fputc('"', file);
}

static void fb_platform_string(cl_platform_id platform, char *value,
                               size_t value_size)
{
    cl_int error;
    if (value_size == 0) return;
    error = clGetPlatformInfo(platform, CL_PLATFORM_NAME, value_size, value,
                              NULL);
    if (error != CL_SUCCESS) strcpy(value, "unavailable");
    else value[value_size - 1u] = '\0';
}

static void fb_device_string(cl_device_id device, cl_device_info info,
                             char *value, size_t value_size)
{
    cl_int error;
    if (value_size == 0) return;
    error = clGetDeviceInfo(device, info, value_size, value, NULL);
    if (error != CL_SUCCESS) strcpy(value, "unavailable");
    else value[value_size - 1u] = '\0';
}

static int fb_metadata(const fb_files *files, const fb_config *config,
                       const ocl_runtime *runtime, const char *status)
{
    FILE *file;
    char platform[256] = "unavailable";
    char device[256] = "unavailable";
    char driver[256] = "unavailable";
    char device_version[256] = "unavailable";
    char opencl_c[256] = "unavailable";
    size_t i;

    file = fopen(files->metadata_path, "wb");
    if (file == NULL) return -1;
    if (runtime != NULL && runtime->platform != NULL && runtime->device != NULL) {
        fb_platform_string(runtime->platform, platform, sizeof(platform));
        fb_device_string(runtime->device, CL_DEVICE_NAME, device, sizeof(device));
        fb_device_string(runtime->device, CL_DRIVER_VERSION, driver, sizeof(driver));
        fb_device_string(runtime->device, CL_DEVICE_VERSION, device_version,
                         sizeof(device_version));
        fb_device_string(runtime->device, CL_DEVICE_OPENCL_C_VERSION, opencl_c,
                         sizeof(opencl_c));
    }
    fprintf(file, "{\n  \"status\":");
    fb_json_string(file, status);
    fprintf(file, ",\n  \"source_version\":");
    fb_json_string(file, FB_SOURCE_VERSION);
    fprintf(file, ",\n  \"seed\":%u,\n  \"sizes\":[", config->seed);
    for (i = 0; i < config->size_count; ++i)
        fprintf(file, "%s%zu", i == 0 ? "" : ",", config->sizes[i]);
    fprintf(file, "],\n  \"kernels\":");
    fb_json_string(file, fb_selection_name(config->selection));
    fprintf(file, ",\n  \"warmups\":%zu,\n  \"measured_runs\":%zu,\n"
            "  \"tile\":16,\n  \"build_options\":\"-cl-std=CL1.2\",\n"
            "  \"input_range\":\"[-8,7]\",\n"
            "  \"kernel_order\":\"INT32,INT8,INT4\",\n"
            "  \"timing\":{\"kernel_ms\":\"OpenCL event END-START\","
            "\"device_path_ms\":\"host monotonic H2D-through-D2H\","
            "\"pack_a_b_ms\":\"host monotonic for both INT4 packs\"},\n"
            "  \"platform\":", config->warmups, config->runs);
    fb_json_string(file, platform);
    fprintf(file, ",\n  \"device\":");
    fb_json_string(file, device);
    fprintf(file, ",\n  \"driver_version\":");
    fb_json_string(file, driver);
    fprintf(file, ",\n  \"device_version\":");
    fb_json_string(file, device_version);
    fprintf(file, ",\n  \"opencl_c_version\":");
    fb_json_string(file, opencl_c);
    fprintf(file, ",\n  \"environment\":\"board Adreno development container\"\n}\n");
    fclose(file);
    return 0;
}

static int fb_raw_row(FILE *file, const fb_data *data, fb_kind kind,
                      size_t trial, uint32_t seed, const gpu_timing *timing,
                      size_t input_bytes, size_t mismatches)
{
    if (fprintf(file, "%zu,%s,%zu,%u,%.9f,%.9f,%zu,%zu,%zu\n", data->n,
                fb_kind_name(kind), trial, seed, timing->kernel_ms,
                timing->device_path_ms, input_bytes, data->bytes,
                mismatches) < 0) return -1;
    return fflush(file) == 0 ? 0 : -1;
}

static int fb_packing_row(FILE *file, const fb_data *data, uint32_t seed,
                          double packing_ms)
{
    if (fprintf(file, "%zu,%u,%.9f,%zu\n", data->n, seed, packing_ms,
                data->packed_bytes * 2u) < 0) return -1;
    return fflush(file) == 0 ? 0 : -1;
}

static int fb_invoke(fb_kind kind, const ocl_runtime *runtime, cl_kernel kernel,
                    const fb_data *data, gpu_timing *timing)
{
    if (kind == FB_KIND_INT32)
        return int32_gpu_gemm(runtime, kernel, data->a, data->b, data->actual,
                              data->n, timing);
    if (kind == FB_KIND_INT8)
        return int8_gpu_gemm(runtime, kernel, data->a8, data->b8, data->actual,
                             data->n, timing);
    return int4_gpu_gemm(runtime, kernel, data->a4, data->b4, data->actual,
                         data->n, timing);
}

static size_t fb_input_bytes(const fb_data *data, fb_kind kind)
{
    if (kind == FB_KIND_INT32) return data->elements * sizeof(*data->a) * 2u;
    if (kind == FB_KIND_INT8) return data->elements * sizeof(*data->a8) * 2u;
    return data->packed_bytes * 2u;
}

static void fb_print_failure(const fb_data *data, fb_kind kind,
                             size_t mismatches, size_t first)
{
    size_t row = first / data->n;
    size_t col = first % data->n;
    fprintf(stderr, "%s mismatch: N=%zu count=%zu first=(%zu,%zu) "
            "expected=%d actual=%d\n", fb_kind_name(kind), data->n,
            mismatches, row, col, data->expected[first], data->actual[first]);
}

static int fb_run_kernel(const fb_config *config, const ocl_runtime *runtime,
                         cl_program program, fb_files *files,
                         const fb_data *data, fb_kind kind)
{
    cl_int error;
    cl_kernel kernel;
    size_t trial;

    kernel = clCreateKernel(program, fb_kernel_function(kind), &error);
    if (error != CL_SUCCESS || kernel == NULL) {
        fprintf(stderr, "%s clCreateKernel failed: %s (%d)\n",
                fb_kind_name(kind), ocl_error_name(error), error);
        return -1;
    }
    for (trial = 0; trial < config->warmups; ++trial) {
        gpu_timing timing;
        size_t first;
        size_t mismatches;
        if (fb_invoke(kind, runtime, kernel, data, &timing) != 0 ||
            !fb_timing_valid(&timing)) {
            fprintf(stderr, "%s warm-up %zu failed\n", fb_kind_name(kind), trial + 1u);
            goto failure;
        }
        mismatches = fb_count_mismatches(data->expected, data->actual,
                                         data->elements, &first);
        if (mismatches != 0) {
            fb_print_failure(data, kind, mismatches, first);
            goto failure;
        }
    }
    for (trial = 0; trial < config->runs; ++trial) {
        gpu_timing timing;
        size_t first;
        size_t mismatches;
        if (fb_invoke(kind, runtime, kernel, data, &timing) != 0 ||
            !fb_timing_valid(&timing)) {
            fprintf(stderr, "%s measured run %zu failed\n", fb_kind_name(kind),
                    trial + 1u);
            goto failure;
        }
        mismatches = fb_count_mismatches(data->expected, data->actual,
                                         data->elements, &first);
        if (fb_raw_row(files->raw, data, kind, trial + 1u, config->seed,
                       &timing, fb_input_bytes(data, kind), mismatches) != 0) {
            fprintf(stderr, "Writing raw.csv failed\n");
            goto failure;
        }
        if (mismatches != 0) {
            fb_print_failure(data, kind, mismatches, first);
            goto failure;
        }
    }
    printf("%s benchmark complete: N=%zu, warmups=%zu, measured=%zu\n",
           fb_kind_name(kind), data->n, config->warmups, config->runs);
    (void)clReleaseKernel(kernel);
    return 0;

failure:
    if (files->log != NULL) {
        fprintf(files->log, "failure=kernel=%s size=%zu\n",
                fb_kind_name(kind), data->n);
        fflush(files->log);
    }
    (void)clReleaseKernel(kernel);
    return -1;
}

static int fb_run_size(const fb_config *config, const ocl_runtime *runtime,
                       cl_program program, fb_files *files, size_t n)
{
    fb_data data;
    double packing_ms;
    fb_kind kind;

    if (fb_prepare(n, config->seed, &data, &packing_ms) != 0) {
        fprintf(stderr, "Preparation failed for N=%zu\n", n);
        return -1;
    }
    if (fb_packing_row(files->packing, &data, config->seed, packing_ms) != 0) {
        fprintf(stderr, "Writing packing.csv failed\n");
        fb_free_data(&data);
        return -1;
    }
    if (files->log != NULL) {
        fprintf(files->log, "size=%zu pack_a_b_ms=%.9f\n", n, packing_ms);
        fflush(files->log);
    }
    for (kind = FB_KIND_INT32; kind < FB_KIND_COUNT; ++kind) {
        if (fb_selected(config->selection, kind) &&
            fb_run_kernel(config, runtime, program, files, &data, kind) != 0) {
            fb_free_data(&data);
            return -1;
        }
    }
    fb_free_data(&data);
    return 0;
}

static void fb_write_command(FILE *log, int argc, char **argv)
{
    int i;
    if (log == NULL) return;
    fputs("command=", log);
    for (i = 0; i < argc; ++i)
        fprintf(log, "%s%s", i == 0 ? "" : " ", argv[i]);
    fputc('\n', log);
    fflush(log);
}

int formal_benchmark_main(int argc, char **argv)
{
    fb_config config;
    fb_files files;
    ocl_runtime runtime;
    cl_program program = NULL;
    size_t i;
    int size_set = 0;
    int result = EXIT_FAILURE;

    memset(&config, 0, sizeof(config));
    memset(&files, 0, sizeof(files));
    memset(&runtime, 0, sizeof(runtime));
    config.sizes[0] = FB_DEFAULT_SIZE;
    config.size_count = 1;
    config.seed = FB_DEFAULT_SEED;
    config.warmups = FB_DEFAULT_WARMUPS;
    config.runs = FB_DEFAULT_RUNS;
    config.selection = FB_SELECTION_ALL;

    for (i = 1; i < (size_t)argc; ++i) {
        if (strcmp(argv[i], "--output") == 0 && i + 1u < (size_t)argc) {
            config.output_dir = argv[++i];
        } else if (strcmp(argv[i], "--sizes") == 0 && i + 1u < (size_t)argc) {
            if (size_set || fb_parse_sizes(argv[++i], config.sizes,
                                           &config.size_count) != 0) {
                fprintf(stderr, "Invalid or conflicting --sizes value\n");
                return EXIT_FAILURE;
            }
            size_set = 1;
        } else if (strcmp(argv[i], "--size") == 0 && i + 1u < (size_t)argc) {
            if (size_set || fb_parse_size(argv[++i], &config.sizes[0]) != 0) {
                fprintf(stderr, "Invalid or conflicting --size value\n");
                return EXIT_FAILURE;
            }
            config.size_count = 1;
            size_set = 1;
        } else if (strcmp(argv[i], "--kernels") == 0 && i + 1u < (size_t)argc) {
            if (fb_parse_selection(argv[++i], &config.selection) != 0) {
                fprintf(stderr, "Invalid --kernels value\n");
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1u < (size_t)argc) {
            unsigned long long parsed;
            if (fb_parse_unsigned(argv[++i], UINT32_MAX, &parsed) != 0 ||
                parsed == 0) {
                fprintf(stderr, "Invalid --seed value\n");
                return EXIT_FAILURE;
            }
            config.seed = (uint32_t)parsed;
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1u < (size_t)argc) {
            unsigned long long parsed;
            if (fb_parse_unsigned(argv[++i], 1000000u, &parsed) != 0) {
                fprintf(stderr, "Invalid --warmup value\n");
                return EXIT_FAILURE;
            }
            config.warmups = (size_t)parsed;
        } else if (strcmp(argv[i], "--runs") == 0 && i + 1u < (size_t)argc) {
            unsigned long long parsed;
            if (fb_parse_unsigned(argv[++i], 1000000u, &parsed) != 0 ||
                parsed == 0) {
                fprintf(stderr, "Invalid --runs value\n");
                return EXIT_FAILURE;
            }
            config.runs = (size_t)parsed;
        } else if (strcmp(argv[i], "--list-devices") == 0) {
            config.list_devices = 1;
        } else if (strcmp(argv[i], "--check-only") == 0) {
            config.check_only = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            fb_usage(argv[0]);
            return EXIT_SUCCESS;
        } else {
            fb_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }
    if (config.list_devices) {
        return ocl_print_devices() == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (config.output_dir == NULL || config.output_dir[0] == '\0' ||
        config.check_only) {
        fprintf(stderr, "Formal mode requires --output and does not use --check-only\n");
        return EXIT_FAILURE;
    }
    for (i = 0; i < config.size_count; ++i) {
        if (config.sizes[i] % 16u != 0) {
            fprintf(stderr, "Size %zu is unsupported: N must be divisible by 16\n",
                    config.sizes[i]);
            return EXIT_FAILURE;
        }
    }
    if (fb_open_outputs(&config, &files) != 0) return EXIT_FAILURE;
    fb_write_command(files.log, argc, argv);
    if (fb_metadata(&files, &config, NULL, "running") != 0) goto cleanup;
    if (ocl_init(&runtime) != 0) goto cleanup;
    if (fb_metadata(&files, &config, &runtime, "running") != 0) goto cleanup;
    if (ocl_build_program_from_file(&runtime, "kernels/gemm_all.cl",
                                    "-cl-std=CL1.2", &program) != 0) goto cleanup;
    if (files.log != NULL) {
        fprintf(files.log, "device=initialized source=kernels/gemm_all.cl\n");
        fflush(files.log);
    }
    for (i = 0; i < config.size_count; ++i) {
        if (fb_run_size(&config, &runtime, program, &files,
                        config.sizes[i]) != 0) goto cleanup;
    }
    if (fb_metadata(&files, &config, &runtime, "complete") != 0) goto cleanup;
    if (files.log != NULL) {
        fprintf(files.log, "status=complete\n");
        fflush(files.log);
    }
    printf("Formal benchmark complete: results written to %s\n",
           config.output_dir);
    result = EXIT_SUCCESS;

cleanup:
    if (result != EXIT_SUCCESS) {
        if (files.log != NULL) {
            fprintf(files.log, "status=failed\n");
            fflush(files.log);
        }
        (void)fb_metadata(&files, &config,
                          runtime.device == NULL ? NULL : &runtime, "failed");
    }
    if (program != NULL) (void)clReleaseProgram(program);
    ocl_release(&runtime);
    fb_close_outputs(&files);
    return result;
}
