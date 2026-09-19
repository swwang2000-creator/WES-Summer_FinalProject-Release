#include "opencl_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int get_platform_name(cl_platform_id platform, char *name,
                             size_t name_size)
{
    cl_int error;
    size_t required = 0;

    error = clGetPlatformInfo(platform, CL_PLATFORM_NAME, name_size, name,
                              &required);
    if (error != CL_SUCCESS || name_size == 0) {
        return -1;
    }
    name[name_size - 1] = '\0';
    return 0;
}

static int get_device_string(cl_device_id device, cl_device_info info,
                             char *value, size_t value_size)
{
    cl_int error;
    size_t required = 0;

    error = clGetDeviceInfo(device, info, value_size, value, &required);
    if (error != CL_SUCCESS || value_size == 0) {
        return -1;
    }
    value[value_size - 1] = '\0';
    return 0;
}

const char *ocl_error_name(cl_int error)
{
    switch (error) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
    case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    default: return "CL_UNKNOWN_ERROR";
    }
}

static int print_one_platform(cl_platform_id platform, cl_uint platform_index)
{
    cl_int error;
    cl_uint device_count = 0;
    cl_device_id *devices = NULL;
    char platform_name[256] = {0};
    cl_uint i;

    (void)get_platform_name(platform, platform_name, sizeof(platform_name));
    printf("Platform %u: %s\n", platform_index, platform_name);

    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL,
                           &device_count);
    if (error == CL_DEVICE_NOT_FOUND) {
        printf("  no devices\n");
        return 0;
    }
    if (error != CL_SUCCESS || device_count == 0) {
        fprintf(stderr, "  clGetDeviceIDs failed: %s (%d)\n",
                ocl_error_name(error), error);
        return -1;
    }

    devices = calloc(device_count, sizeof(*devices));
    if (devices == NULL) {
        return -1;
    }
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, device_count,
                           devices, NULL);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "  clGetDeviceIDs(list) failed: %s (%d)\n",
                ocl_error_name(error), error);
        free(devices);
        return -1;
    }
    for (i = 0; i < device_count; ++i) {
        char device_name[256] = {0};
        cl_device_type type = 0;
        (void)get_device_string(devices[i], CL_DEVICE_NAME,
                                device_name, sizeof(device_name));
        (void)clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(type),
                              &type, NULL);
        printf("  Device %u: %s (%s)\n", i, device_name,
               (type & CL_DEVICE_TYPE_GPU) != 0 ? "GPU" : "non-GPU");
    }
    free(devices);
    return 0;
}

int ocl_print_devices(void)
{
    cl_int error;
    cl_uint platform_count = 0;
    cl_platform_id *platforms = NULL;
    cl_uint i;
    int result = 0;

    error = clGetPlatformIDs(0, NULL, &platform_count);
    if (error != CL_SUCCESS || platform_count == 0) {
        fprintf(stderr, "clGetPlatformIDs failed: %s (%d)\n",
                ocl_error_name(error), error);
        return -1;
    }
    platforms = calloc(platform_count, sizeof(*platforms));
    if (platforms == NULL) {
        return -1;
    }
    error = clGetPlatformIDs(platform_count, platforms, NULL);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "clGetPlatformIDs(list) failed: %s (%d)\n",
                ocl_error_name(error), error);
        free(platforms);
        return -1;
    }
    for (i = 0; i < platform_count; ++i) {
        if (print_one_platform(platforms[i], i) != 0) {
            result = -1;
        }
    }
    free(platforms);
    return result;
}

static int validate_device_limits(cl_device_id device)
{
    cl_int error;
    size_t max_work_group = 0;
    cl_uint dimensions = 0;
    size_t *work_item_sizes = NULL;
    cl_ulong local_memory = 0;

    error = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                            sizeof(max_work_group), &max_work_group, NULL);
    if (error != CL_SUCCESS) {
        return -1;
    }
    error = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                            sizeof(dimensions), &dimensions, NULL);
    if (error != CL_SUCCESS || dimensions < 2) {
        fprintf(stderr, "GPU has fewer than two work-item dimensions\n");
        return -1;
    }
    work_item_sizes = calloc(dimensions, sizeof(*work_item_sizes));
    if (work_item_sizes == NULL) {
        return -1;
    }
    error = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                            dimensions * sizeof(*work_item_sizes),
                            work_item_sizes, NULL);
    if (error != CL_SUCCESS) {
        free(work_item_sizes);
        return -1;
    }
    error = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE,
                            sizeof(local_memory), &local_memory, NULL);
    if (error != CL_SUCCESS) {
        free(work_item_sizes);
        return -1;
    }
    if (max_work_group < 256 || work_item_sizes[0] < 16 ||
        work_item_sizes[1] < 16 || local_memory < 2048) {
        free(work_item_sizes);
        fprintf(stderr, "GPU does not meet the 16x16 INT32 tile requirements\n");
        return -1;
    }
    free(work_item_sizes);
    printf("GPU limits: max_work_group=%zu, local_memory=%llu bytes\n",
           max_work_group, (unsigned long long)local_memory);
    return 0;
}

int ocl_init(ocl_runtime *runtime)
{
    cl_int error;
    cl_uint platform_count = 0;
    cl_platform_id *platforms = NULL;
    cl_uint i;

    if (runtime == NULL) {
        return -1;
    }
    memset(runtime, 0, sizeof(*runtime));
    error = clGetPlatformIDs(0, NULL, &platform_count);
    if (error != CL_SUCCESS || platform_count == 0) {
        fprintf(stderr, "clGetPlatformIDs failed: %s (%d)\n",
                ocl_error_name(error), error);
        return -1;
    }
    platforms = calloc(platform_count, sizeof(*platforms));
    if (platforms == NULL) {
        return -1;
    }
    error = clGetPlatformIDs(platform_count, platforms, NULL);
    if (error != CL_SUCCESS) {
        free(platforms);
        return -1;
    }
    for (i = 0; i < platform_count && runtime->device == NULL; ++i) {
        cl_uint device_count = 0;
        cl_device_id device = NULL;
        error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1,
                               &device, &device_count);
        if (error == CL_SUCCESS && device_count > 0) {
            runtime->platform = platforms[i];
            runtime->device = device;
        } else if (error != CL_DEVICE_NOT_FOUND) {
            fprintf(stderr, "GPU enumeration failed on platform %u: %s (%d)\n",
                    i, ocl_error_name(error), error);
        }
    }
    free(platforms);
    if (runtime->device == NULL) {
        fprintf(stderr, "No OpenCL GPU device was found\n");
        return -1;
    }

    {
        char platform_name[256] = {0};
        char device_name[256] = {0};
        (void)get_platform_name(runtime->platform, platform_name,
                                sizeof(platform_name));
        (void)get_device_string(runtime->device, CL_DEVICE_NAME, device_name,
                                sizeof(device_name));
        printf("Selected platform: %s\nSelected device: %s\n",
               platform_name, device_name);
    }
    if (validate_device_limits(runtime->device) != 0) {
        return -1;
    }

    {
        cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)runtime->platform, 0
        };
        runtime->context = clCreateContext(properties, 1, &runtime->device,
                                            NULL, NULL, &error);
    }
    if (error != CL_SUCCESS || runtime->context == NULL) {
        fprintf(stderr, "clCreateContext failed: %s (%d)\n",
                ocl_error_name(error), error);
        ocl_release(runtime);
        return -1;
    }
    runtime->queue = clCreateCommandQueue(runtime->context, runtime->device,
                                          CL_QUEUE_PROFILING_ENABLE, &error);
    if (error != CL_SUCCESS || runtime->queue == NULL) {
        fprintf(stderr, "clCreateCommandQueue failed: %s (%d)\n",
                ocl_error_name(error), error);
        ocl_release(runtime);
        return -1;
    }
    return 0;
}

void ocl_release(ocl_runtime *runtime)
{
    if (runtime == NULL) {
        return;
    }
    if (runtime->queue != NULL) {
        (void)clReleaseCommandQueue(runtime->queue);
    }
    if (runtime->context != NULL) {
        (void)clReleaseContext(runtime->context);
    }
    memset(runtime, 0, sizeof(*runtime));
}

int ocl_build_program_from_file(const ocl_runtime *runtime,
                                const char *path,
                                const char *options,
                                cl_program *program_out)
{
    FILE *file;
    long file_size;
    char *source;
    const char *source_pointer;
    size_t read_count;
    cl_int error;
    cl_program program;

    if (runtime == NULL || path == NULL || program_out == NULL) {
        return -1;
    }
    *program_out = NULL;
    file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Cannot open kernel source: %s\n", path);
        return -1;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return -1;
    }
    file_size = ftell(file);
    if (file_size < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return -1;
    }
    source = malloc((size_t)file_size + 1u);
    if (source == NULL) {
        fclose(file);
        return -1;
    }
    read_count = fread(source, 1, (size_t)file_size, file);
    fclose(file);
    if (read_count != (size_t)file_size) {
        free(source);
        return -1;
    }
    source[file_size] = '\0';
    source_pointer = source;
    program = clCreateProgramWithSource(runtime->context, 1,
                                        &source_pointer, NULL, &error);
    free(source);
    if (error != CL_SUCCESS || program == NULL) {
        fprintf(stderr, "clCreateProgramWithSource failed: %s (%d)\n",
                ocl_error_name(error), error);
        return -1;
    }
    error = clBuildProgram(program, 1, &runtime->device, options,
                           NULL, NULL);
    if (error != CL_SUCCESS) {
        size_t log_size = 0;
        char *log = NULL;
        (void)clGetProgramBuildInfo(program, runtime->device,
                                    CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        log = calloc(log_size + 1u, sizeof(*log));
        if (log != NULL) {
            (void)clGetProgramBuildInfo(program, runtime->device,
                                        CL_PROGRAM_BUILD_LOG, log_size, log,
                                        NULL);
            fprintf(stderr, "OpenCL build log for %s:\n%s\n", path, log);
            free(log);
        }
        (void)clReleaseProgram(program);
        return -1;
    }
    *program_out = program;
    return 0;
}
