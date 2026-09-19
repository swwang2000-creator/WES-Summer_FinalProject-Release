#ifndef OPENCL_RUNTIME_H
#define OPENCL_RUNTIME_H

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif
#include <CL/cl.h>

typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
} ocl_runtime;

int ocl_print_devices(void);
int ocl_init(ocl_runtime *runtime);
void ocl_release(ocl_runtime *runtime);
int ocl_build_program_from_file(const ocl_runtime *runtime,
                                const char *path,
                                const char *options,
                                cl_program *program_out);
const char *ocl_error_name(cl_int error);

#endif
