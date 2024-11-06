#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "log.h"
#include "sinoscope.h"

typedef struct __attribute__((packed)) sinoscope_args {
    // Integer parameters first
    cl_uint width;
    cl_uint height;
    cl_uint taylor;
    cl_uint interval;
    // Float parameters second
    cl_float interval_inverse;
    cl_float time;
    cl_float max;
    cl_float phase0;
    cl_float phase1;
    cl_float dx;
    cl_float dy;
} sinoscope_args_t;

int sinoscope_opencl_init(sinoscope_opencl_t* opencl, cl_device_id opencl_device_id, unsigned int width, unsigned int height) {
    if (opencl == NULL) {
        LOG_ERROR_NULL_PTR();
        return -1;
    }

    cl_int error = CL_SUCCESS;
    opencl->device_id = opencl_device_id;

    opencl->context = clCreateContext(0, 1, &opencl_device_id, NULL, NULL, &error);
    if (error != CL_SUCCESS) {
        LOG_ERROR("Error creating context: %i\n", (int)error);
        goto cleanup;
    }

    opencl->queue = clCreateCommandQueue(opencl->context, opencl_device_id, 0, &error);
    if (error != CL_SUCCESS) {
        LOG_ERROR("Error creating command queue: %i\n", (int)error);
        goto cleanup;
    }

    opencl->buffer = clCreateBuffer(opencl->context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, width * height * 3, NULL, &error);
    if (error != CL_SUCCESS) {
        LOG_ERROR("Error creating buffer: %i\n", (int)error);
        goto cleanup;
    }

    size_t size = 0;
    char* code = NULL;
    opencl_load_kernel_code(&code, &size);
    if (code == NULL) {
        LOG_ERROR("Failed to load kernel code\n");
        goto cleanup;
    }

    cl_program program = clCreateProgramWithSource(opencl->context, 1, (const char**)&code, &size, &error);
    if (error != CL_SUCCESS) {
        LOG_ERROR("Error creating program: %i\n", (int)error);
        free(code);
        goto cleanup;
    }

    error = clBuildProgram(program, 1, &opencl->device_id, "-I " __OPENCL_INCLUDE__, NULL, NULL);
    if (error != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, opencl->device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        
        char* log = malloc(log_size + 1);
        log[log_size] = '\0';

        clGetProgramBuildInfo(program, opencl->device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        LOG_ERROR("Build error: %s\n", log);
        free(log);
        free(code);
        goto cleanup;
    }

    opencl->kernel = clCreateKernel(program, "sinoscope_kernel", &error);
    if (error != CL_SUCCESS) {
        LOG_ERROR("Error creating kernel: %i\n", (int)error);
        free(code);
        goto cleanup;
    }

    free(code);
    clReleaseProgram(program);
    return 0;

cleanup:
    sinoscope_opencl_cleanup(opencl);
    return -1;
}


void sinoscope_opencl_cleanup(sinoscope_opencl_t* opencl) {
    if (opencl->kernel) clReleaseKernel(opencl->kernel);
    if (opencl->queue) clReleaseCommandQueue(opencl->queue);
    if (opencl->buffer) clReleaseMemObject(opencl->buffer);
    if (opencl->context) clReleaseContext(opencl->context);
}

int sinoscope_image_opencl(sinoscope_t* sinoscope) {
    if (sinoscope == NULL) {
        LOG_ERROR_NULL_PTR();
        return -1;
    }

    cl_int error = CL_SUCCESS;

    error = clSetKernelArg(sinoscope->opencl->kernel, 0, sizeof(cl_mem), 
                          &(sinoscope->opencl->buffer));
    if (error != CL_SUCCESS) {
        LOG_ERROR("Error setting buffer arg: %i\n", (int)error);
        return -1;
    }

    // Pack all parameters into the structure
    sinoscope_args_t args = {
        // Integers first
        sinoscope->width,
        sinoscope->height,
        sinoscope->taylor,
        sinoscope->interval,
        // Floats second
        sinoscope->interval_inverse,
        sinoscope->time,
        sinoscope->max,
        sinoscope->phase0,
        sinoscope->phase1,
        sinoscope->dx,
        sinoscope->dy
    };

    error = clSetKernelArg(sinoscope->opencl->kernel, 1, sizeof(args), &args);
    if (error != CL_SUCCESS) {
        LOG_ERROR("Error setting args struct: %i\n", (int)error);
        return -1;
    }

    const size_t global_work_size = sinoscope->width * sinoscope->height;

    error = clEnqueueNDRangeKernel(sinoscope->opencl->queue, 
                                  sinoscope->opencl->kernel,
                                  1, NULL, &global_work_size, 
                                  NULL, 0, NULL, NULL);
    if (error != CL_SUCCESS) {
        LOG_ERROR("Error in enqueue: %i\n", (int)error);
        return -1;
    }

    error = clEnqueueReadBuffer(sinoscope->opencl->queue, 
                              sinoscope->opencl->buffer, CL_TRUE,
                              0, sinoscope->buffer_size, 
                              sinoscope->buffer, 0, NULL, NULL);
    if (error != CL_SUCCESS) {
        LOG_ERROR("Error in read buffer: %i\n", (int)error);
        return -1;
    }

    return 0;
}

