#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <string.h>
#include "log.h"
#include "sinoscope.h"

int sinoscope_opencl_init(sinoscope_opencl_t* opencl, cl_device_id opencl_device_id, 
                         unsigned int width, unsigned int height) {
    if (opencl == NULL) {
        LOG_ERROR_NULL_PTR();
        return -1;
    }

    cl_int error = CL_SUCCESS;

    // Create OpenCL context
    opencl->context = clCreateContext(NULL, 1, &opencl_device_id, NULL, NULL, &error);
    if (error != CL_SUCCESS) {
        LOG_ERROR_OPENCL(error);
        return -1;
    }

    // Create command queue
    opencl->queue = clCreateCommandQueue(opencl->context, opencl_device_id, 0, &error);
    if (error != CL_SUCCESS) {
        LOG_ERROR_OPENCL(error);
        return -1;
    }

    // Create program
    const char* source_path = "sinoscope.cl";
    char* source = NULL;
    size_t source_size = 0;
    
    // Read kernel source (implement file reading logic here)
    // ...

    opencl->program = clCreateProgramWithSource(opencl->context, 1, 
                      (const char**)&source, &source_size, &error);
    if (error != CL_SUCCESS) {
        LOG_ERROR_OPENCL(error);
        return -1;
    }

    // Build program
    error = clBuildProgram(opencl->program, 1, &opencl_device_id, NULL, NULL, NULL);
    if (error != CL_SUCCESS) {
        LOG_ERROR_OPENCL(error);
        return -1;
    }

    // Create kernel
    opencl->kernel = clCreateKernel(opencl->program, "sinoscope_kernel", &error);
    if (error != CL_SUCCESS) {
        LOG_ERROR_OPENCL(error);
        return -1;
    }

    // Create buffer
    opencl->size = width * height * 3;
    opencl->buffer = clCreateBuffer(opencl->context, CL_MEM_WRITE_ONLY, 
                                  opencl->size, NULL, &error);
    if (error != CL_SUCCESS) {
        LOG_ERROR_OPENCL(error);
        return -1;
    }

    return 0;
}

void sinoscope_opencl_cleanup(sinoscope_opencl_t* opencl) {
    if (opencl == NULL) return;

    if (opencl->buffer) clReleaseMemObject(opencl->buffer);
    if (opencl->kernel) clReleaseKernel(opencl->kernel);
    if (opencl->program) clReleaseProgram(opencl->program);
    if (opencl->queue) clReleaseCommandQueue(opencl->queue);
    if (opencl->context) clReleaseContext(opencl->context);
}

int sinoscope_image_opencl(sinoscope_t* sinoscope) {
    if (sinoscope == NULL || sinoscope->opencl == NULL) {
        LOG_ERROR_NULL_PTR();
        return -1;
    }

    cl_int error = CL_SUCCESS;

    // Create parameter structure
    sinoscope_params_t params = {
        .width = sinoscope->width,
        .height = sinoscope->height,
        .taylor = sinoscope->taylor,
        .interval = sinoscope->interval,
        .dx = sinoscope->dx,
        .dy = sinoscope->dy,
        .phase0 = sinoscope->phase0,
        .phase1 = sinoscope->phase1,
        .time = sinoscope->time,
        .interval_inverse = sinoscope->interval_inverse
    };

    // Set kernel arguments
    error = clSetKernelArg(sinoscope->opencl->kernel, 0, sizeof(cl_mem), 
                          &sinoscope->opencl->buffer);
    error |= clSetKernelArg(sinoscope->opencl->kernel, 1, sizeof(sinoscope_params_t), 
                           &params);
    
    if (error != CL_SUCCESS) {
        LOG_ERROR_OPENCL(error);
        return -1;
    }

    // Execute kernel
    size_t global_size = sinoscope->width * sinoscope->height;
    error = clEnqueueNDRangeKernel(sinoscope->opencl->queue, sinoscope->opencl->kernel,
                                  1, NULL, &global_size, NULL, 0, NULL, NULL);
    if (error != CL_SUCCESS) {
        LOG_ERROR_OPENCL(error);
        return -1;
    }

    // Read results
    error = clEnqueueReadBuffer(sinoscope->opencl->queue, sinoscope->opencl->buffer,
                               CL_TRUE, 0, sinoscope->opencl->size, sinoscope->buffer,
                               0, NULL, NULL);
    if (error != CL_SUCCESS) {
        LOG_ERROR_OPENCL(error);
        return -1;
    }

    return 0;
}