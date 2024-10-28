#include "helpers.cl"

typedef struct {
    int width;
    int height;
    int taylor;
    int interval;
    float dx;
    float dy;
    float phase0;
    float phase1;
    float time;
    float interval_inverse;
} sinoscope_params_t;

__kernel void sinoscope_kernel(__global unsigned char* buffer,
                              __constant sinoscope_params_t* params) {
    int i = get_global_id(0) / params->height;
    int j = get_global_id(0) % params->height;
    
    if (i >= params->width) return;
    
    float px = params->dx * j - 2 * M_PI;
    float py = params->dy * i - 2 * M_PI;
    float value = 0;

    for (int k = 1; k <= params->taylor; k += 2) {
        value += sin(px * k * params->phase1 + params->time) / k;
        value += cos(py * k * params->phase0) / k;
    }

    value = (atan(value) - atan(-value)) / M_PI;
    value = (value + 1) * 100;

    pixel_t pixel;
    color_value(&pixel, value, params->interval, params->interval_inverse);

    int index = (i * 3) + (j * 3) * params->width;
    buffer[index + 0] = pixel.bytes[0];
    buffer[index + 1] = pixel.bytes[1];
    buffer[index + 2] = pixel.bytes[2];
}