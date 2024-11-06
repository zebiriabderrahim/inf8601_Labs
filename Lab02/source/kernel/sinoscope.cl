// sinoscope.cl
#include "helpers.cl"

typedef struct __attribute__((packed)) sinoscope_args {
    // Integer parameters first
    unsigned int width;
    unsigned int height;
    unsigned int taylor;
    unsigned int interval;
    // Float parameters second
    float interval_inverse;
    float time;
    float max;
    float phase0;
    float phase1;
    float dx;
    float dy;
} sinoscope_args_t;

__kernel void sinoscope_kernel(
    __global unsigned char* buffer,  // Shared buffer as first parameter 
    sinoscope_args_t args) {        // Structure as second parameter
    
    const int id = get_global_id(0);
    const int total_pixels = args.width * args.height;
    
    // Skip if we're beyond the image dimensions
    if (id >= total_pixels) return;

    // Calculate i and j from the 1D index
    int i = id / args.height;
    int j = id % args.height;
    
    float px = args.dx * j - 2 * M_PI;
    float py = args.dy * i - 2 * M_PI;
    float value = 0;

    for (int k = 1; k <= args.taylor; k += 2) {
        value += sin(px * k * args.phase1 + args.time) / k;
        value += cos(py * k * args.phase0) / k;
    }

    value = (atan(value) - atan(-value)) / M_PI;
    value = (value + 1) * 100;

    pixel_t pixel;
    color_value(&pixel, value, args.interval, args.interval_inverse);

    int index = (j * 3) + (i * 3) * args.width;
    buffer[index + 0] = pixel.bytes[0];
    buffer[index + 1] = pixel.bytes[1];
    buffer[index + 2] = pixel.bytes[2];
}