#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdlib.h>
#include "color.h"
#include "log.h"
#include "sinoscope.h"

int sinoscope_image_openmp(sinoscope_t* sinoscope) {
    if (sinoscope == NULL) {
        LOG_ERROR_NULL_PTR();
        goto fail_exit;
    }

  #pragma omp parallel for schedule(static)
    for (int j = 0; j < sinoscope->height; j++) {
        for (int i = 0; i < sinoscope->width; i++) {
            float px    = sinoscope->dx * j - 2 * M_PI;
            float py    = sinoscope->dy * i - 2 * M_PI;
            float value = 0;

            for (int k = 1; k <= sinoscope->taylor; k += 2) {
                value += sin(px * k * sinoscope->phase1 + sinoscope->time) / k;
                value += cos(py * k * sinoscope->phase0) / k;
            }

            value = (atan(value) - atan(-value)) / M_PI;
            value = (value + 1) * 100;

            pixel_t pixel;
            color_value(&pixel, value, sinoscope->interval, sinoscope->interval_inverse);

            int index = (j * 3) + (i * 3) * sinoscope->width;

            sinoscope->buffer[index + 0] = pixel.bytes[0];
            sinoscope->buffer[index + 1] = pixel.bytes[1];
            sinoscope->buffer[index + 2] = pixel.bytes[2];
        }
    }
        
        return 0;

fail_exit:
    return -1;
}
