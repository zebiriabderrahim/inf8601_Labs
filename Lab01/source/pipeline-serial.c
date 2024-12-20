/* DO NOT EDIT THIS FILE */

#include <stdio.h>

#include "filter.h"
#include "pipeline.h"

int pipeline_serial(image_dir_t *image_dir) {
  pixel_t pixel = {.bytes = {0, 0, 0, 0}};
  while (1) {
    image_t *image1 = image_dir_load_next(image_dir);
    if (image1 == NULL) {
      break;
    }

    image_t *image2 = filter_scale_up(image1, 3);
    image_destroy(image1);
    if (image2 == NULL) {
      goto fail_exit;
    }

    pixel.bytes[0] = (pixel.bytes[0] + 4) % 256;
    image_t *image3 = filter_add_pixel(image2, &pixel);
    image_destroy(image2);
    if (image3 == NULL) {
      goto fail_exit;
    }

    image_dir_save(image_dir, image3);
    printf(".");
    fflush(stdout);
    image_destroy(image3);
  }

  printf("\n");
  return 0;

fail_exit:
  return -1;
}
