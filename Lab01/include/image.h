/* DO NOT EDIT THIS FILE */

#ifndef INCLUDE_IMAGE_H_
#define INCLUDE_IMAGE_H_

#include <dirent.h>
#include <stdbool.h>
#include <stddef.h>
#include <sys/types.h>

typedef struct pixel {
  unsigned char bytes[4];
} pixel_t;

typedef struct image {
  size_t id;
  size_t width;
  size_t height;
  pixel_t *pixels;
} image_t;

static inline pixel_t *image_get_pixel(image_t *image, unsigned int x,
                                       unsigned int y) {
  if (x >= image->width || y >= image->height) {
    return NULL;
  }

  return &image->pixels[x + y * image->width];
}

image_t *image_create(size_t id, size_t width, size_t height);
image_t *image_create_from_png(char *filename);
image_t *image_copy(image_t *image);
void image_destroy(image_t *image);
int image_save_png(image_t *image, char *filename);

typedef struct image_dir {
  const char *input_dir_name;
  const char *output_dir_name;
  const char *save_prefix;
  size_t load_current;
  bool stop;
} image_dir_t;

image_t *image_dir_load_next(image_dir_t *image_dir);
int image_dir_save(image_dir_t *image_dir, image_t *image);

void image_dir_reset(image_dir_t *image_dir, const char *input_dir_name,
                     const char *output_dir_name, const char *save_prefix);

#endif /* INCLUDE_IMAGE_H_ */
