#include "tbb/pipeline.h"
#include <stdio.h>

extern "C" {
#include "filter.h"
#include "pipeline.h"
}

class TBBLoadNext {
  image_dir_t *image_dir;

public:
  TBBLoadNext(image_dir_t *image_dir) : image_dir(image_dir) {}

  image_t *operator()(tbb::flow_control &fc) const {
    image_t *out = image_dir_load_next(image_dir);
    if (out != NULL) {
      return out;
    } else {
      fc.stop();
      return NULL;
    }
  }
};

class TBBScaleUp {
public:
  TBBScaleUp() {}
  image_t *operator()(image_t *in) const {
    image_t *out = filter_scale_up(in, 3);
    if (out != NULL) {
      image_destroy(in);
      return out;
    }
    fprintf(stderr, "Error scaling up image %zu\n", in->id);
    return NULL;
  }
};

class TBBAddPixel {
public:
  TBBAddPixel() {}
  image_t *operator()(image_t *in) const {
    pixel_t pixel = {0};
    pixel.bytes[0] = (4 * (in->id + 1)) % 256;

    image_t *out = filter_add_pixel(in, &pixel);
    if (out != NULL) {
      image_destroy(in);
      return out;
    }
    fprintf(stderr, "Error adding pixel to image %zu\n", in->id);
    return NULL;
  }
};

class TBBSave {
  image_dir_t *image_dir;

public:
  TBBSave(image_dir_t *image_dir) : image_dir(image_dir) {}

  void operator()(image_t *in) const {
    image_dir_save(image_dir, in);
    image_destroy(in);
    printf(".");
    fflush(stdout);
  }
};

int pipeline_tbb(image_dir_t *image_dir) {
  tbb::parallel_pipeline(
      16, tbb::make_filter<void, image_t *>(tbb::filter::serial,
                                            TBBLoadNext(image_dir)) &
              tbb::make_filter<image_t *, image_t *>(tbb::filter::parallel,
                                                     TBBScaleUp()) &
              tbb::make_filter<image_t *, image_t *>(tbb::filter::parallel,
                                                     TBBAddPixel()) &
              tbb::make_filter<image_t *, void>(tbb::filter::parallel,
                                                TBBSave(image_dir)));
  printf("\n");
  return 0;
}
