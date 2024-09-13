#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <threads.h>
#include <unistd.h>

#include "filter.h"
#include "pipeline.h"
#include "queue.h"

#define QUEUE_SIZE 100

struct thread_args {
  queue_t *input_queue;
  queue_t **output_queues;
  int num_output_queues;
  int thread_id;
};

struct load_args {
  image_dir_t *image_dir;
  queue_t **output_queues;
  int num_output_queues;
  int thread_id;
};

struct save_args {
  queue_t *input_queue;
  image_dir_t *image_dir;
  int thread_id;
};

int calculate_optimal_threads_num(long num_cores) {
  int optimal_threads = (num_cores * 12) / 4;
  return (optimal_threads < 2)    ? 2
         : (optimal_threads > 27) ? 27
                                  : optimal_threads;
}

void *image_load_wrapper(void *arg) {
  struct load_args *args = (struct load_args *)arg;
  int image_count = 0;
  while (1) {
    image_t *image = image_dir_load_next(args->image_dir);
    if (image == NULL) {
      for (int i = 0; i < args->num_output_queues; i++) {
        queue_push(args->output_queues[i], NULL);
      }
      break;
    }
    queue_push(args->output_queues[image_count % args->num_output_queues],
               image);
    image_count++;
  }
  return NULL;
}

void *filter_scale_up_wrapper(void *arg) {
  struct thread_args *args = (struct thread_args *)arg;
  while (1) {
    image_t *image = queue_pop(args->input_queue);
    if (image == NULL) {
      queue_push(args->output_queues[args->thread_id], NULL);
      break;
    }
    image_t *scaled_image = filter_scale_up(image, 3);
    image_destroy(image);
    queue_push(args->output_queues[args->thread_id], scaled_image);
  }
  return NULL;
}

void *filter_add_pixel_wrapper(void *arg) {
  pixel_t pixel = {.bytes = {0, 0, 0, 0}};
  struct thread_args *args = (struct thread_args *)arg;
  while (1) {
    image_t *image = queue_pop(args->input_queue);
    if (image == NULL) {
      queue_push(args->output_queues[args->thread_id], NULL);
      break;
    }
    pixel.bytes[0] = (unsigned char)((4 * (image->id + 1)) % 256);
    image_t *pixel_added_image = filter_add_pixel(image, &pixel);
    image_destroy(image);
    queue_push(args->output_queues[args->thread_id], pixel_added_image);
  }
  return NULL;
}

void *image_save_wrapper(void *arg) {
  struct save_args *args = (struct save_args *)arg;
  while (1) {
    image_t *image = queue_pop(args->input_queue);
    if (image == NULL) {
      break;
    }
    image_dir_save(args->image_dir, image);
    printf(".");
    fflush(stdout);
    image_destroy(image);
  }
  return NULL;
}

int pipeline_pthread(image_dir_t *image_dir) {
  long num_cores = sysconf(_SC_NPROCESSORS_ONLN);
  printf("Number of cores: %ld\n", num_cores);
  const int NUM_THREADS = calculate_optimal_threads_num(num_cores);
  printf("Optimal number of threads: %d\n", NUM_THREADS);

  queue_t *loaded_img_queue[NUM_THREADS];
  queue_t *scaled_img_queue[NUM_THREADS];
  queue_t *pixel_added_img_queue[NUM_THREADS];

  pthread_t load_thread;
  pthread_t scale_threads[NUM_THREADS];
  pthread_t pixel_threads[NUM_THREADS];
  pthread_t save_threads[NUM_THREADS];

  struct load_args load_args = {image_dir, loaded_img_queue, NUM_THREADS, 0};
  struct thread_args scale_args[NUM_THREADS];
  struct thread_args pixel_args[NUM_THREADS];
  struct save_args save_args[NUM_THREADS];

  // Create queues
  for (int i = 0; i < NUM_THREADS; i++) {
    loaded_img_queue[i] = queue_create(QUEUE_SIZE);
    scaled_img_queue[i] = queue_create(QUEUE_SIZE);
    pixel_added_img_queue[i] = queue_create(QUEUE_SIZE);
    if (!loaded_img_queue[i] || !scaled_img_queue[i] ||
        !pixel_added_img_queue[i]) {
      fprintf(stderr, "Failed to create queue %d\n", i);
      goto cleanup;
    }
  }

  // Create load thread
  if (pthread_create(&load_thread, NULL, image_load_wrapper, &load_args) != 0) {
    fprintf(stderr, "Error creating load thread\n");
    goto cleanup;
  }

  // Create scale, pixel, and save threads
  for (int i = 0; i < NUM_THREADS; i++) {
    scale_args[i] = (struct thread_args){loaded_img_queue[i], scaled_img_queue,
                                         NUM_THREADS, i};
    pixel_args[i] = (struct thread_args){scaled_img_queue[i],
                                         pixel_added_img_queue, NUM_THREADS, i};
    save_args[i] = (struct save_args){pixel_added_img_queue[i], image_dir, i};

    if (pthread_create(&scale_threads[i], NULL, filter_scale_up_wrapper,
                       &scale_args[i]) != 0 ||
        pthread_create(&pixel_threads[i], NULL, filter_add_pixel_wrapper,
                       &pixel_args[i]) != 0 ||
        pthread_create(&save_threads[i], NULL, image_save_wrapper,
                       &save_args[i]) != 0) {
      fprintf(stderr, "Error creating thread %d\n", i);
      goto cleanup;
    }
  }

  // Join threads
  pthread_join(load_thread, NULL);
  for (int i = 0; i < NUM_THREADS; i++) {
    pthread_join(scale_threads[i], NULL);
    pthread_join(pixel_threads[i], NULL);
    pthread_join(save_threads[i], NULL);
  }

  printf("\n");

cleanup:
  for (int i = 0; i < NUM_THREADS; i++) {
    if (loaded_img_queue[i])
      queue_destroy(loaded_img_queue[i]);
    if (scaled_img_queue[i])
      queue_destroy(scaled_img_queue[i]);
    if (pixel_added_img_queue[i])
      queue_destroy(pixel_added_img_queue[i]);
  }
  return 0;
}
