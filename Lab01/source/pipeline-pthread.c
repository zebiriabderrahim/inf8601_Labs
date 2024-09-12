#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

#include "filter.h"
#include "pipeline.h"
#include "queue.h"

#define QUEUE_SIZE 100
#define NUM_SCALE_THREADS 27
#define NUM_PIXEL_THREADS 27
#define NUM_SAVE_THREADS 27

// Debug macro
#if 0
#define DEBUG_PRINT(fmt, ...) \
    do { fprintf(stderr, "[%s:%d] " fmt, __func__, __LINE__, ##__VA_ARGS__); } while (0)

#else
#define DEBUG_PRINT(fmt, ...){}
#endif

struct thread_args {
    queue_t* input_queue;
    queue_t** output_queues;
    int num_output_queues;
    int thread_id;
};

struct load_args {
    image_dir_t* image_dir;
    queue_t** output_queues;
    int num_output_queues;
    int thread_id;
};

struct save_args {
    queue_t* input_queue;
    image_dir_t* image_dir;
    int thread_id;
};

void* image_load_wrapper(void* arg) {
    struct load_args* args = (struct load_args*)arg;
    int image_count = 0;
    int next_queue = 0;
    while (1) {
        image_t* image = image_dir_load_next(args->image_dir);
        if (image == NULL) {
            for (int i = 0; i < args->num_output_queues; i++) {
                queue_push(args->output_queues[i], NULL);
            }
            DEBUG_PRINT("Load thread: Pushed NULL to all queues, terminating\n");
            break;
        }
        queue_push(args->output_queues[next_queue], image);
        next_queue = (next_queue + 1) % args->num_output_queues;
        image_count++;
        DEBUG_PRINT("Load thread: Loaded and pushed image %d\n", image_count);
    }
    return NULL;
}

void* filter_scale_up_wrapper(void* arg) {
    struct thread_args* args = (struct thread_args*)arg;
    int processed_count = 0;
    int next_queue = 0;
    while (1) {
        image_t* image = queue_pop(args->input_queue);
        if (image == NULL) {
            for (int i = 0; i < args->num_output_queues; i++) {
                queue_push(args->output_queues[i], NULL);
            }
            DEBUG_PRINT("Scale thread %d: Received NULL, pushed NULL to all queues, terminating\n", args->thread_id);
            break;
        }
        image_t* scaled_image = filter_scale_up(image, 3);
        image_destroy(image);
        queue_push(args->output_queues[next_queue], scaled_image);
        next_queue = (next_queue + 1) % args->num_output_queues;
        processed_count++;
        DEBUG_PRINT("Scale thread %d: Processed image %d\n", args->thread_id, processed_count);
    }
    return NULL;
}

void* filter_add_pixel_wrapper(void* arg) {
    pixel_t pixel = {.bytes = {0, 0, 0, 0}};
    struct thread_args* args = (struct thread_args*)arg;
    int processed_count = 0;
    int next_queue = 0;
    while (1) {
        image_t* image = queue_pop(args->input_queue);
        if (image == NULL) {
            for (int i = 0; i < args->num_output_queues; i++) {
                queue_push(args->output_queues[i], NULL);
            }
            DEBUG_PRINT("Pixel thread %d: Received NULL, pushed NULL to all queues, terminating\n", args->thread_id);
            break;
        }
        pixel.bytes[0] = (unsigned char)((4 * (image->id + 1)) % 256);
        image_t* pixel_added_image = filter_add_pixel(image, &pixel);
        image_destroy(image);
        queue_push(args->output_queues[next_queue], pixel_added_image);
        next_queue = (next_queue + 1) % args->num_output_queues;
        processed_count++;
        DEBUG_PRINT("Pixel thread %d: Processed image %d\n", args->thread_id, processed_count);
    }
    return NULL;
}

void* image_save_wrapper(void* arg) {
    struct save_args* args = (struct save_args*)arg;
    int saved_count = 0;
    while (1) {
        image_t* image = queue_pop(args->input_queue);
        if (image == NULL) {
            DEBUG_PRINT("Save thread %d: Received NULL, terminating\n", args->thread_id);
            break;
        }
        image_dir_save(args->image_dir, image);
        printf(".");
        fflush(stdout);
        image_destroy(image);
        saved_count++;
        DEBUG_PRINT("Save thread %d: Saved image %d\n", args->thread_id, saved_count);
    }
    return NULL;
}

int pipeline_pthread(image_dir_t* image_dir) {
    DEBUG_PRINT("Starting pipeline_pthread\n");
    queue_t* loaded_img_queue[NUM_SCALE_THREADS];
    queue_t* scaled_img_queue[NUM_PIXEL_THREADS];
    queue_t* pixel_added_img_queue[NUM_SAVE_THREADS];
    
    for (int i = 0; i < NUM_SCALE_THREADS; i++) {
        loaded_img_queue[i] = queue_create(QUEUE_SIZE);
        if (!loaded_img_queue[i]) {
            fprintf(stderr, "Failed to create loaded_img_queue %d\n", i);
            goto cleanup;
        }
    }
    for (int i = 0; i < NUM_PIXEL_THREADS; i++) {
        scaled_img_queue[i] = queue_create(QUEUE_SIZE);
        if (!scaled_img_queue[i]) {
            fprintf(stderr, "Failed to create scaled_img_queue %d\n", i);
            goto cleanup;
        }
    }
    for (int i = 0; i < NUM_SAVE_THREADS; i++) {
        pixel_added_img_queue[i] = queue_create(QUEUE_SIZE);
        if (!pixel_added_img_queue[i]) {
            fprintf(stderr, "Failed to create pixel_added_img_queue %d\n", i);
            goto cleanup;
        }
    }

    pthread_t load_thread;
    pthread_t scale_threads[NUM_SCALE_THREADS];
    pthread_t pixel_threads[NUM_PIXEL_THREADS];
    pthread_t save_threads[NUM_SAVE_THREADS];

    struct load_args load_args = {image_dir, loaded_img_queue, NUM_SCALE_THREADS, 0};
    struct thread_args scale_args[NUM_SCALE_THREADS];
    struct thread_args pixel_args[NUM_PIXEL_THREADS];
    struct save_args save_args[NUM_SAVE_THREADS];

    // Create load thread
    if (pthread_create(&load_thread, NULL, image_load_wrapper, &load_args) != 0) {
        fprintf(stderr, "Error creating load thread\n");
        goto cleanup;
    }
    DEBUG_PRINT("Created load thread\n");

    // Create scale threads
    for (int i = 0; i < NUM_SCALE_THREADS; i++) {
        scale_args[i] = (struct thread_args){loaded_img_queue[i], scaled_img_queue, NUM_PIXEL_THREADS, i};
        if (pthread_create(&scale_threads[i], NULL, filter_scale_up_wrapper, &scale_args[i]) != 0) {
            fprintf(stderr, "Error creating scale thread %d\n", i);
            goto cleanup;
        }
        DEBUG_PRINT("Created scale thread %d\n", i);
    }

    // Create pixel addition threads
    for (int i = 0; i < NUM_PIXEL_THREADS; i++) {
        pixel_args[i] = (struct thread_args){scaled_img_queue[i], pixel_added_img_queue, NUM_SAVE_THREADS, i};
        if (pthread_create(&pixel_threads[i], NULL, filter_add_pixel_wrapper, &pixel_args[i]) != 0) {
            fprintf(stderr, "Error creating pixel thread %d\n", i);
            goto cleanup;
        }
        DEBUG_PRINT("Created pixel thread %d\n", i);
    }

    // Create save threads
    for (int i = 0; i < NUM_SAVE_THREADS; i++) {
        save_args[i] = (struct save_args){pixel_added_img_queue[i], image_dir, i};
        if (pthread_create(&save_threads[i], NULL, image_save_wrapper, &save_args[i]) != 0) {
            fprintf(stderr, "Error creating save thread %d\n", i);
            goto cleanup;
        }
        DEBUG_PRINT("Created save thread %d\n", i);
    }

    // Join threads
    DEBUG_PRINT("Waiting for threads to complete\n");
    pthread_join(load_thread, NULL);
    DEBUG_PRINT("Load thread joined\n");
    for (int i = 0; i < NUM_SCALE_THREADS; i++) {
        pthread_join(scale_threads[i], NULL);
        DEBUG_PRINT("Scale thread %d joined\n", i);
    }
    for (int i = 0; i < NUM_PIXEL_THREADS; i++) {
        pthread_join(pixel_threads[i], NULL);
        DEBUG_PRINT("Pixel thread %d joined\n", i);
    }
    for (int i = 0; i < NUM_SAVE_THREADS; i++) {
        pthread_join(save_threads[i], NULL);
        DEBUG_PRINT("Save thread %d joined\n", i);
    }

    printf("\n");

cleanup:
    for (int i = 0; i < NUM_SCALE_THREADS; i++) {
        if (loaded_img_queue[i]) queue_destroy(loaded_img_queue[i]);
    }
    for (int i = 0; i < NUM_PIXEL_THREADS; i++) {
        if (scaled_img_queue[i]) queue_destroy(scaled_img_queue[i]);
    }
    for (int i = 0; i < NUM_SAVE_THREADS; i++) {
        if (pixel_added_img_queue[i]) queue_destroy(pixel_added_img_queue[i]);
    }
    DEBUG_PRINT("Pipeline completed\n");
    return 0;
}
