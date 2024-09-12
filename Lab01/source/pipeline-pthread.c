#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

#include "filter.h"
#include "pipeline.h"
#include "queue.h"

#define QUEUE_SIZE 400
#define NUM_SCALE_THREADS 27
#define NUM_PIXEL_THREADS 27
#define NUM_SAVE_THREADS 27


struct thread_args {
    queue_t* input_queue;
    queue_t** output_queues;
    int num_output_queues;
};

struct load_args {
    image_dir_t* image_dir;
    queue_t** output_queues;
    int num_output_queues;
};

struct save_args {
    queue_t* input_queue;
    image_dir_t* image_dir;
};

void initialize_queues(queue_t* loaded_img_queue[], queue_t* scaled_img_queue[], queue_t* pixel_added_img_queue[]) {
        for (int i = 0; i < NUM_SCALE_THREADS; i++) {
            loaded_img_queue[i] = queue_create(QUEUE_SIZE);
            if (!loaded_img_queue[i]) {
                fprintf(stderr, "Failed to create loaded_img_queue %d\n", i);
                exit(EXIT_FAILURE);
            }
        }
        for (int i = 0; i < NUM_PIXEL_THREADS; i++) {
            scaled_img_queue[i] = queue_create(QUEUE_SIZE);
            if (!scaled_img_queue[i]) {
                fprintf(stderr, "Failed to create scaled_img_queue %d\n", i);
                exit(EXIT_FAILURE);
            }
        }
        for (int i = 0; i < NUM_SAVE_THREADS; i++) {
            pixel_added_img_queue[i] = queue_create(QUEUE_SIZE);
            if (!pixel_added_img_queue[i]) {
                fprintf(stderr, "Failed to create pixel_added_img_queue %d\n", i);
                exit(EXIT_FAILURE);
            }
        }
    }

void* image_load_wrapper(void* arg) {
    struct load_args* args = (struct load_args*)arg;
    int next_queue = 0;
    while (1) {
        image_t* image = image_dir_load_next(args->image_dir);
        if (image == NULL) {
            for (int i = 0; i < args->num_output_queues; i++) {
                queue_push(args->output_queues[i], NULL);
            }
            break;
        }
        queue_push(args->output_queues[next_queue], image);
        next_queue = (next_queue + 1) % args->num_output_queues;
    }
    return NULL;
}

void* filter_scale_up_wrapper(void* arg) {
    struct thread_args* args = (struct thread_args*)arg;
    int next_queue = 0;
    while (1) {
        image_t* image = queue_pop(args->input_queue);
        if (image == NULL) {
            for (int i = 0; i < args->num_output_queues; i++) {
                queue_push(args->output_queues[i], NULL);
            }
            break;
        }
        image_t* scaled_image = filter_scale_up(image, 3);
        image_destroy(image);
        queue_push(args->output_queues[next_queue], scaled_image);
        next_queue = (next_queue + 1) % args->num_output_queues;
    }
    return NULL;
}

void* filter_add_pixel_wrapper(void* arg) {
    pixel_t pixel = {.bytes = {0, 0, 0, 0}};
    struct thread_args* args = (struct thread_args*)arg;
    int next_queue = 0;
    while (1) {
        image_t* image = queue_pop(args->input_queue);
        if (image == NULL) {
            for (int i = 0; i < args->num_output_queues; i++) {
                queue_push(args->output_queues[i], NULL);
            }
            break;
        }
        pixel.bytes[0] = (unsigned char)((4 * (image->id + 1)) % 256);
        image_t* pixel_added_image = filter_add_pixel(image, &pixel);
        image_destroy(image);
        queue_push(args->output_queues[next_queue], pixel_added_image);
        next_queue = (next_queue + 1) % args->num_output_queues;
    }
    return NULL;
}

void* image_save_wrapper(void* arg) {
    struct save_args* args = (struct save_args*)arg;
    while (1) {
        image_t* image = queue_pop(args->input_queue);
        if (image == NULL) {
            break;
        }
        image_dir_save(args->image_dir, image);
        printf(".");
        fflush(stdout);
        image_destroy(image);
    }
    printf("\n");
    return NULL;
}

int pipeline_pthread(image_dir_t* image_dir) {
    queue_t* loaded_img_queue[NUM_SCALE_THREADS] = {NULL};
    queue_t* scaled_img_queue[NUM_PIXEL_THREADS] = {NULL};
    queue_t* pixel_added_img_queue[NUM_SAVE_THREADS] = {NULL};
    
    
    initialize_queues(loaded_img_queue, scaled_img_queue, pixel_added_img_queue);

    pthread_t load_thread;
    pthread_t scale_threads[NUM_SCALE_THREADS];
    pthread_t pixel_threads[NUM_PIXEL_THREADS];
    pthread_t save_threads[NUM_SAVE_THREADS];

    struct load_args load_args = {image_dir, loaded_img_queue, NUM_SCALE_THREADS};
    struct thread_args scale_args[NUM_SCALE_THREADS];
    struct thread_args pixel_args[NUM_PIXEL_THREADS];
    struct save_args save_args[NUM_SAVE_THREADS];

    // Create load thread
    if (pthread_create(&load_thread, NULL, image_load_wrapper, &load_args) != 0) {
        fprintf(stderr, "Error creating load thread\n");
        goto cleanup;
    }

    // Create scale threads
    for (int i = 0; i < NUM_SCALE_THREADS; i++) {
        scale_args[i] = (struct thread_args){loaded_img_queue[i], scaled_img_queue, NUM_PIXEL_THREADS};
        if (pthread_create(&scale_threads[i], NULL, filter_scale_up_wrapper, &scale_args[i]) != 0) {
            fprintf(stderr, "Error creating scale thread %d\n", i);
            goto cleanup;
        }
    }

    // Create pixel addition threads
    for (int i = 0; i < NUM_PIXEL_THREADS; i++) {
        pixel_args[i] = (struct thread_args){scaled_img_queue[i], pixel_added_img_queue, NUM_SAVE_THREADS};
        if (pthread_create(&pixel_threads[i], NULL, filter_add_pixel_wrapper, &pixel_args[i]) != 0) {
            fprintf(stderr, "Error creating pixel thread %d\n", i);
            goto cleanup;
        }
    }
    // Create save threads
    for (int i = 0; i < NUM_SAVE_THREADS; i++) {
        save_args[i] = (struct save_args){pixel_added_img_queue[i], image_dir};
        if (pthread_create(&save_threads[i], NULL, image_save_wrapper, &save_args[i]) != 0) {
            fprintf(stderr, "Error creating save thread %d\n", i);
            goto cleanup;
        }
    }

    // Join threads
    pthread_join(load_thread, NULL);
    for (int i = 0; i < NUM_SCALE_THREADS; i++) {
        pthread_join(scale_threads[i], NULL);
    }
    for (int i = 0; i < NUM_PIXEL_THREADS; i++) {
        pthread_join(pixel_threads[i], NULL);
    }
    for (int i = 0; i < NUM_SAVE_THREADS; i++) {
        pthread_join(save_threads[i], NULL);
    }

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
    return 0;
}