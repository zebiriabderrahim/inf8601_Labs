/* DO NOT EDIT THIS FILE */

#ifndef INCLUDE_QUEUE_H_
#define INCLUDE_QUEUE_H_

#include <pthread.h>
#include <stddef.h>

#include "queue.h"

typedef struct queue_node queue_node_t;

typedef struct queue_node {
  void *value;
  queue_node_t *prev;
} queue_node_t;

typedef struct queue {
  size_t size;
  size_t used;
  queue_node_t *tail;
  queue_node_t *head;
  pthread_mutex_t mutex;
  pthread_cond_t modified_item_pushed;
  pthread_cond_t modified_item_poped;
} queue_t;

queue_t *queue_create(size_t size);
void queue_destroy(queue_t *queue);
int queue_push(queue_t *queue, void *ptr);
void *queue_pop(queue_t *queue);

#endif /* INCLUDE_QUEUE_H_ */
