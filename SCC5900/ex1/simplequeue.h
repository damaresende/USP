/*
 * Exercise 1. Queue implementation
 *
 *    Created on: Apr 5, 2019
 *        Author: Damares Resende
 *  Organization: University of Sao Paulo (USP)
 *                Institute of Mathematics and Computer Science (ICMC)
 *                Project of Algorithms Class (SCC5900)
 */
#ifndef SIMPLEQUEUE_H_
#define SIMPLEQUEUE_H_

#define MAX_COUPLES 30
#define HASH_SIZE 256

typedef struct node s_node;

struct node {
	int value;
	s_node *next;
};

typedef struct queue s_queue;

struct queue {
	s_node *start;
	s_node *end;
};

s_queue* create();
int is_empty(s_queue *q);
void push(int value, s_queue *q);
int pop(s_queue *q);

#endif /* SIMPLEQUEUE_H_ */
