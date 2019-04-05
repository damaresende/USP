/*
 * Exercise 1. Queue implementation
 *
 *    Created on: Apr 5, 2019
 *        Author: Damares Resende
 *  Organization: University of Sao Paulo (USP)
 *                Institute of Mathematics and Computer Science (ICMC)
 *                Project of Algorithms Class (SCC5900)
 */
#ifndef CHARQUEUE_H_
#define CHARQUEUE_H_

typedef struct node c_node;

struct node {
	char value;
	c_node *next;
};

typedef struct queue c_queue;

struct queue {
	c_node *start;
	c_node *end;
};

c_queue* create();
int is_empty(c_queue *q);
void push(char value, c_queue *q);
char pop(c_queue *q);

#endif /* CHARQUEUE_H_ */
