/*
 * Exercise 1. Queue implementation
 *
 *    Created on: Apr 5, 2019
 *        Author: Damares Resende
 *  Organization: University of Sao Paulo (USP)
 *                Institute of Mathematics and Computer Science (ICMC)
 *                Project of Algorithms Class (SCC5900)
 */
#include <stdlib.h>

#include "simplequeue.h"

s_queue* create() {
	s_queue *q = (s_queue*)malloc(sizeof(s_queue));
	q->start = NULL;
	q->end = NULL;
	return q;
}

int is_empty(s_queue *q) {
	if (q->end == NULL && q->start == NULL) {
		return 1;
	}
	return 0;
}

void push(int value, s_queue *q) {
	s_node *n = (s_node*)malloc(sizeof(s_node));
	n->next = NULL;
	n->value = value;

	if (is_empty(q) == 1){
		q->start = n;
		q->end = n;
	} else {
		q->end->next = n;
		q->end = n;
	}
}

int pop(s_queue *q) {
	if (is_empty(q)) {
		return -1;
	}
	int value = q->start->value;
	s_node *aux = q->start;
	q->start = q->start->next;
	free(aux);

	return value;
}

