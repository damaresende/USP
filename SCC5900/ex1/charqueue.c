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
#include "charqueue.h"

c_queue* create() {
	c_queue *q = (c_queue*)malloc(sizeof(c_queue));
	q->start = NULL;
	q->end = NULL;
	return q;
}

int is_empty(c_queue *q) {
	if (q->end == NULL && q->start == NULL) {
		return 1;
	}
	return 0;
}

void push(char value, c_queue *q){
	c_node *n = (c_node*)malloc(sizeof(c_node));
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

char pop(c_queue *q){
	if (is_empty(q)) {
		return ' ';
	}

	char value = q->start->value;
	c_node *aux = q->start;
	q->start = q->start->next;
	free(aux);

	return value;
}




