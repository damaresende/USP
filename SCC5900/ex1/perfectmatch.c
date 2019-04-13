/*
 * Exercise 1. Implementation of Gale-Shapley algorithm
 *
 *    Created on: Apr 2, 2019
 *        Author: Damares Resende
 *  Organization: University of Sao Paulo (USP)
 *                Institute of Mathematics and Computer Science (ICMC)
 *                Project of Algorithms Class (SCC5900)
 */

#include <stdio.h>
#include <stdlib.h>
#include "simplequeue.h"

void find_matches(int men[MAX_COUPLES][2], int women[MAX_COUPLES][2], s_queue *mn_preferences[MAX_COUPLES],
		s_queue *wm_preferences[MAX_COUPLES], int n_couples) {

	int i;
	s_queue *free_men = create();

	for(i = 0; i < n_couples; i++) {
		push(men[i], free_men);
	}

	while(!is_empty(free_men)) {
		int m = pop(free_men);
		for (i = 0; i < n_couples; i++) {
			int w = pop(mn_preferences[0]);
		}
	}

}

int main(void) {
	char name;
	int men[MAX_COUPLES][2];
	int women[MAX_COUPLES][2];

	char names_hash[HASH_SIZE];
	int n_tests, t, i, j, n_couples;
	s_queue *mn_preferences[MAX_COUPLES];
	s_queue *wm_preferences[MAX_COUPLES];

	scanf("%d\n", &n_tests);

	for (t = 0; t < n_tests; t++) {
		scanf("%d\n", &n_couples);

		for(i = 0; i < n_couples; i++) {
			scanf("%c ", &name);
			men[i][1] = -1;
			men[i][0] = name % HASH_SIZE;
			names_hash[name % HASH_SIZE] = name;
		}

		for(i = 0; i < n_couples; i++) {
			scanf("%c ", &name);
			women[i][1] = -1;
			women[i][0] = name % HASH_SIZE;
			names_hash[name % HASH_SIZE] = name;
		}

		for(i = 0; i < n_couples; i++) {
			scanf("%c:", &name);
			mn_preferences[i] = create();

			for(j = 0; j < n_couples; j++) {
				scanf("%c ", &name);
				push(name % HASH_SIZE, mn_preferences[i]);
			}
		}

		for(i = 0; i < n_couples; i++) {
			scanf("%c:", &name);
			wm_preferences[i] = create();

			for(j = 0; j < n_couples; j++) {
				scanf("%c ", &name);
				push(name % HASH_SIZE, wm_preferences[i]);
			}
		}
	}

	return 0;
}





