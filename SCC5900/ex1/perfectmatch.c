/*
 * Exercise 1. Implement Gale-Shapley algorithm
 *
 *    Created on: Apr 2, 2019
 *        Author: Damares Resende
 *  Organization: University of Sao Paulo (USP)
 *                Institute of Mathematics and Computer Science (ICMC)
 *                Project of Algorithms Class (SCC5900)
 */

#include <stdio.h>
#include <stdlib.h>
#include "charqueue.h"

int main(void) {
	int n_tests, t, i, j;
	int n_couples;
	char name;

	scanf("%d\n", &n_tests);

	for (t = 0; t < n_tests; t++) {
		scanf("%d\n", &n_couples);
		char *mn_names = (char*) malloc(n_couples * sizeof(char));
		char *wm_names = (char*) malloc(n_couples * sizeof(char));

		c_queue **mn_preferences = (c_queue**) malloc(n_couples * sizeof(c_queue*));
		c_queue **wm_preferences = (c_queue**) malloc(n_couples * sizeof(c_queue*));

		for(i = 0; i < n_couples; i++) {
			scanf("%c ", mn_names + i);
		}

		for(i = 0; i < n_couples; i++) {
			scanf("%c ", wm_names + i);
		}

		for(i = 0; i < n_couples; i++) {
			scanf("%c:", &name);
			*mn_preferences = *mn_preferences + i;
			*mn_preferences = create();

			for(j = 0; j < n_couples; j++) {
				scanf("%c ", &name);
				push(name, *mn_preferences);
			}
		}

		for(i = 0; i < n_couples; i++) {
			scanf("%c:", &name);
			*mn_preferences = *mn_preferences + i;
			*wm_preferences = create();

			for(j = 0; j < n_couples; j++) {
				scanf("%c ", &name);
				push(name, *wm_preferences);
			}
		}

		for(i = 0; i < n_couples; i++) {
			printf("%c %c\n", *(mn_names + i), *(wm_names + i));
		}

		free(mn_names);
		free(wm_names);
		free(mn_preferences);
		free(wm_preferences);
	}

	return 0;
}





