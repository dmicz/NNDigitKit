#include <stdio.h>	/* printf()	*/
#include <math.h>	/* exp()	*/
#include <stdlib.h>	/* rand()	*/

/*
 * generate_std_norm_dist
 * --------------------
 * computes an approximate random variable from a standard distribution, 
 * N(0,1), using an Irwin-Hall distribution with n=12 and rand() normalized
 * to (0.0,1.0).
 */
double generate_std_norm_dist() {
	double irwin_hall_var = 0.0;
	for (int i = 0; i < 12; i++) {
		irwin_hall_var += (double)rand() / (double)RAND_MAX;
	}
	return irwin_hall_var - 6;
}

double sigmoid(double z) {
	return 1 / (1 + exp(-z));
}

struct Vector {
	int length;
	double* elements;
};

int main() {
	srand(1337); /* set to constant for debugging purposes */
	
	return 0;
}