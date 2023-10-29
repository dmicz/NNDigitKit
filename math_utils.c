#include "math_utils.h"
#include <math.h>
#include <stdlib.h>

double generate_std_norm_dist() {
	double irwin_hall_var = 0.0;
	for (int i = 0; i < 12; i++) {
		irwin_hall_var += (double)rand() / (double)RAND_MAX;
	}
	return irwin_hall_var - 6;
}

double sigmoid(const double z) {
	return 1 / (1 + exp(-z));
}