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

double sigmoid_prime(const double z) {
	double ez = exp(-z);
	return ez / ((1 + ez) * (1 + ez));
}

int byte_array_to_big_endian(unsigned char* bytes) {
	return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | (bytes[3]);
}