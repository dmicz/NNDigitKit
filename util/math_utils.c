#include "math_utils.h"
#include <math.h>
#include <stdlib.h>

float generate_std_norm_dist() {
	float irwin_hall_var = 0.0;
	for (int i = 0; i < 12; i++) {
		irwin_hall_var += (float)rand() / (float)RAND_MAX;
	}
	return min(1,max(-1,irwin_hall_var - 6));
}

float sigmoid(const float z) {
	return 1 / (1 + exp(-z));
}

float sigmoid_prime(const float z) {
	float ez = exp(-z);
	return ez / ((1 + ez) * (1 + ez));
}

float tanh_prime(const float z) {
	return (1. - tanh(z) * tanh(z));
}

int byte_array_to_big_endian(unsigned char* bytes) {
	return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | (bytes[3]);
}