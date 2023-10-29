#ifndef MULTILAYER_PERCEPTRON_H
#define MULTILAYER_PERCEPTRON_H

#include "linalg/matrix.h"
#include "linalg/vector.h"

void random_init_vector(struct Vector* vector);
void random_init_matrix(struct Matrix* matrix);

void feed_forward(struct Vector** biases, struct Matrix** weights, 
	struct Vector* input, struct Vector* output);

#endif