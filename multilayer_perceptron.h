#ifndef MULTILAYER_PERCEPTRON_H
#define MULTILAYER_PERCEPTRON_H

#include "linalg/matrix.h"
#include "linalg/vector.h"

void random_init_vector(struct Vector* vector);
void random_init_matrix(struct Matrix* matrix);

struct Vector* feed_forward(int layer_count, const struct Vector** biases, const struct Matrix** weights,
	const struct Vector* input);

#endif