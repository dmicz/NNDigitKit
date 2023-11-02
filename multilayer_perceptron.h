#ifndef MULTILAYER_PERCEPTRON_H
#define MULTILAYER_PERCEPTRON_H

#include "linalg/matrix.h"
#include "linalg/vector.h"

struct Vector create_label_vector(const int nodes, const int label_value);

struct Vector feed_forward(const int layer_count, const struct Matrix* weights,
	const struct Vector* biases, const struct Vector* input);

void sgd(struct Matrix* weights, struct Vector* biases, 
	const struct Vector* training_images, const char* training_labels,
	const struct Vector* test_images, const char* test_labels, const int test_size,
	const int minibatch_size, const int training_data_size, const int layer_count, const double learning_rate);

#endif