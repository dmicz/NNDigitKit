#ifndef MULTILAYER_PERCEPTRON_H
#define MULTILAYER_PERCEPTRON_H

#include "linalg/matrix.h"
#include "linalg/vector.h"

void random_init_vector(struct Vector* vector);
void random_init_matrix(struct Matrix* matrix);

struct Vector* create_label_vector(const int nodes, const int label_value);

struct Vector* feed_forward(const int layer_count, const struct Matrix** weights,
	const struct Vector** biases, const struct Vector* input);

void backprop(struct Matrix* nabla_weights, struct Vector* nabla_biases,
	const struct Matrix** weights, const struct Vector** biases,
	const struct Vector* input, const struct Vector* y, const int layer_count);

void minibatch_training(const struct Matrix** weights, const struct Vector** biases,
	const struct Vector** training_images, const char* training_labels, 
	const int* labels_to_train, const int minibatch_size, const int layer_count);

void sgd(const struct Matrix** weights,	const struct Vector** biases, 
	const struct Vector** training_images, const char* training_labels,
	const int minibatch_size, const int training_data_size, const int layer_count);

#endif