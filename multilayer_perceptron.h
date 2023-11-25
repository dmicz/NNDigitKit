#ifndef MULTILAYER_PERCEPTRON_H
#define MULTILAYER_PERCEPTRON_H

#include "linalg/matrix.h"
#include "linalg/vector.h"
#include "util/file.h"

struct MultilayerPerceptron {
	struct Matrix* weights;
	struct Vector* biases;
	int* layer_sizes;
	int layer_count;
};

struct MultilayerPerceptron multilayerperceptron_create(int layer_count, int* layer_sizes);

void multilayerperceptron_free(struct MultilayerPerceptron* mlp);

struct Vector create_label_vector(const int nodes, const int label_value);

struct Vector feed_forward(const int layer_count, const struct Matrix* weights,
	const struct Vector* biases, const struct Vector* input);

void sgd(struct MultilayerPerceptron model, const struct LabeledData training_data,
	const struct LabeledData testing_data, const int minibatch_size,
	const int epochs, const float learning_rate);

#endif