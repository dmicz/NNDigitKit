#include "multilayer_perceptron.h"
#include "util/math_utils.h"
#include "linalg/vector.h"
#include <stdlib.h>

typedef double vec4[4];

void random_init_vector(struct Vector* vector) {
	for (int i = 0; i < vector->length; i++) {
		vector->elements[i] = generate_std_norm_dist();
	}
}

void random_init_matrix(struct Matrix* matrix) {
	for (int i = 0; i < matrix->rows; i++) {
		for (int j = 0; j < matrix->columns; j++) {
			matrix->elements[i][j] = generate_std_norm_dist();
		}
	}
}

struct Vector* create_label_vector(const int nodes, const int label_value) {
	struct Vector* label_vector = allocate_vector(nodes);
	zero_vector(label_vector);
	label_vector->elements[label_value] = 1.0;
	return label_vector;
}

struct Vector* feed_forward(const int layer_count, const struct Matrix** weights,
	const struct Vector** biases,
	const struct Vector* input) {
	struct Vector* activation = allocate_vector(input->length);

	for (int i = 0; i < input->length; i++) {
		activation->elements[i] = input->elements[i];
	}

	for (int i = 0; i < layer_count - 1; i++) {
		struct Vector* weighted = matrix_vector_multiply(weights[i], activation);
		struct Vector* z = add_vectors(weighted, biases[i]);
		free_vector(weighted);

		struct Vector* sigmoided = sigmoid_vector(z);
		free_vector(z);

		free_vector(activation);
		activation = sigmoided;
	}

	return activation;
}

void backprop(struct Matrix*** nabla_weights, struct Vector*** nabla_biases,
	const struct Matrix** weights, const struct Vector** biases,
	const struct Vector* input, const char y, const int layer_count) {

	struct Vector* activation = allocate_vector(input->length);
	struct Vector** activations = malloc(layer_count * sizeof(struct Vector*));
	struct Vector** z_vectors = malloc(layer_count * sizeof(struct Vector*));

	for (int i = 0; i < input->length; i++) {
		activation->elements[i] = input->elements[i];
	}

	for (int i = 0; i < layer_count - 1; i++) {
		activations[i] = activation;
		struct Vector* weighted = matrix_vector_multiply(weights[i], activation);
		struct Vector* z = add_vectors(weighted, biases[i]);
		free_vector(weighted);

		z_vectors[i] = z;
		struct Vector* sigmoided = sigmoid_vector(z);


		free_vector(activation);
		activation = sigmoided;
	}

	struct Vector* y_vector = create_label_vector(biases[layer_count - 2]->length, y - '0');
	struct Vector* cost = subtract_vectors(activations[layer_count - 1], y_vector);
	free_vector(y_vector);
	struct Vector* sp = sigmoid_prime_vector(z_vectors[layer_count - 1]);
	struct Vector* delta = elementwise_product(cost, sp);
	free_vector(sp);
	free_vector(cost);

	nabla_biases[layer_count - 2] = delta;
	nabla_weights[layer_count - 2] = dot_product(delta, activations[layer_count - 1]);

	for (int i = layer_count - 2; i > 0; i--) {
		struct Vector* z = z_vectors[i];
		struct Vector* sp = sigmoid_prime_vector(z);
		free_vector(z);
		struct Vector* temp = dot_product(weights[i - 1], delta);
		struct Vector* delta = elementwise_product(temp, sp);

		free_vector(sp);
		free_vector(temp);
		nabla_biases[i - 1] = delta;
		nabla_weights[i - 1] = dot_product(delta, activations[i]);
	}

	for (int i = 0; i < layer_count - 1; i++) {
		free_vector(activations[i]);
		free_vector(z_vectors[i]);
	}
	free(activations);
	free(z_vectors);
}

void minibatch_training(const struct Matrix** weights, const struct Vector** biases,
	const struct Vector** training_images, const char* training_labels,
	const int* labels_to_train, const int minibatch_size, const int layer_count) {

	struct Matrix** nabla_weights = malloc((layer_count - 1) * sizeof(struct Matrix*));
	struct Vector** nabla_biases = malloc((layer_count - 1) * sizeof(struct Vector*));

	for (int i = 0; i < layer_count - 1; i++) {
		nabla_weights[i] = allocate_matrix(weights[i]->rows, weights[i]->columns);
		nabla_biases[i] = allocate_vector(biases[i]->length);
		zero_matrix(nabla_weights);
		zero_vector(nabla_biases);
	}

	for (int i = 0; i < minibatch_size; i++) {
		backprop(&nabla_weights, &nabla_biases, weights, biases, 
			training_images[labels_to_train[i]], training_labels[labels_to_train[i]], layer_count);

	}


	for (int i = 0; i < layer_count - 1; i++) {
		free_matrix(nabla_weights[i]);
		free_vector(nabla_biases[i]);
	}
	free(nabla_weights);
	free(nabla_biases);
}

void sgd(const struct Matrix** weights, const struct Vector** biases,
	const struct Vector** training_images, const char* training_labels,
	const int minibatch_size, const int training_data_size, const int layer_count) {

	int* labels = malloc(training_data_size * sizeof(int));
	for (int i = 0; i < training_data_size; i++) {
		labels[i] = i;
	}
	/* TODO: shuffle */

	for (int i = 0; i < training_data_size / minibatch_size; i++) {
		for (int j = 0; j < minibatch_size; j++) {
			minibatch_training(weights, biases, training_images,
				training_labels, labels + i * minibatch_size, minibatch_size, layer_count);
		}
	}
	free(labels);
}