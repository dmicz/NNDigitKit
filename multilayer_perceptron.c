#include "multilayer_perceptron.h"
#include "util/math_utils.h"
#include "linalg/vector.h"
#include <stdlib.h>


struct Vector create_label_vector(const int nodes, const int label_value) {
	struct Vector label_vector = create_vector(nodes);
	apply_vector_unary_operation(label_vector, &zero_vector);
	label_vector.elements[label_value] = 1.0;
	return label_vector;
}

struct Vector feed_forward(const int layer_count, const struct Matrix* weights,
	const struct Vector* biases,
	const struct Vector* input) {
	struct Vector activation = create_vector(input->length);

	for (int i = 0; i < input->length; i++) {
		activation.elements[i] = input->elements[i];
	}

	for (int i = 0; i < layer_count - 1; i++) {
		struct Vector weighted = matrix_vector_multiply(weights[i], activation);
		apply_vector_binary_operation(weighted, biases[i], &add_vectors);
		apply_vector_unary_operation(weighted, &sigmoid);

		free_vector(activation);
		activation = weighted;
	}

	return activation;
}


void sgd(struct Matrix* weights, struct Vector* biases,
	const struct Vector* training_images, const char* training_labels,
	const int minibatch_size, const int training_data_size,
	const int layer_count, const double learning_rate) {

	int* labels = malloc(training_data_size * sizeof(int));
	for (int i = 0; i < training_data_size; i++) {
		labels[i] = i;
	}
	/* TODO: shuffle */

	/* Iterate through different mini-batches */
	for (int i = 0; i < training_data_size / minibatch_size; i++) {
		/* Allocate parameter change variables */
		struct Matrix* nabla_weights = malloc((layer_count - 1) * sizeof(struct Matrix));
		struct Matrix* delta_nabla_weights = malloc((layer_count - 1) * sizeof(struct Matrix));
		struct Vector* nabla_biases = malloc((layer_count - 1) * sizeof(struct Vector));
		struct Vector* delta_nabla_biases = malloc((layer_count - 1) * sizeof(struct Vector));
		for (int j = 0; j < layer_count - 1; j++) {
			nabla_weights[j] = create_matrix(weights[j].rows, weights[j].columns);
			nabla_biases[j] = create_vector(biases[j].length);
			zero_matrix(&nabla_weights[j]);
			apply_vector_unary_operation(biases[j], &zero_vector);
		}
		/* Backprop through each training sample */
		for (int j = 0; j < minibatch_size; j++) {
			struct Vector image = training_images[labels[i * minibatch_size + j]];
			struct Vector* activations = malloc(layer_count * sizeof(struct Vector));
			activations[0] = image;
			struct Vector* z_vectors = malloc((layer_count - 1) * sizeof(struct Vector));
			for (int k = 0; k < layer_count - 1; k++) {
				z_vectors[k] = matrix_vector_multiply(weights[k], activations[k]);
				apply_vector_binary_operation(z_vectors[k], biases[k], &add_vectors);

				activations[k + 1] = vector_unary_operation(z_vectors[k], &sigmoid);
			}
			struct Vector y = create_label_vector(activations[layer_count - 1].length,
				training_labels[labels[i * minibatch_size + j]]);
			struct Vector delta = vector_binary_operation(activations[layer_count - 1], y, &subtract_vectors);
			apply_vector_unary_operation(z_vectors[layer_count - 2], &sigmoid_prime);
			apply_vector_binary_operation(delta, z_vectors[layer_count - 2], &hadamard_product_vectors);
			delta_nabla_biases[layer_count - 2] = delta;
			delta_nabla_weights[layer_count - 2] = outer_product(delta, activations[layer_count - 2]);

			for (int k = layer_count - 2; k > 0; k--) {
				struct Vector sp = vector_unary_operation(z_vectors[k - 1], &sigmoid_prime);
				struct Matrix tw = transpose(weights[k]);
				delta_nabla_biases[k - 1] = matrix_vector_multiply(tw, delta);
				apply_vector_binary_operation(delta_nabla_biases[k - 1], sp, &hadamard_product_vectors);
				delta = delta_nabla_biases[k - 1];
				delta_nabla_weights[k - 1] = outer_product(delta, activations[k - 1]);
				free_vector(sp);
				free_matrix(tw);
			}

			free_vector(y);
			for (int j = 0; j < layer_count - 1; j++) {
				apply_vector_binary_operation(nabla_biases[j], delta_nabla_biases[j], &add_vectors);
				apply_matrix_binary_operation(nabla_weights[j], delta_nabla_weights[j], &add_vectors);
				free_vector(delta_nabla_biases[j]);
				free_matrix(delta_nabla_weights[j]);
			}
		}

		/* Change parameters according to minibatch */
		for (int j = 0; j < layer_count - 1; j++) {
			for (int k = 0; k < nabla_biases[j].length; k++) {
				nabla_biases[j].elements[k] *= (learning_rate / (double)minibatch_size);
			}
			for (int k = 0; k < nabla_weights[j].rows; k++) {
				for (int l = 0; l < nabla_weights[j].columns; l++) {
					nabla_weights[j].elements[k][l] *= (learning_rate / (double)minibatch_size);
				}
			}
			apply_vector_binary_operation(biases[j], nabla_biases[j], &subtract_vectors);
			apply_matrix_binary_operation(weights[j], nabla_weights[j], &subtract_vectors);
		}
	}
	free(labels);
}

#if 0
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
#endif