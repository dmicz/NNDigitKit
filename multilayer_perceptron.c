#include "multilayer_perceptron.h"
#include "util/math_utils.h"
#include "linalg/vector.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


struct Vector create_label_vector(const int nodes, const int label_value) {
	struct Vector label_vector = vector_create(nodes);
	vector_apply_unary_operation(label_vector, &func_zero_float);
	label_vector.elements[label_value] = 1.0;
	return label_vector;
}

struct Vector feed_forward(const int layer_count, const struct Matrix* weights,
	const struct Vector* biases,
	const struct Vector* input) {
	struct Vector activation = vector_create(input->length);

	for (int i = 0; i < input->length; i++) {
		activation.elements[i] = input->elements[i];
	}

	for (int i = 0; i < layer_count - 1; i++) {
		struct Vector weighted = matrix_vector_multiply(weights[i], activation);
		vector_apply_binary_operation(weighted, biases[i], &func_add_floats);
		vector_apply_unary_operation(weighted, &sigmoid);

		vector_free(activation);
		activation = weighted;
	}

	return activation;
}


void sgd(struct Matrix* weights, struct Vector* biases,
	const struct Vector* training_images, const char* training_labels,
	const struct Vector* test_images, const char* test_labels, const int test_size,
	const int training_size, const int minibatch_size, const int epochs,
	const int layer_count, const float learning_rate) {
	int training_data_size = training_size;
	int* labels = malloc(training_data_size * sizeof(int));
	for (int i = 0; i < training_data_size; i++) {
		labels[i] = i;
	}

	/* Iterate through different mini-batches */
	for (int epoch = 0; epoch < epochs; epoch++) {
		for (int i = 0; i < training_data_size - 1; i++) {
			int j = i + rand() / (RAND_MAX / (training_data_size - i) + 1);
			int t = labels[j];
			labels[j] = labels[i];
			labels[i] = t;
		}
		for (int i = 0; i < training_size / minibatch_size; i++) {
			/* Allocate parameter change variables */
			struct Matrix* nabla_weights = malloc((layer_count - 1) * sizeof(struct Matrix));
			struct Matrix* delta_nabla_weights = malloc((layer_count - 1) * sizeof(struct Matrix));
			struct Vector* nabla_biases = malloc((layer_count - 1) * sizeof(struct Vector));
			struct Vector* delta_nabla_biases = malloc((layer_count - 1) * sizeof(struct Vector));
			for (int j = 0; j < layer_count - 1; j++) {
				nabla_weights[j] = matrix_create(weights[j].rows, weights[j].columns);
				nabla_biases[j] = vector_create(biases[j].length);
				matrix_zero(&nabla_weights[j]);
				vector_apply_unary_operation(nabla_biases[j], &func_zero_float);
			}
			/* Backprop through each training sample */
			for (int j = 0; j < minibatch_size; j++) {
				struct Vector image = training_images[labels[i * minibatch_size + j]];
				struct Vector* activations = malloc(layer_count * sizeof(struct Vector));
				activations[0] = image;
				struct Vector* z_vectors = malloc((layer_count - 1) * sizeof(struct Vector));
				for (int k = 0; k < layer_count - 1; k++) {
					z_vectors[k] = matrix_vector_multiply(weights[k], activations[k]);
					vector_apply_binary_operation(z_vectors[k], biases[k], &func_add_floats);

					activations[k + 1] = vector_unary_operation(z_vectors[k], &sigmoid);
				}
				struct Vector y = create_label_vector(activations[layer_count - 1].length,
					training_labels[labels[i * minibatch_size + j]]);
				struct Vector delta = vector_binary_operation(activations[layer_count - 1], y, &func_subtract_floats);
				vector_apply_unary_operation(z_vectors[layer_count - 2], &sigmoid_prime);
				vector_apply_binary_operation(delta, z_vectors[layer_count - 2], &func_hadamard_product);
				delta_nabla_biases[layer_count - 2] = delta;
				delta_nabla_weights[layer_count - 2] = matrix_outer_product(delta, activations[layer_count - 2]);

				for (int k = layer_count - 2; k > 0; k--) {
					struct Vector sp = vector_unary_operation(z_vectors[k - 1], &sigmoid_prime);
					struct Matrix tw = matrix_transpose(weights[k]);
					delta_nabla_biases[k - 1] = matrix_vector_multiply(tw, delta);
					vector_apply_binary_operation(delta_nabla_biases[k - 1], sp, &func_hadamard_product);
					delta = delta_nabla_biases[k - 1];
					delta_nabla_weights[k - 1] = matrix_outer_product(delta, activations[k - 1]);
					vector_free(sp);
					free_matrix(tw);
				}

				vector_free(y);
				for (int j = 0; j < layer_count - 1; j++) {
					vector_free(z_vectors[j]);
					vector_apply_binary_operation(nabla_biases[j], delta_nabla_biases[j], &func_add_floats);
					matrix_apply_binary_operation(nabla_weights[j], delta_nabla_weights[j], &func_add_floats);
					vector_free(delta_nabla_biases[j]);
					free_matrix(delta_nabla_weights[j]);
					vector_free(activations[j + 1]);
				}
				free(activations);
				free(z_vectors);
			}

			/* Change parameters according to minibatch */
			for (int j = 0; j < layer_count - 1; j++) {
				for (int k = 0; k < nabla_biases[j].length; k++) {
					nabla_biases[j].elements[k] *= (learning_rate / (float)minibatch_size);
				}
				for (int k = 0; k < nabla_weights[j].rows; k++) {
					for (int l = 0; l < nabla_weights[j].columns; l++) {
						nabla_weights[j].elements[k][l] *= (learning_rate / (float)minibatch_size);
					}
				}
				vector_apply_binary_operation(biases[j], nabla_biases[j], &func_subtract_floats);
				matrix_apply_binary_operation(weights[j], nabla_weights[j], &func_subtract_floats);
				free_matrix(nabla_weights[j]);
				vector_free(nabla_biases[j]);
			}
			free(nabla_weights);
			free(nabla_biases);
			free(delta_nabla_weights);
			free(delta_nabla_biases);

		}
		int correct = 0;
		float avg_cost = 0.;
		struct Vector output;
		for (int test = 0; test < test_size; test++) {
			output = feed_forward(layer_count, weights, biases, &test_images[test]);
			int ans = 0;
			float cost = 0.;
			for (int j = 0; j < output.length; j++) {
				if (output.elements[j] > output.elements[ans]) ans = j;
				cost += pow(output.elements[j] - (test_labels[test] == j ? 1 : 0), 2);
			}
			if (ans == test_labels[test]) correct++;
			avg_cost += cost;
			vector_free(output);
		}
		avg_cost /= test_size;
		printf("Correct: %d/%d\tAccuracy: %f\tCost: %f\tEpoch: %d/%d\n", correct, test_size, (float)correct / test_size, avg_cost, epoch + 1, epochs);
	}
	free(labels);
}