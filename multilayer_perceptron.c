#include "multilayer_perceptron.h"
#include "util/math_utils.h"
#include "linalg/vector.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


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
	const struct Vector* test_images, const char* test_labels, const int test_size,
	const int training_size, const int minibatch_size, const int epochs,
	const int layer_count, const double learning_rate) {
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
				nabla_weights[j] = create_matrix(weights[j].rows, weights[j].columns);
				nabla_biases[j] = create_vector(biases[j].length);
				zero_matrix(&nabla_weights[j]);
				apply_vector_unary_operation(nabla_biases[j], &zero_vector);
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
					free_vector(z_vectors[j]);
					apply_vector_binary_operation(nabla_biases[j], delta_nabla_biases[j], &add_vectors);
					apply_matrix_binary_operation(nabla_weights[j], delta_nabla_weights[j], &add_vectors);
					free_vector(delta_nabla_biases[j]);
					free_matrix(delta_nabla_weights[j]);
					if(j>0)	free_vector(activations[j]);
				}
				free(activations);
				free(z_vectors);
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
				free_matrix(nabla_weights[j]);
				free_vector(nabla_biases[j]);
			}
			free(nabla_weights);
			free(nabla_biases);
			free(delta_nabla_weights);
			free(delta_nabla_biases);

		}
		int correct = 0;
		double avg_cost = 0.;
		struct Vector output;
		for (int test = 0; test < test_size; test++) {
			output = feed_forward(layer_count, weights, biases, &test_images[test]);
			int ans = 0;
			double cost = 0.;
			for (int j = 0; j < output.length; j++) {
				if (output.elements[j] > output.elements[ans]) ans = j;
				cost += pow(output.elements[j] - (test_labels[test] == j ? 1 : 0), 2);
			}
			if (ans == test_labels[test]) correct++;
			avg_cost += cost;
			free_vector(output);
		}
		avg_cost /= test_size;
		printf("Correct: %d/%d\tAccuracy: %f\tCost: %f\tEpoch:%d/%d\n", correct, test_size, (double)correct / test_size, avg_cost, epoch + 1, epochs);
	}
	free(labels);
}