#include "multilayer_perceptron.h"
#include "util/math_utils.h"
#include "linalg/vector.h"
#include <stdlib.h>

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

struct Vector* feed_forward(int layer_count, const struct Vector** biases, const struct Matrix** weights,
	const struct Vector* input) {
    struct Vector* activation = allocate_vector(input->length);

    for (int i = 0; i < input->length; i++) {
        activation->elements[i] = input->elements[i];
    }

    for (int i = 0; i < layer_count - 1; i++) {
        struct Vector* weighted = matrix_vector_multiply(weights[i], activation);
        struct Vector* biased = add_vectors(weighted, biases[i]);
        free_vector(weighted);

        struct Vector* sigmoided = sigmoid_vector(biased);
        free_vector(biased);

        free_vector(activation);
        activation = sigmoided;
    }

    return activation;
}