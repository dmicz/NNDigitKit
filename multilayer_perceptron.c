#include "multilayer_perceptron.h"
#include "math_utils.h"

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

void feed_forward(struct Vector** biases, struct Matrix** weights,
	struct Vector* input, struct Vector* output) {

}