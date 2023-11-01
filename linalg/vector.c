#include "vector.h"

#include <stdio.h>
#include <stdlib.h>
#include "../util/math_utils.h"

struct Vector create_vector(int length) {
	struct Vector vector;
	vector.length = length;
	vector.elements = malloc(length * sizeof(double));
	if (vector.elements == NULL) {
		printf("Error allocating vector of length %d", length);
		return (struct Vector) { 0, NULL };
	}
	return vector;
}

struct Vector vector_binary_operation(const struct Vector vector1, const struct Vector vector2, binary_operation operator) {
	if (vector1.length != vector2.length) {
		printf("Error performing binary operation on vectors: vectors are different sizes\n");
		return (struct Vector) { 0, NULL };
	}
	struct Vector result = create_vector(vector1.length);

	for (int i = 0; i < result.length; i++) {
		result.elements[i] = operator(vector1.elements[i], vector2.elements[i]);
	}
	return result;
}

struct Vector vector_unary_operation(const struct Vector vector, unary_operation operator) {
	struct Vector result = create_vector(vector.length);
	for (int i = 0; i < result.length; i++) {
		result.elements[i] = operator(vector.elements[i]);
	}
	return result;
}

void apply_vector_binary_operation(struct Vector vector1, const struct Vector vector2, binary_operation operator) {
	if (vector1.length != vector2.length) {
		printf("Error performing binary operation on vectors: vectors are different sizes\n");
		return;
	}

	for (int i = 0; i < vector1.length; i++) {
		vector1.elements[i] = operator(vector1.elements[i], vector2.elements[i]);
	}
}

void apply_vector_unary_operation(struct Vector vector, unary_operation operator) {
	for (int i = 0; i < vector.length; i++) {
		vector.elements[i] = operator(vector.elements[i]);
	}
}

void free_vector(struct Vector vector) {
	free(vector.elements);
}

double dot_product(const struct Vector vector1, const struct Vector vector2) {
	if (vector1.length != vector2.length) {
		printf("Error performing binary operation on vectors: vectors are different sizes\n");
		return 0;
	}
	int sum = 0;
	for (int i = 0; i < vector1.length; i++) {
		sum += vector1.elements[i] * vector2.elements[i];
	}
}

double add_vectors(double a, double b) { return a + b; }

double subtract_vectors(double a, double b) { return a - b; }

double hadamard_product_vectors(double a, double b) { return a * b; }

double negate_vector(double a) { return -a; }

double zero_vector(double a) { return 0; }

double random_init_vector(double a) { return generate_std_norm_dist(); }