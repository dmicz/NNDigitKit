#include "vector.h"

#include <stdio.h>
#include <stdlib.h>
#include "../util/math_utils.h"

struct Vector vector_create(int length) {
	struct Vector vector;
	vector.length = length;
	vector.elements = malloc(length * sizeof(float));
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
	struct Vector result = vector_create(vector1.length);

	for (int i = 0; i < result.length; i++) {
		result.elements[i] = operator(vector1.elements[i], vector2.elements[i]);
	}
	return result;
}

struct Vector vector_unary_operation(const struct Vector vector, unary_operation operator) {
	struct Vector result = vector_create(vector.length);
	for (int i = 0; i < result.length; i++) {
		result.elements[i] = operator(vector.elements[i]);
	}
	return result;
}

void vector_apply_binary_operation(struct Vector vector1, const struct Vector vector2, binary_operation operator) {
	if (vector1.length != vector2.length) {
		printf("Error performing binary operation on vectors: vectors are different sizes\n");
		return;
	}

	for (int i = 0; i < vector1.length; i++) {
		vector1.elements[i] = operator(vector1.elements[i], vector2.elements[i]);
	}
}

void vector_apply_unary_operation(struct Vector vector, unary_operation operator) {
	for (int i = 0; i < vector.length; i++) {
		vector.elements[i] = operator(vector.elements[i]);
	}
}

void vector_free(struct Vector vector) {
	free(vector.elements);
}

float dot_product(const struct Vector vector1, const struct Vector vector2) {
	if (vector1.length != vector2.length) {
		printf("Error performing binary operation on vectors: vectors are different sizes\n");
		return 0;
	}
	int sum = 0;
	for (int i = 0; i < vector1.length; i++) {
		sum += vector1.elements[i] * vector2.elements[i];
	}
}

float func_add_floats(float a, float b) { return a + b; }

float func_subtract_floats(float a, float b) { return a - b; }

float func_hadamard_product(float a, float b) { return a * b; }

float func_negate_float(float a) { return -a; }

float func_zero_float(float a) { return 0; }

float func_std_norm_dist(float a) { return generate_std_norm_dist(); }