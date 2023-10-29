#include <stdlib.h>
#include "vector.h"
#include "../util/math_utils.h"

struct Vector* allocate_vector(const int length) {
	struct Vector* vector = malloc(sizeof(struct Vector));
	vector->length = length;
	vector->elements = malloc(length * sizeof(double));
	return vector;
}

void free_vector(struct Vector* vector) {
	free(vector->elements);
	free(vector);
}

struct Vector* dot_product(const struct Vector* vector1, const struct Vector* vector2) {
	if (vector1->length != vector2->length) {
		return NULL;
	}
	struct Vector* result = allocate_vector(vector1->length);
	for (int i = 0; i < vector1->length; i++) {
		result->elements[i] = vector1->elements[i] * vector2->elements[i];
	}
	return result;
}


struct Vector* add_vectors(const struct Vector* vector1, const struct Vector* vector2) {
	if (vector1->length != vector2->length) {
		return NULL;
	}
	struct Vector* result = allocate_vector(vector1->length);
	for (int i = 0; i < vector1->length; i++) {
		result->elements[i] = vector1->elements[i] + vector2->elements[i];
	}
	return result;
}

struct Vector* sigmoid_vector(const struct Vector* vector) {
	struct Vector* result = allocate_vector(vector->length);
	for (int i = 0; i < vector->length; i++) {
		result->elements[i] = sigmoid(vector->elements[i]);
	}
	return result;
}