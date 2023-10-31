#include <stdlib.h>
#include "vector.h"
#include "../util/math_utils.h"

struct Vector* allocate_vector(const int length) {
	struct Vector* vector = malloc(sizeof(struct Vector));
	vector->length = length;
	vector->elements = malloc(length * sizeof(double));
	return vector;
}

void zero_vector(struct Vector* vector) {
	for (int i = 0; i < vector->length; i++) {
		vector->elements[i] = 0.;
	}
}

void free_vector(struct Vector* vector) {
	free(vector->elements);
	free(vector);
}

void dot_product(const struct Vector* vector1, const struct Vector* vector2, struct Vector* result) {
	if (vector1->length != vector2->length) {
		return NULL;
	}
	if (result != vector1) {
		if (result->length != vector1->length) {
			free(result->elements);
			result->elements = malloc(vector1->length * sizeof(double));
			result->length = vector1->length;
		}

		for (int i = 0; i < vector1->length; i++) {
			result->elements[i] = vector1->elements[i] * vector2->elements[i];
		}
	}
	else {
		for (int i = 0; i < vector1->length; i++) {
			result->elements[i] *= vector2->elements[i];
		}
	}
}


void add_vectors(const struct Vector* vector1, const struct Vector* vector2, struct Vector* result) {
	if (vector1->length != vector2->length) {
		return NULL;
	}
	if (result != vector1) {
		if (result->length != vector1->length) {
			free(result->elements);
			result->elements = malloc(vector1->length * sizeof(double));
			result->length = vector1->length;
		}

		for (int i = 0; i < vector1->length; i++) {
			result->elements[i] = vector1->elements[i] + vector2->elements[i];
		}
	}
	else {
		for (int i = 0; i < vector1->length; i++) {
			result->elements[i] += vector2->elements[i];
		}
	}
}

void subtract_vectors(const struct Vector* vector1, const struct Vector* vector2, struct Vector* result) {
	if (vector1->length != vector2->length) {
		return NULL;
	}
	if (result != vector1) {
		if (result->length != vector1->length) {
			free(result->elements);
			result->elements = malloc(vector1->length * sizeof(double));
			result->length = vector1->length;
		}

		for (int i = 0; i < vector1->length; i++) {
			result->elements[i] = vector1->elements[i] - vector2->elements[i];
		}
	}
	else {
		for (int i = 0; i < vector1->length; i++) {
			result->elements[i] -= vector2->elements[i];
		}
	}
}

void elementwise_product(const struct Vector* vector1, const struct Vector* vector2, struct Vector* result) {
	if (vector1->length != vector2->length) {
		return NULL;
	}
	if (result != vector1) {
		if (result->length != vector1->length) {
			free(result->elements);
			result->elements = malloc(vector1->length * sizeof(double));
			result->length = vector1->length;
		}

		for (int i = 0; i < vector1->length; i++) {
			result->elements[i] = vector1->elements[i] * vector2->elements[i];
		}
	}
	else {
		for (int i = 0; i < vector1->length; i++) {
			result->elements[i] *= vector2->elements[i];
		}
	}
}

void sigmoid_vector(const struct Vector* vector, struct Vector* result) {
	for (int i = 0; i < vector->length; i++) {
		result->elements[i] = sigmoid(vector->elements[i]);
	}
}

void sigmoid_prime_vector(const struct Vector* vector, struct Vector* result) {
	for (int i = 0; i < vector->length; i++) {
		result->elements[i] = sigmoid_prime(vector->elements[i]);
	}
}