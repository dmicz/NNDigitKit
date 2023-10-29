#include <stdlib.h>
#include "vector.h"

void allocate_vector(struct Vector* vector, const int length) {
	vector->length = length;
	vector->elements = malloc(length * sizeof(double));
	return;
}

void free_vector(struct Vector* vector) {
	free(vector->elements);
}

struct Vector* dot_product(const struct Vector* vector1, const struct Vector* vector2) {
	if (vector1->length != vector2->length) {
		return NULL;
	}
	struct Vector* result = malloc(sizeof(struct Vector));
	allocate_vector(result, vector1->length);
	for (int i = 0; i < vector1->length; i++) {
		result->elements[i] = vector1->elements[i] * vector2->elements[i];
	}
	return result;
}