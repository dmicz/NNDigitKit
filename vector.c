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
