#include "matrix.h"
#include <stdlib.h>

struct Matrix* allocate_matrix(const int rows, const int columns) {
	struct Matrix* matrix = malloc(sizeof(struct Matrix));
	matrix->rows = rows;
	matrix->columns = columns;
	matrix->elements = malloc(rows * sizeof(double*));
	for (int i = 0; i < rows; i++) {
		matrix->elements[i] = malloc(columns * sizeof(double));
	}
	return matrix;
}

void zero_matrix(struct Matrix* matrix) {
	for (int i = 0; i < matrix->rows; i++) {
		for (int j = 0; j < matrix->columns; j++) {
			matrix->elements[i][j] = 0.;
		}
	}
}

void free_matrix(struct Matrix* matrix) {
	for (int i = 0; i < matrix->rows; i++) {
		free(matrix->elements[i]);
	}
	free(matrix->elements);
	free(matrix);
}

struct Vector* matrix_vector_multiply(const struct Matrix* matrix, const struct Vector* vector) {
	if (matrix->columns != vector->length) {
		return NULL;
	}
	struct Vector* result = allocate_vector(matrix->rows);
	for (int i = 0; i < matrix->rows; i++) {
		result->elements[i] = 0;
		for (int j = 0; j < matrix->columns; j++) {
			result->elements[i] += matrix->elements[i][j] * vector->elements[j];
		}
	}
	return result;
}