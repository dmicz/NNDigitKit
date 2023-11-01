#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>

struct Matrix create_matrix(const int rows, const int columns) {
	struct Matrix matrix;
	matrix.rows = rows;
	matrix.columns = columns;
	matrix.elements = malloc(rows * sizeof(double*));
	if (matrix.elements == NULL) {
		printf("Error allocating vector of length %d", rows);
		return (struct Matrix) { 0, 0, NULL };
	}

	for (int i = 0; i < rows; i++) {
		matrix.elements[i] = malloc(columns * sizeof(double));
		if (matrix.elements[i] == NULL) {
			printf("Error allocating vector of length %d", columns);
			return (struct Matrix) { 0, 0, NULL };
		}
	}
	return matrix;
}

void free_matrix(struct Matrix* matrix) {
	for (int i = 0; i < matrix->rows; i++) {
		free(matrix->elements[i]);
	}
	free(matrix->elements);
}

struct Vector matrix_vector_multiply(const struct Matrix matrix, const struct Vector vector) {
	if (matrix.columns != vector.length) {
		printf("Error multiplying matrix and vector: sizes not compatible");
		return (struct Vector) { 0, NULL };
	}

	struct Vector result = create_vector(matrix.rows);
	for (int i = 0; i < matrix.rows; i++) {
		result.elements[i] = 0;
		for (int j = 0; j < matrix.columns; j++) {
			result.elements[i] += matrix.elements[i][j] * vector.elements[j];
		}
	}

	return result;
}

void zero_matrix(struct Matrix* matrix) {
	for (int i = 0; i < matrix->rows; i++) {
		for (int j = 0; j < matrix->columns; j++) {
			matrix->elements[i][j] = 0.;
		}
	}
}