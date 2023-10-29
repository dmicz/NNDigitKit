#include "matrix.h"

void initialize_matrix(struct Matrix* matrix, const int rows, const int columns) {
	matrix->rows = rows;
	matrix->columns = columns;
	matrix->elements = malloc(rows * sizeof(double*));
	for (int i = 0; i < rows; i++) {
		(matrix->elements)[i] = malloc(columns * sizeof(double));
	}
}

void free_matrix(struct Matrix* matrix) {
	for (int i = 0; i < matrix->rows; i++) {
		free(matrix->elements[i]);
	}
	free(matrix->elements);
}