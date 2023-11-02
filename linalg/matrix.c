#include "matrix.h"

#include <stdlib.h>
#include <stdio.h>
#include "../util/math_utils.h"

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

void free_matrix(struct Matrix matrix) {
	for (int i = 0; i < matrix.rows; i++) {
		free(matrix.elements[i]);
	}
	free(matrix.elements);
}

struct Vector matrix_vector_multiply(const struct Matrix matrix, const struct Vector vector) {
	if (matrix.columns != vector.length) {
		printf("Error multiplying matrix and vector: sizes not compatible");
		return (struct Vector) { 0, NULL };
	}

	struct Vector result = create_vector(matrix.rows);
	for (int i = 0; i < matrix.rows; i++) {
		result.elements[i] = 0.;
		for (int j = 0; j < matrix.columns; j++) {
			result.elements[i] += matrix.elements[i][j] * vector.elements[j];
		}
	}

	return result;
}

struct Matrix matrix_binary_operation(const struct Matrix matrix1, const struct Matrix matrix2,
	binary_operation operator) {
	if (matrix1.rows != matrix2.rows || matrix1.columns != matrix2.columns) {
		printf("Error performing binary operation on matrices: matrices are different sizes\n");
		return (struct Matrix) { 0, 0, NULL };
	}
	struct Matrix result = create_matrix(matrix1.rows, matrix1.columns);
	for (int i = 0; i < matrix1.rows; i++) {
		for (int j = 0; j < matrix1.columns; j++) {
			result.elements[i][j] = operator(matrix1.elements[i][j], matrix2.elements[i][j]);
		}
	}
	return result;
}

struct Matrix matrix_unary_operation(const struct Matrix matrix, unary_operation operator) {
	struct Matrix result = create_matrix(matrix.rows, matrix.columns);
	for (int i = 0; i < matrix.rows; i++) {
		for (int j = 0; j < matrix.columns; j++) {
			result.elements[i][j] = operator(matrix.elements[i][j]);
		}
	}
	return result;
}

struct Matrix outer_product(const struct Vector vector1, const struct Vector vector2) {
	struct Matrix result = create_matrix(vector1.length, vector2.length);
	for (int i = 0; i < vector1.length; i++) {
		for (int j = 0; j < vector2.length; j++) {
			result.elements[i][j] = vector1.elements[i] * vector2.elements[j];
		}
	}
	return result;
}

struct Matrix transpose(const struct Matrix matrix) {
	struct Matrix result = create_matrix(matrix.columns, matrix.rows);
	for (int i = 0; i < matrix.columns; i++) {
		for (int j = 0; j < matrix.rows; j++) {
			result.elements[i][j] = matrix.elements[j][i];
		}
	}
	return result;
}

void apply_matrix_binary_operation(struct Matrix matrix1, const struct Matrix matrix2,
	binary_operation operator) {
	if (matrix1.rows != matrix2.rows || matrix1.columns != matrix2.columns) {
		printf("Error performing binary operation on matrices: matrices are different sizes\n");
		return;
	}
	for (int i = 0; i < matrix1.rows; i++) {
		for (int j = 0; j < matrix1.columns; j++) {
			matrix1.elements[i][j] = operator(matrix1.elements[i][j], matrix2.elements[i][j]);
		}
	}
}

void apply_matrix_unary_operation(struct Matrix matrix, unary_operation operator) {
	for (int i = 0; i < matrix.rows; i++) {
		for (int j = 0; j < matrix.columns; j++) {
			matrix.elements[i][j] = operator(matrix.elements[i][j]);
		}
	}
}

void zero_matrix(struct Matrix* matrix) {
	for (int i = 0; i < matrix->rows; i++) {
		for (int j = 0; j < matrix->columns; j++) {
			matrix->elements[i][j] = 0.;
		}
	}
}

void random_init_matrix(struct Matrix* matrix) {
	for (int i = 0; i < matrix->rows; i++) {
		for (int j = 0; j < matrix->columns; j++) {
			matrix->elements[i][j] = generate_std_norm_dist();
		}
	}
}