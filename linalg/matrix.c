#include "matrix.h"

#include <stdlib.h>
#include <stdio.h>
#include <xmmintrin.h>
#include "../util/math_utils.h"

struct Matrix matrix_create(const int rows, const int columns) {
	struct Matrix matrix;
	matrix.rows = rows;
	matrix.columns = columns;
	matrix.elements = malloc(rows * sizeof(float*));
	if (matrix.elements == NULL) {
		printf("Error allocating vector of length %d", rows);
		return (struct Matrix) { 0, 0, NULL };
	}

	for (int i = 0; i < rows; i++) {
		matrix.elements[i] = malloc(columns * sizeof(float));
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

	struct Vector result = vector_create(matrix.rows);

	for (int i = 0; i < matrix.rows; i++) {
		result.elements[i] = 0.;
		float sum = 0.;
		int j = 0;
		for (; j < matrix.columns - 3; j += 4) {
			sum += matrix.elements[i][j] * vector.elements[j] +
				matrix.elements[i][j + 1] * vector.elements[j + 1] +
				matrix.elements[i][j + 2] * vector.elements[j + 2] +
				matrix.elements[i][j + 3] * vector.elements[j + 3];
		}	
		for (; j < matrix.columns; j++) {
			sum += matrix.elements[i][j] * vector.elements[j];
		}
		result.elements[i] = sum;
	}

	return result;
}

struct Matrix matrix_binary_operation(const struct Matrix matrix1, const struct Matrix matrix2,
	binary_operation operator) {
	if (matrix1.rows != matrix2.rows || matrix1.columns != matrix2.columns) {
		printf("Error performing binary operation on matrices: matrices are different sizes\n");
		return (struct Matrix) { 0, 0, NULL };
	}
	struct Matrix result = matrix_create(matrix1.rows, matrix1.columns);
	for (int i = 0; i < matrix1.rows; i++) {
		for (int j = 0; j < matrix1.columns; j++) {
			result.elements[i][j] = operator(matrix1.elements[i][j], matrix2.elements[i][j]);
		}
	}
	return result;
}

struct Matrix matrix_unary_operation(const struct Matrix matrix, unary_operation operator) {
	struct Matrix result = matrix_create(matrix.rows, matrix.columns);
	for (int i = 0; i < matrix.rows; i++) {
		for (int j = 0; j < matrix.columns; j++) {
			result.elements[i][j] = operator(matrix.elements[i][j]);
		}
	}
	return result;
}

struct Matrix matrix_outer_product(const struct Vector vector1, const struct Vector vector2) {
	struct Matrix result = matrix_create(vector1.length, vector2.length);
	for (int i = 0; i < vector1.length; i++) {
		int j = 0;
		for (; j < vector2.length - 3; j += 4) {
			result.elements[i][j] = vector1.elements[i] * vector2.elements[j];
			result.elements[i][j + 1] = vector1.elements[i] * vector2.elements[j + 1];
			result.elements[i][j + 2] = vector1.elements[i] * vector2.elements[j + 2];
			result.elements[i][j + 3] = vector1.elements[i] * vector2.elements[j + 3];
			}
		for (; j < vector2.length; j++) {
			result.elements[i][j] = vector1.elements[i] * vector2.elements[j];
		}
		/*
		for (int j = 0; j < vector2.length; j++) {
			result.elements[i][j] = vector1.elements[i] * vector2.elements[j];
		}
		*/
	}
	return result;
}

struct Matrix matrix_transpose(const struct Matrix matrix) {
	struct Matrix result = matrix_create(matrix.columns, matrix.rows);
	for (int i = 0; i < matrix.columns; i++) {
		for (int j = 0; j < matrix.rows; j++) {
			result.elements[i][j] = matrix.elements[j][i];
		}
	}
	return result;
}

void matrix_apply_binary_operation(struct Matrix matrix1, const struct Matrix matrix2,
	binary_operation operator) {
	if (matrix1.rows != matrix2.rows || matrix1.columns != matrix2.columns) {
		printf("Error performing binary operation on matrices: matrices are different sizes\n");
		return;
	}
	for (int i = 0; i < matrix1.rows; i++) {
		int j = 0;
		for (; j < matrix1.columns - 3; j += 4) {
			matrix1.elements[i][j] = operator(matrix1.elements[i][j], matrix2.elements[i][j]);
			matrix1.elements[i][j + 1] = operator(matrix1.elements[i][j + 1], matrix2.elements[i][j + 1]);
			matrix1.elements[i][j + 2] = operator(matrix1.elements[i][j + 2], matrix2.elements[i][j + 2]);
			matrix1.elements[i][j + 3] = operator(matrix1.elements[i][j + 3], matrix2.elements[i][j + 3]);
		}
		for (; j < matrix1.columns; j++) {
			matrix1.elements[i][j] = operator(matrix1.elements[i][j], matrix2.elements[i][j]);
		}
		/*for (int j = 0; j < matrix1.columns; j++) {
			matrix1.elements[i][j] = operator(matrix1.elements[i][j], matrix2.elements[i][j]);
		}*/
	}
}

void matrix_apply_unary_operation(struct Matrix matrix, unary_operation operator) {
	for (int i = 0; i < matrix.rows; i++) {
		for (int j = 0; j < matrix.columns; j++) {
			matrix.elements[i][j] = operator(matrix.elements[i][j]);
		}
	}
}

void matrix_zero(struct Matrix* matrix) {
	for (int i = 0; i < matrix->rows; i++) {
		for (int j = 0; j < matrix->columns; j++) {
			matrix->elements[i][j] = 0.;
		}
	}
}

void matrix_random_init(struct Matrix* matrix) {
	for (int i = 0; i < matrix->rows; i++) {
		for (int j = 0; j < matrix->columns; j++) {
			matrix->elements[i][j] = generate_std_norm_dist();
		}
	}
}