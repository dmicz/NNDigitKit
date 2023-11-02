#ifndef MATRIX_H
#define MATRIX_H

#include "vector.h"

struct Matrix {
	int rows, columns;
	double** elements;
};

struct Matrix create_matrix(const int rows, const int columns);

void free_matrix(struct Matrix matrix);

struct Vector matrix_vector_multiply(const struct Matrix matrix, const struct Vector vector);

struct Matrix matrix_binary_operation(const struct Matrix matrix1, const struct Matrix matrix2,
	binary_operation operator);

struct Matrix matrix_unary_operation(const struct Matrix matrix, unary_operation operator);

struct Matrix outer_product(const struct Vector vector1, const struct Vector vector2);

struct Matrix transpose(const struct Matrix matrix);

void apply_matrix_binary_operation(struct Matrix matrix1, const struct Matrix matrix2,
	binary_operation operator);

void apply_matrix_unary_operation(struct Matrix matrix, unary_operation operator);

void zero_matrix(struct Matrix* matrix);

void random_init_matrix(struct Matrix* matrix);

#endif