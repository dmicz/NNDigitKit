#ifndef MATRIX_H
#define MATRIX_H

#include "vector.h"

struct Matrix {
	int rows, columns;
	float** elements;
};

struct Matrix matrix_create(const int rows, const int columns);
void free_matrix(struct Matrix matrix);
struct Vector matrix_vector_multiply(const struct Matrix matrix, 
	const struct Vector vector);
struct Matrix matrix_binary_operation(const struct Matrix matrix1, 
	const struct Matrix matrix2, binary_operation operator);
struct Matrix matrix_unary_operation(const struct Matrix matrix, 
	unary_operation operator);
struct Matrix matrix_outer_product(const struct Vector vector1, 
	const struct Vector vector2);
struct Matrix matrix_transpose(const struct Matrix matrix);
void matrix_apply_binary_operation(struct Matrix matrix1, 
	const struct Matrix matrix2, binary_operation operator);
void matrix_apply_unary_operation(struct Matrix matrix, 
	unary_operation operator);
void matrix_zero(struct Matrix* matrix);
void matrix_random_init(struct Matrix* matrix);

#endif