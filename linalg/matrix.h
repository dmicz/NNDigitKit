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

void zero_matrix(struct Matrix* matrix);

void random_init_matrix(struct Matrix* matrix);

#endif