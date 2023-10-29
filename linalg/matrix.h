#ifndef MATRIX_H
#define MATRIX_H

struct Matrix {
	int rows, columns;
	double** elements;
};

void allocate_matrix(struct Matrix* matrix, const int rows, const int columns);
void free_matrix(struct Matrix* matrix);


#endif