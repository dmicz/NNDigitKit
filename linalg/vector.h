#ifndef VECTOR_H
#define VECTOR_H

struct Vector {
	int length;
	double* elements;
};

void allocate_vector(struct Vector* vector, const int length);

void free_vector(struct Vector* vector);

#endif