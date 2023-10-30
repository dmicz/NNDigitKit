#ifndef VECTOR_H
#define VECTOR_H

struct Vector {
	int length;
	double* elements;
};

struct Vector* allocate_vector(const int length);

void zero_vector(struct Vector* vector);

void free_vector(struct Vector* vector);

struct Vector* dot_product(const struct Vector* vector1, const struct Vector* vector2);

struct Vector* add_vectors(const struct Vector* vector1, const struct Vector* vector2);

struct Vector* subtract_vectors(const struct Vector* vector1, const struct Vector* vector2);

struct Vector* elementwise_product(const struct Vector* vector1, const struct Vector* vector2);

struct Vector* sigmoid_vector(const struct Vector* vector);

struct Vector* sigmoid_prime_vector(const struct Vector* vector);


#endif