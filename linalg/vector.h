#ifndef VECTOR_H
#define VECTOR_H

struct Vector {
	int length;
	double* elements;
};

struct Vector* allocate_vector(const int length);

void zero_vector(struct Vector* vector);

void free_vector(struct Vector* vector);

void dot_product(const struct Vector* vector1, const struct Vector* vector2, struct Vector* result);

void add_vectors(const struct Vector* vector1, const struct Vector* vector2, struct Vector* result);

void subtract_vectors(const struct Vector* vector1, const struct Vector* vector2, struct Vector* result);

void elementwise_product(const struct Vector* vector1, const struct Vector* vector2, struct Vector* result);

void sigmoid_vector(const struct Vector* vector, struct Vector* result);

void sigmoid_prime_vector(const struct Vector* vector, struct Vector* result);


#endif