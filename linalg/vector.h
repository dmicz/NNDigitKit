#ifndef VECTOR_H
#define VECTOR_H

struct Vector {
	int length;
	double* elements;
};

typedef double (*binary_operation)(double, double);
typedef double (*unary_operation)(double);

struct Vector create_vector(int length);

struct Vector vector_binary_operation(const struct Vector vector1, const struct Vector vector2, 
	binary_operation operator);

struct Vector vector_unary_operation(const struct Vector vector, unary_operation operator);

void free_vector(struct Vector vector);

double dot_product(const struct Vector vector1, const struct Vector vector2);

double add_vectors(double a, double b);

double subtract_vectors(double a, double b);

double hadamard_product_vectors(double a, double b);

double negate_vector(double a);


#endif