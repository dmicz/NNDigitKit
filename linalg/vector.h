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

void apply_vector_binary_operation(struct Vector vector1, const struct Vector vector2,
	binary_operation operator);

void apply_vector_unary_operation(struct Vector vector, unary_operation operator);

void free_vector(struct Vector vector);

double dot_product(const struct Vector vector1, const struct Vector vector2);

double func_add_doubles(double a, double b);

double func_subtract_doubles(double a, double b);

double func_hadamard_product(double a, double b);

double func_negate_double(double a);

double func_zero_double(double a);

double func_std_norm_dist(double a);

#endif