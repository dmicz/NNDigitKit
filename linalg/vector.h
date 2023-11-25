#ifndef VECTOR_H
#define VECTOR_H

struct Vector {
	int length;
	float* elements;
};

typedef float (*binary_operation)(float, float);
typedef float (*unary_operation)(float);

struct Vector vector_create(int length);
void vector_free(struct Vector vector);
struct Vector vector_binary_operation(const struct Vector vector1, 
	const struct Vector vector2, binary_operation operator);
struct Vector vector_unary_operation(const struct Vector vector, 
	unary_operation operator);
void vector_apply_binary_operation(struct Vector vector1, 
	const struct Vector vector2, binary_operation operator);
void vector_apply_unary_operation(struct Vector vector, 
	unary_operation operator);

float dot_product(const struct Vector vector1, const struct Vector vector2);
float func_add_floats(float a, float b);
float func_subtract_floats(float a, float b);
float func_hadamard_product(float a, float b);
float func_negate_float(float a);
float func_zero_float(float a);
float func_std_norm_dist(float a);

#endif