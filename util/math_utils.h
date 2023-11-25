#ifndef MATH_UTILS_H
#define MATH_UTILS_H

float generate_std_norm_dist();

float sigmoid(const float z);

float sigmoid_prime(const float z);

float tanh_prime(const float z);

int byte_array_to_big_endian(unsigned char* bytes);

#endif