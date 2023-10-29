#include <stdio.h>	/* printf()	*/
#include <math.h>	/* exp()	*/
#include <stdlib.h>	/* rand()	*/

#include "math_utils.h"
#include "vector.h"

struct Matrix {
	int rows, columns;
	double** elements;
};

void initialize_matrix(struct Matrix* matrix, const int rows, const int columns) {
	matrix->rows = rows;
	matrix->columns = columns;
	matrix->elements = (double**)malloc(rows * sizeof(double*));
	for (int i = 0; i < rows; i++) {
		(matrix->elements)[i] = (double*)malloc(columns * sizeof(double));
	}
}

void free_matrix(struct Matrix* matrix) {
	for (int i = 0; i < matrix->rows; i++) {
		free(matrix->elements[i]);
	}
	free(matrix->elements);
}

int byte_array_to_big_endian(unsigned char* bytes) {
	return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | (bytes[3]);
}

int read_image_file(const FILE* file, struct Vector** images) {
	unsigned char* magic_number_bytes = malloc(4 * sizeof(char));
	unsigned char* image_count_bytes = malloc(4 * sizeof(char));
	unsigned char* row_count_bytes = malloc(4 * sizeof(char));
	unsigned char* column_count_bytes = malloc(4 * sizeof(char));

	fread((void*)magic_number_bytes, sizeof(char), 4, file);
	fread((void*)image_count_bytes, sizeof(char), 4, file);
	fread((void*)row_count_bytes, sizeof(char), 4, file);
	fread((void*)column_count_bytes, sizeof(char), 4, file);

	int image_count = byte_array_to_big_endian(image_count_bytes);
	int row_count = byte_array_to_big_endian(row_count_bytes);
	int column_count = byte_array_to_big_endian(column_count_bytes);

	int pixel_count = image_count * row_count * column_count;

	unsigned char* new_images_bytes = malloc(pixel_count * sizeof(char));
	fread((void*)new_images_bytes, sizeof(char), pixel_count, file);

	int vector_size = row_count * column_count;
	struct Vector* new_images = malloc(image_count * sizeof(struct Vector));
	for (int i = 0; i < image_count; i++) {
		initialize_vector(&(new_images[i]), vector_size);
		for (int j = 0; j < vector_size; j++) {
			new_images[i].elements[j] = ((double)new_images_bytes[i * vector_size + j]) / 256.;
		}
	}

	free(magic_number_bytes);
	free(image_count_bytes);
	free(row_count_bytes);
	free(column_count_bytes);
	free(new_images_bytes);

	*images = new_images;
	return image_count;
}

int read_label_file(const FILE* file, char** labels) {
	unsigned char* magic_number_bytes = malloc(4 * sizeof(char));
	unsigned char* label_count_bytes = malloc(4 * sizeof(char));

	fread((void*)magic_number_bytes, sizeof(char), 4, file);
	fread((void*)label_count_bytes, sizeof(char), 4, file);

	int label_count = byte_array_to_big_endian(label_count_bytes);

	char* new_labels = malloc(label_count * sizeof(char));
	fread((void*)new_labels, sizeof(char), label_count, file);
	*labels = new_labels;

	free(magic_number_bytes);
	free(label_count_bytes);

	return label_count;
}

int main() {
	srand(1337); /* set to constant for debugging purposes */

	int layer_sizes[] = { 784, 16, 16, 10 };
	int layer_count = sizeof(layer_sizes) / sizeof(int);

	FILE* training_images_file = fopen("mnist/train-images.idx3-ubyte", "rb");
	FILE* training_labels_file = fopen("mnist/train-labels.idx1-ubyte", "rb");
	if (training_images_file == NULL | training_labels_file == NULL) {
		printf("Could not open training data files.");
		return -1;
	}

	struct Vector* training_images = NULL;
	char* training_labels = NULL;
	int training_labels_count = read_label_file(training_labels_file, &training_labels);
	int training_images_count = read_image_file(training_images_file, &training_images);

	fclose(training_labels_file);
	fclose(training_images_file);
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			if (training_images[16584].elements[i * 28 + j] > 0.5) {
				printf("#");
			}
			else {
				printf(".");
			}
		}
		printf("\n");
	}
	for (int i = 0; i < training_images_count; i++) {
		free_vector(&training_images[i]);
	}
	free(training_images);
	free(training_labels);
	return 0;
}