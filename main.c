#include <stdio.h>	/* printf()	*/
#include <math.h>	/* exp()	*/
#include <stdlib.h>	/* rand()	*/

/*
 * generate_std_norm_dist
 * --------------------
 * computes an approximate random variable from a standard distribution,
 * N(0,1), using an Irwin-Hall distribution with n=12 and rand() normalized
 * to (0.0,1.0).
 */
double generate_std_norm_dist() {
	double irwin_hall_var = 0.0;
	for (int i = 0; i < 12; i++) {
		irwin_hall_var += (double)rand() / (double)RAND_MAX;
	}
	return irwin_hall_var - 6;
}

double sigmoid(const double z) {
	return 1 / (1 + exp(-z));
}

struct Vector {
	int length;
	double* elements;
};

void initialize_vector(struct Vector* vector, const int length) {
	vector->length = length;
	vector->elements = (double*)malloc(length * sizeof(double));
}

void free_vector(struct Vector* vector) {
	free(vector->elements);
}

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

void read_image_file(const FILE* file, struct Vector* images) {

}

void read_label_file(const FILE* file, char** labels) {
	unsigned char* magic_number_bytes = malloc(4 * sizeof(char));
	unsigned char* num_labels_bytes = malloc(4 * sizeof(char));
	fread((void*)magic_number_bytes, sizeof(char), 4, file);
	fread((void*)num_labels_bytes, sizeof(char), 4, file);
	int num_labels = (num_labels_bytes[0] << 24) | (num_labels_bytes[1] << 16) | (num_labels_bytes[2] << 8) | (num_labels_bytes[3]);
	char* new_labels = malloc(num_labels * sizeof(char));
	fread((void*)new_labels, sizeof(char), num_labels, file);
	*labels = new_labels;
#if 0
	labels = (int*)malloc(num_labels * sizeof(int));

	for (int i = 0; i < &num_labels; i++) {
		fread((void))
	}
#endif
}

int main() {
	srand(1337); /* set to constant for debugging purposes */

	int layer_sizes[] = { 784, 16, 16, 10 };
	int num_layers = sizeof(layer_sizes) / sizeof(int);

	FILE* training_images_file = fopen("mnist/train-images.idx3-ubyte", "rb");
	FILE* training_labels_file = fopen("mnist/train-labels.idx1-ubyte", "rb");
	if (training_images_file == NULL | training_labels_file == NULL) {
		printf("Could not open training data files.");
		return -1;
	}

	struct Vector* training_images = NULL;
	char* training_labels = NULL;
	read_label_file(training_labels_file, &training_labels);
	printf("%c", training_labels[0] + '0');
	return 0;
}