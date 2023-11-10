#include <stdio.h>	/* printf()	*/
#include <math.h>	/* exp()	*/
#include <stdlib.h>	/* rand()	*/

#include "util/math_utils.h"
#include "util/file.h"
#include "linalg/vector.h"
#include "linalg/matrix.h"
#include "multilayer_perceptron.h"

int main() {
	srand(1698931523); /* set to constant for debugging purposes */

	int layer_sizes[] = { 784, 100, 10 };
	int layer_count = sizeof(layer_sizes) / sizeof(int);

	FILE* training_images_file = fopen("mnist/train-images.idx3-ubyte", "rb");
	FILE* training_labels_file = fopen("mnist/train-labels.idx1-ubyte", "rb");
	if (training_images_file == NULL | training_labels_file == NULL) {
		printf("Could not open training data files.");
		return -1;
	}

	struct Vector* training_images = NULL;
	char* training_labels = NULL;
	int training_images_count = read_image_file(training_images_file, &training_images);
	int training_labels_count = read_label_file(training_labels_file, &training_labels);

	fclose(training_images_file);
	fclose(training_labels_file);


	struct Matrix* weights = malloc((layer_count - 1) * sizeof(struct Matrix));
	struct Vector* biases = malloc((layer_count - 1) * sizeof(struct Vector));

	for (int i = 0; i < layer_count - 1; i++) {
		weights[i] = matrix_create(layer_sizes[i + 1], layer_sizes[i]);
		biases[i] = vector_create(layer_sizes[i + 1]);
		vector_apply_unary_operation(biases[i], &func_std_norm_dist);
		matrix_random_init(&weights[i]);
	}

	FILE* test_images_file = fopen("mnist/t10k-images.idx3-ubyte", "rb");
	FILE* test_labels_file = fopen("mnist/t10k-labels.idx1-ubyte", "rb");
	if (test_images_file == NULL | test_labels_file == NULL) {
		printf("Could not open training data files.");
		return -1;
	}

	struct Vector* test_images = NULL;
	char* test_labels = NULL;
	int test_images_count = read_image_file(test_images_file, &test_images);
	int test_labels_count = read_label_file(test_labels_file, &test_labels);

	fclose(test_images_file);
	fclose(test_labels_file);

	sgd(weights, biases, training_images, training_labels, test_images, test_labels, 10000, training_images_count, 10, 30, layer_count, 3);


	for (int i = 0; i < layer_count - 1; i++) {
		free_matrix(weights[i]);
		vector_free(biases[i]);
	}
	free(weights);
	free(biases);

	for (int i = 0; i < training_images_count; i++) {
		vector_free(training_images[i]);
	}
	free(training_images);
	free(training_labels);
	for (int i = 0; i < test_images_count; i++) {
		vector_free(test_images[i]);
	}
	free(test_images);
	free(test_labels);
	return 0;
}