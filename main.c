#include <stdio.h>	/* printf()	*/
#include <math.h>	/* exp()	*/
#include <stdlib.h>	/* rand()	*/

#include "util/math_utils.h"
#include "util/file.h"
#include "linalg/vector.h"
#include "linalg/matrix.h"
#include "multilayer_perceptron.h"

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


	struct Matrix* weights = malloc((layer_count - 1) * sizeof(struct Matrix));
	struct Vector* biases = malloc((layer_count - 1) * sizeof(struct Vector));

	for (int i = 0; i < layer_count - 1; i++) {
		weights[i] = create_matrix(layer_sizes[i + 1], layer_sizes[i]);
		biases[i] = create_vector(layer_sizes[i + 1]);
		apply_vector_unary_operation(biases[i], &random_init_vector);
		random_init_matrix(&weights[i]);
	}

	struct Vector output = feed_forward(layer_count, weights, biases, &training_images[0]);
	for (int i = 0; i < output.length; i++) {
		printf("%d: %f\n", i, output.elements[i]);
	}
	free_vector(output);

	sgd(weights, biases, training_images, training_labels, 20, 100, layer_count, 0.1);
	output = feed_forward(layer_count, weights, biases, &training_images[0]);
	for (int i = 0; i < output.length; i++) {
		printf("%d: %f\n", i, output.elements[i]);
	}
	free_vector(output);

	for (int i = 0; i < layer_count - 1; i++) {
		free_matrix(weights[i]);
		free_vector(biases[i]);
	}
	free(weights);
	free(biases);

	for (int i = 0; i < training_images_count; i++) {
		free_vector(training_images[i]);
	}
	free(training_images);
	free(training_labels);
	return 0;
}