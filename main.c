#include <stdio.h>	/* printf()	*/
#include <math.h>	/* exp()	*/
#include <stdlib.h>	/* rand()	*/

#include "math_utils.h"
#include "vector.h"
#include "matrix.h"
#include "file.h"

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