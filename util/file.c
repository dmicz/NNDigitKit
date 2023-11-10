#include "file.h"

#include <stdlib.h>
#include "../linalg/vector.h"

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
		new_images[i] = vector_create(vector_size);
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