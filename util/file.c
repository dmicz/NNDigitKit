#include "file.h"

#include <stdio.h>
#include <stdlib.h>
#include "../linalg/vector.h"

struct LabeledData read_labeled_image_files(const char* image_file_name, const char* label_file_name) {
	FILE* image_file = fopen(image_file_name, "rb");
	struct LabeledData read_data;

	unsigned char* magic_number_bytes = malloc(4 * sizeof(char));
	unsigned char* image_count_bytes = malloc(4 * sizeof(char));
	unsigned char* row_count_bytes = malloc(4 * sizeof(char));
	unsigned char* column_count_bytes = malloc(4 * sizeof(char));

	fread((void*)magic_number_bytes, sizeof(char), 4, image_file);
	fread((void*)image_count_bytes, sizeof(char), 4, image_file);
	fread((void*)row_count_bytes, sizeof(char), 4, image_file);
	fread((void*)column_count_bytes, sizeof(char), 4, image_file);

	int image_count = byte_array_to_big_endian(image_count_bytes);
	int row_count = byte_array_to_big_endian(row_count_bytes);
	int column_count = byte_array_to_big_endian(column_count_bytes);

	int pixel_count = image_count * row_count * column_count;

	unsigned char* new_images_bytes = malloc(pixel_count * sizeof(char));
	fread((void*)new_images_bytes, sizeof(char), pixel_count, image_file);

	int vector_size = row_count * column_count;
	struct Vector* new_images = malloc(image_count * sizeof(struct Vector));
	for (int i = 0; i < image_count; i++) {
		new_images[i] = vector_create(vector_size);
		for (int j = 0; j < vector_size; j++) {
			new_images[i].elements[j] = ((float)new_images_bytes[i * vector_size + j]) / 256.;
		}
	}

	free(magic_number_bytes);
	free(image_count_bytes);
	free(row_count_bytes);
	free(column_count_bytes);
	free(new_images_bytes);

	fclose(image_file);

	read_data.images = new_images;
	read_data.size = image_count;

	FILE* label_file = fopen(label_file_name, "rb");
	magic_number_bytes = malloc(4 * sizeof(char));
	unsigned char* label_count_bytes = malloc(4 * sizeof(char));

	fread((void*)magic_number_bytes, sizeof(char), 4, label_file);
	fread((void*)label_count_bytes, sizeof(char), 4, label_file);

	int label_count = byte_array_to_big_endian(label_count_bytes);

	char* new_labels = malloc(label_count * sizeof(char));
	fread((void*)new_labels, sizeof(char), label_count, label_file);
	read_data.labels = new_labels;

	free(magic_number_bytes);
	free(label_count_bytes);
	
	fclose(label_file);

	return read_data;
}

void free_labeled_data(struct LabeledData data) {
	for (int i = 0; i < data.size; i++) {
		vector_free(data.images[i]);
	}
	free(data.images);
	free(data.labels);
}