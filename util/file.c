#include "file.h"

#include <stdio.h>
#include <stdlib.h>
#include "../linalg/vector.h"
#include "../util/math_utils.h"
#include "../multilayer_perceptron.h"

struct LabeledData read_labeled_image_files(const char* image_file_name, const char* label_file_name) {
	FILE* image_file = fopen(image_file_name, "rb");
	struct LabeledData read_data;

	unsigned char magic_number_bytes[4];
	unsigned char image_count_bytes[4];
	unsigned char row_count_bytes[4];
	unsigned char column_count_bytes[4];

	fread(magic_number_bytes, sizeof(char), 4, image_file);
	fread(image_count_bytes, sizeof(char), 4, image_file);
	fread(row_count_bytes, sizeof(char), 4, image_file);
	fread(column_count_bytes, sizeof(char), 4, image_file);

	int image_count = byte_array_to_big_endian(image_count_bytes);
	int row_count = byte_array_to_big_endian(row_count_bytes);
	int column_count = byte_array_to_big_endian(column_count_bytes);

	int pixel_count = image_count * row_count * column_count;

	unsigned char* new_images_bytes = malloc(pixel_count * sizeof(char));
	fread(new_images_bytes, sizeof(char), pixel_count, image_file);

	int vector_size = row_count * column_count;
	struct Vector* new_images = malloc(image_count * sizeof(struct Vector));
	for (int i = 0; i < image_count; i++) {
		new_images[i] = vector_create(vector_size);
		for (int j = 0; j < vector_size; j++) {
			new_images[i].elements[j] = ((float)new_images_bytes[i * vector_size + j]) / 256.;
		}
	}
	free(new_images_bytes);

	fclose(image_file);

	read_data.images = new_images;
	read_data.size = image_count;

	FILE* label_file = fopen(label_file_name, "rb");
	unsigned char* label_count_bytes[4];

	fread(magic_number_bytes, sizeof(char), 4, label_file);
	fread(label_count_bytes, sizeof(char), 4, label_file);

	int label_count = byte_array_to_big_endian(label_count_bytes);

	char* new_labels = malloc(label_count * sizeof(char));
	fread(new_labels, sizeof(char), label_count, label_file);
	read_data.labels = new_labels;
	
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

void save_multilayerperceptron(struct MultilayerPerceptron model) {
	FILE* model_file = fopen("multilayer-perceptron.bin", "wb");

	fwrite(&model.layer_count, 1, sizeof(int), model_file);
	fwrite(model.layer_sizes, model.layer_count, sizeof(int), model_file);
	for (int i = 0; i < model.layer_count - 1; i++) {
		for (int j = 0; j < model.weights[i].rows; j++) {
			fwrite(model.weights[i].elements[j], model.weights[i].columns, sizeof(float), model_file);
		}
		fwrite(model.biases[i].elements, model.biases[i].length, sizeof(float), model_file);
	}

	fclose(model_file);
}

struct MultilayerPerceptron load_multilayerperceptron(const char* model_file_name) {
	FILE* model_file = fopen(model_file_name, "rb");

	int layer_count;
	fread(&layer_count, 1, sizeof(int), model_file);
	int* layer_sizes = malloc(layer_count * sizeof(int));
	fread(layer_sizes, layer_count, sizeof(int), model_file);
	
	struct MultilayerPerceptron model;
	model = multilayerperceptron_create(layer_count, layer_sizes);
	for (int i = 0; i < model.layer_count - 1; i++) {
		for (int j = 0; j < model.layer_sizes[i + 1]; j++) {
			fread(model.weights[i].elements[j], model.layer_sizes[i], sizeof(float), model_file);
		}
		fread(model.biases[i].elements, model.layer_sizes[i + 1], sizeof(float), model_file);
	}
	
	fclose(model_file);
	return model;
}
