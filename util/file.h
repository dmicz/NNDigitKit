#ifndef FILE_H
#define FILE_H

#include <stdio.h>
#include "../linalg/vector.h"

struct LabeledData {
	struct Vector* images;
	char* labels;
	int size;
};

struct LabeledData read_labeled_image_files(const char* image_file_name, const char* label_file_name);

void free_labeled_data(struct LabeledData data);

#endif