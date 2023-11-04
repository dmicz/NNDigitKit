#ifndef FILE_H
#define FILE_H

#include <stdio.h>
#include "../linalg/vector.h"

int read_image_file(const FILE* file, struct Vector** images);
int read_label_file(const FILE* file, char** labels);

#endif