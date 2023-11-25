#include <stdio.h>
#include <math.h>
#include <stdlib.h>


#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#define CIMGUI_USE_GLFW
#define CIMGUI_USE_OPENGL3
#include <cimgui/cimgui.h>
#include <cimgui/cimgui_impl.h>

#define GLFW_INCLUDE_NONE
#include <glad/gl.h>
#include <GLFW/glfw3.h>


#include "util/math_utils.h"
#include "util/file.h"
#include "linalg/vector.h"
#include "linalg/matrix.h"
#include "multilayer_perceptron.h"	

void error_callback(int error, const char* description)
{
	fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}
}


int main(int argc, char* argv[]) {
	glfwSetErrorCallback(error_callback);
	if (!glfwInit()) {
		return 1;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	GLFWwindow* window = glfwCreateWindow(640, 480, "NNDigitKit", NULL, NULL);
	if (!window)
	{
		return 1;
	}
	glfwSetKeyCallback(window, key_callback);

	glfwMakeContextCurrent(window);
	gladLoadGL(glfwGetProcAddress);
	glfwSwapInterval(1);

	struct ImGuiContext* ctx = igCreateContext(NULL);
	struct ImGuiIO* io = igGetIO();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330 core");

	igStyleColorsDark(NULL);

	int width, height;

	int seed = 1698931523;

	const int MAX_LAYER_COUNT = 6;
	const int MAX_LAYER_SIZE = 1000;
	int* layer_sizes = calloc(MAX_LAYER_COUNT, sizeof(int));
	layer_sizes[0] = 784;
	layer_sizes[1] = 100;
	layer_sizes[2] = 10;
	int layer_count = 3;


	double time = glfwGetTime();
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		igNewFrame();

		igBegin("NNDigitKit Config", NULL, 0);

		igInputInt("Seed", &seed, NULL, NULL, NULL);

		if (igSliderInt("Layer count", &layer_count, 2, 6, NULL, ImGuiSliderFlags_None)) {
			layer_sizes[layer_count - 1] = 10;
		}
		for (int i = 0; i < layer_count; i++) {
			char label[8];
			snprintf(label, sizeof(label), "Layer %d", i + 1);
			if (igInputInt(label, &layer_sizes[i], NULL, NULL, ((i == 0 || i == layer_count - 1) ? ImGuiInputTextFlags_ReadOnly : ImGuiInputTextFlags_None))) {
				if (layer_sizes[i] < 1) layer_sizes[i] = 1;
				else if (layer_sizes[i] > MAX_LAYER_SIZE) layer_sizes[i] = MAX_LAYER_SIZE;
			}
		}

		igPushStyleColor_U32(ImGuiCol_Button, 0xCC0000FF);
		igPushStyleColor_U32(ImGuiCol_ButtonHovered, 0xAA0000FF);
		igPushStyleColor_U32(ImGuiCol_ButtonActive, 0x663333FF);
		if (igButton("Start SGD", (struct ImVec2) { 0, 0 })) {
			glfwSetWindowShouldClose(window, true);
		}
		igPopStyleColor(3);
		igEnd();

		// igShowDemoWindow(NULL);

		glfwGetFramebufferSize(window, &width, &height);
		glViewport(0, 0, width, height);
		glClear(GL_COLOR_BUFFER_BIT);
		glClearColor(0.0, 0.0, 0.0, 0.0);

		igRender();
		ImGui_ImplOpenGL3_RenderDrawData(igGetDrawData());



		glfwSwapBuffers(window);
	}
	glfwDestroyWindow(window);

	srand(seed);

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

	free(layer_sizes);

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


	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	glfwTerminate();
	return 0;
}