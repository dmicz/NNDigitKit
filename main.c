#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

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

void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "Error: %s\n", description);
}

static void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}
}


int main(int argc, char* argv[]) {
	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit()) {
		return 1;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	GLFWwindow* window = glfwCreateWindow(480, 480, "NNDigitKit", NULL, NULL);
	if (!window)
	{
		printf("GLFW could not create window");
		return 1;
	}
	glfwSetKeyCallback(window, glfw_key_callback);

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

	int minibatch_size = 10, epochs = 30, learning_rate = 3;

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		igNewFrame();

		const ImGuiViewport* viewport = igGetMainViewport();
		igSetNextWindowPos(viewport->WorkPos, 0, (struct ImVec2) { 0, 0 });
		igSetNextWindowSize(viewport->WorkSize, 0);

		igBegin("NNDigitKit Config", NULL, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
		igTextColored((struct ImVec4) { 0, 0, 255, 255 }, "NNDigitKit Config");
		
		igText("Random seeding");
		igInputInt("Seed", &seed, NULL, NULL, NULL);
		if (igButton("Use time()", (struct ImVec2) { 0, 0 })) {
			seed = time(NULL);
		}
		igSeparator();

		igText("Layer Settings");

		if (igSliderInt("Layer count", &layer_count, 2, 6, NULL, ImGuiSliderFlags_None)) {
			layer_sizes[layer_count - 1] = 10;
		}

		igSeparator();

		for (int i = 0; i < layer_count; i++) {
			char label[13];
			if (i == 0) {
				igPushStyleColor_U32(ImGuiCol_FrameBg, 0xAA3a2817);
				snprintf(label, sizeof(label), "Input Layer");
			}
			else if (i == layer_count - 1) {
				igPushStyleColor_U32(ImGuiCol_FrameBg, 0xAA3a2817);
				snprintf(label, sizeof(label), "Output Layer");
			}
			else {
				igPushStyleColor_U32(ImGuiCol_FrameBg, 0xFF482E1D);
				snprintf(label, sizeof(label), "Layer %d", i + 1);
			}

			if (igInputInt(label, &layer_sizes[i], NULL, NULL, ((i == 0 || i == layer_count - 1) ? ImGuiInputTextFlags_ReadOnly : ImGuiInputTextFlags_None))) {
				if (layer_sizes[i] < 1) layer_sizes[i] = 1;
				else if (layer_sizes[i] > MAX_LAYER_SIZE) layer_sizes[i] = MAX_LAYER_SIZE;
			}
			igPopStyleColor(1);
		}

		igSeparator();

		igText("Hyperparameters");
		if (igInputInt("Mini-batch Size", &minibatch_size, NULL, NULL, ImGuiInputTextFlags_None)) {
			if (minibatch_size < 1) minibatch_size = 1;
			else if (minibatch_size > 60000) minibatch_size = 60000;
		}
		if (igInputInt("Epochs", &epochs, NULL, NULL, ImGuiInputTextFlags_None)) {
			if (epochs < 1) epochs = 1;
			else if (epochs > 100) epochs = 100;
		}
		if (igInputInt("Learning Rate", &learning_rate, NULL, NULL, ImGuiInputTextFlags_None)) {
			if (learning_rate < 0) learning_rate = 0;
			else if (learning_rate > 100) learning_rate = 100;
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

	struct MultilayerPerceptron neural_network = multilayerperceptron_create(layer_count, layer_sizes);
	srand(seed);

	printf("Loading training data...\n");

	struct LabeledData training_data = read_labeled_image_files("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte");
	struct LabeledData testing_data = read_labeled_image_files("mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte");

	for (int i = 0; i < layer_count - 1; i++) {
		vector_apply_unary_operation(neural_network.biases[i], &func_std_norm_dist);
		matrix_random_init(&neural_network.weights[i]);
	}
	printf("[ ==== TRAINING ==== ]\n");

	sgd(neural_network, training_data, testing_data, minibatch_size, epochs, learning_rate);

	printf("[ ==== TRAINING COMPLETE ==== ]\n");
	multilayerperceptron_free(&neural_network);

	free_labeled_data(training_data);
	free_labeled_data(testing_data);


	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	glfwTerminate();
	return 0;
}