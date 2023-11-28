# NNDigitKit
![GitHub License](https://img.shields.io/github/license/dmicz/NNDigitKit)

NNDigitKit is a digit recognition system designed to analyze 28x28 monochrome images using a multilayer perceptron trained on the popular MNIST dataset.

## Table of Contents

- [Project Status](#project-status)
- [Overview](#overview)
- [Building](#building)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Status

⚠️ Early Development: NNDigitKit is currently in its initial phases and may undergo significant changes. Please note that functionalities and features might be under active development or subject to refinement. Contributions, feedback, and suggestions are highly appreciated as we work towards improving and stabilizing the project.

## Overview

![image](https://github.com/dmicz/NNDigitKit/assets/107702866/eab3ae8d-20cb-4b29-a4c9-a25aadf8afed)

NNDigitKit makes training neural networks on the MNIST database simple, featuring a simple GUI to configure various parameters of the neural network, allowing quicker configuration and iterative improvement on neural network hyperparameters.

## Building

The repository includes comprehensive build settings for MSVC located in the `.vcxproj` file.

To get started, follow these steps:

- Clone the repository:
  ```
  git clone https://github.com/dmicz/NNDigitKit.git
  ```
- Open the `.vcxproj` file in Microsoft Visual Studio.
- Build the project using either debug configuration for debugging or release for an optimized executable.

## Usage

The default multilayer perceptron model is configurable through the GUI, which currently has options for layer count and size, training batch sizes, number of epochs, and learning rate.

The GUI allows fine-tuning of critical parameters. Parameters and their functions are listed below:
**Random seeding**
- **Seed**: Integer passed to `srand()` to initialize parameters of neural network, click `Use time()` to replace the seed with the current Unix timestamp.

**Layer settings**
- **Layer count**: Number of total layers (hidden + 1 input + 1 output) of the network. Currently limited from 2 (no hidden layers) to 6.
- **Layer x**: Number of neurons in each hidden layer. Neurons in input/output layers currently cannot be edited (784 input for each pixel in 28x28 image, 10 output for each decimal digit). Currently limited from 1 neuron to 1000 neurons.

**Other hyperparameters**
- **Mini-batch size**: Size of mini-batch used in stochastic gradient descent, limited from 1 to 60000 (size of training data). If training set size isn't divisible my mini-batch size, the remaining examples are ignored (all training data is shuffled every epoch).
- **Epochs**: Number of iterations of SGD before terminating the training.
- **Learning rate**: Float value multiplied by SGD deltas to speed/slow learning. Must be at least 0, is scaled automatically by mini-batch size.

The model is saved in `multilayer-perceptron.bin` according to the `save_multilayerperceptron` in [`file.c`](util/file.c). Use the `load_multilayerperceptron` in the `main` function to load a model from file and continue training.

## Contributing

Contributions to enhance and extend NNDigitKit are welcome! Feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the [MIT license](https://opensource.org/license/mit/).

## Acknowledgments

### Libraries Used
- [GLFW](https://github.com/glfw/glfw) is used for window and context management.
- [Glad](https://github.com/Dav1dde/glad) is used for OpenGL loading.
- [cimgui](https://github.com/cimgui/cimgui) is used for integrating Dear ImGui with C.
- [Dear ImGui](https://github.com/ocornut/imgui) is used to create the GUI.
