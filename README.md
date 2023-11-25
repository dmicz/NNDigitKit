# NNDigitKit
![GitHub License](https://img.shields.io/github/license/dmicz/NNDigitKit)

NNDigitKit is a digit recognition system designed to analyze 28x28 monochrome images using a multilayer perceptron trained on the popular MNIST dataset.

## Table of Contents

- [Project Status](#project-status)
- [Overview](#overview)
- [Parameters](#parameters)
- [Building](#building)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Status

⚠️ Early Development: NNDigitKit is currently in its initial phases and may undergo significant changes. Please note that functionalities and features might be under active development or subject to refinement. Contributions, feedback, and suggestions are highly appreciated as we work towards improving and stabilizing the project.

## Overview

Currently, the core functionality of NNDigitKit in within the sgd function, which trains a configurable multilayer perceptron. This function has various parameters that are editable through the configuration GUI prior to training.

## Parameters

The `sgd` function in main enables modification of parameters to any desired value:
```c
void sgd(struct Matrix* weights, struct Vector* biases,
	const struct Vector* training_images, const char* training_labels,
	const struct Vector* test_images, const char* test_labels, const int test_size,
	const int training_size, const int minibatch_size, const int epochs,
	const int layer_count, const double learning_rate)
```

The function and GUI allows fine-tuning of critical parameters. The default parameters, when used with the default `srand` seed, achieve a maximum of 97% accuracy.

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
The default multilayer perceptron model is configurable through the GUI, which currently has options for layer count and size.

![image](https://github.com/dmicz/NNDigitKit/assets/107702866/60eb31e2-40c3-4d87-b457-ed2c973e9431)

## Contributing

Contributions to enhance and extend NNDigitKit are welcome! Feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the [MIT license](https://opensource.org/license/mit/).
