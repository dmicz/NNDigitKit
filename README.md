# NNDigitKit

Digit recognition of 28x28 monochrome images using a multilayer perceptron, trained on MNIST.

## Parameters
The `sgd` function in main enables modification of parameters to any desired value:
```c
void sgd(struct Matrix* weights, struct Vector* biases,
	const struct Vector* training_images, const char* training_labels,
	const struct Vector* test_images, const char* test_labels, const int test_size,
	const int training_size, const int minibatch_size, const int epochs,
	const int layer_count, const double learning_rate)
```

The default parameters achieve a maximum 97% accuracy on the default `srand` seed:
![image](https://github.com/dmicz/NNDigitKit/assets/107702866/c8b67a95-0b27-45b2-ba6c-b114db51e472)

## Building
All build settings for building in MSVC are located in the `.vcxproj` file.
