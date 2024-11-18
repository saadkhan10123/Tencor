#pragma once
#include "Tensor.h"

enum Activation {
	LINEAR,
	RELU,
	SIGMOID,
	TANH,
	SOFTMAX
};

class Layer {
public:
	virtual Tensor2D<float> forward(Tensor2D<float>& input) = 0;
	virtual int getOutputSize() = 0;
};