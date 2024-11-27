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
	virtual Tensor2D<double> forward(Tensor2D<double>& input) = 0;
	virtual int getOutputSize() = 0;
};