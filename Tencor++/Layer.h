#pragma once
#include "Tensor.h"

enum Activation {
	LINEAR,
	RELU,
	SIGMOID,
	TANH,
	SOFTMAX
};

Tensor2<double> applyActivation(Tensor2<double>& input, Activation activation) {
	switch (activation) {
	case LINEAR:
		return input;
	case SIGMOID:
		return input.apply([](double x) { return 1.0 / (1.0 + exp(-x)); });
	case RELU:
		return input.apply([](double x) { return x > 0 ? x : 0.0; });
	case TANH:
		return input.apply([](double x) { return tanh(x); });
	default:
		throw std::invalid_argument("Invalid activation function");
	}
}

Tensor2<double> applyActivationDerivative(const Tensor2<double>& dA, Tensor2<double>& Z, Activation activation) {
	switch (activation) {
	case LINEAR:
		return dA;
	case SIGMOID: {
		Tensor2<double> s = Z.apply([](double x) { return 1.0 / (1.0 + exp(-x)); });
		return dA * s * (1.0 - s);
	}
	case RELU:
		return dA * Z.apply([](double x) { return x > 0 ? 1.0 : 0.0; });
	}
	
}

class Layer {
public:
	virtual Tensor2<double> forward(const Tensor2<double>& input, bool training = false) = 0;
	virtual Tensor2<double> backward(const Tensor2<double>& outputGradient, double learningRate) = 0;
	//int getOutputSize();
};