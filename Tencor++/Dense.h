#pragma once
#include "Layer.h"
#include "Tensor.h"

class Dense : public Layer {
public:
    Dense(int inputSize, int outputSize, Activation activation = LINEAR) : inputSize(inputSize), outputSize(outputSize) {
        weights = Tensor2D<double>({ inputSize, outputSize }, InitType::Random);
        biases = Tensor2D<double>({ 1, outputSize }, InitType::Random);
    }

    Tensor2D<double> forward(const Tensor2D<double>& input) const {
        return Tensor2D<double>::dot(input, weights) + biases;
    }

    Tensor2D<double> backward(const Tensor2D<double>& input, const Tensor2D<double>& outputGradient, double learningRate) {
        Tensor2D<double> dW = Tensor2D<double>::dot(Tensor2D<double>::transpose(input), outputGradient) / input.shape[0];
        Tensor2D<double> db = Tensor2D<double>::sum(outputGradient, 0) / input.shape[0];
        Tensor2D<double> inputGradient = Tensor2D<double>::dot(outputGradient, Tensor2D<double>::transpose(weights));

        weights -= dW * learningRate;
        biases -= db * learningRate;

        return inputGradient;
    }

private:
    int inputSize;
    int outputSize;
    Tensor2D<double> weights;
    Tensor2D<double> biases;
};