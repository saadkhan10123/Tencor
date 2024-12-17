#pragma once
#include "Layer.h"
#include "Tensor.h"

class Dense : public Layer {
public:
    Dense(int inputSize, int outputSize, Activation activation = LINEAR) : inputSize(inputSize), outputSize(outputSize) {
        weights = Tensor2<double>({ outputSize, inputSize }, InitType::Random);
        biases = Tensor2<double>({ outputSize, 1 }, InitType::Random);
        this->activation = activation;
        cache = nullptr;
    }

    Tensor2<double> forward(const Tensor2<double>& input, bool training = false) override {
        Tensor2<double> z = Tensor2<double>::dot(weights, input) + biases;
        Tensor2<double> a = applyActivation(z, activation);

        if (training) {
            if (cache != nullptr) {
                delete cache;
                cache = nullptr;
            }
            
			cache = new Cache();
            cache->input = input;
			cache->activationCache = z;
		}
        else if (cache != nullptr) {
			delete cache;
			cache = nullptr;
		}

		return a;
    }

    Tensor2<double> backward(const Tensor2<double>& dA, double learningRate) override {
		// Activation backward calculations
		Tensor2<double> dZ = applyActivationDerivative(dA, cache->activationCache, activation);


        // Linear backward calculations
		Tensor2<double> APrev = cache->input;

		Tensor2<double> dW = Tensor2<double>::dot(dZ, Tensor2<double>::transpose(APrev));
		Tensor2<double> dB = Tensor2<double>::sum(dZ, 1);
		Tensor2<double> dAPrev = Tensor2<double>::dot(Tensor2<double>::transpose(weights), dZ);

        weights -= dW * learningRate;
        biases -= dB * learningRate;

        return dAPrev;
    }

    struct Cache {
        Tensor2<double> input;
        Tensor2<double> activationCache;
    };
private:
    int inputSize;
    int outputSize;
    Tensor2<double> weights;
    Tensor2<double> biases;
    Activation activation;
    Cache* cache;
};