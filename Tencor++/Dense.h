#pragma once
#include "Layer.h"
#include "Tensor.h"
#include "Model.h"
#include <fstream>  // Include this header for file I/O

class Dense : public Layer {
public:
    Dense(int inputSize, int outputSize, Activation activation = LINEAR) : inputSize(inputSize), outputSize(outputSize) {
        weights = Tensor2<double>({ outputSize, inputSize }, InitType::Random);
        biases = Tensor2<double>({ outputSize, 1 }, InitType::Random);
        this->activation = activation;
        cache = nullptr;
    }

    // Save weights and biases to file
    void saveWeightsAndBiases(const std::string& weightsFile, const std::string& biasesFile) {
        std::cout << "Attempting to save weights to " << weightsFile << " and biases to " << biasesFile << std::endl;

        // Save weights
        std::ofstream weightsOut(weightsFile, std::ios::binary);
        if (!weightsOut) {
            std::cerr << "Error: Could not open weights file for writing.\n";
            return;
        }
        for (int i = 0; i < weights.getShape()[0]; ++i) {
            for (int j = 0; j < weights.getShape()[1]; ++j) {
                weightsOut.write(reinterpret_cast<char*>(&weights({i, j})), sizeof(double));
            }
        }
        weightsOut.close();

        // Save biases
        std::ofstream biasesOut(biasesFile, std::ios::binary);
        if (!biasesOut) {
            std::cerr << "Error: Could not open biases file for writing.\n";
            return;
        }
        for (int i = 0; i < biases.getShape()[0]; ++i) {
            biasesOut.write(reinterpret_cast<char*>(&biases({i, 0})), sizeof(double));
        }
        biasesOut.close();

        std::cout << "Weights and biases successfully saved." << std::endl;
    }

    // Load weights and biases from file
    void loadWeightsAndBiases(const std::string& weightsFile, const std::string& biasesFile) {
        std::cout << "Attempting to load weights from " << weightsFile << " and biases from " << biasesFile << std::endl;

        // Load weights
        std::ifstream weightsIn(weightsFile, std::ios::binary);
        if (!weightsIn) {
            std::cerr << "Error: Could not open weights file for reading.\n";
            return;
        }
        for (int i = 0; i < weights.getShape()[0]; ++i) {
            for (int j = 0; j < weights.getShape()[1]; ++j) {
                weightsIn.read(reinterpret_cast<char*>(&weights({i, j})), sizeof(double));
            }
        }
        weightsIn.close();

        // Load biases
        std::ifstream biasesIn(biasesFile, std::ios::binary);
        if (!biasesIn) {
            std::cerr << "Error: Could not open biases file for reading.\n";
            return;
        }
        for (int i = 0; i < biases.getShape()[0]; ++i) {
            biasesIn.read(reinterpret_cast<char*>(&biases({i, 0})), sizeof(double));
        }
        biasesIn.close();

        std::cout << "Weights and biases successfully loaded." << std::endl;
    }

    Tensor2<double> forward(const Tensor2<double>& input, bool training = false) override {
        model->forwardStack.push(this);
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
        } else if (cache != nullptr) {
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

    // Getter and Setter for weights and biases
    Tensor2<double> getWeights() const {
        return weights;
    }

    void setWeights(const Tensor2<double>& newWeights) {
        weights = newWeights;
    }

    Tensor2<double> getBiases() const {
        return biases;
    }

    void setBiases(const Tensor2<double>& newBiases) {
        biases = newBiases;
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
