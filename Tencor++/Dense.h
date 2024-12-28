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

    void saveWeightsAndBiases(const std::string& weightsFile, const std::string& biasesFile) {
        std::ofstream weightsOut(weightsFile, std::ios::binary);
        std::ofstream biasesOut(biasesFile, std::ios::binary);

        if (!weightsOut.is_open() || !biasesOut.is_open()) {
            throw std::runtime_error("Error opening files for saving weights and biases");
        }

        // Get the weights and biases from this layer
        const auto& weights = this->getWeights();
        const auto& biases = this->getBiases();

        // Write the weights to the file
        for (int i = 0; i < weights.getShape()[0]; ++i) {
            for (int j = 0; j < weights.getShape()[1]; ++j) {
                weightsOut.write(reinterpret_cast<const char*>(&weights({i, j})), sizeof(double));
            }
        }

        // Write the biases to the file
        for (int i = 0; i < biases.getShape()[0]; ++i) {
            biasesOut.write(reinterpret_cast<const char*>(&biases({i, 0})), sizeof(double));
        }
    }

    // Load weights and biases from files
    void loadWeightsAndBiases(const std::string& weightsFile, const std::string& biasesFile) {
        std::ifstream weightsIn(weightsFile, std::ios::binary);
        std::ifstream biasesIn(biasesFile, std::ios::binary);

        if (!weightsIn.is_open() || !biasesIn.is_open()) {
            throw std::runtime_error("Error opening files for loading weights and biases");
        }

        // Get references to the weights and biases of the current layer
        auto& weights = this->getWeights();
        auto& biases = this->getBiases();

        // Read the weights from the file
        for (int i = 0; i < weights.getShape()[0]; ++i) {
            for (int j = 0; j < weights.getShape()[1]; ++j) {
                weightsIn.read(reinterpret_cast<char*>(&weights({i, j})), sizeof(double));
            }
        }

        // Read the biases from the file
        for (int i = 0; i < biases.getShape()[0]; ++i) {
            biasesIn.read(reinterpret_cast<char*>(&biases({i, 0})), sizeof(double));
        }

        std::cout << "Weights and biases loaded successfully for this layer." << std::endl;
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
    Tensor2<double>& getWeights() {
        return weights;
    }

    void setWeights(const Tensor2<double>& newWeights) {
        weights = newWeights;
    }

    Tensor2<double>& getBiases() {
        return biases;
    }

    void setBiases(const Tensor2<double>& newBiases) {
        biases = newBiases;
    }
    std::string getName() const {
        return name;
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
    std::string name;
};