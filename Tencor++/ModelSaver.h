#pragma once
#include "Tensor.h"
#include "Dense.h"
#include "Hash.h" 
#include "Model.h"
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>

class ModelSaver {
public:
    ModelSaver() {}

    // Save weights and biases to hash table and files
    void saveWeightsAndBiases(const Model& model, HashTable& hashTable, 
                              const std::string& weightsFile, const std::string& biasesFile) {
        std::cout << "Saving weights and biases to hash table and files.\n";

        std::ofstream weightsOut(weightsFile, std::ios::binary);
        std::ofstream biasesOut(biasesFile, std::ios::binary);

        if (!weightsOut.is_open() || !biasesOut.is_open()) {
            throw std::runtime_error("Failed to open weights or biases file for writing.");
        }

        for (const auto& layerPair : model.getLayers()) {
            if (Dense* denseLayer = dynamic_cast<Dense*>(layerPair.second)) {
                // Save weights and biases
                const auto& weights = denseLayer->getWeights();
                const auto& biases = denseLayer->getBiases();

                // Store in hash table
                hashTable.insert(denseLayer->getName() + "_weights",tensorToVector(weights) );
                hashTable.insert(denseLayer->getName() + "_biases", tensorToVector(biases));

                // Write to files
                writeTensorToFile(weights, weightsOut);
                writeTensorToFile(biases, biasesOut);
            }
        }

        weightsOut.close();
        biasesOut.close();
        std::cout << "Weights and biases successfully saved.\n";
    }

    // Load weights and biases from hash table and files
    void loadWeightsAndBiases(Model& model, HashTable& hashTable, 
                              const std::string& weightsFile, const std::string& biasesFile) {
        std::cout << "Loading weights and biases from hash table and files.\n";

        std::ifstream weightsIn(weightsFile, std::ios::binary);
        std::ifstream biasesIn(biasesFile, std::ios::binary);

        if (!weightsIn.is_open() || !biasesIn.is_open()) {
            throw std::runtime_error("Failed to open weights or biases file for reading.");
        }

        for (auto& layerPair : model.getLayers()) {
            if (Dense* denseLayer = dynamic_cast<Dense*>(layerPair.second)) {
                // Read from files
                auto weights = readTensorFromFile(weightsIn);
                auto biases = readTensorFromFile(biasesIn);

                // Store in hash table
                hashTable.insert(denseLayer->getName() + "_weights", tensorToVector(weights));
                hashTable.insert(denseLayer->getName() + "_biases", tensorToVector(biases));

                // Set the layer's weights and biases
                denseLayer->setWeights(weights);
                denseLayer->setBiases(biases);
            }
        }

        weightsIn.close();
        biasesIn.close();
        std::cout << "Weights and biases successfully loaded.\n";
    }

    // Save weights and biases after a training step
    void saveAfterTrainingStep(const Model& model, HashTable& hashTable, 
                               const std::string& weightsFile, const std::string& biasesFile, int epoch) {
        std::cout << "Saving weights and biases after epoch " << epoch << ".\n";
        saveWeightsAndBiases(model, hashTable, weightsFile, biasesFile);
    }

private:
// Helper to convert a Tensor2 object to a vector
        std::vector<double> tensorToVector(const Tensor2<double>& tensor) {
            std::vector<double> vec;
            const auto& shape = tensor.getShape();
            vec.reserve(shape[0] * shape[1]);
            for (int i = 0; i < shape[0]; ++i) {
                for (int j = 0; j < shape[1]; ++j) {
                    vec.push_back(tensor({i, j}));
                }
            }
            return vec;
        }
    // Helper to write a Tensor2 object to file
    void writeTensorToFile(const Tensor2<double>& tensor, std::ofstream& outFile) {
        const auto& shape = tensor.getShape();
        outFile.write(reinterpret_cast<const char*>(&shape[0]), sizeof(int));
        outFile.write(reinterpret_cast<const char*>(&shape[1]), sizeof(int));

        for (int i = 0; i < shape[0]; ++i) {
            for (int j = 0; j < shape[1]; ++j) {
                double value = tensor({i, j});
                outFile.write(reinterpret_cast<const char*>(&value), sizeof(double));
            }
        }
    }

    // Helper to read a Tensor2 object from file
    Tensor2<double> readTensorFromFile(std::ifstream& inFile) {
        int rows, cols;
        inFile.read(reinterpret_cast<char*>(&rows), sizeof(int));
        inFile.read(reinterpret_cast<char*>(&cols), sizeof(int));

        Tensor2<double> tensor({rows, cols}, InitType::Default);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                double value;
                inFile.read(reinterpret_cast<char*>(&value), sizeof(double));
                tensor({i, j}) = value;
            }
        }
        return tensor;
    }
};
