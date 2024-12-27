#pragma once
#include "Tensor.h"
#include "Dense.h"
#include <string>
#include <vector> 

class Sequential; 

class ModelSaver {
public:
    // Constructor to initialize with model details
    ModelSaver() {}

    // Method to save weights and biases to file
    void saveWeightsAndBiases(const std::string& weightsFile, const std::string& biasesFile) {
        std::cout << "Attempting to save weights and biases to files." << std::endl;
        // Iterate over the layers stored in the vector
        for (Layer* layer : layers) {
            if (auto denseLayer = dynamic_cast<Dense*>(layer)) {
                std::cout << "Saving weights to " << weightsFile << " and biases to " << biasesFile << std::endl;
                denseLayer->saveWeightsAndBiases(weightsFile, biasesFile);  
            }
        }
        std::cout << "Completed saving weights and biases." << std::endl;
    }

    // Method to load weights and biases from file
    void loadWeightsAndBiases(const std::string& weightsFile, const std::string& biasesFile) {
        // Iterate over the layers stored in the vector
        for (Layer* layer : layers) {
            if (auto denseLayer = dynamic_cast<Dense*>(layer)) {
                denseLayer->loadWeightsAndBiases(weightsFile, biasesFile);  // Load weights and biases for Dense layers
            }
        }
    }

    // Save weights and biases after a training step
    void trainStep(Sequential& model, const Tensor2<double>& input, const Tensor2<double>& target, int epoch) {
        // Optionally, you could log the epoch number or any other details here
        std::cout << "Saving weights and biases after epoch " << epoch << std::endl;
        saveWeightsAndBiases("weights.dat", "biases.dat");  // Save after every epoch
    }

    void addLayer(Layer* layer) {
        layers.push_back(layer);  // Add the layer to the vector
    }

private:
    std::vector<Layer*> layers;  // Store layers that need saving/loading
};
