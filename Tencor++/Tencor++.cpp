#include <iostream>
#include <exception>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <opencv2/opencv.hpp>

#include "Sequential.h"
#include "Dense.h"
#include "Loss.h"
#include "ModelSaver.h"
#include "MNISTDataLoader.h"


using namespace std;


Tensor2<double> oneHotEncode(const Tensor2<double>& target, int numClasses) {
    Tensor2<double> oneHot = Tensor2<double>({ numClasses, target.getShape()[1] });

    for (int i = 0; i < target.getShape()[1]; i++) {
        int classIndex = target({ 0, i });
        oneHot({ classIndex, i }) = 1.0;
    }

    return oneHot;
}

  void verifyWeightsAndBiases(const std::string& weightsFile, const std::string& biasesFile, Dense* layer) {
    std::ifstream weightsIn(weightsFile, std::ios::binary);
    std::ifstream biasesIn(biasesFile, std::ios::binary);

    if (!weightsIn.is_open() || !biasesIn.is_open()) {
        std::cerr << "Error opening files for verification." << std::endl;
        return;
    }

    Tensor2<double> loadedWeights(layer->getWeights().getShape());
    Tensor2<double> loadedBiases(layer->getBiases().getShape());

    for (int i = 0; i < loadedWeights.getShape()[0]; ++i) {
        for (int j = 0; j < loadedWeights.getShape()[1]; ++j) {
            weightsIn.read(reinterpret_cast<char*>(&loadedWeights({i, j})), sizeof(double));
        }
    }

    for (int i = 0; i < loadedBiases.getShape()[0]; ++i) {
        biasesIn.read(reinterpret_cast<char*>(&loadedBiases({i, 0})), sizeof(double));
    }

    // Comparison with tolerance
    double epsilon = 1e-6;
    bool weightsMatch = true;
    bool biasesMatch = true;

    for (int i = 0; i < loadedWeights.getShape()[0]; ++i) {
        for (int j = 0; j < loadedWeights.getShape()[1]; ++j) {
            if (std::abs(loadedWeights({i, j}) - layer->getWeights()({i, j})) > epsilon) {
                weightsMatch = false;
            }
        }
    }

    for (int i = 0; i < loadedBiases.getShape()[0]; ++i) {
        if (std::abs(loadedBiases({i, 0}) - layer->getBiases()({i, 0})) > epsilon) {
            biasesMatch = false;
        }
    }

    if (weightsMatch && biasesMatch) {
        std::cout << "Verification successful: Weights and biases match." << std::endl;
    } else {
        std::cerr << "Verification failed: Weights or biases do not match." << std::endl;
    }
}


void MNISTTest() {
    std::string trainImagesPath = "../Dataset/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte";
	std::string trainLabelsPath = "../Dataset/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte";
    MNISTDataLoader testLoader(trainImagesPath, trainLabelsPath);

    testLoader.printSummary();
    testLoader.normalizeImages();

    std::cout << "Data preparation complete" << std::endl;

    // Define layers
    auto* firstLayer = new Dense(784, 128, Activation::RELU);
    auto* secondLayer = new Dense(128, 64, Activation::RELU);
    auto* thirdLayer = new Dense(64, 10, Activation::SOFTMAX);

    std::cout << "Layers created" << std::endl;

    // Create Sequential model
    Sequential model;
    model.add(firstLayer);
    model.add(secondLayer);
    model.add(thirdLayer);

    std::cout << "Layers added to Sequential model" << std::endl;

    // Compile the model
    model.compile(new CategoricalCrossEntropy<double>());

    // Flatten and preprocess data
    Tensor2<double> flattenedImages = testLoader.getImages().flatten(1);
    flattenedImages = Tensor2<double>::transpose(flattenedImages);
    auto labels = oneHotEncode(testLoader.getLabels().squeeze(), 10);

    std::cout << "Data flattened and processed" << std::endl;

    // Train the model
    model.fit(flattenedImages, labels, 5, 0.001, 10);
    std::cout << "Model training complete" << std::endl;

    // Save weights and biases
    ModelSaver modelSaver;
    modelSaver.saveWeightsAndBiases(model, "weights.dat", "biases.dat");
    std::cout << "Weights and biases saved" << std::endl;

    // Save weights and biases for each layer individually
    std::string weightsFile1 = "weights_layer1.dat";
    std::string biasesFile1 = "biases_layer1.dat";
    firstLayer->saveWeightsAndBiases(weightsFile1, biasesFile1);
    
    std::string weightsFile2 = "weights_layer2.dat";
    std::string biasesFile2 = "biases_layer2.dat";
    secondLayer->saveWeightsAndBiases(weightsFile2, biasesFile2);
    
    std::string weightsFile3 = "weights_layer3.dat";
    std::string biasesFile3 = "biases_layer3.dat";
    thirdLayer->saveWeightsAndBiases(weightsFile3, biasesFile3);


    // Verify using the same layer instances
    verifyWeightsAndBiases("weights_layer1.dat", "biases_layer1.dat", firstLayer);
    verifyWeightsAndBiases("weights_layer2.dat", "biases_layer2.dat", secondLayer);
    verifyWeightsAndBiases("weights_layer3.dat", "biases_layer3.dat", thirdLayer);
}

int main() {
    try {
        MNISTTest();
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }

    return 0;
}