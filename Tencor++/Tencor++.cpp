#include <iostream>
#include <chrono>
#include <exception>

#include "Tensor.h"
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "Sequential.h"
#include "Dense.h"
#include "Activation.h"
#include "Loss.h"

#include "Layer.h"

#include "ModelSaver.h"
#include "MNISTDataLoader.h"
#include <stdexcept>
#include "Model.h"
#include "opencv2/ml.hpp"

#include <filesystem>

using namespace std;

void printVector(const vector<int>& vec) {
	for (int i = 0; i < vec.size(); i++) {
		cout << vec[i] << " ";
	}
	cout << endl;
}

Tensor2<double> oneHotEncode(const Tensor2<double>& target, int numClasses) {
    Tensor2<double> oneHot = Tensor2<double>({ numClasses, target.shape[1] });

	for (int i = 0; i < target.shape[1]; i++) {
		int classIndex = target({ 0, i });
		oneHot[classIndex][i] = 1.0;
	}

	return oneHot;
}

void verifyWeightsAndBiases(const std::string& weightsFile, const std::string& biasesFile, const Dense* layer) {
    std::ifstream weightsIn(weightsFile, std::ios::binary);
    std::ifstream biasesIn(biasesFile, std::ios::binary);

    if (!weightsIn.is_open() || !biasesIn.is_open()) {
        std::cerr << "Error opening files for verification." << std::endl;
        return;
    }

    Tensor2<double> loadedWeights(layer->getWeights().getShape());
    Tensor2<double> loadedBiases(layer->getBiases().getShape());

    // Load weights
    for (int i = 0; i < loadedWeights.getShape()[0]; ++i) {
        for (int j = 0; j < loadedWeights.getShape()[1]; ++j) {
            weightsIn.read(reinterpret_cast<char*>(&loadedWeights({i, j})), sizeof(double));
        }
    }

    // Load biases
    for (int i = 0; i < loadedBiases.getShape()[0]; ++i) {
        biasesIn.read(reinterpret_cast<char*>(&loadedBiases({i, 0})), sizeof(double));
    }

    // Compare with original weights and biases
    if (loadedWeights == layer->getWeights() && loadedBiases == layer->getBiases()) {
        std::cout << "Verification successful: Weights and biases match." << std::endl;
    } else {
        std::cerr << "Verification failed: Weights and biases do not match." << std::endl;
    }
}


void MNISTTest() {
    std::string trainImagesPath = "../Dataset/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte";
	std::string trainLabelsPath = "../Dataset/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte";
    MNISTDataLoader testLoader(trainImagesPath, trainLabelsPath);
    testLoader.printSummary();
    testLoader.normalizeImages();

    std::cout << "Data preparation complete" << std::endl;

    Dense* firstLayer = new Dense(784, 128, Activation::RELU);
    Dense* secondLayer = new Dense(128, 64, Activation::RELU);
    Dense* thirdLayer = new Dense(64, 10, Activation::SOFTMAX);

    // Create a Sequential model
    Sequential model;
    model.add(firstLayer);
    model.add(secondLayer);
    model.add(thirdLayer);

    // Compile the model with loss function
    model.compile(new CategoricalCrossEntropy<double>());

    //flattening
    Tensor2<double> test = testLoader.getImages().flatten(1);
	test = Tensor2<double>::transpose(test);
	cout << test.shape[0] << " " << test.shape[1] << endl;

    // Train the model
    // model.fit(test, oneHotEncode(testLoader.getLabels().squeeze(), 10), 5, 0.001, 10);

    // Create a ModelSaver instance
    ModelSaver modelSaver;
    modelSaver.addLayer(firstLayer);
    modelSaver.addLayer(secondLayer);
    modelSaver.addLayer(thirdLayer);

    // Save weights and biases
    modelSaver.saveWeightsAndBiases("weights.dat", "biases.dat");
    std::cout << "Trained Weights and biases saved" << std::endl << std::endl;

    // Verify weights and biases
    verifyWeightsAndBiases("weights.dat", "biases.dat", firstLayer);
    verifyWeightsAndBiases("weights.dat", "biases.dat", secondLayer);
    verifyWeightsAndBiases("weights.dat", "biases.dat", thirdLayer);
    std::cout << "Weights and biases verification completed" << std::endl;

    // Load weights and biases
    modelSaver.loadWeightsAndBiases("weights.dat", "biases.dat");
    std::cout << "Weights and biases loaded" << std::endl;
}
//Predict the output of the model using MNIST test data
void predictMNIST() {
    // Load test images and labels
    std::string testImagesPath = "C:\\Users\\USMAN-PC\\Tencor\\mnist\\t10k-images.idx3-ubyte";
    std::string testLabelsPath = "C:\\Users\\USMAN-PC\\Tencor\\mnist\\t10k-labels.idx1-ubyte";
    MNISTDataLoader testLoader(testImagesPath, testLabelsPath);
    testLoader.printSummary();
    testLoader.normalizeImages();

    // Create model layers
    Dense* firstLayer = new Dense(784, 128, Activation::RELU);
    Dense* secondLayer = new Dense(128, 64, Activation::RELU);
    Dense* thirdLayer = new Dense(64, 10, Activation::SOFTMAX);

    // Create and compile Sequential model
    Sequential model;
    model.add(firstLayer);
    model.add(secondLayer);
    model.add(thirdLayer);
    model.compile(new CategoricalCrossEntropy<double>());

    // Load pre-trained weights and biases
    ModelSaver modelSaver;
    modelSaver.addLayer(firstLayer);
    modelSaver.addLayer(secondLayer);
    modelSaver.addLayer(thirdLayer);
    modelSaver.loadWeightsAndBiases("weights.dat", "biases.dat");

    // Prepare test data for prediction
    Tensor2<double> test = testLoader.getImages().flatten(1);
    test = Tensor2<double>::transpose(test);
    
    // Perform prediction
    Tensor2<double> output = model.forward(test);

    // Print predicted classes and actual labels
    std::cout << "Predicted classes:" << std::endl;
    std::cout << Tensor2<double>::argmax(output, 0) << std::endl;
    std::cout << "Actual labels:" << std::endl;
    std::cout << testLoader.getLabels().squeeze() << std::endl;
    // Get predicted classes 
    // Tensor2<double> predictedClasses = Tensor2<double>::argmax(output, 0); 
    // Tensor2<double> actualLabels = testLoader.getLabels().squeeze(); 
    
    // std::cout << "Prediction completed." << std::endl;

    // // Iterate through the first 50 images and print their actual labels and predicted classes
    // for (int i = 0; i < 50; ++i) {
    //     try {
    //         std::cout << "Displaying Image #" << i + 1 << std::endl;
    //         std::cout << "Actual Label: ";
    //         std::cout << actualLabels({i}) << std::endl;
    //         std::cout << "Predicted Class: ";
    //         std::cout << predictedClasses({i}) << std::endl;
    //     } catch (const std::exception& e) {
    //         std::cerr << "Error displaying image #" << i + 1 << ": " << e.what() << std::endl;
    //     }
    // }
    // // Print predicted classes
    // std::cout << "Argmax of the output tensor:" << std::endl;
    // std::cout << Tensor2<double>::argmax(output, 0) << std::endl;
}

void predictAndDisplayMNIST() {
    // Load test images and labels
    std::string testImagesPath = "C:\\Users\\USMAN-PC\\Tencor\\mnist\\t10k-images.idx3-ubyte";
    std::string testLabelsPath = "C:\\Users\\USMAN-PC\\Tencor\\mnist\\t10k-labels.idx1-ubyte";
    MNISTDataLoader testLoader(testImagesPath, testLabelsPath);
    testLoader.printSummary();
    testLoader.normalizeImages();

    std::cout << "Data loaded and normalized." << std::endl;

    // Create model layers
    Dense* firstLayer = new Dense(784, 128, Activation::RELU);
    Dense* secondLayer = new Dense(128, 64, Activation::RELU);
    Dense* thirdLayer = new Dense(64, 10, Activation::SOFTMAX);

    // Create and compile Sequential model
    Sequential model;
    model.add(firstLayer);
    model.add(secondLayer);
    model.add(thirdLayer);
    model.compile(new CategoricalCrossEntropy<double>());

    std::cout << "Model created and compiled." << std::endl;

    // Load pre-trained weights and biases
    ModelSaver modelSaver;
    modelSaver.addLayer(firstLayer);
    modelSaver.addLayer(secondLayer);
    modelSaver.addLayer(thirdLayer);
    modelSaver.loadWeightsAndBiases("weights.dat", "biases.dat");

    std::cout << "Pre-trained weights and biases loaded." << std::endl;

    // Prepare test data for prediction
    Tensor2<double> test = testLoader.getImages().flatten(1);
    test = Tensor2<double>::transpose(test);
    std::cout << "Test data prepared for prediction." << std::endl;

    // Perform prediction
    Tensor2<double> output = model.forward(test);

    // Get predicted classes 
    Tensor2<double> predictedClasses = Tensor2<double>::argmax(output, 0); 
    Tensor2<double> actualLabels = testLoader.getLabels().squeeze(); 
    
    std::cout << "Prediction completed." << std::endl;

    // Display each test image using OpenCV
    Tensor3<double> images = testLoader.getImages();
    for (int i = 0; i < 10; ++i) {  // Display the first 10 images
    std::cout << "Displaying Image #" << i + 1 << std::endl;
    std::cout << "Actual Label: " << actualLabels({i}) << std::endl;
    std::cout << "Predicted Class: " << predictedClasses({i}) << std::endl;

    // Extract and normalize the image
    cv::Mat image(28, 28, CV_64F);
    for (int j = 0; j < 28; ++j) {
        for (int k = 0; k < 28; ++k) {
            image.at<double>(j, k) = images({i, j, k});
        }
    }

    cv::normalize(image, image, 0, 255, cv::NORM_MINMAX);
    image.convertTo(image, CV_8U);

    // Resize for better visibility
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(280, 280), 0, 0, cv::INTER_LINEAR);

    // Display the image
    cv::imshow("MNIST Test Image", resizedImage);
    cv::waitKey(0); // Wait for a key press
}

}

int main() {
    MNISTTest();
    return 0;
}

