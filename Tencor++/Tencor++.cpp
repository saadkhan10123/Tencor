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
    std::string trainImagesPath = "C:\\Users\\USMAN-PC\\Desktop\\Tencor\\mnist\\train-images.idx3-ubyte";
    std::string trainLabelsPath = "C:\\Users\\USMAN-PC\\Desktop\\Tencor\\mnist\\train-labels.idx1-ubyte";
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

// void PredictTest() {
//     try {
//         // Paths to test dataset
//         std::string testImagesPath = "C:\\Users\\USMAN-PC\\Desktop\\Tencor\\mnist\\t10k-images.idx3-ubyte";
//         std::string testLabelsPath = "C:\\Users\\USMAN-PC\\Desktop\\Tencor\\mnist\\t10k-labels.idx1-ubyte";

//         // Load test data
//         MNISTDataLoader testLoader(testImagesPath, testLabelsPath);
        
//         testLoader.printSummary();
//         testLoader.normalizeImages();
        
//         std::cout << "Data preparation complete" << std::endl;

//         // Define layers
//         auto* firstLayer = new Dense(784, 128, Activation::RELU);
//         auto* secondLayer = new Dense(128, 64, Activation::RELU);
//         auto* thirdLayer = new Dense(64, 10, Activation::SOFTMAX);

//         std::cout << "Layers created" << std::endl;

//         // Create Sequential model
//         Sequential model;
//         model.add(firstLayer);
//         model.add(secondLayer);
//         model.add(thirdLayer);

//         std::cout << "Layers added to Sequential model" << std::endl;

//         ModelSaver modelSaver;
//         modelSaver.loadWeightsAndBiases(model, "weights.dat", "biases.dat");
//         std::cout << "Weights and biases loaded" << std::endl;

//         // Flatten and preprocess the images
//         Tensor2<double> flattenedImages = testLoader.getImages().flatten(1);
//         flattenedImages = Tensor2<double>::transpose(flattenedImages);
//         std::cout << "Flattened Images Shape: " << flattenedImages.getShape()[0] << " x " << flattenedImages.getShape()[1] << std::endl;

//         // Run predictions
//         Tensor2<double> predictions = model.forward(flattenedImages);
//         std::cout << "Predictions complete" << std::endl;

//         // Debugging: Check softmax validity and prediction values
//         for (int i = 0; i < 10; i++) {
//             double sum = 0;
//             std::cout << "Predictions for sample " << i << ": ";
//             for (int c = 0; c < 10; c++) {
//                 double prob = predictions({c, i});
//                 sum += prob;
//                 std::cout << prob << " ";
//             }
//             std::cout << " | Sum: " << sum << std::endl;
//             if (std::abs(sum - 1.0) > 1e-6) {
//                 std::cerr << "Warning: Softmax probabilities for sample " << i << " do not sum to 1!" << std::endl;
//             }
//         }

//         // Calculate accuracy
//         int validPredictions = 0;
//         int correctPredictions = 0;
//         int totalSamples = testLoader.getLabels().getShape()[1];

//         for (int i = 0; i < totalSamples; i++) {
//             // Find the class with the highest probability
//             int predictedClass = -1;
//             double maxProb = -1.0;

//             for (int c = 0; c < 10; c++) {
//                 double prob = predictions({c, i});
//                 if (!std::isnan(prob) && prob > maxProb) {
//                     maxProb = prob;
//                     predictedClass = c;
//                 }
//             }

//                         // Get the actual label
//             int actualLabel = static_cast<int>(testLoader.getLabels()({0, i}));

//             // Check for valid predictions
//             if (predictedClass >= 0) {
//                 validPredictions++;
//                 if (predictedClass == actualLabel) {
//                     correctPredictions++;
//                 }
//             } else {
//                 std::cerr << "Invalid prediction for sample " << i << ": NaN or no valid class found." << std::endl;
//             }

//             // Print results for the first 10 samples
//             if (i < 10) {
//                 std::cout << "Sample " << i + 1 << ": Predicted = " << predictedClass
//                           << " (Prob = " << maxProb << "), Actual = " << actualLabel << std::endl;
//             }
//         }

//         // Print overall accuracy
//         if (validPredictions > 0) {
//             double accuracy = (static_cast<double>(correctPredictions) / validPredictions) * 100.0;
//             std::cout << "\nAccuracy on test set: " << accuracy << "%" << std::endl;
//         } else {
//             std::cerr << "\nNo valid predictions were made." << std::endl;
//         }

//     } catch (const std::exception& ex) {
//         std::cerr << "Error during prediction: " << ex.what() << std::endl;
//     }
// }

#include <iomanip> // For std::setw and std::setprecision

void PredictTest() {
    try {
        // Paths to test dataset
        std::string testImagesPath = "C:\\Users\\USMAN-PC\\Desktop\\Tencor\\mnist\\t10k-images.idx3-ubyte";
        std::string testLabelsPath = "C:\\Users\\USMAN-PC\\Desktop\\Tencor\\mnist\\t10k-labels.idx1-ubyte";

        // Load test data
        MNISTDataLoader testLoader(testImagesPath, testLabelsPath);

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

        std::cout << "Layers added to Sequential model!~" << std::endl;

        ModelSaver modelSaver;
        modelSaver.loadWeightsAndBiases(model, "weights.dat", "biases.dat");
        std::cout << "Weights and biases loaded" << std::endl;

        // Flatten and preprocess the images
        Tensor2<double> flattenedImages = testLoader.getImages().flatten(1);
        flattenedImages = Tensor2<double>::transpose(flattenedImages);

        // Run predictions
        Tensor2<double> predictions = model.forward(flattenedImages);

        // Calculate accuracy
        int validPredictions = 0;
        int correctPredictions = 0;
        int totalSamples = testLoader.getLabels().getShape()[0];
        int progressBarWidth = 50; // Width of the progress bar

        for (int i = 0; i < totalSamples; i++) {
            // Find the class with the highest probability
            int predictedClass = -1;
            double maxProb = -1.0;

            for (int c = 0; c < 10; c++) {
                double prob = predictions({c, i});
                if (!std::isnan(prob) && prob > maxProb) {
                    maxProb = prob;
                    predictedClass = c;
                }
            }

            // Get the actual label
            int actualLabel = static_cast<int>(testLoader.getLabels()({i, 0}));

            // Check for valid predictions
            if (predictedClass >= 0) {
                validPredictions++;
                if (predictedClass == actualLabel) {
                    correctPredictions++;
                }
            }

            // Update progress bar
            float progress = static_cast<float>(i + 1) / totalSamples;
            int pos = static_cast<int>(progress * progressBarWidth);

            std::cout << "\r[";
            for (int j = 0; j < progressBarWidth; ++j) {
                if (j < pos) std::cout << "="; // Filled part
                else if (j == pos) std::cout << ">"; // Current position
                else std::cout << " "; // Empty part
            }
            std::cout << "] " << int(progress * 100) << "% completed" << std::flush;
        }
        std::cout << std::endl; // Move to the next line after progress bar

        // Print overall accuracy
        if (validPredictions > 0) {
            double accuracy = (static_cast<double>(correctPredictions) / validPredictions) * 100.0;
            
            std::cout << "\nAccuracy on test set: " << accuracy << "%" << std::endl<<std::endl;
        } else {
            std::cerr << "\nNo valid predictions were made." << std::endl;
        }

    } catch (const std::exception& ex) {
        std::cerr << "Error during prediction: " << ex.what() << std::endl;
    }
}
void predictAndDisplayMNIST() {  
    // Load test images and labels  
    std::string testImagesPath = "C:\\Users\\USMAN-PC\\Tencor\\mnist\\t10k-images.idx3-ubyte";  
    std::string testLabelsPath = "C:\\Users\\USMAN-PC\\Tencor\\mnist\\t10k-labels.idx1-ubyte";  

    MNISTDataLoader testLoader(testImagesPath, testLabelsPath);  
    testLoader.printSummary();  
    testLoader.normalizeImages();  
    //std::cout << "Data loaded and normalized." << std::endl;  

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
    //std::cout << "Model created and compiled." << std::endl;  

    // Load pre-trained weights and biases  
    ModelSaver modelSaver;  
    modelSaver.loadWeightsAndBiases(model, "weights.dat", "biases.dat");  
    //std::cout << "Pre-trained weights and biases loaded." << std::endl;  

    // Prepare test data for prediction  
    Tensor2<double> test = testLoader.getImages().flatten(1);  
    test = Tensor2<double>::transpose(test);  
    //std::cout << "Test data prepared for prediction." << std::endl;  

    // Perform prediction  
    Tensor2<double> output = model.forward(test);  

    // Get predicted classes   
    Tensor2<double> predictedClasses = Tensor2<double>::argmax(output, 0);   
    Tensor2<double> actualLabels = testLoader.getLabels().squeeze();   

    //std::cout << "Prediction completed." << std::endl;  

    // Display only 10 correctly predicted test images  
    Tensor3<double> images = testLoader.getImages();  
    int displayedCount = 0; // To track how many images displayed  

    for (int i = 0; i < images.getShape()[0] && displayedCount < 10; ++i) {  
        if (predictedClasses({i}) == actualLabels({i})) { // Only display correct predictions
        std::cout<<std::endl;  
            std::cout << "Displaying Image #" << displayedCount + 1 << std::endl;  
            std::cout << "Actual Label: " << actualLabels({i}) << std::endl;  
            std::cout << "Predicted Class: " << predictedClasses({i}) << std::endl;  

            // Create a Mat for the image  
            cv::Mat tempImg = cv::Mat::zeros(cv::Size(28, 28), CV_64F);  

            // Extract the image data  
            for (int j = 0; j < 28; ++j) {  
                for (int k = 0; k < 28; ++k) {  
                    tempImg.at<double>(j, k) = images({i, j, k});  
                }  
            }  

            // Normalize and prepare the image for display  
            cv::normalize(tempImg, tempImg, 0, 255, cv::NORM_MINMAX);  
            tempImg.convertTo(tempImg, CV_8U);  

            // Resize the image for better visibility  
            cv::Mat resizedImg;  
            cv::resize(tempImg, resizedImg, cv::Size(560, 560), 0, 0, cv::INTER_LINEAR);  

            // Display the image in a window  
            cv::imshow("MNIST Test Image", resizedImg);  
            int key = cv::waitKey(0);  // Wait indefinitely for a key press  

            // Optional: Break the loop if the user presses 'q' or 'ESC'  
            if (key == 'q' || key == 27) {  // 27 is the ASCII code for the ESC key  
                std::cout << "Terminating image display early." << std::endl;  
                break;  
            }  

            displayedCount++; // Increment the displayed count  
        }  
    }  
}
int main() {
    try {
        //MNISTTest();
        PredictTest();
        predictAndDisplayMNIST();
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }

    return 0;
}