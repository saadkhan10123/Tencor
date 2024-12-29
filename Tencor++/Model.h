#pragma once
#include <iostream>
#include <string>
#include "Stack.h"
#include "Loss.h"
#include "Layer.h"
#include "Hash.h"

class Model {
private:
	HashTable<Layer*> layers;
    Loss<double>* lossFunc;
    int layerCount;

public:
    Model() : lossFunc(nullptr), layerCount(0) {} // Initialize layerCount to 0 and lossFunc to nullptr

    void addLayer(std::string name, Layer* layer) {
		layers.put(name, layer);
        layer->setModel(this);
        layerCount++;
    }
    const HashTable<Layer*>& getLayers() const {
        return layers;
    }

    void addLayer(Layer* layer) {
        addLayer("layer " + std::to_string(layerCount), layer);
        layer->setModel(this);
    }

    virtual Tensor2<double> forward(Tensor2<double> input, bool training = false) = 0;

    void backward(Tensor2<double> grad, double learningRate) {
        while (!forwardStack.isEmpty()) {
            Layer* layer = forwardStack.pop();
            grad = layer->backward(grad, learningRate);
        }
    }

    void printProgress(int epoch, int epochs, int batch, int total, double loss, bool endOfEpoch = false) {
        int progressBarWidth = 50; // Width of the progress bar
        float progress = static_cast<float>(batch) / total; // Calculate progress
        int pos = static_cast<int>(progress * progressBarWidth); // Calculate position in the bar

        std::cout << "\rEpoch [" << epoch + 1 << "/" << epochs << "] "
            << "Batch [" << batch << "/" << total << "] "
            << "[";
        for (int i = 0; i < progressBarWidth; ++i) {
            if (i < pos) std::cout << "="; // Filled part
            else std::cout << " "; // Empty part
        }
        std::cout << "] " << int(progress * 100) << "% "
            << "Loss: " << loss << std::flush; // Print loss

        if (endOfEpoch) {
            std::cout << std::endl; // Print a newline at the end of the epoch
        }
        std::cout.flush(); // Ensure the output is displayed immediately
    }

    void printEpochDetails(int epoch, int epochs, double loss) {
        std::cout << "\rEpoch: " << epoch + 1 << " Epoch Loss: " << loss << std::endl << std::endl; // Print loss
        std::cout.flush(); // Ensure the output is displayed immediately
    }

    void fit(Tensor2<double> input, Tensor2<double> target, int epochs, double learningRate, int batchSize = -1) {
        if (batchSize == -1) {
            batchSize = input.getShape()[1];
        } else if (batchSize > input.getShape()[1]) {
            std::cerr << "Batch size cannot be greater than the number of samples" << std::endl;
            throw std::invalid_argument("Batch size cannot be greater than the number of samples");
        } else if (batchSize <= 0) {
            std::cerr << "Batch size must be greater than 0" << std::endl;
            throw std::invalid_argument("Batch size must be greater than 0");
        }

        if (!lossFunc) {
            std::cerr << "Loss function not set" << std::endl;
            throw std::invalid_argument("Loss function not set");
        }

        for (int i = 0; i < epochs; i++) {
            double overallLoss = 0.0;
            for (int j = 0; j < input.getShape()[1]; j += batchSize) {
                Tensor2<double> output = forward(input.slice(j, j + batchSize, 1), true);
                Tensor2<double> grad = lossFunc->backward(output, target.slice(j, j + batchSize, 1));
                double loss = lossFunc->forward(output, target.slice(j, j + batchSize, 1));
                backward(grad, learningRate);
                printProgress(i, epochs, j, input.getShape()[1], loss);
                overallLoss += loss;
            }
            // Run the remaining samples
            if (input.getShape()[1] % batchSize != 0) {
                Tensor2<double> output = forward(input.slice(input.getShape()[1] - (input.getShape()[1] % batchSize), input.getShape()[1], 1), true);
                Tensor2<double> grad = lossFunc->backward(output, target.slice(input.getShape()[1] - (input.getShape()[1] % batchSize), input.getShape()[1], 1));
                backward(grad, learningRate);
                double loss = lossFunc->forward(output, target.slice(input.getShape()[1] - (input.getShape()[1] % batchSize), input.getShape()[1], 1));
                printProgress(i, epochs, input.getShape()[1], input.getShape()[1], loss);
                overallLoss += loss;
            }
            int totalBatches = input.getShape()[1] / batchSize;
            if (input.getShape()[1] % batchSize != 0) {
                totalBatches++;
            }
            double avgLoss = overallLoss / totalBatches;
            printProgress(i, epochs, input.getShape()[1], input.getShape()[1], avgLoss, true); // End of epoch
            printEpochDetails(i, epochs, avgLoss);
        }
    }

    void compile(Loss<double>* loss) {
        lossFunc = loss;
    }

    Stack<Layer*> forwardStack;
};