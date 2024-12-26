#pragma once
#include <iostream>
#include <unordered_map>
#include <string>
#include "Stack.h"
#include "Loss.h"
#include "Layer.h"

class Model {
private:
    std::unordered_map<std::string, Layer*> layers;
    Loss<double>* lossFunc;
    int layerCount;

public:
    void addLayer(std::string name, Layer* layer) {
        layers[name] = layer;
        layer->setModel(this);
        layerCount++;
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

    void fit(Tensor2<double> input, Tensor2<double> target, int epochs, double learningRate) {
        if (!lossFunc) {
            std::cerr << "Loss function not set" << std::endl;
            throw std::invalid_argument("Loss function not set");
        }
        for (int i = 0; i < epochs; i++) {
            std::cout << "Epoch " << i + 1 << ": ";
            Tensor2<double> output = forward(input, true);
            Tensor2<double> grad = lossFunc->backward(output, target);
            std::cout << "Loss: " << lossFunc->forward(output, target) << std::endl;
            backward(grad, learningRate);
        }
    }

    void compile(Loss<double>* loss) {
        lossFunc = loss;
    }

    Stack<Layer*> forwardStack;
};

