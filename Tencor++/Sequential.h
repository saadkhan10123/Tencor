#pragma once
#include <iostream>
#include "List.h"
#include "Layer.h"
#include "Loss.h"
#include <initializer_list>

class Sequential {
public:
	Sequential() {
		layers = new List<Layer*>();
		lossFunc = nullptr;
	}

	~Sequential() {
		delete layers;
		if (lossFunc != nullptr) {
			delete lossFunc;
		}
	}

	void add(Layer* layer) {
		layers->append(layer);
	}

	void add(std::initializer_list<Layer*> layerList) {
		for (Layer* layer : layerList) {
			layers->append(layer);
		}
	}

	Tensor2<double> forward(Tensor2<double> input, bool training = false) {
		Tensor2<double> output = input;

		for (auto layer : *layers) {
			output = layer->forward(output, training);
		}

		return output;
	}

	void backward(Tensor2<double> grad, double learningRate) {
		Node<Layer*>* node = layers->getTail();

		while (node != nullptr) {
			grad = node->data->backward(grad, learningRate);
			node = node->prev;
		}
	}

	void compile(Loss<double>* loss) {
		lossFunc = loss;
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

private:
	List<Layer*>* layers;
	Loss<double>* lossFunc;
};