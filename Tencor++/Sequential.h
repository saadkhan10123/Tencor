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
	}

	~Sequential() {
		delete layers;
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
		for (int i = 0; i < epochs; i++) {
			Tensor2<double> output = forward(input, true);
			Tensor2<double> grad = lossFunc->backward(output, target);
			backward(grad, learningRate);
		}
	}

private:
	List<Layer*>* layers;
	Loss<double>* lossFunc;
};