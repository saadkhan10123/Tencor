#pragma once
#include <iostream>
#include "List.h"
#include "Layer.h"
#include "Loss.h"
#include "Model.h"
#include <initializer_list>

class Sequential : public Model {
public:
	Sequential() {
		lossFunc = nullptr;
	}

	void add(Layer* layer) {
		addLayer(layer);
		order.append(layer);
	}

	void add(std::initializer_list<Layer*> layerList) {
		for (Layer* layer : layerList) {
			addLayer(layer);
			order.append(layer);
		}
	}

	Tensor2<double> forward(Tensor2<double> input, bool training = false) {
		Tensor2<double> output = input;

		for (Layer* layer : order) {
			output = layer->forward(output, training);
		}

		return output;
	}

private:
	Loss<double>* lossFunc;
	List<Layer*> order;
};

