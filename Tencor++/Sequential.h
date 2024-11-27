#pragma once
#include <iostream>
#include "stack.h"
#include "List.h"
#include "Layer.h"
#include "Loss.h"
#include <initializer_list>

// A struct to store the cache for backpropagation
struct cache {
	Tensor2D<double> input;
	Tensor2D<double> output;
	Tensor2D<double> activation;
	Layer *layer;
};

class Sequential : public Layer {
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

	Tensor2D<double> forward(Tensor2D<double> input) {
		Tensor2D<double> output = input;

		for (auto layer : *layers) {
			output = layer->forward(output);
		}

		return output;
	}

	void forward(Tensor2D<double> input, Stack<cache*>& cacheStack) {
		Tensor2D<double> output = input;

		for (auto layer : *layers) {
			cache* c = new cache();
			c->input = output;
			c->output = output;
			c->activation = output;
			c->layer = layer;
			cacheStack.push(c);
			output = layer->forward(output);
		}
	}

	void backward(Tensor2D<double> grad, Stack<cache*>& cacheStack) {
		Tensor2D<double> gradient = grad;

		while (!cacheStack.isEmpty()) {
			cache* c = cacheStack.pop();
			
		}


	}


private:
	List<Layer*>* layers;
	Loss<double>* lossFunc;
};