#pragma once

enum Activation {
	LINEAR,
	RELU,
	SIGMOID,
	TANH,
	SOFTMAX
};

Tensor relu(Tensor& input) {
	Tensor result(input.getRows(), input.getCols());
	for (int i = 0; i < input.getRows(); i++) {
		for (int j = 0; j < input.getCols(); j++) {
			result(i, j) = input(i, j) > 0 ? input(i, j) : 0;
		}
	}
	return result;
}

Tensor sigmoid(Tensor& input) {
	Tensor result(input.getRows(), input.getCols());
	for (int i = 0; i < input.getRows(); i++) {
		for (int j = 0; j < input.getCols(); j++) {
			result(i, j) = 1 / (1 + exp(-input(i, j)));
		}
	}
	return result;
}

Tensor tanh(Tensor& input) {
	Tensor result(input.getRows(), input.getCols());
	for (int i = 0; i < input.getRows(); i++) {
		for (int j = 0; j < input.getCols(); j++) {
			result(i, j) = tanh(input(i, j));
		}
	}
	return result;
}

Tensor softmax(Tensor& input) {
	Tensor result(input.getRows(), input.getCols());
	for (int i = 0; i < input.getRows(); i++) {
		float sum = 0;
		for (int j = 0; j < input.getCols(); j++) {
			result(i, j) = exp(input(i, j));
			sum += result(i, j);
		}
		for (int j = 0; j < input.getCols(); j++) {
			result(i, j) /= sum;
		}
	}
	return result;
}

class Layer {
public:
	virtual Tensor forward(Tensor& input) = 0;
	virtual int getOutputSize() = 0;

};

class DenseLayer : public Layer {
public:
	DenseLayer(int inputSize, int hiddenUnits, Activation activation = LINEAR) {
		this->hiddenUnits = hiddenUnits;
		this->inputSize = inputSize;
		this->activation = activation;
		weights = new Tensor(hiddenUnits, inputSize);
		bias = new Tensor(hiddenUnits, 1);
	}

	~DenseLayer() {
		delete weights;
		delete bias;
	}

	Tensor forward(Tensor& input) {
		Tensor result = (*weights * input) + *bias;
		switch (activation) {
		case RELU:
			result = relu(result);
			break;
		case SIGMOID:
			result = sigmoid(result);
			break;
		case TANH:
			result = tanh(result);
			break;
		case SOFTMAX:
			result = softmax(result);
			break;
		default:
			break;
		}
		return result;
	}

	int getOutputSize() {
		return hiddenUnits;
	}

private:
	int hiddenUnits;
	int inputSize;
	Tensor* weights;
	Tensor* bias;
	Activation activation;
};

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

	Tensor forward(Tensor& input) {
		Tensor result = input;
		for (Layer* layer : *layers) {
			result = layer->forward(result);
		}
		return result;
	}

private:
	List<Layer*>* layers;
};