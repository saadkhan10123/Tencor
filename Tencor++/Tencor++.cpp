#include <iostream>
#include <chrono>
#include <exception>

#include "Tensor.h"
#include <vector>

#include "Sequential.h"
#include "Dense.h"
#include "Activation.h"
#include "Loss.h"

#include "Layer.h"

using namespace std;

void printVector(const vector<int>& vec) {
	for (int i = 0; i < vec.size(); i++) {
		cout << vec[i] << " ";
	}
	cout << endl;
}

void test2DTensor() {
    Tensor2<double> t1 = {
		{1, 2, 3},
		{4, 5, 6}
	};
	printVector(t1.shape);

    Tensor2<double> t2 = {
        {1, 2, 3}
    };
	printVector(t2.shape);

	cout << t1 + t2 << endl;
}

Tensor2<double> oneHotEncode(const Tensor2<double>& target, int numClasses) {
    Tensor2<double> oneHot = Tensor2<double>({ numClasses, target.shape[1] });

	for (int i = 0; i < target.shape[1]; i++) {
		int classIndex = target({ 0, i });
		oneHot[classIndex][i] = 1.0;
	}

	return oneHot;
}

void testModel() {
    // Define the input tensor(10 features x 10 samples)
    Tensor2<double> input = {
		{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0},
		{0.4, 0.3, 0.2, 0.1, 0.0, 0.9, 0.8, 0.7, 0.6, 0.5},
		{0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.9, 0.8},
    };

	cout << input << endl;

    // Define the target tensor(one - hot encoded, 3 classes x 10 samples)
    Tensor2<double> target = {
        { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 }
    };

	target = oneHotEncode(target, 2);

    // Create a Sequential model
    Sequential model;
    model.add(new Dense(3, 50, Activation::RELU));
    model.add(new Dense(50, 20, Activation::RELU));
    model.add(new Dense(20, 2, Activation::SOFTMAX));

    // Compile the model with Cross Entropy loss
    model.compile(new CategoricalCrossEntropy<double>());

    // Train the model
    model.fit(input, target, 20, 0.001);

    // Test the model with a new tensor
	Tensor2<double> test = {
		{ 0.2, 0.9 },
		{ 0.9, 0.1 },
		{ 0.3, 0.7 }
	};

    Tensor2<double> output = model.forward(test);

    // Print argmax of the output tensor
    cout << "Argmax of the output tensor:" << endl;
    cout << Tensor2<double>::argmax(output, 0) << endl;
}


void testDot() {
	Tensor2<double> t1 = {
		{1, 2, 3},
		{4, 5, 6}
	};

	Tensor2<double> t2 = {
		{1, 2},
		{3, 4},
		{5, 6}
	};

	cout << Tensor2<double>::dot(t1, t2) << endl;
}

void testSoftmax() {
	Tensor2<double> t1 = {
        {2, 5, 7},
		{2, 6, 8},
		{3, 7, 9}
	};

	cout << t1 << endl;
	cout << applyActivation(t1, SOFTMAX) << endl;
}

int main() {
	// test2DTensor();

    testModel();

    //testDot();

    // testSoftmax();

    return 0;
}
