#include <iostream>
#include <chrono>
#include <exception>

#include "Tensor.h"
#include <vector>

#include "Sequential.h"
#include "Dense.h"
#include "Activation.h"
#include "Loss.h"

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

void testModel() {
    Tensor2<double> input = {
		{ 0.1, 0.9, 0.5, 0.4 },
	    { 0.3, 0.6, 0.9, 0.8 },
	    { 0.9, 0.1, 0.2, 0.9 },
		{ 0.8, 0.5, 0.2, 0.7 },
		{ 0.2, 0.3, 0.4, 0.5 },
    };

    Tensor2<double> target = {
        {1, 0, 0, 1}
    };

	Tensor2<double> test = {
		{ 0.15 },
		{ 0.25 },
		{ 0.85 },
		{ 0.95 },
		{ 0.4 }
	};

    // Create a Sequential model
    Sequential model;
    model.add(new Dense(5, 10, Activation::RELU));
	model.add(new Dense(10, 20, Activation::RELU));
	model.add(new Dense(20, 6, Activation::RELU));
    model.add(new Dense(6, 1, Activation::SIGMOID));

    // Compile the model with Mean Squared Error loss
    model.compile(new MeanSquaredError<double>());

    // Train the model
    model.fit(input, target, 200, 0.01);

    // Test the model with the same input
    Tensor2<double> output = model.forward(test);

    // Print the output tensor
    cout << "Output Tensor after training:" << endl;
    cout << output << endl;
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

int main() {
	// test2DTensor();

    testModel();

    //testDot();

    return 0;
}
