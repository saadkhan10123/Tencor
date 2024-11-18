#include <iostream>
#include <chrono>
#include "List.h"
#include "Tensor.h"
#include "Tencor.h"
#include <vector>

using namespace std;

void printVector(const vector<int>& vec) {
	for (int i = 0; i < vec.size(); i++) {
		cout << vec[i] << " ";
	}
	cout << endl;
}

void test2DTensor() {
    Tensor2D<double> t1 = { 
        {1, 2} 
    };

    Tensor2D<float> weights = {
        {0.1, 0.2},
        {0.3, 0.4}
    };
    Tensor2D<double> bias = {
        { 1, 2 },
    };

    t1 -= bias;
}

int main() {
    try {
        // Start timer
        auto start = std::chrono::high_resolution_clock::now();

        // Test 2D Tensor
        test2DTensor();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " ms" << std::endl;
    }
    catch (const std::exception& e) {

		std::cerr << "Error Occurred: " << e.what() << std::endl;
	}

    return 0;
}
