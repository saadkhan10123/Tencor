#include <iostream>
#include <chrono>
#include "List.h"
#include "Tensor.h"
#include <vector>

using namespace std;

void printVector(const vector<int>& vec) {
	for (int i = 0; i < vec.size(); i++) {
		cout << vec[i] << " ";
	}
	cout << endl;
}

void test2DTensor() {
    // Create a 2D tensor
    Tensor2D<int> tensor({ 3, 3 }, InitType::Default);

    // Print the tensor
    cout << "Initial Tensor:" << endl;
    cout << tensor << endl;

    // Modify the tensor
    tensor({ 0, 0 }, 1);
    tensor({ 1, 1 }, 2);
    tensor({ 2, 2 }, 3);

    // Print the modified tensor
    cout << "Modified Tensor:" << endl;
    cout << tensor << endl;

    // Create another 2D tensor
    Tensor2D<int> tensor2({ 3, 1 }, InitType::Random);

    // Test Dot Product
    Tensor2D<int> dotProduct = Tensor2D<int>::dot(tensor2, tensor);

    // Print the dot product
    cout << "Tensor2:" << endl;
    cout << tensor2 << endl;
    cout << "Dot Product:" << endl;
    cout << dotProduct << endl;

    // Test addition
    Tensor2D<int> sum = tensor + tensor2;

    // Print the sum
    cout << "Tensor:" << endl;
    cout << tensor << endl;
    cout << "Tensor2:" << endl;
    cout << tensor2 << endl;
    cout << "Sum:" << endl;
    cout << sum << endl;

    sum += tensor2;

    // Print the sum
    cout << "Sum:" << endl;
    cout << sum << endl;
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
