#include <iostream>
#include <chrono>
#include "List.h"
#include "Tensor.h"

using namespace std;

int main() {
	// Start timer
	auto start = std::chrono::high_resolution_clock::now();

	Tensor1D<int>* t1 = new Tensor1D<int>({ 4 }, InitType::Default);
	Tensor1D<int>* t2 = new Tensor1D<int>({ 4 }, InitType::Ones);

	(*t2)({ 0 }, 4);

	cout << "t1: " << *t1 << endl;
	cout << "t2: " << *t2 << endl;

	Tensor1D<int> t3 = (*t1) - (*t2);

	cout << "t3: " << t3 << endl;

	t3 = (*t1) * (*t2);

	cout << "t3: " << t3 << endl;


	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed = end - start;
	std::cout << "Elapsed time: " << elapsed.count() << " ms" << std::endl;
	
	return 0;
}

