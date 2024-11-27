#pragma once
#include "Tensor.h"

template <typename T>
class Loss {
public:
	virtual T forward(Tensor2D<T>& Y_True, Tensor2D<T>& Y_Pred) = 0;
	virtual Tensor2D<T> backward(Tensor2D<T>& Y_True, Tensor2D<T> Y_Tred) = 0;
};

template <typename T>
class MeanSquaredError : public Loss {
public:
	T forward(Tensor2D<T>& Y_True, Tensor2D<T>& Y_Pred) override {
		T loss = Tensor2D<T>::sum(Tensor2D<T>::square(Y_True - Y_Pred)) / Y_True.getShape()[1];

		return loss;
	}
	Tensor2D<T> backward(Tensor2D<T>& Y_True, Tensor2D<T> Y_Tred) override {
		Tensor2D<T> dY = 2.0 * (Y_Tred - Y_True) / Y_True.getShape()[1];

		return dY;
	})
};