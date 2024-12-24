#pragma once
#include "Tensor.h"

template <typename T>
class Loss {
public:
	virtual T forward(Tensor2<T>& Y_Pred, Tensor2<T>& Y_True) = 0;
	virtual Tensor2<T> backward(Tensor2<T> Y_Pred, Tensor2<T>& Y_True) = 0;
};

template <typename T>
class MeanSquaredError : public Loss<T> {
public:
	T forward(Tensor2<T>& Y_Pred, Tensor2<T>& Y_True) override {
		T loss = Tensor2<T>::sum(Tensor2<T>::square(Y_True - Y_Pred)) / Y_True.getShape()[1];

		return loss;
	}
	Tensor2<T> backward(Tensor2<T> Y_Pred, Tensor2<T>& Y_True) override {
		Tensor2<T> dY = (Y_Pred - Y_True) * 2.0 / Y_True.getShape()[1];

		return dY;
	}
};

template <typename T>
class BinaryCrossEntropy : public Loss<T> {
	T forward(Tensor2<T>& Y_Pred, Tensor2<T>& Y_True) override {
		Tensor2<T> loss = -Y_True * Tensor2<T>::log(Y_Pred) - (1.0 - Y_True) * Tensor2<T>::log(1.0 - Y_Pred);
		T meanLoss = Tensor2<T>::sum(loss) / Y_True.getShape()[1];

		return meanLoss;
	}

	Tensor2<T> backward(Tensor2<T> Y_Pred, Tensor2<T>& Y_True) override {
		Tensor2<T> dY = (Y_Pred - Y_True) / (Y_Pred * (1.0 - Y_Pred)) / Y_True.getShape()[1];

		return dY;
	}
};

template <typename T>
class CategoricalCrossEntropy : public Loss<T> {
	T forward(Tensor2<T>& Y_Pred, Tensor2<T>& Y_True) override {
		Tensor2<T> loss = -Y_True * Tensor2<T>::log(Y_Pred);
		T meanLoss = Tensor2<T>::sum(loss) / Y_True.getShape()[1];

		return meanLoss;
	}

	Tensor2<T> backward(Tensor2<T> Y_Pred, Tensor2<T>& Y_True) override {
		Tensor2<T> dY = Y_Pred - Y_True;

		return dY;
	}
};