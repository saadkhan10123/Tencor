#pragma once
#include <iostream>
#include <vector>

template <typename T> class Tensor1D;
template <typename T> class Tensor2D;
template <typename T> class Tensor3D;
template <typename T> class Tensor4D;

enum class InitType {
	Default,
	Ones,
	Random
};

template <typename T>
class Tensor {
    // Common properties and methods for all tensors
public:

    Tensor(const std::vector<int>& dimensions)
        : dimensions(dimensions)
    {
    }

    virtual ~Tensor() = default;

    virtual T& operator()(const std::vector<int>& indices) = 0;
    virtual const T& operator()(const std::vector<int>& indices) const = 0;
    virtual void operator()(const std::vector<int>& indices, const T& value) = 0;

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        tensor.print(os);
        return os;
    }

    virtual void print(std::ostream& os) const = 0;

    static Tensor<T> dot(const Tensor<T>& tensor1, const Tensor<T>& tensor2) {
        // Dot product of two tensors
        if (tensor1.dimensions.size() != tensor2.dimensions.size()) {
            throw std::invalid_argument("Tensors must have the same number of dimensions");
        }

        switch (tensor1.dimensions.size()) {
        case 1:
            return Tensor1D<T>::dot(tensor1, tensor2);
        default:
            throw std::invalid_argument("Unsupported number of dimensions");
        }
    }

    std::vector<int> getDimensions() const {
		return dimensions;
	}

protected:
    std::vector<int> dimensions;
};

template <typename T>
class Tensor1D : public Tensor<T> {
    // Properties and methods specific to 1D tensors
public:
    Tensor1D(const std::vector<int>& dimensions, InitType init = InitType::Default)
        : Tensor<T>(dimensions)
    {
        data = new T[dimensions[0]];

        switch (init) {
        case InitType::Default:
            for (int i = 0; i < dimensions[0]; ++i) {
                data[i] = 0;
            }
            break;
        case InitType::Ones:
            for (int i = 0; i < dimensions[0]; ++i) {
                data[i] = 1;
            }
            break;
        case InitType::Random:
            for (int i = 0; i < dimensions[0]; ++i) {
                data[i] = rand() % 100;
            }
            break;
        default:
            throw std::invalid_argument("Unsupported initialization type");
        }
    }

    Tensor1D(const Tensor1D& other)
        : Tensor<T>(other.dimensions)
    {
        data = new T[other.dimensions[0]];
        for (int i = 0; i < other.dimensions[0]; ++i) {
            data[i] = other.data[i];
        }
    }

    Tensor1D& operator=(const Tensor1D& other) {
        if (this == &other) {
            return *this;
        }

        delete[] data;

        this->dimensions = other.dimensions;
        data = new T[other.dimensions[0]];
        for (int i = 0; i < other.dimensions[0]; ++i) {
            data[i] = other.data[i];
        }

        return *this;
    }

    ~Tensor1D() {
        delete[] data;
    }

    T& operator()(const std::vector<int>& indices) override {
        return data[indices[0]];
    }

    const T& operator()(const std::vector<int>& indices) const override {
        return data[indices[0]];
    }

    void operator()(const std::vector<int>& indices, const T& value) override {
        if (indices.size() != 1 || indices[0] < 0 || indices[0] >= this->dimensions[0]) {
            throw std::out_of_range("Index out of range");
        }
        data[indices[0]] = value;
    }

    Tensor1D operator+(const Tensor1D& other) const {
        if (this->dimensions != other.dimensions) {
			throw std::invalid_argument("Dimensions must match for addition");
		}
		Tensor1D result(this->dimensions);
        for (int i = 0; i < this->dimensions[0]; ++i) {
			result({ i }) = data[i] + other({ i });
		}
		return result;
	}

    Tensor1D& operator+=(const Tensor1D& other) {
        if (this->dimensions != other.dimensions) {
            throw std::invalid_argument("Dimensions must match for addition");
        }
        for (int i = 0; i < this->dimensions[0]; ++i) {
			data[i] += other({ i });
		}
		return *this;
	}

    Tensor1D operator-(const Tensor1D& other) const {
        if (this->dimensions != other.dimensions) {
            throw std::invalid_argument("Dimensions must match for subtraction");
        }
        Tensor1D result(this->dimensions);
        for (int i = 0; i < this->dimensions[0]; ++i) {
            result({ i }) = data[i] - other({ i });
        }
        return result;
    }


    Tensor1D& operator-=(const Tensor1D& other) {
        if (this->dimensions != other.dimensions) {
			throw std::invalid_argument("Dimensions must match for subtraction");
		}
        for (int i = 0; i < this->dimensions[0]; ++i) {
			data[i] -= other({ i });
		}
		return *this;
	}

    Tensor1D operator*(const Tensor1D& other) const {
        if (this->dimensions != other.dimensions) {
			throw std::invalid_argument("Dimensions must match for multiplication");
		}
		Tensor1D result(this->dimensions);
        for (int i = 0; i < this->dimensions[0]; ++i) {
			result({ i }) = data[i] * other({ i });
		}
		return result;
	}

    Tensor1D& operator*=(const Tensor1D& other) {
        if (this->dimensions != other.dimensions) {
            throw std::invalid_argument("Dimensions must match for multiplication");
        }
        for (int i = 0; i < this->dimensions[0]; ++i) {
            data[i] *= other({ i });
        }
        return *this;
    }

    void print(std::ostream& os) const override {
        os << "{ ";
        for (int i = 0; i < this->dimensions[0]; ++i) {
            os << data[i] << ", ";
        }
        os << " }";
    }

    static Tensor1D<T> dot(const Tensor<T>& tensor1, const Tensor<T>& tensor2) {
        // Implement dot product of two 1D tensors
        const Tensor1D<T>& t1 = dynamic_cast<const Tensor1D<T>&>(tensor1);
        const Tensor1D<T>& t2 = dynamic_cast<const Tensor1D<T>&>(tensor2);

        if (t1.dimensions[0] != t2.dimensions[0]) {
            if (t1.dimensions[0] % t2.dimensions[0] == 0) {
                return dotProjected(t1, t2);
            }
            else if (t2.dimensions[0] % t1.dimensions[0] == 0) {
                return dotProjected(t2, t1);
            }
            else {
                throw std::invalid_argument("Dimensions must match for dot product");
            }
        }

        Tensor1D<T> result(t1.dimensions);
        for (int i = 0; i < t1.dimensions[0]; ++i) {
            result({ i }) = t1({ i }) * t2({ i });
        }
        return result;
    }

private:
    T* data;

    static Tensor1D<T> dotProjected(const Tensor1D<T>& t1, const Tensor1D<T>& t2) {
        Tensor1D<T> result(t1.dimensions);
        for (int i = 0; i < t1.dimensions[0]; ++i) {
            result({ i }) = t1({ i }) * t2({ i % t2.dimensions[0] });
        }
        return result;
    }
};

template <typename T>
Tensor1D<T> operator+(const Tensor1D<T>& lhs, const Tensor1D<T>& rhs) {
    Tensor1D<T> result(lhs);
    result += rhs;
    return result;
}

template <typename T>
Tensor1D<T> operator-(const Tensor1D<T>& lhs, const Tensor1D<T>& rhs) {
    Tensor1D<T> result(lhs);
    result -= rhs;
    return result;
}