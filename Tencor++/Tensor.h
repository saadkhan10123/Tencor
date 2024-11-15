#pragma once
#include <iostream>
#include <vector>
#include <initializer_list>
#include <cassert>


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
public:
    Tensor() = default;

    Tensor(const std::vector<int>& dimensions) : dimensions(dimensions)
    { }

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

public:
    std::vector<int> dimensions;
};

template <typename T>
class Tensor1D : public Tensor<T> {
    // Properties and methods specific to 1D tensors
public:
    Tensor1D() = default;

    Tensor1D(const std::vector<int>& dimensions, InitType init = InitType::Default) : Tensor<T>(dimensions)
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

    Tensor1D(const Tensor1D& other) : Tensor<T>(other.dimensions)
    {
        data = new T[other.dimensions[0]];
        for (int i = 0; i < other.dimensions[0]; ++i) {
            data[i] = other.data[i];
        }
    }

	// Add this constructor to Tensor1D class
	Tensor1D(std::initializer_list<T> values) : Tensor<T>({ static_cast<int>(values.size()) }) {
		data = new T[values.size()];
		int i = 0;
		for (const T& value : values) {
			data[i++] = value;
		}
	}

    ~Tensor1D() {
        delete[] data;
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
        if (this->dimensions == other.dimensions) {
			Tensor1D result(this->dimensions);

			for (int i = 0; i < this->dimensions[0]; ++i) {
				result({ i }) = data[i] + other({ i });
			}

			return result;
		}
		else if (this->dimensions[0] % other.dimensions[0] == 0) {
			Tensor1D result(this->dimensions);

			for (int i = 0; i < this->dimensions[0]; ++i) {
				result({ i }) = data[i] + other({ i % other.dimensions[0] });
			}

			return result;
		}
		else if (other.dimensions[0] % this->dimensions[0] == 0) {
			Tensor1D result(other.dimensions);

			for (int i = 0; i < other.dimensions[0]; ++i) {
				result({ i }) = data[i % this->dimensions[0]] + other({ i });
			}

			return result;
		}
		else {
			throw std::invalid_argument("Dimensions must match for addition");
		}
	}

    Tensor1D& operator+=(const Tensor1D& other) {
        if (this->dimensions == other.dimensions) {
			for (int i = 0; i < this->dimensions[0]; ++i) {
				data[i] += other({ i });
			}
		} 
		else if (this->dimensions[0] % other.dimensions[0] == 0) {
			for (int i = 0; i < this->dimensions[0]; ++i) {
				data[i] += other({ i % other.dimensions[0] });
			}
		}
		else {
			throw std::invalid_argument("Dimensions must match for addition");
		}
		return *this;
	}

    Tensor1D operator-(const Tensor1D& other) const {
		if (this->dimensions == other.dimensions) {
			Tensor1D result(this->dimensions);

			for (int i = 0; i < this->dimensions[0]; ++i) {
				result({ i }) = data[i] - other({ i });
			}

			return result;
		}
		else if (this->dimensions[0] % other.dimensions[0] == 0) {
			Tensor1D result(this->dimensions);

			for (int i = 0; i < this->dimensions[0]; ++i) {
				result({ i }) = data[i] - other({ i % other.dimensions[0] });
			}

			return result;
		}
		else if (other.dimensions[0] % this->dimensions[0] == 0) {
			Tensor1D result(other.dimensions);

			for (int i = 0; i < other.dimensions[0]; ++i) {
				result({ i }) = data[i % this->dimensions[0]] - other({ i });
			}

			return result;
		}
		else {
			throw std::invalid_argument("Dimensions must match for subtraction");
		}
    }


    Tensor1D& operator-=(const Tensor1D& other) {
		if (this->dimensions == other.dimensions) {
			for (int i = 0; i < this->dimensions[0]; ++i) {
				data[i] -= other({ i });
			}
		}
		else if (this->dimensions[0] % other.dimensions[0] == 0) {
			for (int i = 0; i < this->dimensions[0]; ++i) {
				data[i] -= other({ i % other.dimensions[0] });
			}
		}
		else {
			throw std::invalid_argument("Dimensions must match for subtraction");
		}
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
            os << data[i];
            if (i < this->dimensions[0] - 1) {
				os << ", ";
			}
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
				std::cerr << "Dimensions must match for dot product\n";
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
    T* data = nullptr;

    static Tensor1D<T> dotProjected(const Tensor1D<T>& t1, const Tensor1D<T>& t2) {
        Tensor1D<T> result(t1.dimensions);
        for (int i = 0; i < t1.dimensions[0]; ++i) {
            result({ i }) = t1({ i }) * t2({ i % t2.dimensions[0] });
        }
        return result;
    }
};

template <typename T>
class Tensor2D : public Tensor<T> {
public:
    Tensor2D(const std::vector<int>& dimensions, InitType init = InitType::Default) : Tensor<T>(dimensions)
	{
		data = new Tensor1D<T>[dimensions[0]];
        for (int i = 0; i < dimensions[0]; ++i) {
            data[i] = Tensor1D<T>({ dimensions[1] }, init);
        }
	}

	Tensor2D(const Tensor2D& other) : Tensor<T>(other.dimensions)
	{
		data = new Tensor1D<T>[other.dimensions[0]];
		for (int i = 0; i < other.dimensions[0]; ++i) {
			data[i] = other.data[i];
		}

	}

    ~Tensor2D() {
        delete[] data;
    }

	Tensor2D& operator=(const Tensor2D& other) {
		if (this == &other) {
			return *this;
		}

		delete[] data;

		this->dimensions = other.dimensions;
		data = new T[other.dimensions[0] * other.dimensions[1]];
		for (int i = 0; i < other.dimensions[0] * other.dimensions[1]; ++i) {
			data[i] = other.data[i];
		}

		return *this;
	}

	T& operator()(const std::vector<int>& indices) override {
		return data[indices[0]]({ indices[1] });
	}

	const T& operator()(const std::vector<int>& indices) const override {
        return data[indices[0]]({ indices[1] });
	}

	void operator()(const std::vector<int>& indices, const T& value) override {
		if (indices.size() != 2 || indices[0] < 0 || indices[0] >= this->dimensions[0] || indices[1] < 0 || indices[1] >= this->dimensions[1]) {
			throw std::out_of_range("Index out of range");
		}
        data[indices[0]]({ indices[1] }) = value;
	}

    Tensor2D operator+(const Tensor2D& other) const {

		std::vector<int> resultDimensions;
		try {
			resultDimensions = getDimensionsOp(*this, other);
		}
		catch (const std::invalid_argument& e) {
			std::cerr << "Dimensions must match for addition\n";
			throw std::invalid_argument("Dimensions must match for addition");
		}

		Tensor2D result(resultDimensions);

		if (this->dimensions == other.dimensions) {
			for (int i = 0; i < this->dimensions[0]; ++i) {
				result.data[i] = data[i] + other.data[i];
			}
		}
		else if (this->dimensions[0] % other.dimensions[0] == 0) {
			for (int i = 0; i < this->dimensions[0]; ++i) {
				result.data[i] = data[i] + other.data[i % other.dimensions[0]];
			}
		}
		else if (other.dimensions[0] % this->dimensions[0] == 0) {
			for (int i = 0; i < other.dimensions[0]; ++i) {
				result.data[i] = data[i % this->dimensions[0]] + other.data[i];
			}
		}
		else {
			throw std::invalid_argument("Dimensions must match for addition");
		}

		return result;
	}

    Tensor2D& operator+=(const Tensor2D& other) {
		if (this->dimensions == other.dimensions) {
			for (int i = 0; i < this->dimensions[0]; ++i) {
				data[i] += other.data[i];
			}
		}
		else if (this->dimensions[0] % other.dimensions[0] == 0) {
			for (int i = 0; i < this->dimensions[0]; ++i) {
				data[i] += other.data[i % other.dimensions[0]];
			}
		}
		else {
			throw std::invalid_argument("Dimensions must match for addition");
		}

		return *this;
	}

    Tensor2D operator-(const Tensor2D& other) const {
		std::vector<int> resultDimensions;
		try {
			resultDimensions = getDimensionsOp(*this, other);
		}
		catch (const std::invalid_argument& e) {
			std::cerr << "Dimensions must match for subtraction\n";
			throw std::invalid_argument("Dimensions must match for subtraction");
		}

		Tensor2D result(resultDimensions);

		if (this->dimensions == other.dimensions) {
			for (int i = 0; i < this->dimensions[0]; ++i) {
				result.data[i] = data[i] - other.data[i];
			}
		}
		else if (this->dimensions[0] % other.dimensions[0] == 0) {
			for (int i = 0; i < this->dimensions[0]; ++i) {
				result.data[i] = data[i] - other.data[i % other.dimensions[0]];
			}
		}
		else if (other.dimensions[0] % this->dimensions[0] == 0) {
			for (int i = 0; i < other.dimensions[0]; ++i) {
				result.data[i] = data[i % this->dimensions[0]] - other.data[i];
			}
		}
		else {
			throw std::invalid_argument("Dimensions must match for subtraction");
		}

		return result;
	}

	Tensor2D& operator-=(const Tensor2D& other) {
		if (this->dimensions == other.dimensions) {
			for (int i = 0; i < this->dimensions[0]; ++i) {
				data[i] -= other.data[i];
			}
		}
		else if (this->dimensions[0] % other.dimensions[0] == 0) {
			for (int i = 0; i < this->dimensions[0]; ++i) {
				data[i] -= other.data[i % other.dimensions[0]];
			}
		}
		else {
			throw std::invalid_argument("Dimensions must match for subtraction");
		}
		return *this;
	}

	Tensor2D operator*(const Tensor2D& other) const {
		if (this->dimensions != other.dimensions) {
			throw std::invalid_argument("Dimensions must match for multiplication");
		}
		Tensor2D result(this->dimensions);
		for (int i = 0; i < this->dimensions[0]; ++i) {
			result.data[i] = data[i] * other.data[i];
		}
		return result;
	}

	Tensor2D& operator*=(const Tensor2D& other) {
		if (this->dimensions != other.dimensions) {
			throw std::invalid_argument("Dimensions must match for multiplication");
		}
		for (int i = 0; i < this->dimensions[0]; ++i) {
			data[i] *= other.data[i];
		}
		return *this;
	}

    void print(std::ostream& os) const override {
		os << "{\n";
		for (int i = 0; i < this->dimensions[0]; ++i) {
            os << "  ";
			data[i].print(os);
            os << ",\n";
		}
		os << "}";
	}

    void setRow(const Tensor1D<T>& row, int rowNumber) {
        if (row.getDimensions()[0] != this->dimensions[1]) {
			std::cerr << "Row dimensions must match tensor dimensions\n";
			throw std::invalid_argument("Row dimensions must match tensor dimensions");
		}
        this->data[rowNumber] = row;
    }

    Tensor1D<T> getRow(int rowNumber) const {
		if (rowNumber < 0 || rowNumber >= this->dimensions[0]) {
			std::cerr << "Row index out of range\n";
			throw std::out_of_range("Row index out of range");
		}

        return data[rowNumber];
	}

    static Tensor2D<T> dot(const Tensor<T>& tensor1, const Tensor<T>& tensor2) {
        // Implement dot product of two 2D tensors
        const Tensor2D<T>& t1 = dynamic_cast<const Tensor2D<T>&>(tensor1);
        const Tensor2D<T>& t2 = dynamic_cast<const Tensor2D<T>&>(tensor2);

        // Get dimensions for result vector and find dot product
        std::vector<int> resultDimensions;
		try {
			resultDimensions = getDimensionsOp(t1, t2);
		}
		catch (const std::invalid_argument& e) {
			std::cerr << "Dimensions must match for dot product";
			throw std::invalid_argument("Dimensions must match for dot product");
		}

		Tensor2D<T> result(resultDimensions);

        if (t1.dimensions[0] == t2.dimensions[0]) {
            for (int i = 0; i < t1.dimensions[0]; ++i) {
                result.setRow(Tensor1D<T>::dot(t1.getRow(i), t2.getRow(i)), i);
			}
		}
		else if (t1.dimensions[1] == t2.dimensions[1]) {
			if (t1.dimensions[0] % t2.dimensions[0] == 0) {
				for (int i = 0; i < resultDimensions[0]; ++i) {
					result.setRow(t1.getRow(i % t1.dimensions[0]) * t2.getRow(i), i);
				}
			}
			else if (t2.dimensions[0] % t1.dimensions[0] == 0) {
				for (int i = 0; i < resultDimensions[0]; ++i) {
					result.setRow(t1.getRow(i) * t2.getRow(i % t2.dimensions[0]), i);
				}

			}
		}
		return result;
    }
private:
    Tensor1D<T>* data = nullptr;

	static std::vector<int> getDimensionsOp(const Tensor2D<T> A, const Tensor2D<T> B) {
		if (A.getDimensions()[0] == B.getDimensions()[0]) {
			if (A.getDimensions()[1] == B.getDimensions()[1]) {
				return A.getDimensions();
			}
			else if (A.getDimensions()[1] % B.getDimensions()[1] == 0) {
				return A.getDimensions();
			}
			else if (B.getDimensions()[1] % A.getDimensions()[1] == 0) {
				return B.getDimensions();
			}
			else {
				std::cerr << "Dimension mismatch";
				throw std::invalid_argument("Dimension mismatch");
			}
		}
		else if (A.getDimensions()[0] % B.getDimensions()[0] == 0) {
			if (A.getDimensions()[1] == B.getDimensions()[1]) {
				return B.getDimensions();
			}
			else {
				std::cerr << "Dimension mismatch";
				throw std::invalid_argument("Dimension mismatch");
			}
		}
		else if (B.getDimensions()[0] % A.getDimensions()[0] == 0) {
			if (A.getDimensions()[1] == B.getDimensions()[1]) {
				return A.getDimensions();
			}
			else {
				std::cerr << "Dimension mismatch";
				throw std::invalid_argument("Dimension mismatch");
			}
		}
		else {
			std::cerr << "Dimension mismatch";
			throw std::invalid_argument("Dimension mismatch");
		}

		return { -1, -1 };
	}
};

template <typename T>
class Tensor3D : public Tensor<T> {
    public:
	Tensor3D(const std::vector<int>& dimensions, InitType init = InitType::Default) : Tensor<T>(dimensions)
	{
		data = new Tensor2D<T>[dimensions[0]];
		for (int i = 0; i < dimensions[0]; ++i) {
			data[i] = Tensor2D<T>({ dimensions[1], dimensions[2] }, init);
		}
	}

	Tensor3D(const Tensor3D& other) : Tensor<T>(other.dimensions)
	{
		data = new Tensor2D<T>[other.dimensions[0]];
		for (int i = 0; i < other.dimensions[0]; ++i) {
			data[i] = other.data[i];
		}
	}

	~Tensor3D() {
		delete[] data;
	}

	Tensor3D& operator=(const Tensor3D& other) {
		if (this == &other) {
			return *this;
		}

		delete[] data;

		this->dimensions = other.dimensions;
		data = new Tensor2D<T>[other.dimensions[0]];
		for (int i = 0; i < other.dimensions[0]; ++i) {
			data[i] = other.data[i];
		}

		return *this;
	}

	T& operator()(const std::vector<int>& indices) override {
		return data[indices[0]]({ indices[1], indices[2] });
	}

	const T& operator()(const std::vector<int>& indices) const override {
		return data[indices[0]]({ indices[1], indices[2] });
	}

	void operator()(const std::vector<int>& indices, const T& value) override {
		if (indices.size() != 3 || indices[0] < 0 || indices[0] >= this->dimensions[0] || indices[1] < 0 || indices[1] >= this->dimensions[1] || indices[2] < 0 || indices[2] >= this->dimensions[2]) {
			throw std::out_of_range("Index out of range");
		}
		data[indices[0]]({ indices[1], indices[2] }) = value;
	}

	Tensor3D operator+(const Tensor3D& other) const {
        if (this->dimensions != other.dimensions) {
			throw std::invalid_argument("Dimensions must match for addition");
		}
		Tensor3D result(this->dimensions);
		for (int i = 0; i < this->dimensions[0]; ++i) {
			result.data[i] = data[i] + other.data[i];
		}
		return result;
	}

    Tensor3D& operator+=(const Tensor3D& other) {
		if (this->dimensions != other.dimensions) {
			throw std::invalid_argument("Dimensions must match for addition");
		}
		for (int i = 0; i < this->dimensions[0]; ++i) {
			data[i] += other.data[i];
		}
		return *this;
	}

	Tensor3D operator-(const Tensor3D& other) const {
		if (this->dimensions != other.dimensions) {
			throw std::invalid_argument("Dimensions must match for subtraction");
		}
		Tensor3D result(this->dimensions);
		for (int i = 0; i < this->dimensions[0]; ++i) {
			result.data[i] = data[i] - other.data[i];
		}
		return result;
	}

	Tensor3D& operator-=(const Tensor3D& other) {
		if (this->dimensions != other.dimensions) {
			throw std::invalid_argument("Dimensions must match for subtraction");
		}
		for (int i = 0; i < this->dimensions[0]; ++i) {
			data[i] -= other.data[i];
		}
		return *this;
	}

	Tensor3D operator*(const Tensor3D& other) const {
		if (this->dimensions != other.dimensions) {
			throw std::invalid_argument("Dimensions must match for multiplication");
		}
		Tensor3D result(this->dimensions);
		for (int i = 0; i < this->dimensions[0]; ++i) {
			result.data[i] = data[i] * other.data[i];
		}
		return result;
	}

	Tensor3D& operator*=(const Tensor3D& other) {
		if (this->dimensions != other.dimensions) {
			throw std::invalid_argument("Dimensions must match for multiplication");
		}
		for (int i = 0; i < this->dimensions[0]; ++i) {
			data[i] *= other.data[i];
		}
		return *this;
	}

	void print(std::ostream& os) const override {
		os << "{\n";
		for (int i = 0; i < this->dimensions[0]; ++i) {
			os << "  ";
			data[i].print(os);
			os << ",\n";
		}
		os << "}";
	}

	void setMatrix(Tensor2D<T>& matrix, int matrixNumber) {
		if (matrix.getDimensions()[0] != this->dimensions[1] || matrix.getDimensions()[1] != this->dimensions[2]) {
			throw std::invalid_argument("Matrix dimensions must match tensor dimensions");
		}
		data[matrixNumber] = matrix;
	}

	Tensor2D<T> getMatrix(int matrixNumber) {
		if (matrixNumber < 0 || matrixNumber >= this->dimensions[0]) {
			throw std::out_of_range("Matrix index out of range");
		}

		return data[matrixNumber];
	}

	static Tensor3D<T> dot(const Tensor<T>& tensor1, const Tensor<T>& tensor2) {
		// Implement dot product of two 3D tensors
		const Tensor3D<T>& t1 = dynamic_cast<const Tensor3D<T>&>(tensor1);
		const Tensor3D<T>& t2 = dynamic_cast<const Tensor3D<T>&>(tensor2);

		// Get dimensions for result vector and find dot product
		if (t1.dimensions[0] == t2.dimensions[0]) {
			std::vector<int> resultDimensions;

			if (t1.dimensions[1] == t2.dimensions[1]) {
				if (t1.dimensions[2] == t2.dimensions[2]) {
					resultDimensions = t1.dimensions;
				}
				else if (t1.dimensions[2] % t2.dimensions[2] == 0) {
					resultDimensions = t2.dimensions;
				}
				else if (t2.dimensions[2] % t1.dimensions[2] == 0) {
					resultDimensions = t1.dimensions;
				}
				else {
					std::cerr << "Dimensions do not match for dot product\n";
					throw std::invalid_argument("Dimensions must match for dot product");
				}
			}
			else if (t1.dimensions[1] % t2.dimensions[1] == 0) {
				if (t1.dimensions[2] == t2.dimensions[2]) {
					resultDimensions = t2.dimensions;
				}
				else {
					std::cerr << "Dimensions do not match for dot product\n";
					throw std::invalid_argument("Dimensions must match for dot product");
				}
			}
			else if (t2.dimensions[1] % t1.dimensions[1] == 0) {
				if (t1.dimensions[2] == t2.dimensions[2]) {
					resultDimensions = t1.dimensions;
				}
				else {
					std::cerr << "Dimensions do not match for dot product\n";
					throw std::invalid_argument("Dimensions must match for dot product");
				}
			}
			else {
				std::cerr << "Dimensions do not match for dot product\n";
				throw std::invalid_argument("Dimensions must match for dot product");
			}

			Tensor3D<T> result(resultDimensions);

			for (int i = 0; i < t1.dimensions[0]; ++i) {
				result.setMatrix(Tensor2D<T>::dot(t1.getMatrix(i), t2.getMatrix(i)), i);
			}

			return result;
		}
		else if (t1.dimensions[0] % t2.dimensions[0] == 0) {
			if (t1.dimensions[1] == t2.dimensions[1] && t1.dimensions[2] == t2.dimensions[2]) {
				std::vector<int> resultDimensions = t2.dimensions;

				Tensor3D<T> result(resultDimensions);

				for (int i = 0; i < resultDimensions[0]; ++i) {
					result.setMatrix(t1.getMatrix(i % t1.dimensions[0]) * t2.getMatrix(i), i);
				}

				return result;
			}
			else {
				std::cerr << "Dimensions do not match for dot product\n";
				throw std::invalid_argument("Dimensions must match for dot product");
			}
		} 
		else if (t2.dimensions[0] % t1.dimensions[0] == 0) {
			if (t1.dimensions[1] == t2.dimensions[1] && t1.dimensions[2] == t2.dimensions[2]) {
				std::vector<int> resultDimensions = t1.dimensions;

				Tensor3D<T> result(resultDimensions);

				for (int i = 0; i < resultDimensions[0]; ++i) {
					result.setMatrix(t1.getMatrix(i) * t2.getMatrix(i % t2.dimensions[0]), i);
				}

				return result;
			}
			else {
				std::cerr << "Dimensions do not match for dot product\n";
				throw std::invalid_argument("Dimensions must match for dot product");
			}
		}
		else {
			std::cerr << "Dimensions do not match for dot product\n";
			throw std::invalid_argument("Dimensions must match for dot product");
		}
	}

private:
	Tensor2D<T>* data = nullptr;
};