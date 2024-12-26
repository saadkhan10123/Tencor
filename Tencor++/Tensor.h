#pragma once
#include <iostream>
#include <vector>
#include <initializer_list>
#include <cassert>

template <typename T> class Tensor1;
template <typename T> class Tensor2;
template <typename T> class Tensor3;

enum class InitType {
	Default,
	Ones,
	Random
};

template <typename T>
class Tensor {
public:
    Tensor() = default;

    Tensor(const std::vector<int>& shape) : shape(shape)
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
        if (tensor1.shape.size() != tensor2.shape.size()) {
			std::cerr << "Tensors must have the same number of dimensions\n";
            throw std::invalid_argument("Tensors must have the same number of dimensions");
        }

        switch (tensor1.shape.size()) {
        case 1:
            return Tensor1<T>::dot(tensor1, tensor2);
        default:
			std::cerr << "Unsupported number of dimensions\n";
            throw std::invalid_argument("Unsupported number of dimensions");
        }
    }

    std::vector<int> getShape() const {
		return shape;
	}

public:
    std::vector<int> shape;
};

template <typename T>
class Tensor1 : public Tensor<T> {
    // Properties and methods specific to 1D tensors
public:
    Tensor1() = default;

    Tensor1(const std::vector<int>& shape, InitType init = InitType::Default) : Tensor<T>(shape)
    {
        data = new T[shape[0]];

        switch (init) {
        case InitType::Default:
            for (int i = 0; i < shape[0]; ++i) {
                data[i] = 0;
            }
            break;
        case InitType::Ones:
            for (int i = 0; i < shape[0]; ++i) {
                data[i] = 1;
            }
            break;
        case InitType::Random:
            for (int i = 0; i < shape[0]; ++i) {
				// Random initialization between -1 and 1
				data[i] = static_cast<T>(rand() % 2000 - 1000) / 1000;
            }
            break;
        default:
			std::cerr << "Unsupported initialization type\n";
            throw std::invalid_argument("Unsupported initialization type");
        }
    }

    Tensor1(const Tensor1& other) : Tensor<T>(other.shape)
    {
        data = new T[other.shape[0]];
        for (int i = 0; i < other.shape[0]; ++i) {
            data[i] = other.data[i];
        }
    }

	// Add this constructor to Tensor1D class
	Tensor1(std::initializer_list<T> values) : Tensor<T>({ static_cast<int>(values.size()) }) {
		data = new T[values.size()];
		int i = 0;
		for (const T& value : values) {
			data[i++] = value;
		}
	}

    ~Tensor1() {
        delete[] data;
    }

    Tensor1& operator=(const Tensor1& other) {
        if (this == &other) {
            return *this;
        }

        delete[] data;

        this->shape = other.shape;
        data = new T[other.shape[0]];
        for (int i = 0; i < other.shape[0]; ++i) {
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
        if (indices.size() != 1 || indices[0] < 0 || indices[0] >= this->shape[0]) {
			std::cerr << "Index out of range\n";
            throw std::out_of_range("Index out of range");
        }
        data[indices[0]] = value;
    }

	T& operator[] (int index) {
		return data[index];
	}

    Tensor1 operator+(const Tensor1& other) const {
        if (this->shape == other.shape) {
			Tensor1 result(this->shape);

			for (int i = 0; i < this->shape[0]; ++i) {
				result({ i }) = data[i] + other({ i });
			}

			return result;
		}
		else if (this->shape[0] % other.shape[0] == 0) {
			Tensor1 result(this->shape);

			for (int i = 0; i < this->shape[0]; ++i) {
				result({ i }) = data[i] + other({ i % other.shape[0] });
			}

			return result;
		}
		else if (other.shape[0] % this->shape[0] == 0) {
			Tensor1 result(other.shape);

			for (int i = 0; i < other.shape[0]; ++i) {
				result({ i }) = data[i % this->shape[0]] + other({ i });
			}

			return result;
		}
		else {
			throw std::invalid_argument("Dimensions must match for addition");
		}
	}

    Tensor1& operator+=(const Tensor1& other) {
        if (this->shape == other.shape) {
			for (int i = 0; i < this->shape[0]; ++i) {
				data[i] += other({ i });
			}
		} 
		else if (this->shape[0] % other.shape[0] == 0) {
			for (int i = 0; i < this->shape[0]; ++i) {
				data[i] += other({ i % other.shape[0] });
			}
		}
		else {
			std::cerr << "Dimensions must match for addition\n";
			throw std::invalid_argument("Dimensions must match for addition");
		}
		return *this;
	}

    Tensor1 operator-(const Tensor1& other) const {
		if (this->shape == other.shape) {
			Tensor1 result(this->shape);

			for (int i = 0; i < this->shape[0]; ++i) {
				result({ i }) = data[i] - other({ i });
			}

			return result;
		}
		else if (this->shape[0] % other.shape[0] == 0) {
			Tensor1 result(this->shape);

			for (int i = 0; i < this->shape[0]; ++i) {
				result({ i }) = data[i] - other({ i % other.shape[0] });
			}

			return result;
		}
		else if (other.shape[0] % this->shape[0] == 0) {
			Tensor1 result(other.shape);

			for (int i = 0; i < other.shape[0]; ++i) {
				result({ i }) = data[i % this->shape[0]] - other({ i });
			}

			return result;
		}
		else {
			std::cerr << "Dimensions must match for subtraction\n";
			throw std::invalid_argument("Dimensions must match for subtraction");
		}
    }

	friend Tensor1<T> operator-(T lhs, const Tensor1<T>& rhs) {
		Tensor1<T> result(rhs.shape);
		for (int i = 0; i < rhs.shape[0]; ++i) {
			result({ i }) = lhs - rhs({ i });
		}
		return result;
	}

    Tensor1 operator-() const {
        Tensor1 result(this->shape);
        for (int i = 0; i < this->shape[0]; ++i) {
            result({ i }) = -data[i];
        }
        return result;
    }

    Tensor1& operator-=(const Tensor1& other) {
		if (this->shape == other.shape) {
			for (int i = 0; i < this->shape[0]; ++i) {
				data[i] -= other({ i });
			}
		}
		else if (this->shape[0] % other.shape[0] == 0) {
			for (int i = 0; i < this->shape[0]; ++i) {
				data[i] -= other({ i % other.shape[0] });
			}
		}
		else {
			std::cerr << "Dimensions must match for subtraction\n";
			throw std::invalid_argument("Dimensions must match for subtraction");
		}
	}

	

    Tensor1 operator*(const Tensor1& other) const {
        if (this->shape != other.shape) {
			std::cerr << "Dimensions must match for multiplication";
			throw std::invalid_argument("Dimensions must match for multiplication");
		}
		Tensor1 result(this->shape);
        for (int i = 0; i < this->shape[0]; ++i) {
			result({ i }) = data[i] * other({ i });
		}
		return result;
	}

    Tensor1& operator*=(const Tensor1& other) {
        if (this->shape != other.shape) {
            throw std::invalid_argument("Dimensions must match for multiplication");
        }
        for (int i = 0; i < this->shape[0]; ++i) {
            data[i] *= other({ i });
        }
        return *this;
    }

	Tensor1 operator*(const T& other) const {
		Tensor1 result(this->shape);
		for (int i = 0; i < this->shape[0]; ++i) {
			result({ i }) = data[i] * other;
		}
		return result;
	}

	Tensor1 operator/(const Tensor1& other) const {
		if (this->shape != other.shape) {
			throw std::invalid_argument("Dimensions must match for division");
		}
		Tensor1 result(this->shape);
		for (int i = 0; i < this->shape[0]; ++i) {
			result({ i }) = data[i] / other({ i });
		}
		return result;
	}

	Tensor1 operator/(const T& other) const {
		Tensor1 result(this->shape);
		for (int i = 0; i < this->shape[0]; ++i) {
			result({ i }) = data[i] / other;
		}
		return result;
	}

	Tensor1& operator/=(const T& other) {
		for (int i = 0; i < this->shape[0]; ++i) {
			data[i] /= other;
		}
		return *this;
	}

	Tensor1 apply(T(*func)(T)) const {
		Tensor1 result(this->shape);
		for (int i = 0; i < this->shape[0]; ++i) {
			result({ i }) = func(data[i]);
		}
		return result;
	}

    void print(std::ostream& os) const override {
        os << "{ ";
        for (int i = 0; i < this->shape[0]; ++i) {
            os << data[i];
            if (i < this->shape[0] - 1) {
				os << ", ";
			}
        }
        os << " }";
    }

	Tensor1 slice(int start, int end) const {
		std::vector<int> resultShape = { end - start };
		Tensor1 result(resultShape);
		for (int i = start; i < end; ++i) {
			result({ i - start }) = data[i];
		}
		return result;
	}

	Tensor2<T> squeeze() const {
		Tensor2<T> result({ 1, this->shape[0] });
		for (int i = 0; i < this->shape[0]; ++i) {
			result({ 0, i }) = data[i];
		}
		return result;
	}

    static Tensor1<T> dot(const Tensor<T>& tensor1, const Tensor<T>& tensor2) {
        // Implement dot product of two 1D tensors
        const Tensor1<T>& t1 = dynamic_cast<const Tensor1<T>&>(tensor1);
        const Tensor1<T>& t2 = dynamic_cast<const Tensor1<T>&>(tensor2);

        if (t1.shape[0] != t2.shape[0]) {
            if (t1.shape[0] % t2.shape[0] == 0) {
                return dotProjected(t1, t2);
            }
            else if (t2.shape[0] % t1.shape[0] == 0) {
                return dotProjected(t2, t1);
            }
            else {
				std::cerr << "Dimensions must match for dot product\n";
                throw std::invalid_argument("Dimensions must match for dot product");
            }
        }

        Tensor1<T> result(t1.shape);
        for (int i = 0; i < t1.shape[0]; ++i) {
            result({ i }) = t1({ i }) * t2({ i });
        }
        return result;
    }

	static Tensor1<T> max(const Tensor1<T>& tensor) {
		Tensor1<T> result({ 1 });
		T maxVal = tensor({ 0 });
		for (int i = 1; i < tensor.shape[0]; ++i) {
			if (tensor({ i }) > maxVal) {
				maxVal = tensor({ i });
			}
		}
		result({ 0 }) = maxVal;
		return result;
	}

	static Tensor1<T> argmax(const Tensor1<T>& tensor) {
		Tensor1<T> result({ 1 });
		T maxVal = tensor({ 0 });
		int maxIndex = 0;
		for (int i = 1; i < tensor.shape[0]; ++i) {
			if (tensor({ i }) > maxVal) {
				maxVal = tensor({ i });
				maxIndex = i;
			}
		}
		result({ 0 }) = maxIndex;
		return result;
	}

	static Tensor1<T> log(const Tensor1<T>& tensor) {
		Tensor1<T> result(tensor.shape);
		for (int i = 0; i < tensor.shape[0]; ++i) {
			result({ i }) = std::log(tensor({ i }));
		}
		return result;
	}

private:
    T* data = nullptr;

    static Tensor1<T> dotProjected(const Tensor1<T>& t1, const Tensor1<T>& t2) {
        Tensor1<T> result(t1.shape);
        for (int i = 0; i < t1.shape[0]; ++i) {
            result({ i }) = t1({ i }) * t2({ i % t2.shape[0] });
        }
        return result;
    }
};

template <typename T>
class Tensor2 : public Tensor<T> {
public:
	Tensor2() = default;

	Tensor2(const std::vector<int>& shape, InitType init = InitType::Default) : Tensor<T>(shape)
	{
		data = new Tensor1<T>[shape[0]];
		for (int i = 0; i < shape[0]; ++i) {
			data[i] = Tensor1<T>({ shape[1] }, init);
		}
	}

	Tensor2(const Tensor2& other) : Tensor<T>(other.shape)
	{
		data = new Tensor1<T>[other.shape[0]];
		for (int i = 0; i < other.shape[0]; ++i) {
			data[i] = other.data[i];
		}
	}

	Tensor2(std::initializer_list<std::initializer_list<T>> values) : Tensor<T>({ static_cast<int>(values.size()), static_cast<int>(values.begin()->size()) }) {
		data = new Tensor1<T>[values.size()];
		int i = 0;
		for (const std::initializer_list<T>& row : values) {
			data[i++] = Tensor1<T>(row);
		}
	}

	Tensor2(const std::vector<int>& shape, const Tensor1<T>& data) : Tensor<T>(shape) {
		if (shape[0] == 1) {
			this->data = new Tensor1<T>[shape[0]];
			this->data[0] = data;
		}
		else if (shape[1] == 1) {
			this->data = new Tensor1<T>[shape[0]];
			for (int i = 0; i < shape[0]; ++i) {
				this->data[i] = Tensor1<T>{ { 1 }, data({ i }) };
			}
		}
		else {
			throw std::invalid_argument("Invalid shape for data");
		}
	}

	~Tensor2() {
		delete[] data;
	}

	Tensor2& operator=(const Tensor2& other) {
		if (this == &other) {
			return *this;
		}

		delete[] data;

		this->shape = other.shape;
		data = new Tensor1<T>[other.shape[0]];
		for (int i = 0; i < other.shape[0]; ++i) {
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
		if (indices.size() != 2 || indices[0] < 0 || indices[0] >= this->shape[0] || indices[1] < 0 || indices[1] >= this->shape[1]) {
			throw std::out_of_range("Index out of range");
		}
		data[indices[0]]({ indices[1] }) = value;
	}

	Tensor1<T>& operator[] (int index) {
		return data[index];
	}

	Tensor2 operator+(const Tensor2& other) const {

		std::vector<int> resultDimensions;
		try {
			resultDimensions = getDimensionsOp(*this, other);
		}
		catch (const std::invalid_argument& e) {
			std::cerr << "Dimensions must match for addition\n";
			throw std::invalid_argument("Dimensions must match for addition");
		}

		Tensor2 result(resultDimensions);

		if (this->shape == other.shape) {
			for (int i = 0; i < this->shape[0]; ++i) {
				result.data[i] = data[i] + other.data[i];
			}
		}
		else if (this->shape[0] % other.shape[0] == 0) {
			for (int i = 0; i < this->shape[0]; ++i) {
				// std::cout << this->shape[0] << this->shape[1] << " " << other.shape[0] << other.shape[1] << std::endl;
				result.data[i] = data[i] + other.data[i % other.shape[0]];
			}
		}
		else if (other.shape[0] % this->shape[0] == 0) {
			for (int i = 0; i < other.shape[0]; ++i) {
				result.data[i] = data[i % this->shape[0]] + other.data[i];
			}
		}
		else {
			throw std::invalid_argument("Dimensions must match for addition");
		}

		return result;
	}

	Tensor2 operator+(const T& other) const {
		Tensor2 result(this->shape);
		for (int i = 0; i < this->shape[0]; ++i) {
			result.data[i] = data[i] + other;
		}
		return result;
	}

	Tensor2& operator+=(const Tensor2& other) {
		if (this->shape == other.shape) {
			for (int i = 0; i < this->shape[0]; ++i) {
				data[i] += other.data[i];
			}
		}
		else if (this->shape[0] % other.shape[0] == 0) {
			for (int i = 0; i < this->shape[0]; ++i) {
				data[i] += other.data[i % other.shape[0]];
			}
		}
		else {
			throw std::invalid_argument("Dimensions must match for addition");
		}

		return *this;
	}

	Tensor2 operator-(const Tensor2& other) const {
		std::vector<int> resultDimensions;
		try {
			resultDimensions = getDimensionsOp(*this, other);
		}
		catch (const std::invalid_argument& e) {
			std::cerr << "Dimensions must match for subtraction\n";
			throw std::invalid_argument("Dimensions must match for subtraction");
		}

		Tensor2 result(resultDimensions);

		if (this->shape == other.shape) {
			for (int i = 0; i < this->shape[0]; ++i) {
				result.data[i] = data[i] - other.data[i];
			}
		}
		else if (this->shape[0] % other.shape[0] == 0) {
			for (int i = 0; i < this->shape[0]; ++i) {
				result.data[i] = data[i] - other.data[i % other.shape[0]];
			}
		}
		else if (other.shape[0] % this->shape[0] == 0) {
			for (int i = 0; i < other.shape[0]; ++i) {
				result.data[i] = data[i % this->shape[0]] - other.data[i];
			}
		}
		else {
			throw std::invalid_argument("Dimensions must match for subtraction");
		}

		return result;
	}

	friend Tensor2<T> operator-(T lhs, const Tensor2<T>& rhs) {
		Tensor2<T> result(rhs.shape);
		for (int i = 0; i < rhs.shape[0]; ++i) {
			result.data[i] = lhs - rhs.data[i];
		}
		return result;
	}

	Tensor2 operator-() const {
		Tensor2 result(this->shape);
		for (int i = 0; i < this->shape[0]; ++i) {
			result.data[i] = -data[i];
		}
		return result;
	}

	Tensor2& operator-=(const Tensor2& other) {
		if (this->shape == other.shape) {
			for (int i = 0; i < this->shape[0]; ++i) {
				data[i] -= other.data[i];
			}
		}
		else if (this->shape[0] % other.shape[0] == 0) {
			for (int i = 0; i < this->shape[0]; ++i) {
				data[i] -= other.data[i % other.shape[0]];
			}
		}
		else {
			throw std::invalid_argument("Dimensions must match for subtraction");
		}
		return *this;
	}

	Tensor2 operator*(const T& other) const {
		Tensor2 result(this->shape);
		for (int i = 0; i < this->shape[0]; ++i) {
			result.data[i] = data[i] * other;
		}
		return result;
	}

	Tensor2 operator*(const Tensor2& other) const {
		if (this->shape != other.shape) {
			throw std::invalid_argument("Dimensions must match for multiplication");
		}
		Tensor2 result(this->shape);
		for (int i = 0; i < this->shape[0]; ++i) {
			result.data[i] = data[i] * other.data[i];
		}
		return result;
	}

	Tensor2& operator*=(const Tensor2& other) {
		if (this->shape != other.shape) {
			throw std::invalid_argument("Dimensions must match for multiplication");
		}
		for (int i = 0; i < this->shape[0]; ++i) {
			data[i] *= other.data[i];
		}
		return *this;
	}

	Tensor2 operator/(const Tensor2& other) const {
		if (this->shape == other.shape || this->shape[0] % other.shape[0] == 0) {
			Tensor2 result(this->shape);
			for (int i = 0; i < this->shape[0]; ++i) {
				result.data[i] = data[i] / other.data[i % other.shape[0]];
			}
			return result;
		}
		else if (other.shape[0] % this->shape[0] == 0) {
			Tensor2 result(other.shape);
			for (int i = 0; i < other.shape[0]; ++i) {
				result.data[i] = data[i % this->shape[0]] / other.data[i];
			}
			return result;
		}
		else {
			std::cerr << "Dimensions must match for division\n";
			throw std::invalid_argument("Dimensions must match for division");
		}
	}

	Tensor2 operator/(const T& other) const {
		Tensor2 result(this->shape);
		for (int i = 0; i < this->shape[0]; ++i) {
			result.data[i] = data[i] / other;
		}
		return result;
	}

	Tensor2& operator/=(const T& other) {
		for (int i = 0; i < this->shape[0]; ++i) {
			data[i] /= other;
		}
		return *this;
	}

	void print(std::ostream& os) const override {
		os << "{\n";
		for (int i = 0; i < this->shape[0]; ++i) {
			os << "  ";
			data[i].print(os);
			os << ",\n";
		}
		os << "}";
	}

	void setRow(const Tensor1<T>& row, int rowNumber) {
		if (row.getShape()[0] != this->shape[1]) {
			std::cerr << "Row dimensions must match tensor dimensions\n";
			throw std::invalid_argument("Row dimensions must match tensor dimensions");
		}
		this->data[rowNumber] = row;
	}

	Tensor1<T> getRow(int rowNumber) const {
		if (rowNumber < 0 || rowNumber >= this->shape[0]) {
			std::cerr << "Row index out of range\n";
			throw std::out_of_range("Row index out of range");
		}

		return data[rowNumber];
	}

	Tensor2 apply(T(*func)(T)) const {
		Tensor2 result(this->shape);
		for (int i = 0; i < this->shape[0]; ++i) {
			result.data[i] = data[i].apply(func);
		}
		return result;
	}

	Tensor2 slice(int start, int end, int axis = 0) const {
		if (axis == 0) {
			if (start < 0 || start >= this->shape[0] || end < 0 || end > this->shape[0] || start >= end) {
				std::cerr << "\nInvalid slice indices\n";
				throw std::invalid_argument("Invalid slice indices");
			}
			std::vector<int> resultShape = { end - start, this->shape[1] };

			Tensor2 result(resultShape);
			for (int i = start; i < end; ++i) {
				result.data[i - start] = data[i];
			}
			return result;
		}
		else if (axis == 1) {
			if (start < 0 || start >= this->shape[1] || end < 0 || end > this->shape[1] || start >= end) {
				std::cerr << "Invalid slice indices\n";
				throw std::invalid_argument("Invalid slice indices");
			}
			std::vector<int> resultShape = { this->shape[0], end - start };

			Tensor2 result(resultShape);
			for (int i = 0; i < this->shape[0]; ++i) {
				result.data[i] = data[i].slice(start, end);
			}
			return result;
		}
		else {
			std::cerr << "Invalid axis\n";
			throw std::invalid_argument("Invalid axis");
		}
	}



	static Tensor2<T> dot(const Tensor<T>& tensor1, const Tensor<T>& tensor2) {
		// Implement dot product of two 2D tensors
		const Tensor2<T>& t1 = dynamic_cast<const Tensor2<T>&>(tensor1);
		const Tensor2<T>& t2 = dynamic_cast<const Tensor2<T>&>(tensor2);

		if (tensor1.shape[1] != tensor2.shape[0]) {
			std::cerr << "Dimension mismath!\n";
			throw std::invalid_argument("Dimensions must match for dot product");
		}

		Tensor2<T> result({ tensor1.shape[0], tensor2.shape[1] });

		for (int i = 0; i < tensor1.shape[0]; ++i) {
			for (int j = 0; j < tensor2.shape[1]; ++j) {
				T sum = 0;
				for (int k = 0; k < tensor1.shape[1]; ++k) {
					sum += t1({ i, k }) * t2({ k, j });
				}
				result({ i, j }) = sum;
			}
		}

		return result;
	}

	static Tensor2<T> transpose(const Tensor<T>& tensor) {
		const Tensor2<T>& t1 = dynamic_cast<const Tensor2<T>&>(tensor);

		Tensor2<T> result({ t1.shape[1], t1.shape[0] });

		for (int i = 0; i < t1.shape[0]; ++i) {
			for (int j = 0; j < t1.shape[1]; ++j) {
				result({ j, i }) = t1({ i, j });
			}
		}

		return result;
	}

	static Tensor2<T> sum(const Tensor2<T>& tensor, int axis) {
		if (axis == 0) {
			Tensor2<T> result({ 1, tensor.shape[1] });
			for (int i = 0; i < tensor.shape[1]; ++i) {
				T sum = 0;
				for (int j = 0; j < tensor.shape[0]; ++j) {
					sum += tensor({ j, i });
				}
				result({ 0, i }) = sum;
			}
			return result;
		}
		else if (axis == 1) {
			Tensor2<T> result({ tensor.shape[0], 1 });
			for (int i = 0; i < tensor.shape[0]; ++i) {
				T sum = 0;
				for (int j = 0; j < tensor.shape[1]; ++j) {
					sum += tensor({ i, j });
				}
				result({ i, 0 }) = sum;
			}
			return result;
		}
		else {
			std::cerr << "Invalid axis\n";
			throw std::invalid_argument("Invalid axis");
		}
	}

	static Tensor2<T> sum(const Tensor<T>& tensor, int axis) {
		const Tensor2<T>& t1 = dynamic_cast<const Tensor2<T>&>(tensor);

		return sum(t1, axis);
	}

	static T sum(const Tensor<T>& tensor) {
		const Tensor2<T>& t1 = dynamic_cast<const Tensor2<T>&>(tensor);

		T sum = 0;
		for (int i = 0; i < t1.shape[0]; ++i) {
			for (int j = 0; j < t1.shape[1]; ++j) {
				sum += t1({ i, j });
			}
		}

		return sum;
	}

	static Tensor2<T> square(const Tensor<T>& tensor) {
		const Tensor2<T>& t1 = dynamic_cast<const Tensor2<T>&>(tensor);

		Tensor2<T> result(t1.shape);
		for (int i = 0; i < t1.shape[0]; ++i) {
			for (int j = 0; j < t1.shape[1]; ++j) {
				result({ i, j }) = t1({ i, j }) * t1({ i, j });
			}
		}

		return result;
	}

	static Tensor2<T> max(const Tensor2<T>& tensor, int axis = 0) {
		if (axis == 0) {
			Tensor2<T> result({ 1, tensor.shape[1] });
			for (int i = 0; i < tensor.shape[1]; ++i) {
				T maxVal = tensor({ 0, i });
				for (int j = 1; j < tensor.shape[0]; ++j) {
					if (tensor({ j, i }) > maxVal) {
						maxVal = tensor({ j, i });
					}
				}
				result({ 0, i }) = maxVal;
			}
			return result;
		}
		else if (axis == 1) {
			Tensor2<T> result({ tensor.shape[0], 1 });
			for (int i = 0; i < tensor.shape[0]; ++i) {
				result({ i, 0 }) = Tensor1<T>::max(tensor.getRow(i))({ 0 });
			}
			return result;
		}
		else {
			std::cerr << "Invalid axis\n";
			throw std::invalid_argument("Invalid axis");
		}
	}

	static Tensor2<T> argmax(const Tensor2<T>& tensor, int axis = 0) {
		if (axis == 0) {
			Tensor2<T> result({ 1, tensor.shape[1] });
			for (int i = 0; i < tensor.shape[1]; ++i) {
				T maxVal = tensor({ 0, i });
				int maxIndex = 0;
				for (int j = 1; j < tensor.shape[0]; ++j) {
					if (tensor({ j, i }) > maxVal) {
						maxVal = tensor({ j, i });
						maxIndex = j;
					}
				}
				result({ 0, i }) = maxIndex;
			}
			return result;
		}
		else if (axis == 1) {
			Tensor2<T> result({ tensor.shape[0], 1 });
			for (int i = 0; i < tensor.shape[0]; ++i) {
				result({ i, 0 }) = Tensor1<T>::argmax(tensor.getRow(i))({ 0 });
			}
			return result;
		}
		else {
			std::cerr << "Invalid axis\n";
			throw std::invalid_argument("Invalid axis");
		}
	}

	static Tensor2<T> log(const Tensor2<T>& tensor) {
		Tensor2<T> result(tensor.shape);
		for (int i = 0; i < tensor.shape[0]; ++i) {
			for (int j = 0; j < tensor.shape[1]; ++j) {
				result({ i, j }) = std::log(tensor({ i, j }));
			}
		}
		return result;
	}


private:
	Tensor1<T>* data = nullptr;

	static std::vector<int> getDimensionsOp(const Tensor2<T> A, const Tensor2<T> B) {
		if (A.getShape()[0] == B.getShape()[0]) {
			if (A.getShape()[1] == B.getShape()[1]) {
				return A.getShape();
			}
			else if (A.getShape()[1] % B.getShape()[1] == 0) {
				return A.getShape();
			}
			else if (B.getShape()[1] % A.getShape()[1] == 0) {
				return B.getShape();
			}
			else {
				std::cerr << "Dimension mismatch";
				throw std::invalid_argument("Dimension mismatch");
			}
		}
		else if (A.getShape()[0] % B.getShape()[0] == 0) {
			if (A.getShape()[1] == B.getShape()[1]) {
				return A.getShape();
			}
			else {
				std::cerr << "Dimension mismatch";
				throw std::invalid_argument("Dimension mismatch");
			}
		}
		else if (B.getShape()[0] % A.getShape()[0] == 0) {
			if (A.getShape()[1] == B.getShape()[1]) {
				return B.getShape();
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
class Tensor3 : public Tensor<T> {
public:
	Tensor3(const std::vector<int>& shape, InitType init = InitType::Default) : Tensor<T>(shape)
	{
		data = new Tensor2<T>[shape[0]];
		for (int i = 0; i < shape[0]; ++i) {
			data[i] = Tensor2<T>({ shape[1], shape[2] }, init);
		}
	}

	Tensor3(const Tensor3& other) : Tensor<T>(other.shape)
	{
		data = new Tensor2<T>[other.shape[0]];
		for (int i = 0; i < other.shape[0]; ++i) {
			data[i] = other.data[i];
		}
	}

	~Tensor3() {
		delete[] data;
	}

	Tensor3& operator=(const Tensor3& other) {
		if (this == &other) {
			return *this;
		}

		delete[] data;

		this->shape = other.shape;
		data = new Tensor2<T>[other.shape[0]];
		for (int i = 0; i < other.shape[0]; ++i) {
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
		if (indices.size() != 3 || indices[0] < 0 || indices[0] >= this->shape[0] || indices[1] < 0 || indices[1] >= this->shape[1] || indices[2] < 0 || indices[2] >= this->shape[2]) {
			throw std::out_of_range("Index out of range");
		}
		data[indices[0]]({ indices[1], indices[2] }) = value;
	}

	Tensor3 operator+(const Tensor3& other) const {
		if (this->shape != other.shape) {
			throw std::invalid_argument("Dimensions must match for addition");
		}
		Tensor3 result(this->shape);
		for (int i = 0; i < this->shape[0]; ++i) {
			result.data[i] = data[i] + other.data[i];
		}
		return result;
	}

	Tensor3& operator+=(const Tensor3& other) {
		if (this->shape != other.shape) {
			throw std::invalid_argument("Dimensions must match for addition");
		}
		for (int i = 0; i < this->shape[0]; ++i) {
			data[i] += other.data[i];
		}
		return *this;
	}

	Tensor3 operator-(const Tensor3& other) const {
		if (this->shape != other.shape) {
			throw std::invalid_argument("Dimensions must match for subtraction");
		}
		Tensor3 result(this->shape);
		for (int i = 0; i < this->shape[0]; ++i) {
			result.data[i] = data[i] - other.data[i];
		}
		return result;
	}

	Tensor3& operator-=(const Tensor3& other) {
		if (this->shape != other.shape) {
			throw std::invalid_argument("Dimensions must match for subtraction");
		}
		for (int i = 0; i < this->shape[0]; ++i) {
			data[i] -= other.data[i];
		}
		return *this;
	}

	Tensor3 operator*(const Tensor3& other) const {
		if (this->shape != other.shape) {
			throw std::invalid_argument("Dimensions must match for multiplication");
		}
		Tensor3 result(this->shape);
		for (int i = 0; i < this->shape[0]; ++i) {
			result.data[i] = data[i] * other.data[i];
		}
		return result;
	}

	Tensor3& operator*=(const Tensor3& other) {
		if (this->shape != other.shape) {
			throw std::invalid_argument("Dimensions must match for multiplication");
		}
		for (int i = 0; i < this->shape[0]; ++i) {
			data[i] *= other.data[i];
		}
		return *this;
	}

	Tensor3& operator/=(const T& other) {
		for (int i = 0; i < this->shape[0]; ++i) {
			data[i] /= other;
		}
		return *this;
	}

	void print(std::ostream& os) const override {
		os << "{\n";
		for (int i = 0; i < this->shape[0]; ++i) {
			os << "  ";
			data[i].print(os);
			os << ",\n";
		}
		os << "}";
	}

	Tensor2<T> flatten(int axis = 0) {
		if (axis == 0) {
			std::vector<int> resultShape = { this->shape[0] * this->shape[1], this->shape[2] };
			Tensor2<T> result(resultShape);
			for (int i = 0; i < this->shape[0]; ++i) {
				for (int j = 0; j < this->shape[1]; ++j) {
					for (int k = 0; k < this->shape[2]; ++k) {
						result({ i * this->shape[1] + j, k }) = data[i]({ j, k });
					}
				}
			}
			return result;
		}
		else if (axis == 1) {
			std::vector<int> resultShape = { this->shape[0], this->shape[1] * this->shape[2] };
			Tensor2<T> result(resultShape);
			for (int i = 0; i < this->shape[0]; ++i) {
				for (int j = 0; j < this->shape[1]; ++j) {
					for (int k = 0; k < this->shape[2]; ++k) {
						result({ i, j * this->shape[2] + k }) = data[i]({ j, k });
					}
				}
			}
			return result;
		}
		else {
			std::cerr << "Invalid axis\n";
			throw std::invalid_argument("Invalid axis");
		}

	}


	static Tensor3<T> dot(const Tensor<T>& tensor1, const Tensor<T>& tensor2) {
		// Implement dot product of two 3D tensors
		const Tensor3<T>& t1 = dynamic_cast<const Tensor3<T>&>(tensor1);
		const Tensor3<T>& t2 = dynamic_cast<const Tensor3<T>&>(tensor2);
	}

private:
	Tensor2<T>* data = nullptr;

};