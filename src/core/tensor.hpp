#pragma once

#include <Eigen/Eigen>
#include <unsupported/Eigen/KroneckerProduct>
#include <array>
#include <memory>
#include <tuple>

template<typename Scalar, int Order>
class Tensor
{
public:
	using SizeVector = std::array<int, Order>;

	Tensor() noexcept : m_size{}, m_offsets{}, m_numElements(0), m_capacity(0), m_data(nullptr) {}

	explicit Tensor(const SizeVector& _size)
		: m_size(_size)
	{
		computeOffsets();
		m_numElements = static_cast<size_t>(m_offsets.back()) 
			* static_cast<size_t>(m_size.back());
		m_capacity = m_numElements;

		m_data = std::make_unique<Scalar[]>(m_numElements);
	}

	Tensor(const SizeVector& _size, const Scalar* _data)
		: Tensor(_size)
	{
		if (_data)
			std::copy(_data, _data + m_numElements, m_data.get());
	}

	Tensor(const Tensor& _oth)
		: m_size(_oth.m_size), 
		m_offsets(_oth.m_offsets),
		m_numElements(_oth.m_numElements),
		m_capacity(_oth.m_numElements),
		m_data(std::make_unique<Scalar[]>(m_numElements))
	{
		std::copy(_oth.m_data.get(), _oth.m_data.get() + m_numElements, m_data.get());
	}

	Tensor(Tensor&& _oth) noexcept
		: m_size(_oth.m_size),
		m_offsets(_oth.m_offsets),
		m_numElements(_oth.m_numElements),
		m_capacity(_oth.m_capacity),
		m_data(std::move(_oth.m_data))
	{}

	template<typename StreamT>
	explicit Tensor(StreamT& _stream)
	{
		_stream.read(reinterpret_cast<char*>(m_size.data()), m_size.size() * sizeof(int));

		computeOffsets();
		m_numElements = static_cast<size_t>(m_offsets.back())
			* static_cast<size_t>(m_size.back());
		m_capacity = m_numElements;

		m_data = std::make_unique<Scalar[]>(m_numElements);

		_stream.read(reinterpret_cast<char*>(m_data.get()), m_numElements * sizeof(Scalar));
	}

	Tensor& operator=(const Tensor& _oth)
	{
		m_size = _oth.m_size;
		m_offsets = _oth.m_offsets;
		m_numElements = _oth.m_numElements;
		m_capacity = m_numElements;
		m_data = std::make_unique<Scalar[]>(m_numElements);
		std::copy(_oth.m_data.get(), _oth.m_data.get() + m_numElements, m_data.get());

		return *this;
	}

	Tensor& operator=(Tensor&& _oth) noexcept
	{
		m_size = _oth.m_size;
		m_offsets = _oth.m_offsets;
		m_numElements = _oth.m_numElements;
		m_capacity = _oth.m_capacity;
		m_data = std::move(_oth.m_data);

		return *this;
	}

	// set from a k-flattening
	void set(const Eigen::MatrixX<Scalar>& _flatTensor, int _k)
	{
		assert(_flatTensor.rows() == m_size[_k]);
		assert(_flatTensor.rows() * _flatTensor.cols() == m_numElements);

		for (size_t i = 0; i < m_numElements; ++i)
		{
			const auto& [indK, indOth] = decomposeFlatIndex(i, _k);
			m_data[i] = _flatTensor(indK, indOth);
		}
	}

	//set from a k-flattening with K known at compile time
	template<int K>
	void set(const Eigen::MatrixX<Scalar>& _flatTensor) noexcept
	{
		static_assert(K < Order);
		assert(_flatTensor.rows() == m_size[K]);
		assert(_flatTensor.rows() * _flatTensor.cols() == m_numElements);

		if constexpr (K == 0)
		{
			std::copy(_flatTensor.data(), _flatTensor.data() + m_numElements, m_data.get());
			return;
		}

		for (size_t i = 0; i < m_numElements; ++i)
		{
			const auto& [indK, indOth] = decomposeFlatIndex<K>(i);
			m_data[i] = _flatTensor(indK, indOth);
		}
	}



	template<typename Gen>
	void set(Gen _generator)
	{
		for(size_t i = 0; i < m_numElements; ++i)
			m_data[i] = _generator(index(i));
	}

	void append(const Tensor<Scalar, Order>& _tensor)
	{
		for (int i = 0; i < Order - 1; ++i)
			if (m_size[i] != _tensor.size()[i])
				throw std::string("Incompatible tensor sizes.");

		reserve(m_numElements + _tensor.numElements());
		std::copy(_tensor.data(), _tensor.data() + _tensor.numElements(), m_data.get() + m_numElements);
		m_size.back() += _tensor.size().back();
		m_numElements += _tensor.numElements();
	}

	Eigen::MatrixX<Scalar> flatten(int _k) const
	{
		const size_t othDim = m_numElements / m_size[_k];
		Eigen::MatrixX<Scalar> m(m_size[_k], othDim);

		for (size_t i = 0; i < m_numElements; ++i)
		{
			const auto& [indK, indOth] = decomposeFlatIndex(i, _k);
			m(indK, indOth) = m_data[i];
		}

		return m;
	}

	// If K is known at compile time use this.
	template<int K>
	Eigen::MatrixX<Scalar> flatten() const
	{
		const size_t othDim = m_numElements / m_size[K];
		Eigen::MatrixX<Scalar> m(m_size[K], othDim);

		if constexpr (K == 0)
		{
			std::copy(m_data.get(), m_data.get() + m_numElements, m.data());
		}
		else
		{
			for (size_t i = 0; i < m_numElements; ++i)
			{
				const auto& [indK, indOth] = decomposeFlatIndex<K>(i);
				m(indK, indOth) = m_data[i];
			}
		}

		return m;
	}

	// Change the size of this tensor to _newSize.
	// The data is unspecified afterwards.
	// @param _shrink Shrink the buffer if the new size is smaller.
	void resize(const SizeVector& _newSize, bool _shrink = false)
	{
		m_size = _newSize;
		const std::size_t oldNum = m_numElements;

		computeOffsets();
		m_numElements = static_cast<size_t>(m_offsets.back()) 
			* static_cast<size_t>(m_size.back());

		if (m_capacity < m_numElements || (_shrink && oldNum > m_numElements))
		{
			m_data = std::make_unique<Scalar[]>(m_numElements);
			m_capacity = m_numElements;
		}
	}

	// Ensures that the reserved memory can hold atleast _capacity elements.
	// If the buffer is already larger no allocations take place.
	void reserve(std::size_t _capacity)
	{
		if (_capacity <= m_capacity) return;

		Scalar* newData = new float[_capacity];
		std::copy(m_data.get(), m_data.get() + m_numElements, newData);
		m_data.reset(newData);
		m_capacity = _capacity;
	}

	// ACCESS OPERATIONS

	// vectorization
	Eigen::Map<const Eigen::VectorX<Scalar>> vec() const noexcept
	{
		return { m_data.get(), static_cast<Eigen::Index>(m_numElements) };
	}

	// view which is equivalent to the 0-flattening
	Eigen::Map<const Eigen::MatrixX<Scalar>> mat() const noexcept
	{
		return { m_data.get(),
			static_cast<Eigen::Index>(m_size[0]),
			static_cast<Eigen::Index>(m_numElements / m_size[0]) };
	}

	// index access
	Scalar& operator[](const SizeVector& _index) noexcept { return m_data[flatIndex(_index)]; }
	Scalar operator[](const SizeVector& _index) const noexcept { return m_data[flatIndex(_index)]; }

	// raw access to the underlying memory
	Scalar* data() noexcept { return m_data.get(); }
	const Scalar* data() const noexcept { return m_data.get(); }

	constexpr int order() const noexcept { return Order; }
	const SizeVector& size() const noexcept { return m_size; }
	const std::size_t numElements() const noexcept { return m_numElements; }

	template<int OthOrder>
	bool isSameSize(const Tensor<Scalar, OthOrder>& _oth) const noexcept
	{
		if constexpr(OthOrder != Order) return false;

		for (int i = 0; i < Order; ++i)
			if (m_size[i] != _oth.m_size[i]) return false;

		return true;
	}

	bool operator==(const Tensor& _oth) const noexcept
	{
		if (!isSameSize(_oth)) return false;

		return std::memcmp(m_data.get(), _oth.m_data.get(), m_numElements * sizeof(Scalar)) == 0;
	}

	// ARITHMETIC OPERATORS
	Tensor<Scalar, Order> operator+(const Tensor<Scalar, Order>& _oth) const
	{
		assert(isSameSize(_oth));

		Tensor<Scalar, Order> tensor(m_size);
		for (size_t i = 0; i < m_numElements; ++i)
		{
			tensor.m_data[i] = m_data[i] + _oth.m_data[i];
		}

		return tensor;
	}

	Tensor<Scalar, Order> operator-(const Tensor<Scalar, Order>& _oth) const
	{
		assert(isSameSize(_oth));

		Tensor<Scalar, Order> tensor(m_size);
		for (size_t i = 0; i < m_numElements; ++i)
		{
			tensor.m_data[i] = m_data[i] - _oth.m_data[i];
		}

		return tensor;
	}

	// Frobenius Norm
	Scalar norm() const noexcept
	{
		Scalar s = 0;
		for (size_t i = 0; i < m_numElements; ++i)
			s += m_data[i] * m_data[i];

		return std::sqrt(s);
	}

	size_t flatIndex(const SizeVector& _index) const noexcept
	{
		std::size_t flatInd = _index[0];
		std::size_t dimSize = m_size[0];
		for (std::size_t i = 1; i < _index.size(); ++i)
		{
			flatInd += dimSize * _index[i];
			dimSize *= m_size[i];
		}

		return flatInd;
	}

	SizeVector index(size_t _flatIndex) const noexcept
	{
		SizeVector ind{};
		size_t reminder = _flatIndex;

		for (std::size_t i = 0; i < m_size.size(); ++i)
		{
			ind[i] = reminder % m_size[i];
			reminder /= m_size[i];
		}

		return ind;
	}

	// SERIALIZATION
	template<typename StreamT>
	void save(StreamT& _stream) const
	{
		_stream.write(reinterpret_cast<const char*>(m_size.data()), m_size.size() * sizeof(int));
		_stream.write(reinterpret_cast<const char*>(m_data.get()), 
			m_numElements * sizeof(Scalar));
	}
private:

	void computeOffsets() noexcept
	{
		m_offsets[0] = 1;
		for (std::size_t i = 1; i < m_size.size(); ++i)
		{
			m_offsets[i] = m_offsets[i-1] * m_size[i-1];
		}
	}

	// Compute new indices for a k-flattening from a flatIndex.
	std::pair<size_t, size_t> decomposeFlatIndex(size_t flatIndex, int _k) const noexcept
	{
		SizeVector ind{};
		size_t reminder = flatIndex;

		std::size_t flatInd = 0;
		std::size_t dimSize = 1;
		for (int j = 0; j < _k; ++j)
		{
			ind[j] = reminder % m_size[j];
			reminder /= m_size[j];

			flatInd += dimSize * ind[j];
			dimSize *= m_size[j];
		}

		ind[_k] = reminder % m_size[_k];
		reminder /= m_size[_k];

		flatInd += reminder * dimSize;

		return { ind[_k], flatInd };
	}

	// variant for compile time K
	template<int K>
	std::pair<size_t, size_t> decomposeFlatIndex(size_t flatIndex) const noexcept
	{
		static_assert(K < Order);

		SizeVector ind{};
		size_t reminder = flatIndex;

		std::size_t flatInd = 0;
		std::size_t dimSize = 1;
		for (int j = 0; j < K; ++j)
		{
			ind[j] = reminder % m_size[j];
			reminder /= m_size[j];

			flatInd += dimSize * ind[j];
			dimSize *= m_size[j];
		}

		ind[K] = reminder % m_size[K];
		reminder /= m_size[K];

		flatInd += reminder * dimSize;

		return { ind[K], flatInd };
	}

	SizeVector m_size;
	SizeVector m_offsets; // cumulative sizes
	std::size_t m_numElements;
	std::size_t m_capacity;
	std::unique_ptr<Scalar[]> m_data;
};

namespace details {
	template<int K, typename Scalar, int Order, std::size_t OrderA>
	void multilinearProductImpl(const std::array<Eigen::MatrixX<Scalar>, OrderA>& _matrices,
		Tensor<Scalar, Order>& _tensor,
		bool _transpose)
	{
		static_assert(Order == OrderA);
		{
			Eigen::MatrixX<Scalar> flat = _tensor.template flatten<K>();
			if (_transpose)
				flat = _matrices[K].transpose() * flat;
			else
				flat = _matrices[K] * flat;

			auto sizeVec = _tensor.size();
			sizeVec[K] = static_cast<int>(flat.rows());
			_tensor.resize(sizeVec);
			_tensor.template set<K>(flat);
		}
		if constexpr (K < Order - 1)
			details::multilinearProductImpl<K + 1>(_matrices, _tensor, _transpose);
	}
}

// Multilinear product via k-flattening
// @param _transpose If true, the matrices are multiplied transposed with the tensor.
template<typename Scalar, int Order, std::size_t OrderS>
auto multilinearProduct(const std::array<Eigen::MatrixX<Scalar>, OrderS>& _matrices,
	const Tensor<Scalar, Order>& _tensor,
	bool _transpose = false)
	-> Tensor<Scalar, Order>
{
	static_assert(OrderS == Order);

	auto result = _tensor;

	details::multilinearProductImpl<0>(_matrices, result, _transpose);

	return result;
}

// Multilinear product via Kronecker product
// This method requires massive amounts of memory and should not be used.
template<typename Scalar>
auto multilinearProductKronecker(const std::array<Eigen::MatrixX<Scalar>, 3>& _matrices, 
	const Tensor<Scalar, 3>& _tensor, 
	bool _transpose = false)
	-> Tensor<Scalar, 3>
{
	const Eigen::VectorX<Scalar> core = _transpose ? (kroneckerProduct(_matrices[2].transpose(), kroneckerProduct(_matrices[1].transpose(), _matrices[0].transpose())) * _tensor.vec()).eval()
		: (kroneckerProduct(_matrices[2], kroneckerProduct(_matrices[1], _matrices[0])) * _tensor.vec()).eval();

	typename Tensor<Scalar, 3>::SizeVector sizeVec;
	for (size_t i = 0; i < sizeVec.size(); ++i)
		sizeVec[i] = static_cast<int>(_matrices[i].rows());

	return Tensor<Scalar, 3>(sizeVec, core.data());
}