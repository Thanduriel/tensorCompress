#pragma once

#include <Eigen/Eigen>
#include <unsupported/Eigen/KroneckerProduct>
#include <array>
#include <memory>
#include <tuple>

template<typename Scalar, int NumDimensions>
class Tensor
{
public:
	using SizeVector = std::array<int, NumDimensions>;

	Tensor(const SizeVector& _size) 
		: m_size(_size)
	{
		computeOffsets();
		m_numElements = m_offsets.back() * m_size.back();

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
		m_data(std::make_unique<Scalar[]>(m_numElements))
	{
		std::copy(_oth.m_data.get(), _oth.m_data.get() + m_numElements, m_data.get());
	}

	Tensor(Tensor&& _oth) noexcept
		: m_size(_oth.m_size),
		m_offsets(_oth.m_offsets),
		m_numElements(_oth.m_numElements),
		m_data(std::move(_oth.m_data))
	{}

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
	void set(const Eigen::MatrixX<Scalar>& _flatTensor)
	{
		assert(_flatTensor.rows() == m_size[K]);
		assert(_flatTensor.rows() * _flatTensor.cols() == m_numElements);

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
			m_data[i] = _generator(SizeVector{});
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

		for (size_t i = 0; i < m_numElements; ++i)
		{
			const auto& [indK, indOth] = decomposeFlatIndex<K>(i);
			m(indK, indOth) = m_data[i];
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
		m_numElements = m_offsets.back() * m_size.back();

		if(oldNum < m_numElements || (_shrink && oldNum > m_numElements))
			m_data = std::make_unique<Scalar[]>(m_numElements);
	}

	// ACCESS OPERATIONS

	// vectorization
	Eigen::Map<const Eigen::VectorX<Scalar>> vec() const
	{
		return { m_data.get(), static_cast<Eigen::Index>(m_numElements) };
	}

	// index access
	Scalar& operator[](const SizeVector& _index) { return m_data[flatIndex(_index)]; }
	Scalar operator[](const SizeVector& _index) const { return m_data[flatIndex(_index)]; }

	// raw access to the underlying memory
	Scalar* data() { return m_data.get(); }
	const Scalar* data() const { return m_data.get(); }

	const SizeVector& size() const { return m_size; }
	const std::size_t numElements() const { return m_numElements; }


	template<int OthDim>
	bool isSameSize(const Tensor<Scalar, OthDim>& _oth) const
	{
		if constexpr(OthDim != NumDimensions) return false;

		for (int i = 0; i < NumDimensions; ++i)
			if (m_size[i] != _oth.m_size[i]) return false;

		return true;
	}

	// ARITHMETIC OPERATORS
	Tensor<Scalar, NumDimensions> operator-(const Tensor<Scalar, NumDimensions>& _oth) const
	{
		assert(isSameSize(_oth));

		Tensor<Scalar, NumDimensions> tensor(m_size);
		for (size_t i = 0; i < m_numElements; ++i)
		{
			Scalar a = m_data[i];
			Scalar b = _oth.m_data[i];
			Scalar f = m_data[i] - _oth.m_data[i];
			tensor.m_data[i] = m_data[i] - _oth.m_data[i];
		}

		return tensor;
	}

	// Frobenius Norm
	Scalar norm() const
	{
		Scalar s = 0;
		for (size_t i = 0; i < m_numElements; ++i)
			s += m_data[i] * m_data[i];

		return std::sqrt(s);
	}

	size_t flatIndex(const SizeVector& _index) const
	{
		std::size_t flatInd = _index[0];
		std::size_t dimSize = m_size[0];
		for (std::size_t i = 1; i < _index.size(); ++i)
		{
			flatInd += dimSize * _index[i];
			dimSize *= m_size[i];
		}

		return flatInd;
	/*	std::size_t flatInd = _index[0];
		for (std::size_t i = 1; i < _index.size(); ++i)
		{
			flatInd += m_offsets[i] * _index[i];
		}

		return flatInd;*/
	}

	SizeVector index(size_t _flatIndex) const
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
private:

	size_t flatIndex(const SizeVector& _index, size_t _skip) const
	{
		std::size_t flatInd = 0;
		std::size_t dimSize = 1;
		for (std::size_t i = 0; i < _index.size(); ++i)
		{
			if (i != _skip)
			{
				flatInd += dimSize * _index[i];
				dimSize *= m_size[i];
			}
		}

		return flatInd;
/*		std::size_t flatInd = 0;
		std::size_t flatInd2 = 0;
		
		for (std::size_t i = 0; i < _skip; ++i)
			flatInd += m_offsets[i] * _index[i];
		for (std::size_t i = _skip+1; i < _index.size(); ++i)
			flatInd2 += m_offsets[i] * _index[i];

		return flatInd + flatInd2 / m_size[_skip];*/
	}

	void computeOffsets()
	{
		m_offsets[0] = 1;
		for (std::size_t i = 1; i < m_size.size(); ++i)
		{
			m_offsets[i] = m_offsets[i-1] * m_size[i-1];
		}
	}

	// Compute new indicies for a k-flattening from a flatIndex.
	std::pair<size_t, size_t> decomposeFlatIndex(size_t flatIndex, int _k) const
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
	std::pair<size_t, size_t> decomposeFlatIndex(size_t flatIndex) const
	{
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

	std::size_t m_numElements;
	SizeVector m_size;
	SizeVector m_offsets; // cumulative sizes
	std::unique_ptr<Scalar[]> m_data;
};

// multilinear product via flattenings
// @param _transpose If true the matrices are multiplied transposed with the tensor
template<typename Scalar, int Order>
auto multilinearProduct(const std::array<Eigen::MatrixX<Scalar>, Order>& _matrices,
	const Tensor<Scalar, Order>& _tensor,
	bool _transpose = false)
	-> Tensor<Scalar, Order>
{
	auto result = _tensor;

	for (int k = 0; k < Order; ++k)
	{
		Eigen::MatrixX<Scalar> flat = result.flatten(k);
		if (_transpose)
			flat = _matrices[k].transpose() * flat;
		else
			flat = _matrices[k] * flat;

		auto sizeVec = result.size();
		sizeVec[k] = static_cast<int>(flat.rows());
		result.resize(sizeVec);
		result.set(flat, k);
	}

	return result;
}

// multilinear product via kronecker product
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
