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
		: m_size(_size), m_numElements(1)
	{
		for (auto d : m_size)
			m_numElements *= d;
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
		m_numElements(_oth.m_numElements),
		m_data(std::make_unique<Scalar[]>(m_numElements))
	{
		std::copy(_oth.m_data.get(), _oth.m_data.get() + m_numElements, m_data.get());
	}

	Tensor(Tensor&& _oth) noexcept
		: m_size(_oth.m_size),
		m_numElements(_oth.m_numElements),
		m_data(std::move(_oth.m_data))
	{}

	// set from a k-flattening
	void set(const Eigen::MatrixX<Scalar>& _flatTensor, int _k)
	{
		assert(_flatTensor.rows() == m_size[_k]);
		assert(_flatTensor.rows() * _flatTensor.cols() == m_numElements);

		const size_t othDim = m_numElements / m_size[_k];

		for (size_t i = 0; i < m_numElements; ++i)
		{
			const SizeVector ind = index(i);
			m_data[i] = _flatTensor(ind[_k], flatIndex(ind, _k));
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
			const SizeVector ind = index(i);
			m(ind[_k], flatIndex(ind, _k)) = m_data[i];
		}

		return m;
	}

	// Change the size of this tensor to _newSize.
	// The data is unspecified afterwards.
	// @param _shrink Shrink the buffer if the new size is smaller.
	void resize(const SizeVector& _newSize, bool _shrink = false)
	{
		m_size = _newSize;
		std::size_t oldNum = m_numElements;
		m_numElements = 1;

		for (auto d : m_size)
			m_numElements *= d;

		if(oldNum < m_numElements || (_shrink && oldNum > m_numElements))
			m_data = std::make_unique<Scalar[]>(m_numElements);
	}

	// ACCESS OPERATIONS

	Eigen::Map<const Eigen::VectorX<Scalar>> vec() const
	{
		return { m_data.get(), static_cast<Eigen::Index>(m_numElements) };
	}

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

	std::size_t m_numElements;
	std::array<int, NumDimensions> m_size;
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

// higher order svd
// @param _tol Singular values smaller then this tolerance are truncated.
// @return <array of U matrices, core tensor> so that (U1, ..., Ud) * C == _tensor
template<typename Scalar, int Dims>
auto hosvd(const Tensor<Scalar, Dims>& _tensor, Scalar _tol = 0.f) 
	-> std::tuple < std::array<Eigen::MatrixX<Scalar>, Dims>, Tensor<Scalar, Dims>>
{
	using namespace Eigen;

	std::array<MatrixX<Scalar>, Dims> basis;

	for (int k = 0; k < Dims; ++k)
	{
		const MatrixX<Scalar> m = _tensor.flatten(k);
		BDCSVD< MatrixX<Scalar>> svd(m, ComputeThinU);
		if (_tol > 0.f)
		{
			const auto oldRank = svd.rank();
			svd.setThreshold(_tol);
			std::cout << "truncating " << oldRank << " -> " << svd.rank() << std::endl;
		}
		basis[k] = svd.matrixU().leftCols(svd.rank());
	}

	return { basis, multilinearProduct(basis, _tensor, true)};
}

// higher order svd
// See hosvd for params.
// This method is significantly faster then hosvd if the numeric rank of the input is
// low or truncation due to a high tolerance takes place.
template<typename Scalar, int Dims>
auto hosvdInterlaced(const Tensor<Scalar, Dims>& _tensor, Scalar _tol = 0.f)
-> std::tuple < std::array<Eigen::MatrixX<Scalar>, Dims>, Tensor<Scalar, Dims>>
{
	using namespace Eigen;

	std::array<MatrixX<Scalar>, Dims> basis;

	Tensor<Scalar, Dims> core = _tensor;

	for (int k = 0; k < Dims; ++k)
	{
		const MatrixX<Scalar> m = core.flatten(k);
		JacobiSVD< MatrixX<Scalar>> svd(m, ComputeThinU);
		if (_tol > 0.f)
		{
			const auto oldRank = svd.rank();
			svd.setThreshold(_tol);
			std::cout << "truncating " << oldRank << " -> " << svd.rank() << std::endl;
		}
		basis[k] = svd.matrixU().leftCols(svd.rank());
		const MatrixX<Scalar> flatNext = basis[k].transpose() * m;

		// shrink tensor if truncation took place
		auto size = core.size();
		if (flatNext.rows() < size[k])
		{
			size[k] = static_cast<int>(flatNext.rows());
			core.resize(size);
		}
		
		core.set(flatNext, k);
	}

	return { basis, std::move(core) };
}

