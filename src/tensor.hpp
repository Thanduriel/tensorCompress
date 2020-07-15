#pragma once

#include <Eigen/Eigen>
#include <unsupported/Eigen/KroneckerProduct>
#include <array>
#include <memory>
#include <tuple>
/*
template<typename Scalar, int Rows, int Columns, int... Dimensions>
class Tensor
{};
*/

template<typename Scalar, int NumDimensions>
class Tensor
{
public:
	using SizeVector = std::array<int, NumDimensions>;

	Tensor(const SizeVector& _size, const Scalar* _data = nullptr) 
		: m_size(_size), m_numElements(1)
	{
		for (auto d : m_size)
			m_numElements *= d;
		m_data = std::make_unique<Scalar[]>(m_numElements);

		if (_data)
			std::copy(_data, _data + m_numElements, m_data.get());
	}

/*	Tensor(Tensor&& _oth)
		: m_size(_oth.m_size),
		m_numElements(_oth.m_numElements),
		m_data(std::move(_oth.m_data))
	{}*/

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
		
		for(size_t k = 0; k < m_size[_k]; ++k)
			for (size_t i = 0; i < othDim; ++i)
			{
				m(k, i) = m_data[k*othDim + i];
			}

		return m;
	}

	Eigen::Map<Eigen::VectorX<Scalar>> vec() const
	{
		return { m_data.get(), static_cast<Eigen::Index>(m_numElements) };
	}

	Scalar& operator[](const SizeVector& _index)
	{
		return m_data[flatIndex(_index)];
	}

	Scalar operator[](const SizeVector& _index) const
	{
		return m_data[flatIndex(_index)];
	}

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
	// basic arithmetic operators
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
private:
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

	std::size_t m_numElements;
	std::array<int, NumDimensions> m_size;
	std::unique_ptr<Scalar[]> m_data;
};

// multilinear product
template<typename Scalar, int Dims>
auto multilinearProduct(const std::array<Eigen::MatrixX<Scalar>, Dims>& _matrices, 
	const Tensor<Scalar, Dims>& _tensor, 
	bool _transpose = false)
	-> Tensor<Scalar, Dims>
{
	const Eigen::VectorX<Scalar> core = _transpose ? (kroneckerProduct(_matrices[2].transpose(), kroneckerProduct(_matrices[1].transpose(), _matrices[0].transpose())) * _tensor.vec()).eval()
		: (kroneckerProduct(_matrices[2], kroneckerProduct(_matrices[1], _matrices[0])) * _tensor.vec()).eval();

	return Tensor<Scalar, Dims>(_tensor.size(), core.data());
}

// higher order svd
// @return <array of U matrices, core tensor>
template<typename Scalar, int Dims>
auto hosvd(const Tensor<Scalar, Dims>& _tensor) 
	-> std::tuple < std::array<Eigen::MatrixX<Scalar>, Dims>, Tensor<Scalar, Dims>>
{
	using namespace Eigen;

	std::array<MatrixX<Scalar>, Dims> basis;

	for (int k = 0; k < Dims; ++k)
	{
		const MatrixX<Scalar> m = _tensor.flatten(k);
		BDCSVD< MatrixX<Scalar>> svd(m, ComputeThinU);
		basis[k] = svd.matrixU();
	}

	return { basis, multilinearProduct(basis, _tensor, true)};
}

/*
template<typename Scalar, int M, int N, int P, int Q>
auto kronecker(const Eigen::Matrix<Scalar,M,N>& _mat1, const Eigen::Matrix<Scalar, P, Q>& _mat2)
	-> Eigen::Matrix<Scalar, P * M, Q * N>
{
	Eigen::Matrix<Scalar, P* M, Q* N> result;

	for (int j = 0; j < Q * N; ++j)
		for (int i = 0; i < P * M; ++i)
		{
			result(i,j) = _mat1
		}
}*/