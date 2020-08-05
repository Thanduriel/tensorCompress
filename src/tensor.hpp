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

	Tensor(const Tensor& _oth)
		: m_size(_oth.m_size), 
		m_numElements(_oth.m_numElements),
		m_data(std::make_unique<Scalar[]>(m_numElements))
	{
		std::copy(_oth.m_data.get(), _oth.m_data.get() + m_numElements, m_data.get());
	}

	Tensor(Tensor&& _oth)
		: m_size(_oth.m_size),
		m_numElements(_oth.m_numElements),
		m_data(std::move(_oth.m_data))
	{}

	// set from a k-flattening
	void set(const Eigen::MatrixX<Scalar>& _flatTensor, int _k)
	{
		assert(_flatTensor.rows() == m_size[_k]);

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

	Eigen::Map<const Eigen::VectorX<Scalar>> vec() const
	{
		return { m_data.get(), static_cast<Eigen::Index>(m_numElements) };
	}

	Scalar& operator[](const SizeVector& _index) { return m_data[flatIndex(_index)]; }
	Scalar operator[](const SizeVector& _index) const { return m_data[flatIndex(_index)]; }

	// raw access to the underlying memory
	Scalar* data() { return m_data.get(); }

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
		SizeVector ind;
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
			std::cout << "old rank: " << svd.rank() << std::endl;
			svd.setThreshold(_tol);
			std::cout << "new rank" << svd.rank() << std::endl;
		}
		basis[k] = svd.matrixU().leftCols(svd.rank());
	}

	return { basis, multilinearProduct(basis, _tensor, true)};
}

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
		BDCSVD< MatrixX<Scalar>> svd(m, ComputeThinU | ComputeThinV);
		if (_tol > 0.f)
		{
			std::cout << "old rank: " << svd.rank() << std::endl;
			svd.setThreshold(_tol);
			std::cout << "new rank" << svd.rank() << std::endl;
		}
		basis[k] = svd.matrixU().leftCols(svd.rank());
		const MatrixX<Scalar> flatNext = basis[k].transpose() * m;
		const MatrixX<Scalar> flatNextDif = flatNext - svd.singularValues().asDiagonal() * svd.matrixV().transpose();
		std::cout << svd.singularValues() << "\n";
		std::cout << flatNextDif.norm() << "\n";
		const float no1 = flatNext.norm();
		auto temp = core;
		core.set(flatNext, k);
		float tnorm = (temp - core).norm();
	//	const MatrixX<Scalar> m2 = core.flatten(k+1);
	//	const float no2 = (m - m2).norm();
		int uiae = 12;
	}

	return { basis, std::move(core) };
}

