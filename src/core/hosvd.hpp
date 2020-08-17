#pragma once

#include "tensor.hpp"
#include <iostream>
#include <variant>
#include <limits>

namespace truncation {

	struct Zero
	{
		Eigen::Index operator() (const Eigen::VectorX<float>&, int) const 
		{ 
			return std::numeric_limits<Eigen::Index>::max();
		}
	};

	template<typename Scalar>
	struct Tolerance
	{
		Tolerance(Scalar _tol) : tolerance(_tol) {}

		float tolerance;

		Eigen::Index operator() (const Eigen::VectorX<float>& _singularValues, int) const
		{
			Eigen::Index rank = 0;

			for (Eigen::Index i = 0; i < _singularValues.size() && _singularValues[i] >= tolerance; ++i)
			{
				++rank;
			}
			return rank;
		}
	};

	template<int Order>
	struct Rank
	{
		Rank(const std::array<int, Order>& _rank) : rank(_rank) {}
		std::array<int, Order> rank;

		Eigen::Index operator() (const Eigen::VectorX<float>& _singularValues, int _k) const
		{
			return rank[_k];
		}
	};

}

// higher order svd
// @param _tol Singular values smaller then this tolerance are truncated.
// @return <array of U matrices, core tensor C> such that (U1, ..., Ud) * C == _tensor
template<typename Scalar, int Dims, typename Truncate = truncation::Zero>
auto hosvd(const Tensor<Scalar, Dims>& _tensor, Truncate _truncate = truncation::Zero())
-> std::tuple < std::array<Eigen::MatrixX<Scalar>, Dims>, Tensor<Scalar, Dims>>
{
	using namespace Eigen;

	std::array<MatrixX<Scalar>, Dims> basis;

	for (int k = 0; k < Dims; ++k)
	{
		const MatrixX<Scalar> m = _tensor.flatten(k);
		BDCSVD< MatrixX<Scalar>> svd(m, ComputeThinU);

		const Index newRank = std::min(_truncate(svd.singularValues(), k), svd.rank());
		basis[k] = svd.matrixU().leftCols(newRank);
	}

	return { basis, multilinearProduct(basis, _tensor, true) };
}

// higher order svd
// See hosvd for params.
// This method is significantly faster then hosvd if the numeric rank of the input is
// low or truncation due to a high tolerance takes place.
template<typename Scalar, int Dims, typename Truncate = truncation::Zero>
auto hosvdInterlaced(const Tensor<Scalar, Dims>& _tensor, 
	Truncate _truncate = truncation::Zero())
-> std::tuple < std::array<Eigen::MatrixX<Scalar>, Dims>, Tensor<Scalar, Dims>>
{
	using namespace Eigen;

	std::array<MatrixX<Scalar>, Dims> basis;

	Tensor<Scalar, Dims> core = _tensor;
	details::hosvdInterlacedImpl<0>(core, _truncate, basis);

	return { basis, std::move(core) };
}

namespace details {
	template<int K, typename Scalar, int Dims, typename Truncate>
	void hosvdInterlacedImpl(Tensor<Scalar, Dims>& _tensor, Truncate _truncate,
		std::array<Eigen::MatrixX<Scalar>, Dims>& _basis)
	{
		using namespace Eigen;

		const MatrixX<Scalar> m = _tensor.flatten<K>();
		BDCSVD< MatrixX<Scalar>> svd(m, ComputeThinU); //JacobiSVD, BDCSVD

		const Index newRank = std::min(_truncate(svd.singularValues(), K), svd.rank());
		_basis[K] = svd.matrixU().leftCols(newRank);
		const MatrixX<Scalar> flatNext = _basis[K].transpose() * m;

		// shrink tensor if truncation took place
		auto size = _tensor.size();
		if (flatNext.rows() < size[K])
		{
			size[K] = static_cast<int>(flatNext.rows());
			_tensor.resize(size);
		}

		_tensor.set<K>(flatNext);

		if constexpr (K < Dims - 1)
			hosvdInterlacedImpl<K + 1>(_tensor, _truncate, _basis);
	}
}