#pragma once

#include "tensor.hpp"
#include "truncation.hpp"
#include <iostream>
#include <variant>
#include <limits>

// Higher order singular value decomposition.
// @param _tensor The tensor to decompose.
// @param _tol Singular values smaller than this tolerance are truncated.
// @return <array of U matrices, core tensor C> such that (U1, ..., Ud) * C == _tensor.
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

namespace details {
	template<int K, typename Scalar, int Dims, std::size_t DimsA, typename Truncate>
	void hosvdInterlacedImpl(Tensor<Scalar, Dims>& _tensor,
		const Truncate& _truncate,
		std::array<Eigen::MatrixX<Scalar>, DimsA>& _basis,
		Eigen::BDCSVD< Eigen::MatrixX<Scalar>>& _svd)
	{
		using namespace Eigen;

		// extra scope to enforce release of resources before the recursive call
		{
			const MatrixX<Scalar> m = _tensor.template flatten<K>();
			_svd.compute(m, ComputeThinU);

			const Index newRank = std::min(_truncate(_svd.singularValues(), K), _svd.rank());
			_basis[K] = _svd.matrixU().leftCols(newRank);
			const MatrixX<Scalar> flatNext = _basis[K].transpose() * m;

			// shrink tensor if truncation took place
			auto size = _tensor.size();
			if (flatNext.rows() < size[K])
			{
				size[K] = static_cast<int>(flatNext.rows());
				_tensor.resize(size);
			}

			_tensor.template set<K>(flatNext);
		}

		if constexpr (K < Dims - 1)
			hosvdInterlacedImpl<K + 1>(_tensor, _truncate, _basis, _svd);
	}
}

// Interlaced higher order singular value decomposition.
// See hosvd for a description of the parameters.
// This method is significantly faster than hosvd if the numeric rank of the input is
// low or truncation due to a high tolerance takes place.
template<typename Scalar, int Dims, typename Truncate = truncation::Zero>
auto hosvdInterlaced(const Tensor<Scalar, Dims>& _tensor, 
	const Truncate& _truncate = truncation::Zero())
-> std::tuple < std::array<Eigen::MatrixX<Scalar>, Dims>, Tensor<Scalar, Dims>>
{
	using namespace Eigen;

	std::array<MatrixX<Scalar>, Dims> basis;

	Tensor<Scalar, Dims> core = _tensor;
	BDCSVD< MatrixX<Scalar>> svd; //JacobiSVD, BDCSVD
	details::hosvdInterlacedImpl<0>(core, _truncate, basis, svd);

	return { std::move(basis), std::move(core) };
}