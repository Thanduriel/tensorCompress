#pragma once

#include "tensor.hpp"

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

	return { basis, multilinearProduct(basis, _tensor, true) };
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
	details::hosvdInterlacedImpl<0>(core, _tol, basis);

	return { basis, std::move(core) };
}

namespace details {
	template<int K, typename Scalar, int Dims>
	void hosvdInterlacedImpl(Tensor<Scalar, Dims>& _tensor, Scalar _tol,
		std::array<Eigen::MatrixX<Scalar>, Dims>& _basis)
	{
		using namespace Eigen;

		const MatrixX<Scalar> m = _tensor.flatten<K>();
		JacobiSVD< MatrixX<Scalar>> svd(m, ComputeThinU);
		if (_tol > 0.f)
		{
			const auto oldRank = svd.rank();
			svd.setThreshold(_tol);
			std::cout << "truncating " << oldRank << " -> " << svd.rank() << std::endl;
		}
		_basis[K] = svd.matrixU().leftCols(svd.rank());
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
			hosvdInterlacedImpl<K + 1>(_tensor, _tol, _basis);
	}
}