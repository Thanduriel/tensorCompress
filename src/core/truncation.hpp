#pragma once

namespace truncation {

	struct AbstractTruncation
	{
		virtual ~AbstractTruncation() = default;

		virtual Eigen::Index operator() (const Eigen::VectorX<float>& _singularValues, int _k) const = 0;
	//	virtual Eigen::Index operator() (const Eigen::VectorX<double>& _singularValues, int _k) const = 0;
	};

	template<typename TruncateImpl>
	struct TruncationAdaptor : public AbstractTruncation, TruncateImpl
	{
		TruncationAdaptor(const TruncateImpl& _impl)
			: TruncateImpl(_impl)
		{}

		Eigen::Index operator() (const Eigen::VectorX<float>& _singularValues, int _k) const noexcept override
		{
			return TruncateImpl::operator()(_singularValues, _k);
		}

	/*	Eigen::Index operator() (const Eigen::VectorX<double>& _singularValues, int _k) const noexcept override
		{
			return TruncateImpl::operator()(_singularValues, _k);
		}*/
	};

	struct Zero
	{
		template<typename Scalar>
		Eigen::Index operator() (const Eigen::VectorX<Scalar>&, int) const noexcept
		{
			return std::numeric_limits<Eigen::Index>::max();
		}
	};

	template<typename Scalar>
	struct Tolerance
	{
		explicit Tolerance(Scalar _tol) : tolerance(1, _tol) { }
		explicit Tolerance(const std::initializer_list<Scalar>& _tol) : tolerance(_tol) {}
		explicit Tolerance(const std::vector<Scalar>& _tol) : tolerance(_tol) {}

		std::vector<Scalar> tolerance;

		Eigen::Index operator() (const Eigen::VectorX<Scalar>& _singularValues, int _k) const noexcept
		{
			_k = std::min(static_cast<int>(tolerance.size()) - 1, _k);

			Eigen::Index rank = 0;

			for (Eigen::Index i = 0; i < _singularValues.size()
				&& _singularValues[i] >= tolerance[_k]; ++i)
			{
				++rank;
			}
			return rank;
		}
	};

	template<typename Scalar>
	struct ToleranceSum
	{
		explicit ToleranceSum(Scalar _tol) : tolerance(1, _tol) { }
		explicit ToleranceSum(const std::initializer_list<Scalar>& _tol) : tolerance(_tol) {}
		explicit ToleranceSum(const std::vector<Scalar>& _tol) : tolerance(_tol) {}

		std::vector<Scalar> tolerance;

		Eigen::Index operator() (const Eigen::VectorX<Scalar>& _singularValues, int _k) const noexcept
		{
			_k = std::min(static_cast<int>(tolerance.size()) - 1, _k);

			Eigen::Index rank = _singularValues.size();
			Scalar sum = 0;

			for (Eigen::Index i = _singularValues.size() - 1; i > 0; --i)
			{
				sum += _singularValues[i];
				if (sum > tolerance[_k])
					break;
				--rank;
			}
			return rank;
		}
	};

	struct Rank
	{
		explicit Rank(int _rank) : rank(1, _rank) {}
		explicit Rank(const std::initializer_list<int>& _rank) : rank(_rank) {}
		explicit Rank(const std::vector<int>& _rank) : rank(_rank) {}
		template<int Order>
		Rank(const std::array<int, Order>& _rank)
		{
			rank.reserve(Order);
			for (int i : _rank) rank.push_back(i);
		}

		std::vector<int> rank;

		template<typename Scalar>
		Eigen::Index operator() (const Eigen::VectorX<Scalar>& _singularValues, int _k) const noexcept
		{
			return static_cast<size_t>(_k) < rank.size() ? rank[_k] : rank.back();
		}
	};
}