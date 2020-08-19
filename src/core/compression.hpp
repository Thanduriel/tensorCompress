#pragma once

#include "video/video.hpp"

namespace compression {

	template<typename Scalar>
	class Compressor
	{

	};

	template<typename Scalar>
	class SVDCompressor
	{

	};

	class HOSVDCompressor
	{
	public:
		HOSVDCompressor();

		void encode(const Video& _video);
		Video decode() const;
		void save(const std::string& _fileName);
		void load(const std::string& _fileName);

		void setTargetRank(const std::vector<int>& _targetRank) { m_targetRank = _targetRank; }
	private:
		Video::FrameRate m_frameRate;
		Tensor<float, 4> m_core;
		std::array< Eigen::MatrixX<float>, 4> m_basis;

		std::vector<int> m_targetRank;
	};
}