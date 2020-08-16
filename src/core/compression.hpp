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
		void encode(const Video& _video);
		Video decode();
		void save(const std::string& _fileName);
		void load(const std::string& _fileName);

	private:
		Tensor<float, 4> m_core;
		std::array< Eigen::MatrixX<float>, 4> m_basis;
	};
}