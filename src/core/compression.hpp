#pragma once

#include "video/video.hpp"
#include "truncation.hpp"

namespace compression {

	class HOSVDCompressor
	{
	public:
		HOSVDCompressor();

		void encode(const Video& _video);
		Video decode() const;
		void save(const std::string& _fileName);
		void load(const std::string& _fileName);
		
		template<typename Truncate>
		void setTruncation(Truncate _truncation)
		{
			m_truncation.reset(new truncation::TruncationAdaptor<Truncate>(_truncation));
		}
	private:
		Video::FrameRate m_frameRate;
		Tensor<float, 4> m_core;
		std::array< Eigen::MatrixX<float>, 4> m_basis;

		std::unique_ptr<truncation::AbstractTruncation> m_truncation;
	};
}