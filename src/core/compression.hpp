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

		void setFramesPerBlock(int _numFrames) { m_numFramesPerBlock = _numFrames; }
	private:
		size_t m_numFramesPerBlock;
		Video::FrameRate m_frameRate;
		std::vector<Tensor<float, 4>> m_core;
		std::vector<std::array< Eigen::MatrixX<float>, 4>> m_basis;

		std::unique_ptr<truncation::AbstractTruncation> m_truncation;
	};
}