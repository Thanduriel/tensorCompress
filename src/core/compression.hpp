#pragma once

#include "video/video.hpp"
#include "truncation.hpp"

namespace compression {

	template<typename PixelFormat>
	class HOSVDCompressor
	{
	public:
		using TensorType = typename PixelFormat::TensorType;

		explicit HOSVDCompressor(const PixelFormat& _space);

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

		const std::vector<TensorType>& singularValues() const { return m_core; }
		const std::vector<std::array< Eigen::MatrixX<float>, 4>> basis() const { return m_basis; }
	private:
		PixelFormat m_pixelFormat;
		size_t m_numFramesPerBlock;
		Video::FrameRate m_frameRate;
		std::vector<TensorType> m_core;
		std::vector<std::array< Eigen::MatrixX<float>, 4>> m_basis;

		std::unique_ptr<truncation::AbstractTruncation> m_truncation;
	};
}