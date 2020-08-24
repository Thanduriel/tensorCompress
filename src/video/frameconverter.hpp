#pragma once

#include "ffmpegutils.hpp"
#include <memory>

class FrameConverter
{
public:
	FrameConverter(int _width, int _height, AVPixelFormat _srcFormat, AVPixelFormat _dstFormat);

	AVFrame& getSrcFrame() { return *m_sourceFrame; }
	AVFrame& getDstFrame() { return *m_destinationFrame; }
	void convert();
private:
	std::unique_ptr<AVFrame> m_sourceFrame;
	std::unique_ptr<AVFrame> m_destinationFrame;
	std::unique_ptr<SwsContext> m_swsContext;
};