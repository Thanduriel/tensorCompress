#include "frameconverter.hpp"
#include <iostream>
#include <array>

FrameConverter::FrameConverter(int _width, int _height, AVPixelFormat _srcFormat, AVPixelFormat _dstFormat)
	: m_sourceFrame(av_frame_alloc()),
	m_destinationFrame(av_frame_alloc()),
	m_swsContext(AVCALLRET(sws_getContext,
		_width, _height,
		_srcFormat,
		_width, _height, _dstFormat, SWS_BICUBIC,
		NULL, NULL, NULL))
{
	m_sourceFrame->format = _srcFormat;
	m_sourceFrame->width = _width;
	m_sourceFrame->height = _height;
	if(_srcFormat == AVPixelFormat::AV_PIX_FMT_RGB24)
		m_sourceFrame->linesize[0] = 3 * _width;
	else if (_srcFormat == AVPixelFormat::AV_PIX_FMT_YUV444P)
	{
		m_sourceFrame->linesize[0] = _width;
		m_sourceFrame->linesize[1] = _width;
		m_sourceFrame->linesize[2] = _width;
	}
	else if (_srcFormat == AVPixelFormat::AV_PIX_FMT_YUV420P)
	{
		m_sourceFrame->linesize[0] = _width;
		m_sourceFrame->linesize[1] = _width/2;
		m_sourceFrame->linesize[2] = _width/2;
	}
	av_frame_get_buffer(m_sourceFrame.get(), 0);

	m_destinationFrame->format = _dstFormat;
	m_destinationFrame->width = _width;
	m_destinationFrame->height = _height;
	av_frame_get_buffer(m_destinationFrame.get(), 0);
}

void FrameConverter::convert()
{
	sws_scale(m_swsContext.get(), m_sourceFrame->data, m_sourceFrame->linesize, 0, m_sourceFrame->height,
		m_destinationFrame->data, m_destinationFrame->linesize);
}