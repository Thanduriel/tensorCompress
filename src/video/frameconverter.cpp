#include "frameconverter.hpp"
#include <iostream>

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
	av_frame_get_buffer(m_sourceFrame.get(), 32);
	
	m_destinationFrame->format = _dstFormat;
	m_destinationFrame->width = _width;
	m_destinationFrame->height = _height;
	av_frame_get_buffer(m_destinationFrame.get(), 32);
}

void FrameConverter::convert()
{
	sws_scale(m_swsContext.get(), m_sourceFrame->data, m_sourceFrame->linesize, 0, m_sourceFrame->height,
		m_destinationFrame->data, m_destinationFrame->linesize);
}