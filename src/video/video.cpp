#include "video.hpp"
#include "ffmpegutils.hpp"
#include "frameconverter.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>
#include <fstream>
#include <iostream>

struct AVInit
{
	AVInit()
	{
		avcodec_register_all();
	}
};

static AVInit init;



Video::Video(const std::string& _fileName)
	: m_width(0), m_height(0)
{
	decode("file:" + _fileName);
}

Video::Video(const FrameTensor& _tensor, FrameRate _frameRate)
	: m_width(_tensor.size()[0]),
	m_height(_tensor.size()[1]),
	m_frameSize(_tensor.size()[0]* _tensor.size()[1]*3),
	m_frameRate(_frameRate)
{
	FrameTensor::SizeVector sizeVector{};
	sizeVector.back() = 1;
	const size_t frameOffset = _tensor.flatIndex(sizeVector);
	for (int i = 0; i < _tensor.size().back(); ++i)
	{
		m_frames.emplace_back(new unsigned char[m_frameSize] {});
		const float* begin = _tensor.data() + i * frameOffset;
		for (int j = 0; j < m_frameSize; j+=3)
		{
			m_frames.back()[j] = static_cast<unsigned char>(std::clamp(*begin,0.f,1.f) * 255.f);
			++begin;
		}
	}
}

Tensor<float, 3> Video::SingleChannel::operator()(const Video& _video, 
	int _firstFrame, int _numFrames) const
{
	TensorType tensor({ _video.m_width, _video.m_height, _numFrames });

	float* ptr = tensor.data();
	int count = 0;
	for (int i = _firstFrame; i < _firstFrame+_numFrames; ++i)
	{
		for (int j = static_cast<int>(channel); j < _video.m_frameSize; j += 3)
		{
			*ptr = static_cast<float>(_video.m_frames[i][j]) / 255.f;
			++ptr;
			++count;
		}
	}

	return tensor;
}

Tensor<float, 4> Video::RGB::operator()(const Video& _video,
	int _firstFrame, int _numFrames) const
{
	TensorType tensor({ 3, _video.m_width, _video.m_height, _numFrames });

	float* ptr = tensor.data();
	int count = 0;
	for (int i = _firstFrame; i < _firstFrame + _numFrames; ++i)
	{
		for (int j = 0; j < _video.m_frameSize; ++j)
		{
			*ptr = static_cast<float>(_video.m_frames[i][j]) / 255.f;
			++ptr;
			++count;
		}
	}

	return tensor;
}

void Video::RGB::operator()(const TensorType& _tensor, Video& _video) const
{
	Tensor<float, 4>::SizeVector sizeVector{};
	sizeVector.back() = 1;
	const size_t frameOffset = _tensor.flatIndex(sizeVector);
	for (int i = 0; i < _tensor.size().back(); ++i)
	{
		_video.m_frames.emplace_back(new unsigned char[_video.m_frameSize] {});
		const float* begin = _tensor.data() + i * frameOffset;
		for (int j = 0; j < _video.m_frameSize; ++j)
		{
			_video.m_frames.back()[j] = static_cast<unsigned char>(std::clamp(*begin, 0.f, 1.f) * 255.f);
			++begin;
		}
	}
}

Tensor<float, 4> Video::YUV444::operator()(const Video& _video,
	int _firstFrame, int _numFrames) const
{
	TensorType tensor({ 3, _video.m_width, _video.m_height, _numFrames });

	FrameConverter converter(_video.m_width, _video.m_height,
		AVPixelFormat::AV_PIX_FMT_RGB24, AVPixelFormat::AV_PIX_FMT_YUV444P);

	float* ptr = tensor.data();
	for (int i = _firstFrame; i < _firstFrame + _numFrames; ++i)
	{
		const unsigned char* begin = _video.m_frames[i].get();
		std::copy(begin, begin + _video.m_frameSize, converter.getSrcFrame().data[0]);
		converter.convert();
		for (int j = 0; j < _video.m_width*_video.m_height; ++j)
		{
			*ptr++ = static_cast<float>(converter.getDstFrame().data[0][j]) / 255.f;
			*ptr++ = static_cast<float>(converter.getDstFrame().data[1][j]) / 255.f;
			*ptr++ = static_cast<float>(converter.getDstFrame().data[2][j]) / 255.f;
		}
	}

	return tensor;
}

void Video::YUV444::operator()(const TensorType& _tensor, Video& _video) const
{
	FrameConverter converter(_video.m_width, _video.m_height,
		AVPixelFormat::AV_PIX_FMT_YUV444P, AVPixelFormat::AV_PIX_FMT_RGB24);

	Tensor<float, 4>::SizeVector sizeVector{};
	sizeVector.back() = 1;
	const size_t frameOffset = _tensor.flatIndex(sizeVector);
	for (int i = 0; i < _tensor.size().back(); ++i)
	{
		const float* current = _tensor.data() + i * frameOffset;
		for (int j = 0; j < _video.m_frameSize/3; ++j)
		{
			AVFrame& frame = converter.getSrcFrame();
			frame.data[0][j] = static_cast<unsigned char>(std::clamp(*current++, 0.f, 1.f) * 255.f);
			frame.data[1][j] = static_cast<unsigned char>(std::clamp(*current++, 0.f, 1.f) * 255.f);
			frame.data[2][j] = static_cast<unsigned char>(std::clamp(*current++, 0.f, 1.f) * 255.f);
			/*const float c = current[0] * 255.f - 16;
			const float d = current[1] * 255.f - 128;
			const float e = current[2] * 255.f - 128;
			_video.m_frames.back()[j] = static_cast <unsigned char>(298 * c + 409 * e + 128);
			_video.m_frames.back()[j] = static_cast <unsigned char>(298 * c - 100 - 208*e * e + 128);*/
		}
		converter.convert();
		_video.m_frames.emplace_back(new unsigned char[_video.m_frameSize]);
		std::copy(converter.getDstFrame().data[0], converter.getDstFrame().data[0] + _video.m_frameSize, _video.m_frames.back().get());
	}
}

void Video::saveFrame(const std::string& _fileName, int _frame) const
{
	stbi_write_png(_fileName.c_str(), m_width, m_height, 3, m_frames[_frame].get(), 0);
}

void Video::save(const std::string& _fileName) const
{
	constexpr AVPixelFormat outFormat = AVPixelFormat::AV_PIX_FMT_YUV420P;
	int err = 0;

	AVFormatContext* ofctxTemp;
	AVCALL(avformat_alloc_output_context2, &ofctxTemp, nullptr, nullptr, _fileName.c_str());
	std::unique_ptr<AVFormatContext> ofctx(ofctxTemp);

	AVCodec* codec = AVCALLRET(avcodec_find_encoder_by_name, "ffv1");

	AVStream* videoStream = AVCALLRET(avformat_new_stream, ofctx.get(), codec);

	std::unique_ptr<AVCodecContext> cctx(AVCALLRET(avcodec_alloc_context3, codec));

	videoStream->codecpar->codec_id = codec->id;
	videoStream->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
	videoStream->codecpar->width = m_width;
	videoStream->codecpar->height = m_height;
	videoStream->codecpar->format = outFormat;
//	videoStream->codecpar->bit_rate = 10 * 1000;
	videoStream->time_base = { m_frameRate.num, m_frameRate.den };

	avcodec_parameters_to_context(cctx.get(), videoStream->codecpar);
	cctx->time_base = videoStream->time_base;
//	cctx->max_b_frames = 2;
//	cctx->gop_size = 12;
/*	if (videoStream->codecpar->codec_id == AV_CODEC_ID_H264) {
		av_opt_set(cctx, "preset", "ultrafast", 0);
	}*/
	avcodec_parameters_from_context(videoStream->codecpar, cctx.get());

	AVCALL(avcodec_open2, cctx.get(), codec, NULL);

	AVCALL(avio_open, &ofctx->pb, _fileName.c_str(), AVIO_FLAG_WRITE);

	AVCALL(avformat_write_header, ofctx.get(), NULL);

	av_dump_format(ofctx.get(), 0, _fileName.c_str(), 1);

	FrameConverter converter(cctx->width, cctx->height, AV_PIX_FMT_RGB24, outFormat);

	int count = 0;
	AVPacket pkt;
	av_init_packet(&pkt);
	pkt.data = NULL;
	pkt.size = 0;
	for (auto& frame : m_frames)
	{
		const unsigned char* begin = frame.get();
		std::copy(begin, begin + m_frameSize, converter.getSrcFrame().data[0]);
	//	const int inLinesize[1] = { 3 * cctx->width };
		converter.convert();
		converter.getDstFrame().pts = ++count;
		avcodec_send_frame(cctx.get(), &converter.getDstFrame());

		if (avcodec_receive_packet(cctx.get(), &pkt) == 0) 
		{
		//	pkt.flags |= AV_PKT_FLAG_KEY;
			av_interleaved_write_frame(ofctx.get(), &pkt);
			av_packet_unref(&pkt);
		}
	}

	for (;;) 
	{
		avcodec_send_frame(cctx.get(), NULL);
		if (avcodec_receive_packet(cctx.get(), &pkt) == 0) 
		{
			av_interleaved_write_frame(ofctx.get(), &pkt);
			av_packet_unref(&pkt);
		}
		else {
			break;
		}
	}

	av_write_trailer(ofctx.get());
}

void Video::decode(const std::string& _url)
{
	// open file
	AVFormatContext* formatContextTemp = nullptr;
	AVCALL(avformat_open_input, &formatContextTemp, _url.c_str(), NULL, NULL);
	std::unique_ptr<AVFormatContext> formatContext(formatContextTemp);
	
	AVCALL(avformat_find_stream_info, formatContext.get(), nullptr);

	av_dump_format(formatContext.get(), 0, _url.c_str(), 0);

	constexpr int streamId = 0;
	AVCodecParameters* codecParams = formatContext->streams[streamId]->codecpar;

	AVCodec* codec = AVCALLRET(avcodec_find_decoder, codecParams->codec_id);

	std::unique_ptr<AVCodecContext> codecContext(AVCALLRET(avcodec_alloc_context3, codec));
	AVCALL(avcodec_parameters_to_context, codecContext.get(),
		formatContext->streams[streamId]->codecpar);

	AVCALL(avcodec_open2, codecContext.get(), codec, nullptr);

	AVPacket packet;
	av_init_packet(&packet);

	int frameCount = 0;

	m_frameRate.num = 1;//formatContext->streams[streamId]->r_frame_rate.num;
	m_frameRate.den =  24;// formatContext->streams[streamId]->r_frame_rate.den;
	const int w = codecContext->width;
	const int h = codecContext->height;
	m_width = w;
	m_height = h;
	m_frameSize = w * h * 3;

	FrameConverter converter(w, h, codecContext->pix_fmt, AVPixelFormat::AV_PIX_FMT_RGB24);

	while (av_read_frame(formatContext.get(), &packet) >= 0)
	{
		if (packet.stream_index == streamId)
		{
			int sendPacketResult = avcodec_send_packet(codecContext.get(), &packet);
			if (sendPacketResult == AVERROR(EAGAIN)) {
				// Decoder can't take packets right now. Make sure you are draining it.
			}
			else if (sendPacketResult < 0) {
				// Failed to send the packet to the decoder
			}

			int decodeFrame = avcodec_receive_frame(codecContext.get(), &converter.getSrcFrame());

			if (decodeFrame == AVERROR(EAGAIN)) {
				// The decoder doesn't have enough data to produce a frame
				// Not an error unless we reached the end of the stream
				// Just pass more packets until it has enough to produce a frame
			}
			else if (decodeFrame < 0) {
				// Failed to get a frame from the decoder
			}
			else
			{
				converter.convert();
				m_frames.emplace_back(new unsigned char[m_frameSize]);
				std::copy(converter.getDstFrame().data[0], converter.getDstFrame().data[0] + m_frameSize, m_frames.back().get());
				++frameCount;
			}
		}
		av_packet_unref(&packet);
	}
	std::cout << "Decoded " << frameCount << " frames.";
}