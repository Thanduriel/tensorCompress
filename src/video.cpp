#include "video.hpp"
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
}
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>
#include <fstream>
#include <iostream>

Video::Video(const std::string& _fileName)
	: m_width(0), m_height(0)
{
/*	using namespace std;
	ifstream ifs(_fileName.c_str(), ios::in | ios::binary | ios::ate);

	ifstream::pos_type fileSize = ifs.tellg();
	ifs.seekg(0, ios::beg);

	vector<char> bytes(fileSize);
	ifs.read(bytes.data(), fileSize);
	*/
	avcodec_register_all();
	decode("file:" + _fileName);
}

Video::FrameTensor Video::asTensor(int _firstFrame, int _numFrames)
{
	const int maxFrame = std::min(_firstFrame + _numFrames, 
		static_cast<int>(m_frames.size()));

	FrameTensor tensor({m_width, m_height, _numFrames});

	float* ptr = tensor.data();
	int count = 0;
	for (int i = _firstFrame; i < maxFrame; ++i)
	{
		for (int j = 0; j < m_frameSize; j+=3)
		{
			*ptr = static_cast<float>(m_frames[i][j]) / 255.f;
			++ptr;
			++count;
		}
	}

	return tensor;
}

void Video::decode(const std::string& _url)
{
	// open file
	AVFormatContext* formatContext = nullptr;
	if (avformat_open_input(&formatContext, _url.c_str(), NULL, NULL) < 0) {
		std::cerr << "Could not open file " << _url.c_str() << std::endl;
	}
	if (avformat_find_stream_info(formatContext, nullptr) < 0)
		std::cerr << "Could not find stream info in " << _url.c_str() << std::endl;

	av_dump_format(formatContext, 0, _url.c_str(), 0);

	constexpr int streamId = 0;
	AVCodecParameters* codecParams = formatContext->streams[streamId]->codecpar;

	AVCodec* codec = avcodec_find_decoder(codecParams->codec_id);
	if (!codec) {
		std::cerr << "Unsupported codec!\n";
		return;
	}

	AVCodecContext* codecContext = avcodec_alloc_context3(codec);
	if (avcodec_parameters_to_context(codecContext,
		formatContext->streams[streamId]->codecpar) < 0)
	{
		std::cerr << "Failed to set parameters.";
		avformat_close_input(&formatContext);
		avcodec_free_context(&codecContext);
		return;
	}

	if (avcodec_open2(codecContext, codec, nullptr) < 0)
		std::cerr << "Could not open codec context.\n";

	AVPacket packet;
	av_init_packet(&packet);

	int frameCount = 0;
	AVFrame* frame = av_frame_alloc();
	AVFrame* frameOut = av_frame_alloc();

	const int w = codecContext->width;
	const int h = codecContext->height;
	m_width = w;
	m_height = h;
	m_frameSize = w * h * 3;
	SwsContext* swsContext = sws_getContext(w, h,
		codecContext->pix_fmt,
		w, h, AVPixelFormat::AV_PIX_FMT_RGB24, SWS_BICUBIC,
		NULL, NULL, NULL);
	frameOut->format = AVPixelFormat::AV_PIX_FMT_RGB24;
	frameOut->width = w;
	frameOut->height = h;
	av_frame_get_buffer(frameOut, 0);

	while (av_read_frame(formatContext, &packet) >= 0)
	{
		if (packet.stream_index == streamId)
		{
			int sendPacketResult = avcodec_send_packet(codecContext, &packet);
			if (sendPacketResult == AVERROR(EAGAIN)) {
				// Decoder can't take packets right now. Make sure you are draining it.
			}
			else if (sendPacketResult < 0) {
				// Failed to send the packet to the decoder
			}

			int decodeFrame = avcodec_receive_frame(codecContext, frame);

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
				sws_scale(swsContext, frame->data, frame->linesize, 0, h, frameOut->data, frameOut->linesize);
				m_frames.emplace_back(new char[m_frameSize]);
				std::copy(frameOut->data[0], frameOut->data[0] + m_frameSize, m_frames.back().get());
				++frameCount;
			/*	if (frameCount == 40)
				{
					stbi_write_png("test.png", w, h, 3, frameOut->data[0], 0);
				}*/
			}
		}
	}
	std::cout << "Decoded " << frameCount << " frames.";

	// Free resources
	av_frame_unref(frame);
	av_frame_free(&frame);
	av_frame_unref(frame);
	av_frame_free(&frame);

	avcodec_free_context(&codecContext);
	sws_freeContext(swsContext);
	avformat_close_input(&formatContext);
}