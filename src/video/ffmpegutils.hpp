#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
}

#include <memory>
#include <iostream>
#include <exception>

namespace std
{
	// Enable management of ffmpeg structs through smart pointers.
	template<>
	class default_delete< AVFormatContext >
	{
	public:
		void operator()(AVFormatContext* ptr) const
		{
			avformat_close_input(&ptr);
		}
	};

	template<>
	class default_delete< AVCodecContext >
	{
	public:
		void operator()(AVCodecContext* ptr) const
		{
			avcodec_free_context(&ptr);
		}
	};

	template<>
	class default_delete< AVFrame >
	{
	public:
		void operator()(AVFrame* ptr) const
		{
			av_frame_free(&ptr);
		}
	};

	template<>
	class default_delete< SwsContext >
	{
	public:
		void operator()(SwsContext* ptr) const
		{
			sws_freeContext(ptr);
		}
	};
}


#define AVCALL(fn, ...)	do{								\
	int res = fn(__VA_ARGS__);							\
	if (res < 0)										\
	{													\
		std::cerr << "Call to " << #fn << " failed with error " << res << "\n";	\
		throw std::exception();							\
	}													\
	}while(false)

#define AVCALLRET(fn, ...) 							\
	[&](){											\
			auto res = fn(__VA_ARGS__);				\
			if(!res){								\
				std::cerr << "Call to " << #fn << "failed\n";	\
				throw std::exception();				\
			}										\
			return res;								\
	}()
