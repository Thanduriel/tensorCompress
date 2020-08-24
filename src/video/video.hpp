#pragma once

#include "../core/tensor.hpp"
#include <string>
#include <vector>
#include <memory>

class Video
{
public:
	struct FrameRate
	{
		int num;
		int den;
	};

	using FrameTensor = Tensor<float, 3>;
	Video(const std::string& _fileName);
	Video(const FrameTensor& _tensor, FrameRate _frameRate);
	Video(const Tensor<float, 4>& _tensor, FrameRate _frameRate);

	enum struct PixelChannel
	{
		R = 0, // single color channel for (w x h x f)
		G = 1,
		B = 2,
	};
	struct RGB
	{
		using TensorType = Tensor<float, 4>;
		TensorType operator()(const Video&, int _firstFrame, int _numFrames) const;
	};
	struct SingleChannel
	{
		SingleChannel(PixelChannel _channel) : channel(_channel) {}
		PixelChannel channel;

		using TensorType = Tensor<float, 3>;
		TensorType operator()(const Video&, int _firstFrame, int _numFrames) const;
	};

	struct YUV444
	{
		using TensorType = Tensor<float, 4>;
		TensorType operator()(const Video&, int _firstFrame, int _numFrames) const;
	};
	// Creates a tensor from this video, converting color information to floats in[0,1].
	template<typename Format>
	auto asTensor(int _firstFrame = 0, int _numFrames = 0xfffffff,
		const Format& _format = SingleChannel(PixelChannel::R)) const
	{
		_numFrames = std::min(_numFrames, static_cast<int>(m_frames.size()) - _firstFrame);
		return _format(*this, _firstFrame, _numFrames);
	}
	FrameRate getFrameRate() const { return m_frameRate; }
	int getWidth() const { return m_width; }
	int getHeight() const { return m_height; }

	// save as lossless video
	void save(const std::string& _fileName) const;
	// save a single frame as png
	void saveFrame(const std::string& _fileName, int _frame) const;
private:
	void decode(const std::string& _url);

	int m_width;
	int m_height;
	int m_frameSize; //< in bytes
	FrameRate m_frameRate; // frames per second
	using Frame = std::unique_ptr<unsigned char[]>;
	std::vector<Frame> m_frames;
};