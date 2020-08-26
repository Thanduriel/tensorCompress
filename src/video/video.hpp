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
	using Frame = std::unique_ptr<unsigned char[]>;
	using FrameTensor = Tensor<float, 3>;

	Video(const std::string& _fileName);
	Video(const FrameTensor& _tensor, FrameRate _frameRate);
	template<typename Format, int Order>
	Video(const Tensor<float, Order>& _tensor, FrameRate _frameRate, Format _format)
		: m_width(_tensor.size()[1]),
		m_height(_tensor.size()[2]),
		m_frameSize(_tensor.size()[0] * _tensor.size()[1] * _tensor.size()[2]),
		m_frameRate(_frameRate)
	{
		_format.fromTensor(_tensor, *this);
	}
	template<typename First, typename Second, typename Format>
	Video(const std::pair<First, Second>& _tensors, FrameRate _frameRate, Format _format)
		: m_width(_tensors.first.size()[0]),
		m_height(_tensors.first.size()[1]),
		m_frameSize(_tensors.first.size()[0] * _tensors.first.size()[1] * 3),
		m_frameRate(_frameRate)
	{
		_format.fromTensor(_tensors, *this);
	}

	enum struct PixelChannel
	{
		R = 0, // single color channel for (w x h x f)
		G = 1,
		B = 2,
	};

	struct RGB
	{
		using TensorType = Tensor<float, 4>;
		TensorType toTensor(const Video&, int _firstFrame, int _numFrames) const;
		void fromTensor(const TensorType& _tensor, Video& _video) const;
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
		TensorType toTensor(const Video&, int _firstFrame, int _numFrames) const;
		void fromTensor(const TensorType& _tensor, Video& _video) const;
	};

	struct YUV420
	{
		using TensorType = std::pair<Tensor<float, 3>, Tensor<float, 4>>;
		TensorType toTensor(const Video&, int _firstFrame, int _numFrames) const;
		void fromTensor(const TensorType& _tensor, Video& _video) const;
	};
	// Creates a tensor from this video, converting color information to floats in[0,1].
	template<typename Format>
	auto asTensor(int _firstFrame = 0, int _numFrames = 0xfffffff,
		const Format& _format = SingleChannel(PixelChannel::R)) const
	{
		_numFrames = std::min(_numFrames, static_cast<int>(m_frames.size()) - _firstFrame);
		return _format.toTensor(*this, _firstFrame, _numFrames);
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
	std::vector<Frame> m_frames;
};