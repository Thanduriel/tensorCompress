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

	explicit Video(const std::string& _fileName);
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
	template<typename... Args, typename Format>
	Video(const std::tuple<Args...>& _tensors, FrameRate _frameRate, Format _format)
		: m_width(std::get<0>(_tensors).size()[std::get<0>(_tensors).order() - 3]),
		m_height(std::get<0>(_tensors).size()[std::get<0>(_tensors).order() - 2]),
		m_frameSize(m_width * m_height),
		m_frameRate(_frameRate)
	{
		int numChannels = 0;
		std::apply([&](auto&&... args) { ((numChannels += args.order() == 4 ? args.size()[0] : 1), ...); }, _tensors);
		m_frameSize *= numChannels;

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
		using TensorType = std::tuple<Tensor<float, 3>, Tensor<float, 4>>;
		TensorType toTensor(const Video&, int _firstFrame, int _numFrames) const;
		void fromTensor(const TensorType& _tensor, Video& _video) const;
	};
	// Creates a tensor from this video, converting color information to floats in [0,1].
	template<typename Format>
	auto asTensor(size_t _firstFrame = 0, size_t _numFrames = 0xfffffff,
		const Format& _format = RGB()) const
	{
		_numFrames = std::min(_numFrames, m_frames.size() - _firstFrame);
		return _format.toTensor(*this, static_cast<int>(_firstFrame), static_cast<int>(_numFrames));
	}
	FrameRate getFrameRate() const { return m_frameRate; }
	int getWidth() const { return m_width; }
	int getHeight() const { return m_height; }
	size_t getNumFrames() const { return m_frames.size(); }

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