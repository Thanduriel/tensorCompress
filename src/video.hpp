#pragma once

#include "tensor.hpp"
#include <string>
#include <vector>
#include <memory>

class Video
{
public:
	using FrameTensor = Tensor<float, 3>;
	Video(const std::string& _fileName);
	Video(const FrameTensor& _tensor);

	FrameTensor asTensor(int _firstFrame = 0, int _numFrames = 0xfffffff);
private:
	void decode(const std::string& _url);

	int m_width;
	int m_height;
	int m_frameSize; //< in bytes
	using Frame = std::unique_ptr<unsigned char[]>;
	std::vector<Frame> m_frames;
};