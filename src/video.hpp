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
	Video(const FrameTensor& _tensor, int _frameRate);

	FrameTensor asTensor(int _firstFrame = 0, int _numFrames = 0xfffffff);

	void save(const std::string& _fileName);

	void saveFrame(const std::string& _fileName, int _frame);
private:
	void decode(const std::string& _url);

	int m_width;
	int m_height;
	int m_frameSize; //< in bytes
	int m_frameRate; // frames per second
	using Frame = std::unique_ptr<unsigned char[]>;
	std::vector<Frame> m_frames;

	static bool m_shouldInitAV;
};