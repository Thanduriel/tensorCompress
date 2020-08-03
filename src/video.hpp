#pragma once

#include <string>
#include <vector>
#include <memory>

class Video
{
public:
	Video(const std::string& _fileName);

private:
	void decode(const std::string& _url);

	using Frame = std::unique_ptr<char[]>;
	std::vector<Frame> m_frames;
};