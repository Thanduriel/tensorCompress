#include "compression.hpp"
#include "core/hosvd.hpp"
#include <fstream>

namespace compression {

	HOSVDCompressor::HOSVDCompressor()
		: m_frameRate{1,24},
		m_truncation(new truncation::TruncationAdaptor(truncation::Zero())),
		m_numFramesPerBlock(24)
	{}

	void HOSVDCompressor::encode(const Video& _video)
	{
		m_frameRate = _video.getFrameRate();
		
		size_t numBlocks = _video.getNumFrames() / m_numFramesPerBlock;
		if (_video.getNumFrames() % m_numFramesPerBlock != 0) ++numBlocks;
		numBlocks = 2;
		for (size_t i = 0; i < numBlocks; ++i)
		{
			const int begin = static_cast<int>(i * m_numFramesPerBlock);
			auto tensor = _video.asTensor(begin, begin + m_numFramesPerBlock, Video::YUV444());
			auto UC = hosvdInterlaced(tensor, *m_truncation);
			m_basis.emplace_back(std::move(std::get<0>(UC)));
			m_core.emplace_back(std::move(std::get<1>(UC)));
		}
	}

	Video HOSVDCompressor::decode() const
	{
		Tensor<float, 4> fullTensor(multilinearProduct(m_basis[0], m_core[0]));
		for (size_t i = 1; i < m_basis.size(); ++i)
		{
			fullTensor.append(multilinearProduct(m_basis[i], m_core[i]));
		}
		return Video(fullTensor, m_frameRate, Video::YUV444());
	}

	void HOSVDCompressor::save(const std::string& _fileName)
	{
		std::ofstream file(_fileName, std::ios::binary);
		file.write(reinterpret_cast<const char*>(&m_frameRate), sizeof(Video::FrameRate));
		const int numBlocks = static_cast<int>(m_basis.size());
		file.write(reinterpret_cast<const char*>(&numBlocks), sizeof(int));

		for (size_t i = 0; i < m_basis.size(); ++i)
		{
			file.write(reinterpret_cast<const char*>(&m_frameRate), sizeof(Video::FrameRate));
			m_core[i].save(file);
			for (const auto& m : m_basis[i])
			{
				const Eigen::Index cols = m.cols();
				const Eigen::Index rows = m.rows();
				file.write(reinterpret_cast<const char*>(&cols), sizeof(Eigen::Index));
				file.write(reinterpret_cast<const char*>(&rows), sizeof(Eigen::Index));
				file.write(reinterpret_cast<const char*>(m.data()), m.rows() * m.cols() * sizeof(float));
			}
		}
	}

	void HOSVDCompressor::load(const std::string& _fileName)
	{
		std::ifstream file(_fileName, std::ios::binary);
		
		file.read(reinterpret_cast<char*>(&m_frameRate), sizeof(Video::FrameRate));
		int numBlocks = 0;
		file.read(reinterpret_cast<char*>(&numBlocks), sizeof(int));

		m_core.reserve(numBlocks);
		m_basis.resize(numBlocks);
		for (int j = 0; j < numBlocks; ++j)
		{
			m_core.emplace_back(file);

			for (size_t i = 0; i < m_basis[j].size(); ++i)
			{
				auto& m = m_basis[j][i];

				Eigen::Index cols = 0;
				Eigen::Index rows = 0;
				file.read(reinterpret_cast<char*>(&cols), sizeof(Eigen::Index));
				file.read(reinterpret_cast<char*>(&rows), sizeof(Eigen::Index));
				m.resize(rows, cols);
				file.read(reinterpret_cast<char*>(m.data()), m.rows() * m.cols() * sizeof(float));
			}
		}
	}

}