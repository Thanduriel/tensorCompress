#include "compression.hpp"
#include "core/hosvd.hpp"
#include <fstream>

namespace compression {

	HOSVDCompressor::HOSVDCompressor()
		: m_targetRank(4, std::numeric_limits<int>::max())
	{}

	void HOSVDCompressor::encode(const Video& _video)
	{
		m_frameRate = _video.getFrameRate();
		auto tensor = _video.asTensor(0, 42, Video::YUV444());
		auto UC = hosvdInterlaced(tensor, truncation::Rank(m_targetRank));
		m_basis = std::move(std::get<0>(UC));
		m_core = std::move(std::get<1>(UC));
	}

	Video HOSVDCompressor::decode() const
	{
		return Video(multilinearProduct(m_basis, m_core), m_frameRate, Video::YUV444());
	}

	void HOSVDCompressor::save(const std::string& _fileName)
	{
		std::ofstream file(_fileName, std::ios::binary);
		file.write(reinterpret_cast<const char*>(&m_frameRate), sizeof(Video::FrameRate));
		m_core.save(file);
		for (const auto& m : m_basis)
		{
			const Eigen::Index cols = m.cols();
			const Eigen::Index rows = m.rows();
			file.write(reinterpret_cast<const char*>(&cols), sizeof(Eigen::Index));
			file.write(reinterpret_cast<const char*>(&rows), sizeof(Eigen::Index));
			file.write(reinterpret_cast<const char*>(m.data()), m.rows() * m.cols() * sizeof(float));
		}
	}

	void HOSVDCompressor::load(const std::string& _fileName)
	{
		std::ifstream file(_fileName, std::ios::binary);
		
		file.read(reinterpret_cast<char*>(&m_frameRate), sizeof(Video::FrameRate));
	//	m_frameRate.num = 1;
	//	m_frameRate.den = 24;
		m_core = Tensor<float, 4>(file);

		for (size_t i = 0; i < m_basis.size(); ++i)
		{
			auto& m = m_basis[i];

			Eigen::Index cols = 0;
			Eigen::Index rows = 0;
			file.read(reinterpret_cast<char*>(&cols), sizeof(Eigen::Index));
			file.read(reinterpret_cast<char*>(&rows), sizeof(Eigen::Index));
			m.resize(rows, cols);
			file.read(reinterpret_cast<char*>(m.data()), m.rows() * m.cols() * sizeof(float));
		}
	}

}