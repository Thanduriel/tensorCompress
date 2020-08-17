#include "compression.hpp"
#include "core/hosvd.hpp"

namespace compression {

	void HOSVDCompressor::encode(const Video& _video)
	{
		auto tensor = _video.asTensor(0, 48, Video::RGB());
		auto UC = hosvdInterlaced(tensor, truncation::Tolerance(0.5f));
		m_basis = std::move(std::get<0>(UC));
		m_core = std::move(std::get<1>(UC));
	//	Video videoOut(multilinearProduct(U, C), 8);
		//	Video videoOut(tensor, 8);
	//	videoOut.save("video_1sv.avi");
	}

}