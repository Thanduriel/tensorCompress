#include "core/hosvd.hpp"
#include "video/video.hpp"
#include "utils/utils.hpp"
#include "tests/tests.hpp"
#include "core/compression.hpp"
#include <iostream>
#include <random>
#include <chrono>
#include <Eigen/Eigen>
#include <Eigen/SVD>
#include <fstream>
#include <charconv>
#include <args.hxx>
#include <filesystem>
#include <thread>

// CRT's memory leak detection
#ifndef NDEBUG 
#if defined(_MSC_VER)
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#endif
#endif

void benchmarkSVD(int sx, int sy)
{
	std::default_random_engine rng;
	std::normal_distribution<float> dist;

	using namespace Eigen;
	MatrixXf testMat(sx, sy);
	for (int i = 0; i < sx; ++i)
		for (int j = 0; j < sy; ++j)
			testMat(i, j) = dist(rng);

	auto start = std::chrono::high_resolution_clock::now();
	BDCSVD<MatrixXf> svd(testMat);
	auto end = std::chrono::high_resolution_clock::now();

	std::cout << "SVD takes " << std::chrono::duration<float>(end - start).count() << std::endl;

	std::cout << svd.singularValues().size() << std::endl;
}


template<int K, int Dim>
void flattenTest(Tensor<float, Dim>& tensor, float& sum)
{
	auto start = std::chrono::high_resolution_clock::now();
	auto m = tensor.template flatten<K>();
	auto end = std::chrono::high_resolution_clock::now();
	float t = std::chrono::duration<float>(end - start).count();
	std::cout << "Flatten in dimension " << K << "  \t" << std::chrono::duration<float>(end - start).count() << std::endl;
	//	m *= 2.14f;
	//	sum += m.trace();

	start = std::chrono::high_resolution_clock::now();
	tensor.template set<K>(m);
	end = std::chrono::high_resolution_clock::now();
	std::cout << "Unflatten in dimension " << K << "\t" << std::chrono::duration<float>(end - start).count() << std::endl;
	t += std::chrono::duration<float>(end - start).count();
	sum += t;//tensor.norm()
}

template<int Dim>
void benchmarkTensor(const std::array<int, Dim>& _sizeVec)
{
	std::cout << "Benchmarking a tensor with size " << _sizeVec << "^" << Dim << std::endl;

	std::default_random_engine rng;
	std::uniform_real_distribution<float> dist;

	auto start = std::chrono::high_resolution_clock::now();

	Tensor<float, Dim> tensor(_sizeVec);
	tensor.set([&](auto) { return dist(rng); });
	
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "Fill with random elements " << std::chrono::duration<float>(end - start).count() << std::endl;

	float sum = 0.f;
	//for (int k = 0; k < Dim; ++k)

	flattenTest<0>(tensor, sum);
	flattenTest<1>(tensor, sum);
	flattenTest<2>(tensor, sum);
	flattenTest<3>(tensor, sum);

	start = std::chrono::high_resolution_clock::now();
	const auto&[U, C] = hosvdInterlaced(tensor, truncation::Tolerance(0.05f));
	end = std::chrono::high_resolution_clock::now();
	std::cout << "hosvd                   \t" << std::chrono::duration<float>(end - start).count() << std::endl;
	
	start = std::chrono::high_resolution_clock::now();
	auto tensor2 = multilinearProduct(U, C);
	end = std::chrono::high_resolution_clock::now();
	std::cout << "multilinear product     \t" << std::chrono::duration<float>(end - start).count() << std::endl;

	start = std::chrono::high_resolution_clock::now();
	auto tensor3 = tensor - tensor2;
	end = std::chrono::high_resolution_clock::now();
	std::cout << "subtract                \t" << std::chrono::duration<float>(end - start).count() << std::endl;

	start = std::chrono::high_resolution_clock::now();
	sum += tensor3.norm();
	end = std::chrono::high_resolution_clock::now();
	std::cout << "norm                    \t" << std::chrono::duration<float>(end - start).count() << std::endl;
	
	std::cout << sum + C.norm();
}

enum struct TruncationMode
{
	Zero,
	Rank,
	Tolerance,
	ToleranceSum,
};

const std::unordered_map<std::string, TruncationMode> TRUNCATION_NAMES =
{ {
	{"zero", TruncationMode::Zero},
	{"rank", TruncationMode::Rank},
	{"tolerance", TruncationMode::Tolerance},
	{"tolerance_sum", TruncationMode::ToleranceSum},
} };

enum struct PixelFormat 
{
	RGB,
	YUV444
};

const std::unordered_map<std::string, PixelFormat> PIXEL_FORMATS =
{ {
	{"RGB", PixelFormat::RGB},
	{"YUV444", PixelFormat::YUV444}
} };

int main(int argc, char** args)
{
#ifndef NDEBUG 
#if defined(_MSC_VER)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	//	_CrtSetBreakAlloc(12613);
#endif
	Tests tests;
	tests.run();
#endif

	//	benchmarkSVD(1920, 1080);
	//	benchmarkTensor<4>({3,800,600,14});

	args::ArgumentParser parser("HOSVD video compressor.");
	args::HelpFlag help(parser, "help", "display this help menu", { 'h', "help" });

	args::ValueFlag<std::string> inputFile(parser, "input file",
		"path of the video file to process",
		{ 'i', "input" });
	args::ValueFlag<std::string> outputFile(parser, "output file",
		"name of the output file",
		{ 'o', "output" });

	args::MapFlag<std::string, TruncationMode> truncationMode(parser, "truncation mode",
		"rule which is applied to truncate singular values", { "trunc" }, TRUNCATION_NAMES,
		TruncationMode::Tolerance);
	args::MapFlag<std::string, PixelFormat> pixelFormat(parser, "pixel format",
		"pixel format on which tensors are defined; independent of the format used by input/output videos", { "pix_fmt" }, 
		PIXEL_FORMATS, PixelFormat::YUV444);
	args::PositionalList<float> truncationThreshold(parser, "truncation threshold",
		"values used for truncation in each dimension");
	args::ValueFlag<int> framesPerBlock(parser, "frames per block",
		"number of frames combined to a single tensor; if 0, the whole video is used (larger blocks allow for better compression but reduce encode and decode performance)",
		{ "block_size" }, 24);
	args::ValueFlag<int> numThreads(parser, "max threads",
		"maximum number of threads used during computations",
		{ "num_threads" }, std::thread::hardware_concurrency() / 2);

	try
	{
		parser.ParseCLI(argc, args);
	}
	catch (const args::Help&)
	{
		std::cout << parser;
		return 0;
	}
	catch (const args::ParseError& e)
	{
		std::cerr << e.what() << std::endl;
		return 1;
	}
	catch (const args::ValidationError& e)
	{
		std::cerr << e.what() << std::endl;
		return 1;
	}

	namespace fs = std::filesystem;
	const fs::path inputPath = args::get(inputFile);
	if (!fs::exists(inputPath) || !fs::is_regular_file(inputPath))
	{
		std::cerr << "[Error] The input file " << inputFile << " does not exist.\n";
		return 1;
	}

	Eigen::setNbThreads(args::get(numThreads));

	auto process = [&](const auto& pixelFormat)
	{
		auto compressor = compression::HOSVDCompressor(pixelFormat);
		compressor.setFramesPerBlock(args::get(framesPerBlock));

		std::vector<float> rank = args::get(truncationThreshold);
		if (rank.size() < 4)
		{
			std::cout << "[Warning] Less than 4 truncation threshold values given. Default may not work with every truncation mode.\n";
		}
		switch (args::get(truncationMode))
		{
		case TruncationMode::Zero:
			compressor.setTruncation(truncation::Zero());
			break;
		case TruncationMode::Rank:
			if (rank.empty()) 
				rank.push_back(1);
			compressor.setTruncation(truncation::Rank(std::vector<int>(rank.begin(), rank.end())));
			break;
		case TruncationMode::Tolerance:
			if (rank.empty())
				rank.push_back(0.1f);
			compressor.setTruncation(truncation::Tolerance(rank));
			break;
		case TruncationMode::ToleranceSum:
			if (rank.empty())
				rank.push_back(0.2f);
			compressor.setTruncation(truncation::ToleranceSum(rank));
			break;
		}

		if (inputPath.extension() == "ten")
		{
			std::cout << "Loading tensor file " << args::get(inputFile) << ".\n";
			compressor.load(args::get(inputFile));
		}
		else
		{
			std::cout << "Loading video " << args::get(inputFile) << ".\n";
			Video video(args::get(inputFile));
			std::cout << "Applying HOSVD.\n";
			compressor.encode(video);
		}

		struct Stats
		{
			int min = std::numeric_limits<int>::max();
			int max = 0;
			int sum = 0;
		};
		const auto& singularValues = compressor.singularValues();
		std::vector<Stats> stats(singularValues.front().order());

		for (auto s : singularValues)
		{
			for (size_t dim = 0; dim < s.size().size(); ++dim)
			{
				const int r = s.size()[dim];
				stats[dim].min = std::min(stats[dim].min, r);
				stats[dim].max = std::max(stats[dim].max, r);
				stats[dim].sum += r;
			}
		}
		std::cout << "Statistics of the resulting tensors: \n dimension\\rank min max mean\n";
		for (auto& stat : stats)
			std::cout << stat.min << " " << stat.max << " " << stat.sum / singularValues.size() << "\n";

		namespace fs = std::filesystem;
		const fs::path outputPath = args::get(outputFile);
		if (outputPath.extension() == "ten")
		{
			std::cout << "Saving tensors as " << args::get(outputFile) << ".\n";
			compressor.save(args::get(outputFile));
		}
		else
		{
			std::cout << "Saving video as " << args::get(outputFile) << ".\n";
			Video video = compressor.decode();
			video.save(args::get(outputFile));
		}
	};

	switch (args::get(pixelFormat))
	{
	case PixelFormat::RGB:
		process(Video::RGB());
		break;
	case PixelFormat::YUV444:
		process(Video::YUV444());
		break;
	}

#if false
	Video video("TestScene.mp4");
	/*	auto tensor = video.asTensor(0, 80, Video::YUV420());
		Video video2(tensor, Video::FrameRate{1,24}, Video::YUV420());
		video2.save("TestSceneRestoredYUV420.avi");*/
	/**/compression::HOSVDCompressor compressor;
	compressor.setTruncation(truncation::Rank{ 2,100,100,10 });
	compressor.encode(video);
	Video video2 = compressor.decode();
	video2.save("TestSceneRestoredYUV4.avi");
#endif

	return 0;
}