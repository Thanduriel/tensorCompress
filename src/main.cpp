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
	auto m = tensor.flatten<K>();
	auto end = std::chrono::high_resolution_clock::now();
	float t = std::chrono::duration<float>(end - start).count();
	std::cout << "Flatten in dimension " << K << "  \t" << std::chrono::duration<float>(end - start).count() << std::endl;
	//	m *= 2.14f;
	//	sum += m.trace();

	start = std::chrono::high_resolution_clock::now();
	tensor.set<K>(m);
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
	Rank,
	Tolerance,
	ToleranceSum,
	COUNT
};

const std::unordered_map<std::string, TruncationMode> TRUNCATION_NAMES =
{ {
	{"rank", TruncationMode::Rank},
	{"tolerance", TruncationMode::Tolerance},
	{"tolerance_sum", TruncationMode::ToleranceSum},
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

	args::ValueFlag<std::string> inputPath(parser, "input file",
		"path of the video file to process",
		{ 'i', "input" });

	args::ValueFlag<std::string> outputPath(parser, "output file",
		"name of the output file",
		{ 'o', "output" });


	args::MapFlag<std::string, TruncationMode> truncationMode(parser, "truncation mode",
		"rule which is applied to truncate singular values", { "trunc" }, TRUNCATION_NAMES);
	args::PositionalList<float> truncationThreshold(parser, "truncation threshold",
		"values used for truncation in each dimension");

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

	std::cout << "Loading video " << args::get(inputPath) << ".\n";
	Video video(args::get(inputPath));
	compression::HOSVDCompressor compressor;
	const std::vector<float> rank = args::get(truncationThreshold);

	switch (args::get(truncationMode))
	{
	case TruncationMode::Rank:
		compressor.setTruncation(truncation::Rank(std::vector<int>(rank.begin(), rank.end())));
		break;
	case TruncationMode::Tolerance:
		compressor.setTruncation(truncation::Tolerance(rank));
		break;
	case TruncationMode::ToleranceSum:
		compressor.setTruncation(truncation::ToleranceSum(rank));
		break;
	}

	std::cout << "Applying HOSVD.\n";
	compressor.encode(video);
	std::cout << "Reduced rank: ";
	for (auto r : compressor.core().front().size())
		std::cout << r << " ";
	std::cout << std::endl;

	Video video2 = compressor.decode();
	video2.save(args::get(outputPath));

	if (false)
	{
		Video video("TestScene.mp4");
		/*	auto tensor = video.asTensor(0, 80, Video::YUV420());
			Video video2(tensor, Video::FrameRate{1,24}, Video::YUV420());
			video2.save("TestSceneRestoredYUV420.avi");*/
		compression::HOSVDCompressor compressor;
		compressor.setTruncation(truncation::Rank{ 2,100,100,10 });
		compressor.encode(video);
		Video video2 = compressor.decode();
		video2.save("TestSceneRestoredYUV4.avi");
	}
	return 0;
}