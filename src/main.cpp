#include "core/tensor.hpp"
#include "video/video.hpp"
#include "utils/utils.hpp"
#include "tests/tests.hpp"
#include <iostream>
#include <random>
#include <chrono>
#include <Eigen/Eigen>
#include <Eigen/SVD>

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


template<int Dim>
void benchmarkTensor(int _size)
{
	std::cout << "Benchmarking a tensor with size " << _size << "^" << Dim << std::endl;

	std::default_random_engine rng;
	std::uniform_real_distribution<float> dist;

	auto start = std::chrono::high_resolution_clock::now();

	Tensor<float, Dim>::SizeVector size;
	for (auto& s : size) s = Dim;
	Tensor<float, Dim> tensor(size);
	tensor.set([&](auto) { return dist(rng); });
	
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "Fill with random elements " << std::chrono::duration<float>(end - start).count() << std::endl;

	for (int k = 0; k < Dim; ++k)
	{
		start = std::chrono::high_resolution_clock::now();

		auto m = tensor.flatten(k);

		end = std::chrono::high_resolution_clock::now();
		std::cout << "Flatten in dimension " << k << " " << std::chrono::duration<float>(end - start).count() << std::endl;
		std::cout << m.norm();
	}
}

int main()
{
#ifndef NDEBUG 
#if defined(_MSC_VER)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
//	_CrtSetBreakAlloc(12613);
#endif
#endif
	Tests tests;
	tests.run();

//	benchmarkSVD(1920, 1080);
//	benchmarkTensor<3>(512);

	Video video("TestScene.mp4");
	auto tensor = video.asTensor(40, 48, Video::RGB());
//	auto tensor = randomTensor<3>({ 400,100,48 });
	const auto& [U, C] = hosvdInterlaced(tensor, 0.5f);
	Video videoOut(multilinearProduct(U, C), 8);
//	Video videoOut(tensor, 8);
	videoOut.save("video_1sv.avi");

//	testHosvd(tensor);

	return 0;
}