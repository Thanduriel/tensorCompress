#include "core/hosvd.hpp"
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


template<int K, int Dim>
void flattenTest(Tensor<float, Dim>& tensor, float& sum)
{
	auto start = std::chrono::high_resolution_clock::now();
	auto m = tensor.flatten<K>();
	auto end = std::chrono::high_resolution_clock::now();
	float t = std::chrono::duration<float>(end - start).count();
	std::cout << "Flatten in dimension " << K << " " << std::chrono::duration<float>(end - start).count() << std::endl;
	//	m *= 2.14f;
	//	sum += m.trace();

	start = std::chrono::high_resolution_clock::now();
	tensor.set<K>(m);
	end = std::chrono::high_resolution_clock::now();
	std::cout << "Unflatten in dimension " << K << " " << std::chrono::duration<float>(end - start).count() << std::endl;
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

/*	flattenTest<0>(tensor, sum);
	flattenTest<1>(tensor, sum);
	flattenTest<2>(tensor, sum);
	flattenTest<3>(tensor, sum);*/

	start = std::chrono::high_resolution_clock::now();
	const auto&[U, C] = hosvdInterlaced(tensor, 0.05f);
	end = std::chrono::high_resolution_clock::now();
	std::cout << "hosvd" << std::chrono::duration<float>(end - start).count() << std::endl;
	sum += C.norm();
	std::cout << sum;
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
	benchmarkTensor<4>({3,800,600,14});

/*	Video video("TestScene.mp4");
	auto tensor = video.asTensor(40, 48, Video::RGB());
//	auto tensor = randomTensor<3>({ 400,100,48 });
	const auto& [U, C] = hosvdInterlaced(tensor, 0.5f);
	Video videoOut(multilinearProduct(U, C), 8);
//	Video videoOut(tensor, 8);
	videoOut.save("video_1sv.avi");*/

//	testHosvd(tensor);

	return 0;
}