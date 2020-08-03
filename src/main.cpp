#include "tensor.hpp"
#include "video.hpp"
#include <iostream>
#include <random>
#include <chrono>
#include <Eigen/Eigen>
#include <Eigen/SVD>

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
Tensor<float, Dim> randomTensor(int _size)
{
	std::default_random_engine rng;
	std::uniform_real_distribution<float> dist;

	auto start = std::chrono::high_resolution_clock::now();

	Tensor<float, Dim>::SizeVector size;
	size.fill(Dim);
	Tensor<float, Dim> tensor(size);
	tensor.set([&](auto) { return dist(rng); });

	return tensor;
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
//	benchmarkSVD(1920, 1080);
//	benchmarkTensor<3>(512);

/*	Tensor<float, 3> tensor({4,4,4});

	for (int i = 0; i < 4; ++i)
		tensor[{i, i, i}] = static_cast<float>(i);

	float sum = 0.f;
	for (int z = 0; z < 4; ++z)
		for (int y = 0; y < 4; ++y)
			for (int x = 0; x < 4; ++x)
				sum += tensor[{x, y, z}];

	std::cout << sum << std::endl;*/

	const auto tensor = randomTensor<3>(4);

	const auto&[U, C] = hosvd(tensor);

	const Tensor<float, 3> tensor2 = multilinearProduct(U, C);

	std::cout << tensor.norm() << ", " << tensor2.norm() << std::endl;
	std::cout << (tensor - tensor2).norm();

	Video video("AcaIntro_light.mp4");

	return 0;
}