#include "tensor.hpp"
#include "video.hpp"
#include "utils.hpp"
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
Tensor<float, Dim> randomTensor(const typename Tensor<float, Dim>::SizeVector& _size)
{
	std::default_random_engine rng;
	std::uniform_real_distribution<float> dist;

	auto start = std::chrono::high_resolution_clock::now();

	Tensor<float, Dim> tensor(_size);
	tensor.set([&](auto) { return dist(rng); });

	return tensor;
}

template<int Dim>
Tensor<float, Dim> countTensor(const typename Tensor<float, Dim>::SizeVector& _size)
{
	Tensor<float, Dim> tensor(_size);
	float count = 0;
	tensor.set([&](auto) { return count += 1.f; });

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

template<typename Scalar, int Dims>
void testHosvd(const Tensor<Scalar,Dims>& tensor)
{
//	const auto tensor = randomTensor<3>(4);

	// flatten
	for (int k = 0; k < Dims; ++k)
	{
		auto flat = tensor.flatten(k);
		Tensor<Scalar, Dims> tensor2(tensor.size());
		tensor2.set(flat, k);
//		std::cout << flat;
		std::cout << "k-flattening: " << (tensor - tensor2).norm() << "\n";
	}

	// hosvd
/*	const auto& [U, C] = hosvd(tensor, 0.0f);
	auto tensor2 = multilinearProduct(U, C);
	std::cout << "classic hosvd: " << (tensor - tensor2).norm() << "\n";*/

	const auto& [U2, C2] = hosvdInterlaced(tensor, 0.15f);
	auto tensor3 = multilinearProduct(U2, C2);
	std::cout << tensor.norm() << "\n";
	std::cout << "interlaced hosvd: " << (tensor - tensor3).norm() << "\n";

//	std::cout << tensor.norm() << ", " << tensor2.norm() << std::endl;
}

int main()
{
//	benchmarkSVD(1920, 1080);
//	benchmarkTensor<3>(512);
//	testHosvd(randomTensor<3>({1920,1080,2}));

	Video video("AcaIntro_light.mp4");
	auto tensor = video.asTensor(40, 2);
//	auto tensor2 = randomTensor<3>({ 1920,1080,2 });

	const auto& [U, C] = hosvdInterlaced(tensor, 0.f);
	std::cout << C.size() << "\n";
//	testHosvd(tensor);

	return 0;
}