#include "tests.hpp"
#include "tensor.hpp"
#include <iostream>
#include <random>

# define EXPECT(cond,description)										\
do {																	\
	++testsRun;															\
	if (!(cond)) {														\
	  std::cerr << "FAILED "											\
				<< description << std::endl <<"       " << #cond		\
				<< std::endl											\
				<< "       " << __FILE__ << ':' << __LINE__				\
				<< std::endl;											\
	  ++testsFailed;													\
	}																	\
} while (false)

int testsRun = 0;
int testsFailed = 0;

template<int Dim>
Tensor<float, Dim> randomTensor(const typename Tensor<float, Dim>::SizeVector & _size)
{
	std::default_random_engine rng;
	std::uniform_real_distribution<float> dist;

	Tensor<float, Dim> tensor(_size);
	tensor.set([&](auto) { return dist(rng); });

	return tensor;
}

template<int Dim>
Tensor<float, Dim> countTensor(const typename Tensor<float, Dim>::SizeVector & _size)
{
	Tensor<float, Dim> tensor(_size);
	float count = 0;
	tensor.set([&](auto) { return count += 1.f; });

	return tensor;
}

template<typename Scalar, int Dims>
void testFlattening(const Tensor<Scalar, Dims>& tensor)
{
	// flatten
	for (int k = 0; k < Dims; ++k)
	{
		auto flat = tensor.flatten(k);
		Tensor<Scalar, Dims> tensor2(tensor.size());
		tensor2.set(flat, k);
		EXPECT((tensor - tensor2).norm() == 0.f, "flatten and unflatten");
	}
}

void Tests::run()
{
	// small tensors
	const auto smallTensor = randomTensor<3>({ 2,3,4 });
	testFlattening<float, 3>(smallTensor);
	// hosvd
	const auto& [U, C] = hosvd(smallTensor, 0.0f);
	auto smallT2 = multilinearProduct(U, C);
	auto smallT3 = multilinearProductKronecker(U, C);
	EXPECT((smallT2 - smallT3).norm(), "multilinear product");
	EXPECT((smallTensor - smallT2).norm() / smallTensor.norm() < 0.0001f, "classic hosvd");

	const auto& [U2, C2] = hosvdInterlaced(smallTensor, 0.0f);
	auto tensor3 = multilinearProduct(U2, C2);
	EXPECT((smallTensor - tensor3).norm() / smallTensor.norm() < 0.0001f, "interlaced hosvd");

	std::cout << "\nSuccessfully finished tests " << testsRun - testsFailed << "/" << testsRun << "\n";
}