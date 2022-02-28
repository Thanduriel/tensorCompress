#include "tests.hpp"
#include "../core/hosvd.hpp"
#include <iostream>
#include <random>
#include <fstream>

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
	static std::default_random_engine rng(0x23451);
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
	const auto fixTensor = countTensor<3>({ 2,3,4 });
	EXPECT(fixTensor.norm() == 70.f, "frobenius norm");
	auto index = Tensor<float, 3>::SizeVector({1,2,3});
	EXPECT(fixTensor.index(fixTensor.flatIndex(index))
		== index, "index computation");
	EXPECT(fixTensor == fixTensor, "comparison operator");
	const auto smallTensor = randomTensor<3>({ 2,3,4 });
	testFlattening<float, 3>(smallTensor);

	// save & load
/*	std::ofstream outFile("test.tensor");
	fixTensor.save(outFile);
	outFile.close();
	std::ifstream inFile("test.tensor");
	const Tensor<float, 3> loadTensor(inFile);
	EXPECT(loadTensor == fixTensor, "save and load");
	*/
	// hosvd
	const auto& [U, C] = hosvd(smallTensor);
	auto smallT2 = multilinearProduct(U, C);
	auto smallT3 = multilinearProductKronecker(U, C);
	EXPECT((smallT2 - smallT3).norm(), "multilinear product");
	EXPECT((smallTensor - smallT2).norm() / smallTensor.norm() < 0.0001f, "classic hosvd");

	const auto& [U2, C2] = hosvdInterlaced(smallTensor);
	auto tensor3 = multilinearProduct(U2, C2);
	EXPECT((smallTensor - tensor3).norm() / smallTensor.norm() < 0.0001f, "interlaced hosvd");

	using SizeVec = typename Tensor<float, 3>::SizeVector;
	const SizeVec sizeVec{ 2,2,1 };
	const auto& [U4, C4] = hosvdInterlaced(smallTensor, truncation::Rank(sizeVec));
	EXPECT(sizeVec == C4.size(), "rank based truncation");

	const auto mediumTensor = countTensor<4>({ 16,11,7, 8 });
	testFlattening(mediumTensor);
	const auto& [U3, C3] = hosvdInterlaced(mediumTensor);
	auto tensor4 = multilinearProduct(U3, C3);
	EXPECT((mediumTensor - tensor4).norm() / mediumTensor.norm() < 0.0001f, "interlaced hosvd");

	std::cout << "\nSuccessfully finished tests " << testsRun - testsFailed << "/" << testsRun << "\n";
}