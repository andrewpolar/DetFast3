//Concept: Andrew Polar and Mike Poluektov
//Developer Andrew Polar

// License
// If the end user somehow manages to make billions of US dollars using this code,
// and happens to meet the developer begging for change outside a McDonald's,
// they are under no obligation to buy the developer a sandwich.

// Symmetry Clause
// Likewise, if the developer becomes rich and famous by publishing this code,
// and meets an unfortunate end user who went bankrupt using it,
// the developer is also under no obligation to buy the end user a sandwich.

//Publications:
//https://www.sciencedirect.com/science/article/abs/pii/S0016003220301149
//https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742
//https://link.springer.com/article/10.1007/s10994-025-06800-6

//Website:
//http://OpenKAN.org

//This code is demo of concurrent pretraining and concurrent training on disjoints.
//Processing in Linux almost only half second for dataset 100K, 16 features. 
//Pretraining started...
//Pretraining ended, time 115.0
//Parallel version
//Targets are determinants of random 4 * 4 matrices, 100000 training records
//Loop = 0, time in ms = 267.000
//Loop = 1, time in ms = 414.000
//Loop = 2, time in ms = 558.000
//Validation ...
//Pearson 0.974925

//Makefile

//# Compiler and flags
//CXX = g++
//CXXFLAGS = -O2 - std = c++17 - Wall - pthread
//LDFLAGS = -pthread
//
//# Target name(final executable)
//TARGET = DetFast3
//
//# Source files
//SRCS = DetFast3.cpp
//
//# Object files
//OBJS = $(SRCS:.cpp = .o)
//
//# Default rule
//$(TARGET) : $(OBJS)
//	$(CXX) $(CXXFLAGS) - o $@ $(OBJS) $(LDFLAGS)
//
//# Compile.cpp to.o
//% .o: % .cpp
//	$(CXX) $(CXXFLAGS) - c $ < -o $@
//
//# Clean rule
//clean :
//	rm - f $(OBJS) $(TARGET)

#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include "Helper.h"
#include "Function.h"

double validateFunctions(const std::vector<std::unique_ptr<Function>>& layer0, const std::vector<std::unique_ptr<Function>>& layer1,
	const std::vector<std::vector<double>>& features, const std::vector<double>& targets) {

	int nRecords = (int)features.size();
	int nFeatures = (int)features[0].size();
	int nU0 = (int)layer0.size() / nFeatures;
	int nU1 = 1;

	std::vector<double> models0(nU0);
	std::vector<double> models1(nU1);
	std::vector<double> predictions(nRecords);

	for (int record = 0; record < nRecords; ++record) {
		for (int k = 0; k < nU0; ++k) {
			models0[k] = 0.0;
			for (int j = 0; j < nFeatures; ++j) {
				models0[k] += Compute(features[record][j], true, *layer0[k * nFeatures + j]);
			}
			models0[k] /= nFeatures;
		}
		for (int k = 0; k < nU1; ++k) {
			models1[k] = 0.0;
			for (int j = 0; j < nU0; ++j) {
				models1[k] += Compute(models0[j], true, *layer1[j]);
			}
			models1[k] /= nU0;
		}
		predictions[record] = models1[0];
	}
	double pearson = Pearson(predictions, targets);
	return pearson;
}

void pre_worker(std::vector<std::unique_ptr<Function>>& layer0, std::vector<std::unique_ptr<Function>>& layer1,
	const std::vector<std::vector<double>>& features, const std::vector<double>& targets, 
	int nEpochs, double alpha, int nAddends, int nFirst0, int nHowMany) {

	int nRecords = (int)features.size();
	int nFeatures = (int)features[0].size();
	int nU0 = nHowMany / nFeatures;
	int nU1 = 1;
	int nFirst1 = nFirst0 / nFeatures;

	std::vector<double> models0(nU0);
	std::vector<double> models1(nU1);
	std::vector<double> deltas0(nU0);
	std::vector<double> deltas1(nU1);

	for (int epoch = 0; epoch < nEpochs; ++epoch) {
		for (int record = 0; record < nRecords; ++record) {
			for (int k = 0; k < nU0; ++k) {
				models0[k] = 0.0;
				for (int j = 0; j < nFeatures; ++j) {
					models0[k] += Compute(features[record][j], false, *layer0[k * nFeatures + j + nFirst0]);
				}
				models0[k] /= nFeatures;
			}
			for (int k = 0; k < nU1; ++k) {
				models1[k] = 0.0;
				for (int j = 0; j < nU0; ++j) {
					models1[k] += Compute(models0[j], false, *layer1[j + nFirst1]);
				}
				models1[k] /= nAddends; 
			}
			deltas1[0] = alpha * (targets[record] - models1[0]);
			for (int j = 0; j < nU0; ++j) {
				deltas0[j] = deltas1[0] * ComputeDerivative(*layer1[j + nFirst1]);
			}
			for (int k = 0; k < nU1; ++k) {
				for (int j = 0; j < nU0; ++j) {
					Update(deltas1[k], *layer1[j + nFirst1]);
				}
			}
			for (int k = 0; k < nU0; ++k) {
				for (int j = 0; j < nFeatures; ++j) {
					Update(deltas0[k], *layer0[k * nFeatures + j + nFirst0]);
				}
			}
		}
	}
}

void worker(std::vector<std::unique_ptr<Function>>& inner,
	std::vector<std::unique_ptr<Function>>& outer, const std::vector<std::vector<double>>& features,
	const std::vector<double>& targets, int nInner, int nOuter, int start, int end, int nRecords, double alpha) {

	size_t nFeatures = features[0].size();
	std::vector<double> models0(nInner);
	std::vector<double> models1(nOuter);
	std::vector<double> deltas0(nInner);
	std::vector<double> deltas1(nOuter);

	for (int idx = start; idx < end; ++idx) {
		int record = idx;
		if (record >= nRecords) record -= nRecords;
		for (int k = 0; k < nInner; ++k) {
			models0[k] = 0.0;
			for (size_t j = 0; j < nFeatures; ++j) {
				models0[k] += Compute(features[record][j], false, *inner[k * nFeatures + j]);
			}
			models0[k] /= nFeatures;
		}
		for (int k = 0; k < nOuter; ++k) {
			models1[k] = 0.0;
			for (int j = 0; j < nInner; ++j) {
				models1[k] += Compute(models0[j], false, *outer[j]);
			}
			models1[k] /= nInner;
		}
		deltas1[0] = alpha * (targets[record] - models1[0]);
		for (int j = 0; j < nInner; ++j) {
			deltas0[j] = deltas1[0] * ComputeDerivative(*outer[j]);
		}
		for (int k = 0; k < nOuter; ++k) {
			for (int j = 0; j < nInner; ++j) {
				Update(deltas1[k], *outer[j]);
			}
		}
		for (int k = 0; k < nInner; ++k) {
			for (size_t j = 0; j < nFeatures; ++j) {
				Update(deltas0[k], *inner[k * nFeatures + j]);
			}
		}
	}
}

int main() {
	//data
	const int nTrainingRecords = 100'000;
	const int nValidationRecords = 20'000;
	const int nMatrixSize = 4;
	const int nFeatures = nMatrixSize * nMatrixSize;
	const double min = 0.0;
	const double max = 10.0;

	//model
	const int nInner = 70;
	const int nAddends = nInner;
	const int nOuter = 1;
	double alpha = 0.3;
	const int nInnerPoints = 3;
	const int nOuterPoints = 30;

	//batches
	const int nBatchSize = 39'000;
	const int nBatches = 3;
	const int nLoops = 3;

	auto features_training = GenerateInput(nTrainingRecords, nFeatures, min, max);
	auto features_validation = GenerateInput(nValidationRecords, nFeatures, min, max);
	auto targets_training = ComputeDeterminantTarget(features_training, nMatrixSize, nTrainingRecords);
	auto targets_validation = ComputeDeterminantTarget(features_validation, nMatrixSize, nValidationRecords);

	auto tstart = std::chrono::high_resolution_clock::now();

	double targetMin = *std::min_element(targets_training.begin(), targets_training.end());
	double targetMax = *std::max_element(targets_training.begin(), targets_training.end());

	//Initialize random functions
	std::random_device rd;
	std::mt19937 rng(rd());
	std::vector<std::unique_ptr<Function>> innerFunctions;
	for (int i = 0; i < nInner * nFeatures; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nInnerPoints, min, max, targetMin, targetMax, rng);
		innerFunctions.push_back(std::move(function));
	}
	std::vector<std::unique_ptr<Function>> outerFunctions;
	for (int i = 0; i < nInner; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nOuterPoints, targetMin, targetMax, targetMin, targetMax, rng);
		outerFunctions.push_back(std::move(function));
	}

	printf("Pretraining started...\n");
	std::vector<std::thread> workers;
	for (int i = 0; i < nInner / 2; ++i) {
		int offset = i * 32;

		workers.emplace_back(
			pre_worker,
			std::ref(innerFunctions),
			std::ref(outerFunctions),
			std::ref(features_training),
			std::ref(targets_training),
			2,
			alpha,
			nAddends,
			offset,
			32
		);
	}
	for (auto& t : workers) {
		t.join();
	}
	auto tend_pre = std::chrono::high_resolution_clock::now();
	auto ms_pre = std::chrono::duration_cast<std::chrono::milliseconds>(tend_pre - tstart);
	printf("Pretraining ended, time %2.1f\n", static_cast<double>(ms_pre.count()));

	//create containers sized to nBatches
	std::vector<std::vector<std::unique_ptr<Function>>> inners;
	std::vector<std::vector<std::unique_ptr<Function>>> outers;

	//copy 
	for (int b = 0; b < nBatches; ++b) {
		inners.push_back(CopyVector(innerFunctions));
		outers.push_back(CopyVector(outerFunctions));
	}
	innerFunctions.clear();
	outerFunctions.clear();

	printf("Parallel version\n");
	printf("Targets are determinants of random %d * %d matrices, %d training records\n",
		nMatrixSize, nMatrixSize, nTrainingRecords);
	int start = 0;
	std::vector<std::thread> threads;
	for (int loop = 0; loop < nLoops; ++loop) {
		// concurrent training of model copies
		threads.clear();
		for (int b = 0; b < nBatches; ++b) {
			int threadStart = start;
			int threadEnd = start + nBatchSize;
			// Launch thread to train inners[b] and outers[b]
			threads.emplace_back(worker, std::ref(inners[b]), std::ref(outers[b]),
				std::cref(features_training), std::cref(targets_training),
				nInner, nOuter, threadStart, threadEnd, nTrainingRecords, alpha);

			// advance start for next batch (wrap-around)
			start += nBatchSize;
			if (start >= nTrainingRecords) start -= nTrainingRecords;
		}

		for (auto& t : threads) {
			t.join();
		}

		// merging concurrently trained models into the first slot (inners[0], outers[0])
		for (int b = 1; b < nBatches; ++b) {
			AddVectors(inners[0], inners[b]); // sum into inners[0]
			AddVectors(outers[0], outers[b]); // sum into outers[0]
		}

		// average the summed model
		ScaleVectors(inners[0], 1.0 / static_cast<double>(nBatches));
		ScaleVectors(outers[0], 1.0 / static_cast<double>(nBatches));

		// redistribute averaged model to all batch copies for next loop
		for (int b = 1; b < nBatches; ++b) {
			CopyVector(inners[0], inners[b]);
			CopyVector(outers[0], outers[b]);
		}

		auto tend = std::chrono::high_resolution_clock::now();
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart);
		printf("Loop = %d, time in ms = %2.3f\n", loop, static_cast<double>(ms.count()));
	}
	printf("Validation ...\n");
	double pearson = validateFunctions(inners[0], outers[0], features_validation, targets_validation);
	printf("Pearson %f\n\n", pearson);
}

