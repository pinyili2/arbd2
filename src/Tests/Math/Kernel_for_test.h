#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "Math/Types.h"
#include "Random/Random.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

struct TransformKernel {
	template<typename... Args>
	HOST DEVICE void operator()(size_t i, Args... args) const {
		// The template system will pass the extracted pointers as args
		// For this kernel: args should be (float* input, float* output)
		auto tuple_args = std::make_tuple(args...);
		auto* input = std::get<0>(tuple_args);
		auto* output = std::get<1>(tuple_args);

		// Transform: y = 2*x + 1
		output[i] = 2.0f * input[i] + 1.0f;
	}
};

struct CombineKernel {
	template<typename... Args>
	HOST DEVICE void operator()(size_t i, Args... args) const {
		// The template system will pass the extracted pointers as args
		// For this kernel: args should be (float* uniform, float* gaussian, float* combined)
		auto tuple_args = std::make_tuple(args...);
		auto* uniform = std::get<0>(tuple_args);
		auto* gaussian = std::get<1>(tuple_args);
		auto* combined = std::get<2>(tuple_args);

		// Simple combination: 70% uniform + 30% gaussian
		combined[i] = 0.7f * uniform[i] + 0.3f * gaussian[i];
	}
};
