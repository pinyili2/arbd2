// In a new header, e.g., "RandomKernels.h"
#pragma once
#include "openrand/philox.h" // Your chosen PRNG library
#include "Backend/Resource.h"
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Math/Vector3.h"
#include <cmath>

// --- Functor for Uniform Float Generation ---
namespace ARBD {
template<typename T>
struct UniformFunctor {
    T min_val;
    T max_val;

    HOST DEVICE void operator()(size_t i, openrand::Philox* states, T* output) const {
        output[i] = min_val + states[i].draw_float4() * (max_val - min_val);
    }
};

template<typename T>
struct GaussianFunctor {
    T mean;
    T stddev;
    size_t output_size;

    // The Box-Muller transform using your Vector3_t as a float2 container
    HOST DEVICE ARBD::Vector3_t<T> box_muller(float u1, float u2) const {
        float r = std::sqrt(-2.0f * std::log(u1));
        float theta = 2.0f * 3.1415926535f * u2;
        return ARBD::Vector3_t<T>(r * std::cos(theta), r * std::sin(theta), 0.0f);
    }

    // Device code using standard C++ functions
    HOST DEVICE void operator()(size_t i, openrand::Philox* states, T* output) const {
        auto u = states[i].draw_float4();
        float u1 = std::fmax(u.x, 1e-7f);
        float u2 = u.y;

        ARBD::Vector3_t<T> gauss_pair = box_muller(u1, u2);

        output[i*2] = mean + stddev * gauss_pair.x;
        if ((i*2 + 1) < output_size) {
             output[i*2 + 1] = mean + stddev * gauss_pair.y;
        }
    }
};
}