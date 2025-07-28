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
    uint64_t base_seed;
    uint32_t base_ctr;
    uint32_t global_seed;

    template<typename... Args>
    HOST DEVICE void operator()(size_t i, Args... args) const {
        // The template system will pass the extracted pointers as args
        // For this kernel: args should be (T* output)
        auto tuple_args = std::make_tuple(args...);
        auto* output = std::get<0>(tuple_args);
        
        // Create a fresh Philox instance with deterministic parameters
        // Each thread gets a unique counter value based on its index
        openrand::Philox rng(base_seed, base_ctr + static_cast<uint32_t>(i), global_seed);
        
        auto uniform_vals = rng.draw_float4();
        // Use only the x component for single value generation
        output[i] = min_val + uniform_vals.x * (max_val - min_val);
    }
};

template<typename T>
struct GaussianFunctor {
    T mean;
    T stddev;
    size_t output_size;
    uint64_t base_seed;
    uint32_t base_ctr;
    uint32_t global_seed;

    template<typename... Args>
    HOST DEVICE void operator()(size_t i, Args... args) const {
        // The template system will pass the extracted pointers as args
        // For this kernel: args should be (T* output)
        auto tuple_args = std::make_tuple(args...);
        auto* output = std::get<0>(tuple_args);
        
        if (i >= output_size) return;
        
        // Create a fresh Philox instance with deterministic parameters
        openrand::Philox rng(base_seed, base_ctr + static_cast<uint32_t>(i), global_seed);
        
        auto u = rng.draw_float4();
        float u1 = (u.x < 1e-7f) ? 1e-7f : u.x;  // Avoid log(0)
        float u2 = u.y;

        // Box-Muller transform - generate one value per thread
        float r = sqrt(-2.0f * log(u1));
        float theta = 2.0f * 3.1415926535f * u2;
        float gaussian_val = r * cos(theta);

        output[i] = mean + stddev * gaussian_val;
    }
};

// Specialization for Vector3_t types
template<typename T>
struct GaussianFunctor<ARBD::Vector3_t<T>> {
    ARBD::Vector3_t<T> mean;
    ARBD::Vector3_t<T> stddev;
    size_t output_size;
    uint64_t base_seed;
    uint32_t base_ctr;
    uint32_t global_seed;

    // The Box-Muller transform for generating pairs of Gaussian values
    HOST DEVICE ARBD::Vector3_t<T> box_muller(float u1, float u2) const {
        float r = sqrt(-2.0f * log(u1));
        float theta = 2.0f * 3.1415926535f * u2;
        return ARBD::Vector3_t<T>(r * cos(theta), r * sin(theta), 0.0f);
    }

    // Device code for Vector3_t types
    template<typename... Args>
    HOST DEVICE void operator()(size_t i, Args... args) const {
        // The template system will pass the extracted pointers as args
        // For this kernel: args should be (Vector3_t<T>* output)
        auto tuple_args = std::make_tuple(args...);
        auto* output = std::get<0>(tuple_args);
        
        if (i >= output_size) return;
        
        // Create a fresh Philox instance with deterministic parameters
        openrand::Philox rng(base_seed, base_ctr + static_cast<uint32_t>(i), global_seed);
        
        auto u1 = rng.draw_float4();
        auto u2 = rng.draw_float4();
        
        // Generate three Gaussian values using Box-Muller (needs 3 uniform values)
        float u1_x = (u1.x < 1e-7f) ? 1e-7f : u1.x;
        float u2_x = u1.y;
        float u1_y = (u1.z < 1e-7f) ? 1e-7f : u1.z;  
        float u2_y = u1.w;
        float u1_z = (u2.x < 1e-7f) ? 1e-7f : u2.x;
        float u2_z = u2.y;

        ARBD::Vector3_t<T> gauss_pair1 = box_muller(u1_x, u2_x);
        ARBD::Vector3_t<T> gauss_pair2 = box_muller(u1_y, u2_y);
        ARBD::Vector3_t<T> gauss_pair3 = box_muller(u1_z, u2_z);

        output[i] = ARBD::Vector3_t<T>(
            mean.x + stddev.x * gauss_pair1.x,
            mean.y + stddev.y * gauss_pair2.x,
            mean.z + stddev.z * gauss_pair3.x
        );
    }
};
}