// In a new header, e.g., "Random.h"
#pragma once
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "RandomKernels.h"
#include "openrand/philox.h"

namespace ARBD {

template<typename Backend>
class Random {
private:
    const Resource& resource_;
    DeviceBuffer<openrand::Philox> states_; // Buffer to hold PRNG states on the device
    unsigned long seed_;
    size_t offset_;

public:
    explicit Random(const Resource& resource, size_t num_states)
        : resource_(resource), states_(num_states), seed_(0), offset_(0) {}

    void init(unsigned long seed, size_t offset = 0) {
        seed_ = seed;
        offset_ = offset;
    }

    // --- UNIFORM DISTRIBUTION ---
    template<typename T>
    Event generate_uniform(DeviceBuffer<T>& output, T min_val, T max_val) {
        ARBD::KernelConfig config;
        UniformFunctor<T> func{min_val, max_val};

        return launch_kernel<Backend>(
            resource_,
            output.size(),
            std::tie(),
            std::tie(output),
            config,
            func,
            states_.data()    // Pass the PRNG states buffer
        );
    }

    // --- GAUSSIAN DISTRIBUTION ---
    template<typename T>
    Event generate_gaussian(DeviceBuffer<T>& output, T mean, T stddev) {
        ARBD::KernelConfig config;
        GaussianFunctor<T> func{mean, stddev};
        
        // Launch the kernel, note we only need half as many threads
        // since each thread generates two numbers.
        return launch_kernel<Backend>(
            resource_,
            output.size() / 2,
            std::tie(),
            std::tie(output),
            config,
            func,
            states_.data()
        );
    }
};

}