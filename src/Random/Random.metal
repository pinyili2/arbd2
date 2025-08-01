using namespace ARBD;
using namespace metal;
#include <metal_stdlib>
#include "RandomKernels.h"
#include "Math/Types.h"

// Box-Muller transform for Gaussian distribution
float2 box_muller(float u1, float u2) {
    float r = sqrt(-2.0f * log(max(u1, 1e-7f)));
    float theta = 2.0f * M_PI_F * u2;
    return float2(r * cos(theta), r * sin(theta));
}

// Uniform float generation
kernel void generate_uniform_float(device float* output [[buffer(0)]],
                                 constant uint& count [[buffer(1)]],
                                 constant uint64_t& seed [[buffer(2)]],
                                 constant uint64_t& offset [[buffer(3)]],
                                 constant float& min_val [[buffer(4)]],
                                 constant float& max_val [[buffer(5)]],
                                 uint thread_id [[thread_position_in_grid]]) {
    if (thread_id >= count) return;
    
    PhiloxGenerator rng(seed, thread_id, offset);
    float u = rng.next_float();
    output[thread_id] = min_val + u * (max_val - min_val);
}

// Gaussian float generation
kernel void generate_gaussian_float(device float* output [[buffer(0)]],
                                  constant uint& count [[buffer(1)]],
                                  constant uint64_t& seed [[buffer(2)]],
                                  constant uint64_t& offset [[buffer(3)]],
                                  constant float& mean [[buffer(4)]],
                                  constant float& stddev [[buffer(5)]],
                                  uint thread_id [[thread_position_in_grid]]) {
    uint base_id = thread_id * 2;
    if (base_id >= count) return;
    
    PhiloxGenerator rng(seed, thread_id, offset);
    float2 uniform_vals = rng.next_float2();
    float2 gaussian_vals = box_muller(uniform_vals.x, uniform_vals.y);
    
    output[base_id] = mean + stddev * gaussian_vals.x;
    if (base_id + 1 < count) {
        output[base_id + 1] = mean + stddev * gaussian_vals.y;
    }
}


// Gaussian Vector3 generation
kernel void generate_gaussian_vector3(device Vector3* output [[buffer(0)]],
                                    constant uint& count [[buffer(1)]],
                                    constant uint64_t& seed [[buffer(2)]],
                                    constant uint64_t& offset [[buffer(3)]],
                                    constant float& mean [[buffer(4)]],
                                    constant float& stddev [[buffer(5)]],
                                    uint thread_id [[thread_position_in_grid]]) {
    if (thread_id >= count) return;
    
    PhiloxGenerator rng(seed, thread_id, offset);
    
    // Generate 3 Gaussian values using 2 Box-Muller transforms
    float2 uniform_vals1 = rng.next_float2();
    float2 uniform_vals2 = rng.next_float2();
    
    float2 gaussian_pair1 = box_muller(uniform_vals1.x, uniform_vals1.y);
    float2 gaussian_pair2 = box_muller(uniform_vals2.x, uniform_vals2.y);
    
    output[thread_id] = Vector3{
        mean + stddev * gaussian_pair1.x,
        mean + stddev * gaussian_pair1.y,
        mean + stddev * gaussian_pair2.x
    };
}

// Uniform double generation
kernel void generate_uniform_double(device double* output [[buffer(0)]],
                                  constant uint& count [[buffer(1)]],
                                  constant uint64_t& seed [[buffer(2)]],
                                  constant uint64_t& offset [[buffer(3)]],
                                  constant double& min_val [[buffer(4)]],
                                  constant double& max_val [[buffer(5)]],
                                  uint thread_id [[thread_position_in_grid]]) {
    if (thread_id >= count) return;
    
    PhiloxGenerator rng(seed, thread_id, offset);
    double u = rng.next_double();
    output[thread_id] = min_val + u * (max_val - min_val);
}

// Gaussian double generation
kernel void generate_gaussian_double(device double* output [[buffer(0)]],
                                   constant uint& count [[buffer(1)]],
                                   constant uint64_t& seed [[buffer(2)]],
                                   constant uint64_t& offset [[buffer(3)]],
                                   constant double& mean [[buffer(4)]],
                                   constant double& stddev [[buffer(5)]],
                                   uint thread_id [[thread_position_in_grid]]) {
    uint base_id = thread_id * 2;
    if (base_id >= count) return;
    
    PhiloxGenerator rng(seed, thread_id, offset);
    float2 uniform_vals = rng.next_float2();
    float2 gaussian_vals = box_muller(uniform_vals.x, uniform_vals.y);
    
    output[base_id] = double(mean) + double(stddev) * double(gaussian_vals.x);
    if (base_id + 1 < count) {
        output[base_id + 1] = double(mean) + double(stddev) * double(gaussian_vals.y);
    }
}

// Uniform int generation
kernel void generate_uniform_int(device int* output [[buffer(0)]],
                               constant uint& count [[buffer(1)]],
                               constant uint64_t& seed [[buffer(2)]],
                               constant uint64_t& offset [[buffer(3)]],
                               constant int& min_val [[buffer(4)]],
                               constant int& max_val [[buffer(5)]],
                               uint thread_id [[thread_position_in_grid]]) {
    if (thread_id >= count) return;
    
    PhiloxGenerator rng(seed, thread_id, offset);
    uint range = uint(max_val - min_val + 1);
    
    if (range == 0) {
        output[thread_id] = min_val;
        return;
    }
    
    // Avoid modulo bias
    uint limit = UINT_MAX - (UINT_MAX % range);
    uint u32 = rng.next_uint();
    while (u32 >= limit) {
        u32 = rng.next_uint();
    }
    
    output[thread_id] = min_val + int(u32 % range);
}

// Uniform uint generation
kernel void generate_uniform_uint(device uint* output [[buffer(0)]],
                                constant uint& count [[buffer(1)]],
                                constant uint64_t& seed [[buffer(2)]],
                                constant uint64_t& offset [[buffer(3)]],
                                constant uint& min_val [[buffer(4)]],
                                constant uint& max_val [[buffer(5)]],
                                uint thread_id [[thread_position_in_grid]]) {
    if (thread_id >= count) return;
    
    PhiloxGenerator rng(seed, thread_id, offset);
    uint range = max_val - min_val + 1;
    
    if (range == 0) {
        output[thread_id] = min_val;
        return;
    }
    
    // Avoid modulo bias
    uint limit = UINT_MAX - (UINT_MAX % range);
    uint u32 = rng.next_uint();
    while (u32 >= limit) {
        u32 = rng.next_uint();
    }
    
    output[thread_id] = min_val + (u32 % range);
}