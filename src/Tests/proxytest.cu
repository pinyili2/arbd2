#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_approx.hpp>
#include "../Proxy.h"
#include <cuda_runtime.h>

// Simple CUDA test class 
struct CUDATestObject {
    int value;
    __host__ __device__ CUDATestObject() : value(0) {}
    __host__ __device__ CUDATestObject(int v) : value(v) {}
    
    // Basic operation
    __host__ __device__ void increment() { value++; }

    // Required for Proxy operations
    void clear() {}
};

// Helper function to check CUDA errors
inline void checkCuda(cudaError_t result, const char* file, int line) {
    if (result != cudaSuccess) {
        FAIL(std::string(cudaGetErrorString(result)) + " at " + 
             std::string(file) + ":" + std::to_string(line));
    }
}
#define CHECK_CUDA(x) checkCuda(x, __FILE__, __LINE__)

// Basic kernel to increment value
__global__ void increment_kernel(CUDATestObject* obj) {
    if (threadIdx.x == 0) {
        obj->increment();
    }
}

TEST_CASE("Basic CUDA Operations", "[cuda]") {
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    REQUIRE(deviceCount > 0);
    CHECK_CUDA(cudaSetDevice(0));

    SECTION("Send to GPU and retrieve") {
        Resource gpu_resource{Resource::GPU, 0};
        
        // Create object on CPU
        CUDATestObject host_obj(41);
        
        // Send to GPU
        auto proxy = send(gpu_resource, host_obj);
        REQUIRE(proxy.location.type == Resource::GPU);
        REQUIRE(proxy.addr != nullptr);

        // Run kernel
        increment_kernel<<<1, 1>>>((CUDATestObject*)proxy.addr);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Copy back and verify
        CUDATestObject result;
        CHECK_CUDA(cudaMemcpy(&result, proxy.addr, sizeof(CUDATestObject), 
                             cudaMemcpyDeviceToHost));
        REQUIRE(result.value == 42);
    }
}

TEST_CASE("GPU Memory Management", "[cuda]") {
    CHECK_CUDA(cudaSetDevice(0));
    Resource gpu_resource{Resource::GPU, 0};

    SECTION("Move semantics") {
        CUDATestObject host_obj(42);
        auto proxy1 = send(gpu_resource, host_obj);
        void* original_addr = proxy1.addr;

        // Move construction
        Proxy<CUDATestObject> proxy2(std::move(proxy1));
        REQUIRE(proxy2.addr == original_addr);
        REQUIRE(proxy1.addr == nullptr);

        // Verify data still accessible
        CUDATestObject result;
        CHECK_CUDA(cudaMemcpy(&result, proxy2.addr, sizeof(CUDATestObject), 
                             cudaMemcpyDeviceToHost));
        REQUIRE(result.value == 42);
    }
}

TEST_CASE("GPU Resource Handling", "[cuda]") {
    CHECK_CUDA(cudaSetDevice(0));
    Resource gpu_resource{Resource::GPU, 0};
    Resource cpu_resource{Resource::CPU, 0};

    SECTION("Resource transfer") {
        CUDATestObject host_obj(42);
        auto proxy1 = send(gpu_resource, host_obj);
        
        CUDATestObject verify;
        CHECK_CUDA(cudaMemcpy(&verify, proxy1.addr, sizeof(CUDATestObject), 
                             cudaMemcpyDeviceToHost));
        REQUIRE(verify.value == 42);
    }
}
