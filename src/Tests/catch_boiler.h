#pragma once

#include <iostream>
#include <cstdio>
#include <string>
#include <memory>

// Include backend-specific headers
#ifdef USE_CUDA
#include "SignalManager.h"
#include "Backend/CUDA/CUDAManager.h"
#include <cuda.h>
#include <nvfunctional>
#endif

#ifdef USE_SYCL
#include "Backend/SYCL/SYCLManager.h"
#include <sycl/sycl.hpp>
#endif

#ifdef USE_METAL
#include "Backend/METAL/METALManager.h"
#endif

// Common includes
#include "Backend/Resource.h"
#include "Backend/UnifiedBuffer.h"

#include "ARBDLogger.h"
#include "Math/TypeName.h"
#include "Math/Types.h"

// Use Catch2 v3 headers with quotes
#include "catch2/catch_test_macros.hpp"

// Macro for run_trial function - defines run_trial as an alias to run_trial function
#define DEF_RUN_TRIAL \
    using Tests::run_trial;

namespace Tests {

// =============================================================================
// Backend-specific kernel implementations
// =============================================================================

#if defined(USE_CUDA) && defined(__CUDACC__)
template<typename Op_t, typename R, typename ...T>
__global__ void cuda_op_kernel(R* result, T...args) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *result = Op_t::op(args...);
    }
}
#endif

// =============================================================================
// Unified Backend Manager
// =============================================================================

/**
 * @brief Unified backend manager for test execution across different compute backends
 */
class TestBackendManager {
public:
    enum BackendType { CUDA_BACKEND, SYCL_BACKEND, METAL_BACKEND };
    
private:
    BackendType current_backend_;
    bool initialized_ = false;
    
public:
    TestBackendManager(BackendType backend = SYCL_BACKEND) : current_backend_(backend) {
        initialize();
    }
    
    ~TestBackendManager() {
        finalize();
    }
    
    void initialize() {
        if (initialized_) return;
        
        switch (current_backend_) {
#ifdef USE_CUDA
        case CUDA_BACKEND:
            ARBD::SignalManager::manage_segfault();
            // Initialize CUDA GPU Manager
            ARBD::CUDA::CUDAManager::init();
            ARBD::CUDA::CUDAManager::load_info();
            break;
#endif
#ifdef USE_SYCL
        case SYCL_BACKEND:
            ARBD::SYCL::SYCLManager::init();
            ARBD::SYCL::SYCLManager::load_info();
            break;
#endif
#ifdef USE_METAL
        case METAL_BACKEND:
            ARBD::METAL::METALManager::init();
            ARBD::METAL::METALManager::load_info();
            break;
#endif
        default:
            throw std::runtime_error("Unsupported backend type");
        }
        initialized_ = true;
    }
    
    void finalize() {
        if (!initialized_) return;
        
        switch (current_backend_) {
#ifdef USE_CUDA
        case CUDA_BACKEND:
            // CUDA cleanup is typically automatic
            break;
#endif
#ifdef USE_SYCL
        case SYCL_BACKEND:
            ARBD::SYCL::SYCLManager::finalize();
            break;
#endif
#ifdef USE_METAL
        case METAL_BACKEND:
            ARBD::METAL::METALManager::finalize();
            break;
#endif
        }
        initialized_ = false;
    }
    
    void synchronize() {
        switch (current_backend_) {
#ifdef USE_CUDA
        case CUDA_BACKEND:
            cudaDeviceSynchronize();
            break;
#endif
#ifdef USE_SYCL
        case SYCL_BACKEND:
            ARBD::SYCL::SYCLManager::sync();
            break;
#endif
#ifdef USE_METAL
        case METAL_BACKEND:
            ARBD::METAL::METALManager::sync();
            break;
#endif
        }
    }
    
    BackendType get_backend_type() const { return current_backend_; }
    
    template<typename R>
    R* allocate_device_memory(size_t count) {
        switch (current_backend_) {
#ifdef USE_CUDA
        case CUDA_BACKEND: {
            R* ptr;
            ARBD::check_cuda_error(cudaMalloc((void**)&ptr, count * sizeof(R)), __FILE__, __LINE__);
            // Initialize device memory to zero
            ARBD::check_cuda_error(cudaMemset(ptr, 0, count * sizeof(R)), __FILE__, __LINE__);
            return ptr;
        }
#endif
#ifdef USE_SYCL
        case SYCL_BACKEND: {
            auto& queue = ARBD::SYCL::SYCLManager::get_current_queue();
            return sycl::malloc_device<R>(count, queue.get());
        }
#endif
#ifdef USE_METAL
        case METAL_BACKEND: {
            auto& device = ARBD::METAL::METALManager::get_current_device();
            // Metal uses unified memory, so we can allocate using DeviceMemory
            // For simplicity, we'll use the manager's allocate function
            // Note: This is conceptual - actual Metal allocation would be different
            return static_cast<R*>(std::malloc(count * sizeof(R)));
        }
#endif
        default:
            throw std::runtime_error("Unsupported backend for memory allocation");
        }
    }
    
    template<typename R>
    void free_device_memory(R* ptr) {
        switch (current_backend_) {
#ifdef USE_CUDA
        case CUDA_BACKEND:
            cudaFree(ptr);
            break;
#endif
#ifdef USE_SYCL
        case SYCL_BACKEND: {
            auto& queue = ARBD::SYCL::SYCLManager::get_current_queue();
            sycl::free(ptr, queue.get());
            break;
        }
#endif
#ifdef USE_METAL
        case METAL_BACKEND:
            std::free(ptr);
            break;
#endif
        }
    }
    
    template<typename R>
    void copy_to_device(R* device_ptr, const R* host_ptr, size_t count) {
        switch (current_backend_) {
#ifdef USE_CUDA
        case CUDA_BACKEND:
            ARBD::check_cuda_error(cudaMemcpy(device_ptr, host_ptr, count * sizeof(R), cudaMemcpyHostToDevice), __FILE__, __LINE__);
            break;
#endif
#ifdef USE_SYCL
        case SYCL_BACKEND: {
            auto& queue = ARBD::SYCL::SYCLManager::get_current_queue();
            queue.get().memcpy(device_ptr, host_ptr, count * sizeof(R)).wait();
            break;
        }
#endif
#ifdef USE_METAL
        case METAL_BACKEND:
            std::memcpy(device_ptr, host_ptr, count * sizeof(R));
            break;
#endif
        }
    }
    
    template<typename R>
    void copy_from_device(R* host_ptr, const R* device_ptr, size_t count) {
        switch (current_backend_) {
#ifdef USE_CUDA
        case CUDA_BACKEND:
            ARBD::check_cuda_error(cudaMemcpy(host_ptr, device_ptr, count * sizeof(R), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
            break;
#endif
#ifdef USE_SYCL
        case SYCL_BACKEND: {
            auto& queue = ARBD::SYCL::SYCLManager::get_current_queue();
            queue.get().memcpy(host_ptr, device_ptr, count * sizeof(R)).wait();
            break;
        }
#endif
#ifdef USE_METAL
        case METAL_BACKEND:
            std::memcpy(host_ptr, device_ptr, count * sizeof(R));
            break;
#endif
        }
    }
    
    template<typename Op_t, typename R, typename ...T>
    void execute_kernel(R* result_device, T...args) {
        switch (current_backend_) {
#ifdef USE_CUDA
        case CUDA_BACKEND:
#if defined(__CUDACC__)
            cuda_op_kernel<Op_t, R, T...><<<1,1>>>(result_device, args...);
            ARBD::check_cuda_error(cudaGetLastError(), __FILE__, __LINE__);
            ARBD::check_cuda_error(cudaDeviceSynchronize(), __FILE__, __LINE__);
#else
            // Fallback: execute on host when not compiled with nvcc
            *result_device = Op_t::op(args...);
#endif
            break;
#endif
#ifdef USE_SYCL
        case SYCL_BACKEND: {
            auto& queue = ARBD::SYCL::SYCLManager::get_current_queue();
            queue.submit([=](sycl::handler& h) {
                h.single_task([=]() {
                    *result_device = Op_t::op(args...);
                });
            }).wait();
            break;
        }
#endif
#ifdef USE_METAL
        case METAL_BACKEND:
            // For Metal, we'll execute on CPU for now since compute shaders
            // require more complex setup. In a full implementation, this would
            // dispatch a Metal compute shader.
            *result_device = Op_t::op(args...);
            break;
#endif
        default:
            throw std::runtime_error("Unsupported backend for kernel execution");
        }
    }
};

// =============================================================================
// Unified Test Runner
// =============================================================================

/**
 * @brief Run a test operation across different backends
 */
template<typename Op_t, typename R, typename ...T>
void run_trial(std::string name, R expected_result, T...args) {
    using namespace ARBD;
    
    INFO(name);
    
    // Test CPU execution
    R cpu_result = Op_t::op(args...);
    CAPTURE(cpu_result);
    CAPTURE(expected_result);
    REQUIRE(cpu_result == expected_result);
    
    // Test available backends
#ifdef USE_CUDA
    {
        TestBackendManager cuda_manager(TestBackendManager::CUDA_BACKEND);
        
        R* gpu_result_d = cuda_manager.allocate_device_memory<R>(1);
        
        cuda_manager.execute_kernel<Op_t, R, T...>(gpu_result_d, args...);
        
        R gpu_result;
        cuda_manager.copy_from_device(&gpu_result, gpu_result_d, 1);
        cuda_manager.synchronize();
        
        cuda_manager.free_device_memory(gpu_result_d);
        
        CAPTURE(gpu_result);
        CHECK(cpu_result == gpu_result);
    }
#endif

#ifdef USE_SYCL
    {
        TestBackendManager sycl_manager(TestBackendManager::SYCL_BACKEND);
        
        R* device_result_d = sycl_manager.allocate_device_memory<R>(1);
        
        sycl_manager.execute_kernel<Op_t, R, T...>(device_result_d, args...);
        
        R device_result;
        sycl_manager.copy_from_device(&device_result, device_result_d, 1);
        sycl_manager.synchronize();
        
        sycl_manager.free_device_memory(device_result_d);
        
        CAPTURE(device_result);
        CHECK(cpu_result == device_result);
    }
#endif

#ifdef USE_METAL
    {
        TestBackendManager metal_manager(TestBackendManager::METAL_BACKEND);
        
        R* metal_result_d = metal_manager.allocate_device_memory<R>(1);
        
        metal_manager.execute_kernel<Op_t, R, T...>(metal_result_d, args...);
        
        R metal_result;
        metal_manager.copy_from_device(&metal_result, metal_result_d, 1);
        metal_manager.synchronize();
        
        metal_manager.free_device_memory(metal_result_d);
        
        CAPTURE(metal_result);
        CHECK(cpu_result == metal_result);
    }
#endif
}

/**
 * @brief Run a test operation on a specific backend
 */
template<typename Op_t, typename R, typename ...T>
void run_trial_on_backend(TestBackendManager::BackendType backend,
                         std::string name, R expected_result, T...args) {
    using namespace ARBD;
    
    INFO(name + " [" + std::to_string(static_cast<int>(backend)) + "]");
    
    // Test CPU execution first
    R cpu_result = Op_t::op(args...);
    CAPTURE(cpu_result);
    CAPTURE(expected_result);
    REQUIRE(cpu_result == expected_result);
    
    // Test specified backend
    TestBackendManager manager(backend);
    
    R* device_result_d = manager.allocate_device_memory<R>(1);
    
    manager.execute_kernel<Op_t, R, T...>(device_result_d, args...);
    
    R device_result;
    manager.copy_from_device(&device_result, device_result_d, 1);
    manager.synchronize();
    
    manager.free_device_memory(device_result_d);
    
    CAPTURE(device_result);
    CHECK(cpu_result == device_result);
}

} // namespace Tests

// =============================================================================
// Operation definitions (unchanged from original)
// =============================================================================

namespace Tests::Unary {
    template<typename R, typename T>
    struct NegateOp { 
        HOST DEVICE static R op(T in) { return static_cast<R>(-in); } 
    };

    template<typename R, typename T>
    struct NormalizedOp { 
        HOST DEVICE static R op(T in) { return static_cast<R>(in.normalized()); } 
    };
}

namespace Tests::Binary {
    // R is return type, T and U are types of operands
    template<typename R, typename T, typename U> 
    struct AddOp { 
        HOST DEVICE static R op(T a, U b) { return static_cast<R>(a+b); } 
    };
    
    template<typename R, typename T, typename U> 
    struct SubOp { 
        HOST DEVICE static R op(T a, U b) { return static_cast<R>(a-b); } 
    };
    
    template<typename R, typename T, typename U> 
    struct MultOp { 
        HOST DEVICE static R op(T a, U b) { return static_cast<R>(a*b); } 
    };
    
    template<typename R, typename T, typename U> 
    struct DivOp { 
        HOST DEVICE static R op(T a, U b) { return static_cast<R>(a/b); } 
    };
}