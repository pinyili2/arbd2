#pragma once

#include <tuple>
#include <vector>
#include <memory>
#include <functional>
#include <future>
#ifdef USE_CUDA
#include "CUDA/CUDAManager.h"
#include <cuda_runtime.h>
#endif

#ifdef USE_SYCL
#include "SYCL/SYCLManager.h"
#include <sycl/sycl.hpp>
#endif

#ifdef USE_METAL
#include "METAL/METALManager.h"
#endif
#include "ARBDLogger.h"
#include "ARBDException.h"
#include "Resource.h"
#include "Buffer.h"
#include "Events.h"

namespace ARBD {
namespace Kernels {

// ============================================================================
// Kernel Launch Configuration
// ============================================================================

struct LaunchConfig {
    size_t grid_size = 1;
    size_t block_size = 256;
    size_t shared_memory = 0;
    bool async = false;
};

// Enhanced kernel configuration with dependencies
struct KernelConfig {
    size_t grid_size = 0;      // 0 means auto-calculate
    size_t block_size = 256;   // Default block size
    size_t shared_mem = 0;     // Shared memory size
    bool async = false;        // Async execution
    EventList dependencies;    // Event dependencies
};

// Generic kernel function signature
template<typename... Args>
using KernelFunction = std::function<void(size_t, Args...)>;

// ============================================================================
// Backend-Specific Kernel Wrappers (only when backends are available)
// ============================================================================

#ifdef USE_CUDA
template<typename Kernel, typename... Args>
__global__ void cuda_kernel_wrapper(size_t n, Kernel kernel, Args... args) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        kernel(i, args...);
    }
}
#endif

// ============================================================================
// Kernel Dispatchers (using ifdefs instead of switch statements)
// ============================================================================

/**
 * @brief Generic kernel dispatcher - implementation detail from Adapter.h
 */
template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
Event dispatch_kernel(const Resource& resource,
                     size_t thread_count,
                     const InputTuple& inputs,
                     const OutputTuple& outputs,
                     Functor&& kernel_func,
                     Args... args) {
    
    if (resource.is_device()) {
        // Try each available backend implementation
#ifdef USE_CUDA
        return launch_cuda_kernel(resource, thread_count, inputs, outputs,
                                std::forward<Functor>(kernel_func),
                                std::forward<Args>(args)...);
#elif defined(USE_SYCL)
        return launch_sycl_kernel(resource, thread_count, inputs, outputs,
                                std::forward<Functor>(kernel_func),
                                std::forward<Args>(args)...);
#elif defined(USE_METAL)
        return launch_metal_kernel(resource, thread_count, inputs, outputs,
                                 std::forward<Functor>(kernel_func),
                                 std::forward<Args>(args)...);
#else
        // CPU fallback for device resource when no device backend available
        return launch_cpu_kernel(resource, thread_count, inputs, outputs,
                                std::forward<Functor>(kernel_func),
                                std::forward<Args>(args)...);
#endif
    } else {
        // Host execution
        return launch_cpu_kernel(resource, thread_count, inputs, outputs,
                               std::forward<Functor>(kernel_func),
                               std::forward<Args>(args)...);
    }
}

// ============================================================================
// Backend-Specific Kernel Launchers (only when backends are available)
// ============================================================================

#ifdef USE_CUDA
template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
Event launch_cuda_kernel(const Resource& resource,
                         size_t thread_count,
                         const InputTuple& inputs,
                         const OutputTuple& outputs,
                         Functor&& kernel_func,
                         Args... args) {
    
    size_t block_size = 256;
    size_t grid_size = (thread_count + block_size - 1) / block_size;
    
    dim3 grid(grid_size);
    dim3 block(block_size);
    
    cudaEvent_t event;
    cudaEventCreate(&event);
    
    // Launch kernel with all arguments
    auto all_args = std::tuple_cat(inputs, outputs);
    std::apply([&](auto*... ptrs) {
        cuda_kernel_wrapper<<<grid, block>>>(
            thread_count, kernel_func, ptrs..., args...);
    }, all_args);
    
    cudaEventRecord(event);
    return Event(event, resource);
}
#endif

#ifdef USE_SYCL
template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
Event launch_sycl_kernel(const Resource& resource,
                         size_t thread_count,
                         const InputTuple& inputs,
                         const OutputTuple& outputs,
                         Functor&& kernel_func,
                         Args... args) {
    
    auto& device = SYCL::SYCLManager::get_current_device();
    auto& queue = device.get_next_queue();
    
    auto all_args = std::tuple_cat(inputs, outputs);
    auto event = std::apply([&](auto*... ptrs) {
        return queue.get().parallel_for(
            sycl::range<1>(thread_count),
            [=](sycl::id<1> idx) {
                kernel_func(idx[0], ptrs..., args...);
            }
        );
    }, all_args);
    
    return Event(event, resource);
}
#endif

#ifdef USE_METAL
template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
Event launch_metal_kernel(const Resource& resource,
                          size_t thread_count,
                          const InputTuple& inputs,
                          const OutputTuple& outputs,
                          Functor&& kernel_func,
                          Args... args) {
    
    // Metal kernel launch would go here
    // For now, CPU fallback
    auto all_args = std::tuple_cat(inputs, outputs);
    std::apply([&](auto*... ptrs) {
        for (size_t i = 0; i < thread_count; ++i) {
            kernel_func(i, ptrs..., args...);
        }
    }, all_args);
    
    return Event(nullptr, resource);
}
#endif

// CPU fallback implementation (always available)
template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
Event launch_cpu_kernel(const Resource& resource,
                        size_t thread_count,
                        const InputTuple& inputs,
                        const OutputTuple& outputs,
                        Functor&& kernel_func,
                        Args... args) {
    
    auto all_args = std::tuple_cat(inputs, outputs);
    std::apply([&](auto*... ptrs) {
        for (size_t i = 0; i < thread_count; ++i) {
            kernel_func(i, ptrs..., args...);
        }
    }, all_args);
    
    return Event(nullptr, resource);
}

// ============================================================================
// Unified Kernel Launch Interface
// ============================================================================

/**
 * @brief Unified kernel launcher - the main abstraction (enhanced)
 */
template<typename InputRefs, typename OutputRefs, typename KernelFunc>
Event kernel_call(const Resource& resource,
                 InputRefs&& inputs,
                 OutputRefs&& outputs,
                 size_t n,
                 KernelFunc&& kernel,
                 const LaunchConfig& config = {}) {
    
    EventList deps;
    
    // Get pointers from MultiRef
    auto input_ptrs = inputs.get_pointers(deps);
    auto output_ptrs = outputs.get_pointers(deps);
    
    // Merge tuples and launch kernel
    auto all_ptrs = std::tuple_cat(input_ptrs, output_ptrs);
    
    return std::apply([&](auto*... ptrs) {
        return launch_kernel_impl(resource, n, kernel, config, ptrs...);
    }, all_ptrs);
}

/**
 * @brief Generic kernel launch function from Adapter.h - no type dependencies
 */
template<typename RefIn, typename RefOut, typename Functor, typename... Args>
void kernel_call_generic(const Resource& resource,
                        RefIn input_refs, 
                        RefOut output_refs, 
                        size_t thread_count, 
                        Functor&& kernel_func,
                        Args... args) {
    
    EventList depends_list;
    
    // Get accessors - completely generic
    auto input_ptrs = input_refs.get_read_access(depends_list);
    auto output_ptrs = output_refs.get_write_access(depends_list);
    
    // Backend dispatch - no type knowledge needed
    Event completion_event = dispatch_kernel(resource, thread_count, 
                                           input_ptrs, output_ptrs,
                                           std::forward<Functor>(kernel_func),
                                           std::forward<Args>(args)...);
    
    // Complete event state
    input_refs.complete_event_state(completion_event);
    output_refs.complete_event_state(completion_event);
}

/**
 * @brief Simplified kernel launch function that works with Buffer.h types
 */
template<typename KernelFunc, typename... Buffers>
Event simple_kernel(const Resource& resource,
                    size_t num_elements,
                    KernelFunc&& kernel,
                    Buffers&... buffers) {
    
    EventList deps;
    auto buffer_ptrs = std::make_tuple(buffers.get_write_access(deps)...);
    
    return std::apply([&](auto*... ptrs) {
        return launch_kernel_impl(resource, num_elements, kernel, LaunchConfig{}, ptrs...);
    }, buffer_ptrs);
}

// Backend-specific kernel launch implementations (using ifdefs)
template<typename KernelFunc, typename... Args>
Event launch_kernel_impl(const Resource& resource,
                        size_t n,
                        KernelFunc&& kernel,
                        const LaunchConfig& config,
                        Args*... args) {
    
    if (resource.is_device()) {
        // Try each available device backend
#ifdef USE_CUDA
        dim3 grid((n + config.block_size - 1) / config.block_size);
        dim3 block(config.block_size);
        
        cudaEvent_t event;
        cudaEventCreate(&event);
        
        cuda_kernel_wrapper<<<grid, block, config.shared_memory>>>(
            n, kernel, args...);
        
        cudaEventRecord(event);
        
        if (!config.async) {
            cudaEventSynchronize(event);
        }
        
        return Event(event, resource);
#elif defined(USE_SYCL)
        auto& device = SYCL::SYCLManager::get_current_device();
        auto& queue = device.get_next_queue();
        
        auto event = queue.get().parallel_for(
            sycl::range<1>(n),
            [=](sycl::id<1> idx) {
                kernel(idx[0], args...);
            }
        );
        
        if (!config.async) {
            event.wait();
        }
        
        return Event(event, resource);
#elif defined(USE_METAL)
        // Metal kernel launch would go here
        // For now, CPU fallback
        for (size_t i = 0; i < n; ++i) {
            kernel(i, args...);
        }
        return Event(nullptr, resource);
#else
        // CPU fallback for device resource when no device backend available
        for (size_t i = 0; i < n; ++i) {
            kernel(i, args...);
        }
        return Event(nullptr, resource);
#endif
    } else {
        // Host execution
        for (size_t i = 0; i < n; ++i) {
            kernel(i, args...);
        }
        return Event(nullptr, resource);
    }
}

// ============================================================================
// Kernel Chain for Sequential Execution
// ============================================================================

class KernelChain {
private:
    Resource resource_;
    EventList events_;
    
public:
    explicit KernelChain(const Resource& resource) : resource_(resource) {}
    
    template<typename KernelFunc, typename... Buffers>
    KernelChain& then(size_t num_elements, KernelFunc&& kernel, Buffers&... buffers) {
        KernelConfig config;
        config.dependencies = events_;
        config.async = true;
        
        auto event = simple_kernel(resource_, num_elements,
                                 std::forward<KernelFunc>(kernel), buffers...);
        events_.clear();
        events_.add(event);
        
        return *this;
    }
    
    void wait() {
        events_.wait_all();
    }
    
    EventList get_events() const {
        return events_;
    }
};

// ============================================================================
// Type-Specific Allocators
// ============================================================================

template<typename T>
class TypedAllocator {
public:
    static DeviceBuffer<T> allocate(size_t count, const Resource& resource) {
        return DeviceBuffer<T>(count, resource);
    }
    
    static DeviceBuffer<T> allocate_zeroed(size_t count, const Resource& resource) {
        DeviceBuffer<T> buffer(count, resource);
        
        // Zero initialization kernel
        kernel_call(resource,
                   MultiRef<>{},  // No inputs
                   make_multi_ref(buffer),
                   count,
                   [](size_t i, T* output) {
                       output[i] = T{};
                   });
        
        return buffer;
    }
    
    template<typename InitFunc>
    static DeviceBuffer<T> allocate_initialized(size_t count, 
                                               const Resource& resource,
                                               InitFunc&& init_func) {
        DeviceBuffer<T> buffer(count, resource);
        
        // Custom initialization kernel
        kernel_call(resource,
                   MultiRef<>{},  // No inputs
                   make_multi_ref(buffer),
                   count,
                   [init_func](size_t i, T* output) {
                       output[i] = init_func(i);
                   });
        
        return buffer;
    }
    
    // Specialized allocator for common patterns
    static DeviceBuffer<T> allocate_sequence(size_t count, 
                                            const Resource& resource,
                                            T start = T{0},
                                            T step = T{1}) {
        return allocate_initialized(count, resource,
            [start, step](size_t i) { return start + static_cast<T>(i) * step; });
    }
};

// Convenience aliases
template<typename T>
using Allocator = TypedAllocator<T>;

// ============================================================================
// Utility Functions
// ============================================================================

// Copy between buffers
template<typename T>
Event copy_async(const DeviceBuffer<T>& source,
                DeviceBuffer<T>& destination,
                const Resource& resource) {
    if (source.size() != destination.size()) {
        throw std::runtime_error("Buffer size mismatch in copy");
    }
    
    return kernel_call(resource,
                      make_multi_ref(source),
                      make_multi_ref(destination),
                      source.size(),
                      [](size_t i, const T* src, T* dst) {
                          dst[i] = src[i];
                      },
                      {.async = true});
}

// Fill buffer with value
template<typename T>
Event fill_async(DeviceBuffer<T>& buffer,
                const T& value,
                const Resource& resource) {
    return kernel_call(resource,
                      MultiRef<>{},
                      make_multi_ref(buffer),
                      buffer.size(),
                      [value](size_t i, T* output) {
                          output[i] = value;
                      },
                      {.async = true});
}

// ============================================================================
// Result Wrapper for Kernel Calls
// ============================================================================

template<typename T>
struct KernelResult {
    T result;
    Event completion_event;
    
    KernelResult(T&& res, Event&& event) 
        : result(std::forward<T>(res)), completion_event(std::move(event)) {}
    
    void wait() { completion_event.wait(); }
    bool is_ready() const { return completion_event.is_complete(); }
    
    T get() {
        wait();
        return std::move(result);
    }
};

// ============================================================================
// Kernel Execution Policies
// ============================================================================

struct ExecutionPolicy {
    enum class Type {
        SEQUENTIAL,
        PARALLEL,
        ASYNC
    };
    
    Type type = Type::PARALLEL;
    size_t preferred_block_size = 256;
    bool use_shared_memory = false;
    size_t shared_memory_size = 0;
};

} // namespace Kernels
} // namespace ARBD
