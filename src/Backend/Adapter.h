#pragma once

#include "Backend/Proxy.h"
#include "Backend/Resource.h"
#include <tuple>
#include <vector>
#include <memory>

namespace ARBD {
namespace Backend {

// ============================================================================
// Type-Agnostic DeviceBuffer (Backend Layer)
// ============================================================================

/**
 * @brief Generic DeviceBuffer that works with any type T
 * 
 */
template<typename T>
class DeviceBuffer {
public:
    using value_type = T;
    
    // Constructors
    DeviceBuffer() : size_(0), device_ptr_(nullptr) {}
    
    explicit DeviceBuffer(size_t count, const Resource& resource) 
        : size_(count), resource_(resource), device_ptr_(nullptr) {
        if (count > 0) {
            allocate_device_memory();
        }
    }
    
    // Move semantics
    DeviceBuffer(DeviceBuffer&& other) noexcept 
        : size_(other.size_), resource_(other.resource_), 
          device_ptr_(other.device_ptr_), proxy_(std::move(other.proxy_)) {
        other.device_ptr_ = nullptr;
        other.size_ = 0;
    }
    
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            deallocate_device_memory();
            size_ = other.size_;
            resource_ = other.resource_;
            device_ptr_ = other.device_ptr_;
            proxy_ = std::move(other.proxy_);
            other.device_ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    // No copy semantics (move-only for safety)
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    ~DeviceBuffer() {
        deallocate_device_memory();
    }
    
    // Shamrock-style access patterns (type-agnostic)
    T* get_write_access(EventList& depends_list) {
        // Generic device pointer access
        return device_ptr_;
    }
    
    const T* get_read_access(EventList& depends_list) const {
        return device_ptr_;
    }
    
    void complete_event_state(const Event& e) {
        // Generic event completion
        last_event_ = e;
    }
    
    // Generic buffer operations
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    const Resource& get_resource() const { return resource_; }
    
    // Copy data from host (type-agnostic)
    void copy_from_host(const T* host_data, size_t count) {
        if (count > size_) {
            throw std::runtime_error("Copy size exceeds buffer size");
        }
        // Implementation would use appropriate backend (CUDA memcpy, SYCL copy, etc.)
        copy_host_to_device(host_data, device_ptr_, count * sizeof(T));
    }
    
    // Copy data to host (type-agnostic)
    void copy_to_host(T* host_data, size_t count) const {
        if (count > size_) {
            throw std::runtime_error("Copy size exceeds buffer size");
        }
        // Implementation would use appropriate backend
        copy_device_to_host(device_ptr_, host_data, count * sizeof(T));
    }

private:
    size_t size_;
    Resource resource_;
    T* device_ptr_;                    // Generic device pointer
    std::optional<Proxy<void>> proxy_; // Optional proxy for advanced features
    std::optional<Event> last_event_;  // Track last event for synchronization
    
    void allocate_device_memory() {
        // Generic device allocation based on Resource type
        switch (resource_.type) {
            case Resource::CUDA:
                device_ptr_ = allocate_cuda_memory<T>(size_);
                break;
            case Resource::SYCL:
                device_ptr_ = allocate_sycl_memory<T>(size_, resource_);
                break;
            case Resource::METAL:
                device_ptr_ = allocate_metal_memory<T>(size_);
                break;
            default:
                throw std::runtime_error("Unsupported resource type");
        }
    }
    
    void deallocate_device_memory() {
        if (device_ptr_) {
            switch (resource_.type) {
                case Resource::CUDA:
                    free_cuda_memory(device_ptr_);
                    break;
                case Resource::SYCL:
                    free_sycl_memory(device_ptr_, resource_);
                    break;
                case Resource::METAL:
                    free_metal_memory(device_ptr_);
                    break;
            }
            device_ptr_ = nullptr;
        }
    }
    
    // Backend-specific allocation functions (to be implemented)
    template<typename U> U* allocate_cuda_memory(size_t count);
    template<typename U> U* allocate_sycl_memory(size_t count, const Resource& resource);
    template<typename U> U* allocate_metal_memory(size_t count);
    
    void free_cuda_memory(void* ptr);
    void free_sycl_memory(void* ptr, const Resource& resource);
    void free_metal_memory(void* ptr);
    
    void copy_host_to_device(const void* host, void* device, size_t bytes);
    void copy_device_to_host(const void* device, void* host, size_t bytes);
};

// ============================================================================
// Type-Agnostic Kernel Launch Infrastructure
// ============================================================================

/**
 * @brief Generic event list for dependency tracking
 */
class EventList {
public:
    void add_dependency(const Event& e) {
        dependencies_.push_back(e);
    }
    
    void clear() {
        dependencies_.clear();
        consumed = false;
    }
    
    bool consumed = false;
    std::vector<Event> dependencies_;
};

/**
 * @brief Generic multi-reference adapter
 * Works with any type that provides get_read_access/get_write_access
 */
template<typename... Args>
struct MultiRef {
    std::tuple<Args&...> storage;
    
    MultiRef(Args&... args) : storage(args...) {}
    
    auto get_read_access(EventList& depends_list) {
        return std::apply([&](auto&... args) {
            return std::make_tuple(args.get_read_access(depends_list)...);
        }, storage);
    }
    
    auto get_write_access(EventList& depends_list) {
        return std::apply([&](auto&... args) {
            return std::make_tuple(args.get_write_access(depends_list)...);
        }, storage);
    }
    
    void complete_event_state(const Event& e) {
        std::apply([&](auto&... args) {
            (args.complete_event_state(e), ...);
        }, storage);
    }
};

/**
 * @brief Generic kernel launch function - no type dependencies
 * 
 * This is purely about kernel execution infrastructure,
 * not about specific mathematical types.
 */
template<typename RefIn, typename RefOut, typename Functor, typename... Args>
void kernel_call(const Resource& resource,
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
 * @brief Generic kernel dispatcher - implementation detail
 */
template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
Event dispatch_kernel(const Resource& resource,
                     size_t thread_count,
                     const InputTuple& inputs,
                     const OutputTuple& outputs,
                     Functor&& kernel_func,
                     Args... args) {
    
    switch(resource.type) {
        case Resource::CUDA:
            return launch_cuda_kernel(resource, thread_count, inputs, outputs,
                                    std::forward<Functor>(kernel_func),
                                    std::forward<Args>(args)...);
        case Resource::SYCL:
            return launch_sycl_kernel(resource, thread_count, inputs, outputs,
                                    std::forward<Functor>(kernel_func),
                                    std::forward<Args>(args)...);
        case Resource::METAL:
            return launch_metal_kernel(resource, thread_count, inputs, outputs,
                                     std::forward<Functor>(kernel_func),
                                     std::forward<Args>(args)...);
        default:
            throw std::runtime_error("Unsupported backend for kernel launch");
    }
}

// ============================================================================
// Generic Memory Management Infrastructure
// ============================================================================

/**
 * @brief Generic allocato
 */
template<typename T>
class GenericAllocator {
public:
    static DeviceBuffer<T> allocate(size_t count, const Resource& resource) {
        return DeviceBuffer<T>(count, resource);
    }
    
    static DeviceBuffer<T> allocate_zeroed(size_t count, const Resource& resource) {
        DeviceBuffer<T> buffer(count, resource);
        zero_device_memory(buffer, resource);
        return buffer;
    }
    
    template<typename InitFunc>
    static DeviceBuffer<T> allocate_initialized(size_t count, 
                                               const Resource& resource,
                                               InitFunc&& init_func) {
        DeviceBuffer<T> buffer(count, resource);
        
        // Use kernel_call to initialize - completely generic
        EventList dummy_deps;
        auto output_ptr = buffer.get_write_access(dummy_deps);
        
        // Launch initialization kernel
        kernel_call(resource,
                   MultiRef{},  // No inputs
                   MultiRef{buffer},
                   count,
                   [init_func = std::forward<InitFunc>(init_func)]
                   (size_t i, T* output) {
                       init_func(i, output);
                   });
        
        return buffer;
    }

private:
    static void zero_device_memory(DeviceBuffer<T>& buffer, const Resource& resource) {
        // Backend-specific memory zeroing
        switch(resource.type) {
            case Resource::CUDA:
                zero_cuda_memory(buffer.get_write_access(EventList{}), 
                               buffer.size() * sizeof(T));
                break;
            case Resource::SYCL:
                zero_sycl_memory(buffer.get_write_access(EventList{}), 
                               buffer.size() * sizeof(T), resource);
                break;
            case Resource::METAL:
                zero_metal_memory(buffer.get_write_access(EventList{}), 
                                buffer.size() * sizeof(T));
                break;
        }
    }
    
    // Backend-specific implementations (to be implemented)
    static void zero_cuda_memory(void* ptr, size_t bytes);
    static void zero_sycl_memory(void* ptr, size_t bytes, const Resource& resource);
    static void zero_metal_memory(void* ptr, size_t bytes);
};

// ============================================================================
// Generic Utility Functions
// ============================================================================

/**
 * @brief Copy data between buffers - type-agnostic
 */
template<typename T>
void copy_buffer(const DeviceBuffer<T>& source,
                DeviceBuffer<T>& destination,
                const Resource& resource) {
    if (source.size() != destination.size()) {
        throw std::runtime_error("Buffer size mismatch in copy operation");
    }
    
    kernel_call(resource,
               MultiRef{source},
               MultiRef{destination},
               source.size(),
               [](size_t i, const T* src, T* dst) {
                   dst[i] = src[i];
               });
}

/**
 * @brief Fill buffer with value - type-agnostic
 */
template<typename T>
void fill_buffer(DeviceBuffer<T>& buffer,
                const T& value,
                const Resource& resource) {
    kernel_call(resource,
               MultiRef{},  // No inputs
               MultiRef{buffer},
               buffer.size(),
               [value](size_t i, T* output) {
                   output[i] = value;
               });
}



} // namespace Backend
} // namespace ARBD