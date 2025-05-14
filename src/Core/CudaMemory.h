// CudaMemory.h
#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <memory>
#include <type_traits>
#include "ARBDException.h"

namespace cuda {

/**
 * @brief Checks CUDA error and throws exception if error occurred
 */
inline void check_error(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        throw _ARBDException(std::string(file) + ":" + std::to_string(line), 
                           CUDARuntimeError, 
                           "CUDA error: %s", cudaGetErrorString(error));
    }
}

#define CUDA_CHECK(call) cuda::check_error(call, __FILE__, __LINE__)

/**
 * @brief RAII wrapper for CUDA device memory
 */
template<typename T>
class DeviceMemory {
public:
    DeviceMemory() : ptr_(nullptr), size_(0) {}
    
    explicit DeviceMemory(size_t count) : size_(count) {
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
        } else {
            ptr_ = nullptr;
        }
    }

    ~DeviceMemory() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
    }

    // Disallow copying
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
    
    // Allow moving
    DeviceMemory(DeviceMemory&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    DeviceMemory& operator=(DeviceMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Copy host data to device
    void copyFromHost(const T* host_data, size_t count = 0) {
        if (count == 0) count = size_;
        if (count > size_) {
            throw _ARBDException("", ValueError, "Tried to copy more data than allocated");
        }
        if (!ptr_ || !host_data) return;
        CUDA_CHECK(cudaMemcpy(ptr_, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    // Copy device data to host
    void copyToHost(T* host_data, size_t count = 0) const {
        if (count == 0) count = size_;
        if (count > size_) {
            throw _ARBDException("", ValueError, "Tried to copy more data than allocated");
        }
        if (!ptr_ || !host_data) return;
        CUDA_CHECK(cudaMemcpy(host_data, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    // Accessors
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }
    
    // Conversion operator
    operator T*() { return ptr_; }
    operator const T*() const { return ptr_; }

private:
    T* ptr_;
    size_t size_;
};

/**
 * @brief RAII wrapper for CUDA streams
 */
class Stream {
    public:
        Stream() : stream_(nullptr) {
            CUDA_CHECK(cudaStreamCreate(&stream_));
        }
        
        explicit Stream(unsigned int flags) : stream_(nullptr) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
        }
        
        ~Stream() {
            if (stream_) {
                cudaStreamDestroy(stream_);
                stream_ = nullptr;
            }
        }
        
        // Disallow copying
        Stream(const Stream&) = delete;
        Stream& operator=(const Stream&) = delete;
        
        // Allow moving
        Stream(Stream&& other) noexcept : stream_(other.stream_) {
            other.stream_ = nullptr;
        }
        
        Stream& operator=(Stream&& other) noexcept {
            if (this != &other) {
                if (stream_) {
                    cudaStreamDestroy(stream_);
                }
                stream_ = other.stream_;
                other.stream_ = nullptr;
            }
            return *this;
        }
        
        // Synchronize on this stream
        void synchronize() {
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
        
        // Accessors
        cudaStream_t get() { return stream_; }
        operator cudaStream_t() { return stream_; }
    
    private:
        cudaStream_t stream_;
    };
    
    /**
     * @brief RAII wrapper for CUDA events
     */
class Event {
    public:
        Event() : event_(nullptr) {
            CUDA_CHECK(cudaEventCreate(&event_));
        }
        
        explicit Event(unsigned int flags) : event_(nullptr) {
            CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
        }
        
        ~Event() {
            if (event_) {
                cudaEventDestroy(event_);
                event_ = nullptr;
            }
        }
        
        // Disallow copying
        Event(const Event&) = delete;
        Event& operator=(const Event&) = delete;
        
        // Allow moving
        Event(Event&& other) noexcept : event_(other.event_) {
            other.event_ = nullptr;
        }
        
        Event& operator=(Event&& other) noexcept {
            if (this != &other) {
                if (event_) {
                    cudaEventDestroy(event_);
                }
                event_ = other.event_;
                other.event_ = nullptr;
            }
            return *this;
        }
        
        // Record this event on the given stream
        void record(cudaStream_t stream = nullptr) {
            CUDA_CHECK(cudaEventRecord(event_, stream));
        }
        
        // Synchronize on this event
        void synchronize() {
            CUDA_CHECK(cudaEventSynchronize(event_));
        }
        
        // Check if event is completed
        bool query() {
            cudaError_t result = cudaEventQuery(event_);
            if (result == cudaSuccess) {
                return true;
            } else if (result == cudaErrorNotReady) {
                return false;
            } else {
                CUDA_CHECK(result);
                return false;
            }
        }
        
        // Compute elapsed time between events
        float elapsed(const Event& start) const {
            float time;
            CUDA_CHECK(cudaEventElapsedTime(&time, start.event_, event_));
            return time;
        }
        
        // Accessors
        cudaEvent_t get() { return event_; }
        operator cudaEvent_t() { return event_; }
    
    private:
        cudaEvent_t event_;
    };
    
} // namespace cuda
#endif // USE_CUDA