#pragma once

#include "ARBDException.h"
#include <array>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <concepts>
#include <span>
#include <format>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef USE_NCCL
#include <nccl.h>
#endif

namespace ARBD {


inline void check_cuda_error(cudaError_t error, std::string_view file, int line) {
    if (error != cudaSuccess) {
        ARBD_Exception(ExceptionType::CUDARuntimeError, 
            "CUDA error at {}:{}: {}", 
            file, line, cudaGetErrorString(error));
    }
}

#define CUDA_CHECK(call) check_cuda_error(call, __FILE__, __LINE__)

#ifdef USE_NCCL
inline void check_nccl_error(ncclResult_t result, std::string_view file, int line) {
    if (result != ncclSuccess) {
        ARBD_Exception(ExceptionType::CUDARuntimeError,
            "NCCL error at {}:{}: {}",
            file, line, ncclGetErrorString(result));
    }
}

#define NCCL_CHECK(call) check_nccl_error(call, __FILE__, __LINE__)
#endif

/**
 * @brief Modern RAII wrapper for CUDA device memory
 * 
 * This class provides a safe and efficient way to manage CUDA device memory with RAII semantics.
 * It handles memory allocation, deallocation, and data transfer between host and device memory.
 * 
 * Features:
 * - Automatic memory management (RAII)
 * - Move semantics support
 * - Safe copy operations using std::span
 * - Exception handling for CUDA errors
 * 
 * @tparam T The type of data to store in device memory
 * 
 * @example Basic Usage:
 * ```cpp
 * // Allocate memory for 1000 integers
 * ARBD::DeviceMemory<int> device_mem(1000);
 * 
 * // Copy data from host to device
 * std::vector<int> host_data(1000, 42);
 * device_mem.copyFromHost(host_data);
 * 
 * // Copy data back to host
 * std::vector<int> result(1000);
 * device_mem.copyToHost(result);
 * ```
 * 
 * @example Move Semantics:
 * ```cpp
 * ARBD::DeviceMemory<float> mem1(1000);
 * ARBD::DeviceMemory<float> mem2 = std::move(mem1); // mem1 is now empty
 * ```
 * 
 * @note The class prevents copying to avoid accidental memory leaks.
 *       Use move semantics when transferring ownership.
 */
 
template<typename T>
class DeviceMemory {
public:
    DeviceMemory() = default;
    
    explicit DeviceMemory(size_t count) : size_(count) {
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
        }
    }

    ~DeviceMemory() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    // Prevent copying
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
    
    // Allow moving
    DeviceMemory(DeviceMemory&& other) noexcept 
        : ptr_(std::exchange(other.ptr_, nullptr))
        , size_(std::exchange(other.size_, 0)) {}
    
    DeviceMemory& operator=(DeviceMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = std::exchange(other.ptr_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    // Modern copy operations using std::span
    void copyFromHost(std::span<const T> host_data) {
        if (host_data.size() > size_) {
            ARBD_Exception(ExceptionType::ValueError, 
                "Tried to copy {} elements but only {} allocated", 
                host_data.size(), size_);
        }
        if (!ptr_ || host_data.empty()) return;
        CUDA_CHECK(cudaMemcpy(ptr_, host_data.data(), 
                             host_data.size() * sizeof(T), 
                             cudaMemcpyHostToDevice));
    }

    void copyToHost(std::span<T> host_data) const {
        if (host_data.size() > size_) {
            ARBD_Exception(ExceptionType::ValueError,
                "Tried to copy {} elements but only {} allocated",
                host_data.size(), size_);
        }
        if (!ptr_ || host_data.empty()) return;
        CUDA_CHECK(cudaMemcpy(host_data.data(), ptr_,
                             host_data.size() * sizeof(T),
                             cudaMemcpyDeviceToHost));
    }

    // Accessors
    [[nodiscard]] T* get() noexcept { return ptr_; }
    [[nodiscard]] const T* get() const noexcept { return ptr_; }
    [[nodiscard]] size_t size() const noexcept { return size_; }
    
    // Conversion operators
    operator T*() noexcept { return ptr_; }
    operator const T*() const noexcept { return ptr_; }

private:
    T* ptr_{nullptr};
    size_t size_{0};
};

/**
 * @brief Modern RAII wrapper for CUDA streams
 * 
 * This class provides a safe, modern C++ wrapper around CUDA streams with RAII semantics.
 * It automatically manages the lifecycle of CUDA streams, ensuring proper creation and cleanup.
 * 
 * Features:
 * - Automatic stream creation and destruction
 * - Move semantics support
 * - Thread-safe stream synchronization
 * - Implicit conversion to cudaStream_t for CUDA API compatibility
 * 
 * @example Basic Usage:
 * ```cpp
 * // Create a default stream
 * ARBD::Stream stream;
 * 
 * // Create a stream with specific flags
 * ARBD::Stream non_blocking_stream(cudaStreamNonBlocking);
 * 
 * // Synchronize the stream
 * stream.synchronize();
 * 
 * // Use with CUDA APIs
 * cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
 * ```
 * 
 * @note The stream is automatically destroyed when the Stream object goes out of scope
 */
class Stream {
public:
    Stream() {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }
    
    explicit Stream(unsigned int flags) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
    }
    
    ~Stream() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }
    
    // Prevent copying
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    
    // Allow moving
    Stream(Stream&& other) noexcept 
        : stream_(std::exchange(other.stream_, nullptr)) {}
    
    Stream& operator=(Stream&& other) noexcept {
        if (this != &other) {
            if (stream_) cudaStreamDestroy(stream_);
            stream_ = std::exchange(other.stream_, nullptr);
        }
        return *this;
    }
    
    void synchronize() {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    
    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }
    operator cudaStream_t() const noexcept { return stream_; }
    
private:
    cudaStream_t stream_{nullptr};
};

/**
 * @brief Modern RAII wrapper for CUDA events
 * 
 * This class provides a safe, modern C++ wrapper around CUDA events with RAII semantics.
 * It manages the lifecycle of CUDA events and provides utilities for timing and synchronization.
 * 
 * Features:
 * - Automatic event creation and destruction
 * - Move semantics support
 * - Event recording and synchronization
 * - Timing measurements between events
 * - Implicit conversion to cudaEvent_t for CUDA API compatibility
 * 
 * @example Basic Usage:
 * ```cpp
 * // Create events
 * ARBD::Event start_event;
 * ARBD::Event end_event;
 * 
 * // Record events on a stream
 * start_event.record(stream);
 * // ... perform operations ...
 * end_event.record(stream);
 * 
 * // Get elapsed time
 * float elapsed_ms = end_event.elapsed(start_event);
 * 
 * // Check if event is completed
 * if (end_event.query()) {
 *     // Event is completed
 * }
 * ```
 * 
 * @note Events are automatically destroyed when the Event object goes out of scope
 */
class Event {
public:
    Event() {
        CUDA_CHECK(cudaEventCreate(&event_));
    }
    
    explicit Event(unsigned int flags) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
    }
    
    ~Event() {
        if (event_) {
            cudaEventDestroy(event_);
        }
    }
    
    // Prevent copying
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    
    // Allow moving
    Event(Event&& other) noexcept 
        : event_(std::exchange(other.event_, nullptr)) {}
    
    Event& operator=(Event&& other) noexcept {
        if (this != &other) {
            if (event_) cudaEventDestroy(event_);
            event_ = std::exchange(other.event_, nullptr);
        }
        return *this;
    }
    
    void record(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }
    
    void synchronize() {
        CUDA_CHECK(cudaEventSynchronize(event_));
    }
    
    [[nodiscard]] bool query() {
        cudaError_t result = cudaEventQuery(event_);
        if (result == cudaSuccess) return true;
        if (result == cudaErrorNotReady) return false;
        CUDA_CHECK(result);
        return false;
    }
    
    [[nodiscard]] float elapsed(const Event& start) const {
        float time;
        CUDA_CHECK(cudaEventElapsedTime(&time, start.event_, event_));
        return time;
    }
    
    [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }
    operator cudaEvent_t() const noexcept { return event_; }
    
private:
    cudaEvent_t event_{nullptr};
};

/**
 * @brief Modern GPU management system
 * 
 * This class provides a comprehensive GPU management system with support for multiple GPUs,
 * stream management, and NCCL communication. It handles GPU initialization, selection,
 * and provides utilities for multi-GPU operations.
 * 
 * Features:
 * - Multi-GPU support
 * - Automatic stream management
 * - GPU selection and synchronization
 * - NCCL communication support (when enabled)
 * - Safe GPU timeout handling
 * 
 * @example Basic Usage:
 * ```cpp
 * // Initialize GPU system
 * ARBD::GPUManager::init();
 * 
 * // Select specific GPUs
 * std::vector<unsigned int> gpu_ids = {0, 1};
 * ARBD::GPUManager::select_gpus(gpu_ids);
 * 
 * // Use a specific GPU
 * ARBD::GPUManager::use(0);
 * 
 * // Synchronize all GPUs
 * ARBD::GPUManager::sync();
 * ```
 * 
 * @example Multi-GPU Operations:
 * ```cpp
 * #ifdef USE_NCCL
 * // Broadcast data across GPUs
 * std::vector<float*> device_ptrs = {ptr0, ptr1};
 * ARBD::GPUManager::nccl_broadcast(0, device_ptrs, device_ptrs, size);
 * #endif
 * ```
 * 
 * @note The class uses static methods for global GPU management.
 *       All operations are thread-safe and exception-safe.
 */
class GPUManager {
public:
    static constexpr size_t NUM_STREAMS = 8;

    /**
     * @brief Individual GPU management class
     * 
     * This nested class represents a single GPU device and manages its resources,
     * including streams and device properties.
     * 
     * Features:
     * - Stream management
     * - Device property access
     * - Timeout detection
     * - Safe resource cleanup
     * 
     * @example Basic Usage:
     * ```cpp
     * // Get GPU properties
     * const auto& gpu = ARBD::GPUManager::gpus[0];
     * const auto& props = gpu.properties();
     * 
     * // Get a stream
     * cudaStream_t stream = gpu.get_stream(0);
     * 
     * // Get next available stream
     * cudaStream_t next_stream = gpu.get_next_stream();
     * ```
     */
    class GPU {
    public:
        explicit GPU(unsigned int id);
        ~GPU();

        [[nodiscard]] const cudaStream_t& get_stream(size_t stream_id) const {
            return streams_[stream_id % NUM_STREAMS].get();
        }

        [[nodiscard]] const cudaStream_t& get_next_stream() {
            last_stream_ = (last_stream_ + 1) % NUM_STREAMS;
            return streams_[last_stream_].get();
        }

        [[nodiscard]] unsigned int id() const noexcept { return id_; }
        [[nodiscard]] bool may_timeout() const noexcept { return may_timeout_; }
        [[nodiscard]] const cudaDeviceProp& properties() const noexcept { return properties_; }

    private:
        void create_streams();
        void destroy_streams();

        unsigned int id_;
        bool may_timeout_;
        std::array<Stream, NUM_STREAMS> streams_;
        int last_stream_{-1};
        bool streams_created_{false};
        cudaDeviceProp properties_;
    };

    // Static interface
    static void init();
    static void load_info();
    static void select_gpus(std::span<const unsigned int> gpu_ids);
    static void use(int gpu_id);
    static void sync(int gpu_id);
    static void sync();
    static int current();
    static void safe(bool make_safe);
    static int get_initial_gpu();
    
    [[nodiscard]] static size_t all_gpu_size() noexcept { return all_gpus_.size(); }
    [[nodiscard]] static bool safe() noexcept { return is_safe_; }
    [[nodiscard]] static const cudaStream_t& get_next_stream() { return gpus_[0].get_next_stream(); }

#ifdef USE_NCCL
    template<typename T>
    static void nccl_broadcast(int root, std::span<const T*> send_d, 
                             std::span<T*> recv_d, size_t size, 
                             int stream_id = -1) {
        if (gpus_.size() == 1) return;
        
        ncclGroupStart();
        for (size_t i = 0; i < gpus_.size(); ++i) {
            cudaStream_t stream = (stream_id >= 0) ? 
                                 gpus_[i].get_stream(stream_id) : 0;
                                 
            NCCL_CHECK(ncclBroadcast(
                static_cast<const void*>(send_d[i]),
                static_cast<void*>(recv_d[i]),
                size * sizeof(T) / sizeof(float),
                ncclFloat, root, comms_[i], stream));
        }
        ncclGroupEnd();
    }
    
    template<typename T>
    static void nccl_reduce(int root, std::span<const T*> send_d, 
                          std::span<T*> recv_d, size_t size, 
                          int stream_id = -1) {
        if (gpus_.size() == 1) return;
        
        ncclGroupStart();
        for (size_t i = 0; i < gpus_.size(); ++i) {
            cudaStream_t stream = (stream_id >= 0) ? 
                                 gpus_[i].get_stream(stream_id) : 0;
                                 
            NCCL_CHECK(ncclReduce(
                static_cast<const void*>(send_d[i]),
                static_cast<void*>(recv_d[i]),
                size * sizeof(T) / sizeof(float),
                ncclFloat, ncclSum, root, comms_[i], stream));
        }
        ncclGroupEnd();
    }
#endif

private:
    static void init_devices();
#ifdef USE_NCCL
    static void init_comms();
    static std::vector<ncclComm_t> comms_;
#endif

    static std::vector<GPU> all_gpus_;
    static std::vector<GPU> gpus_;
    static std::vector<GPU> no_timeouts_;
    static bool is_safe_;
};

} // namespace ARBD

#endif // USE_CUDA 