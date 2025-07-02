#pragma once

#ifdef USE_CUDA
#include "ARBDException.h"
#include <algorithm>
#include <array>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace ARBD {
inline void check_cuda_error(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    ARBD_Exception(ExceptionType::CUDARuntimeError, "CUDA error at %s:%d: %s",
                   file, line, cudaGetErrorString(error));
  }
}

#define CUDA_CHECK(call) check_cuda_error(call, __FILE__, __LINE__)

namespace CUDA {


/**
 * @brief Modern RAII wrapper for CUDA device memory
 *
 * This class provides a safe and efficient way to manage CUDA device memory
 * with RAII semantics. It handles memory allocation, deallocation, and data
 * transfer between host and device memory.
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
 * ARBD::CUDA::DeviceMemory<int> device_mem(1000);
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
 * ARBD::CUDA::DeviceMemory<float> mem1(1000);
 * ARBD::CUDA::DeviceMemory<float> mem2 = std::move(mem1); // mem1 is now empty
 * ```
 *
 * @note The class prevents copying to avoid accidental memory leaks.
 *       Use move semantics when transferring ownership.
 */

template <typename T> class DeviceMemory {
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
  DeviceMemory(const DeviceMemory &) = delete;
  DeviceMemory &operator=(const DeviceMemory &) = delete;

  // Allow moving
  DeviceMemory(DeviceMemory &&other) noexcept
      : ptr_(std::exchange(other.ptr_, nullptr)),
        size_(std::exchange(other.size_, 0)) {}

  DeviceMemory &operator=(DeviceMemory &&other) noexcept {
    if (this != &other) {
      if (ptr_)
        cudaFree(ptr_);
      ptr_ = std::exchange(other.ptr_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  // Modern copy operations using std::span
  void copyFromHost(std::span<const T> host_data,
                    cudaStream_t stream = nullptr) {
    if (host_data.size() > size_) {
      ARBD_Exception(ExceptionType::ValueError,
                     "Tried to copy %zu elements but only %zu allocated",
                     host_data.size(), size_);
    }
    if (!ptr_ || host_data.empty())
      return;

    if (stream) {
      CUDA_CHECK(cudaMemcpyAsync(ptr_, host_data.data(),
                                 host_data.size() * sizeof(T),
                                 cudaMemcpyHostToDevice, stream));
    } else {
      CUDA_CHECK(cudaMemcpy(ptr_, host_data.data(),
                            host_data.size() * sizeof(T),
                            cudaMemcpyHostToDevice));
    }
  }

  void copyToHost(std::span<T> host_data, cudaStream_t stream = nullptr) const {
    if (host_data.size() > size_) {
      ARBD_Exception(ExceptionType::ValueError,
                     "Tried to copy %zu elements but only %zu allocated",
                     host_data.size(), size_);
    }
    if (!ptr_ || host_data.empty())
      return;

    if (stream) {
      CUDA_CHECK(cudaMemcpyAsync(host_data.data(), ptr_,
                                 host_data.size() * sizeof(T),
                                 cudaMemcpyDeviceToHost, stream));
    } else {
      CUDA_CHECK(cudaMemcpy(host_data.data(), ptr_,
                            host_data.size() * sizeof(T),
                            cudaMemcpyDeviceToHost));
    }
  }

  // Accessors
  [[nodiscard]] T *get() noexcept { return ptr_; }
  [[nodiscard]] const T *get() const noexcept { return ptr_; }
  [[nodiscard]] size_t size() const noexcept { return size_; }
  [[nodiscard]] size_t bytes() const noexcept { return size_ * sizeof(T); }

  // Conversion operators
  operator T *() noexcept { return ptr_; }
  operator const T *() const noexcept { return ptr_; }

  // Memory operations
  void memset(int value, cudaStream_t stream = nullptr) {
    if (!ptr_)
      return;
    if (stream) {
      CUDA_CHECK(cudaMemsetAsync(ptr_, value, bytes(), stream));
    } else {
      CUDA_CHECK(cudaMemset(ptr_, value, bytes()));
    }
  }

private:
  T *ptr_{nullptr};
  size_t size_{0};
};

/**
 * @brief Modern RAII wrapper for CUDA streams
 *
 * This class provides a safe, modern C++ wrapper around CUDA streams with RAII
 * semantics. It automatically manages the lifecycle of CUDA streams, ensuring
 * proper creation and cleanup.
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
 * ARBD::CUDA::Stream stream;
 *
 * // Create a stream with specific flags
 * ARBD::CUDA::Stream non_blocking_stream(cudaStreamNonBlocking);
 *
 * // Synchronize the stream
 * stream.synchronize();
 *
 * // Use with CUDA APIs
 * cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
 * ```
 *
 * @note The stream is automatically destroyed when the Stream object goes out
 * of scope
 */
class Stream {
public:
  Stream() { CUDA_CHECK(cudaStreamCreate(&stream_)); }

  explicit Stream(unsigned int flags) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
  }

  ~Stream() {
    if (stream_) {
      cudaStreamDestroy(stream_);
    }
  }

  // Prevent copying
  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;

  // Allow moving
  Stream(Stream &&other) noexcept
      : stream_(std::exchange(other.stream_, nullptr)) {}

  Stream &operator=(Stream &&other) noexcept {
    if (this != &other) {
      if (stream_)
        cudaStreamDestroy(stream_);
      stream_ = std::exchange(other.stream_, nullptr);
    }
    return *this;
  }

  void synchronize() { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

  [[nodiscard]] bool query() const {
    cudaError_t result = cudaStreamQuery(stream_);
    if (result == cudaSuccess)
      return true;
    if (result == cudaErrorNotReady)
      return false;
    CUDA_CHECK(result);
    return false;
  }

  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }
  operator cudaStream_t() const noexcept { return stream_; }

private:
  cudaStream_t stream_{nullptr};
};

/**
 * @brief Modern RAII wrapper for CUDA events
 *
 * This class provides a safe, modern C++ wrapper around CUDA events with RAII
 * semantics. It manages the lifecycle of CUDA events and provides utilities for
 * timing and synchronization.
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
 * ARBD::CUDA::Event start_event;
 * ARBD::CUDA::Event end_event;
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
 * @note Events are automatically destroyed when the Event object goes out of
 * scope
 */
class Event {
public:
  Event() { CUDA_CHECK(cudaEventCreate(&event_)); }

  explicit Event(unsigned int flags) {
    CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
  }

  ~Event() {
    if (event_) {
      cudaEventDestroy(event_);
    }
  }

  // Prevent copying
  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;

  // Allow moving
  Event(Event &&other) noexcept
      : event_(std::exchange(other.event_, nullptr)) {}

  Event &operator=(Event &&other) noexcept {
    if (this != &other) {
      if (event_)
        cudaEventDestroy(event_);
      event_ = std::exchange(other.event_, nullptr);
    }
    return *this;
  }

  void record(cudaStream_t stream = nullptr) {
    CUDA_CHECK(cudaEventRecord(event_, stream));
  }

  void synchronize() { CUDA_CHECK(cudaEventSynchronize(event_)); }

  [[nodiscard]] bool query() const {
    cudaError_t result = cudaEventQuery(event_);
    if (result == cudaSuccess)
      return true;
    if (result == cudaErrorNotReady)
      return false;
    CUDA_CHECK(result);
    return false;
  }

  [[nodiscard]] float elapsed(const Event &start) const {
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
 * This class provides a comprehensive GPU management system with support for
 * multiple GPUs, stream management, and device selection. It handles GPU
 * initialization, selection, and provides utilities for multi-GPU operations.
 *
 * Features:
 * - Multi-GPU support
 * - Automatic stream management
 * - GPU selection and synchronization
 * - Peer-to-peer memory access
 * - Safe GPU timeout handling
 *
 * @example Basic Usage:
 * ```cpp
 * // Initialize GPU system
 * ARBD::CUDA::GPUManager::init();
 *
 * // Select specific GPUs
 * std::vector<unsigned int> gpu_ids = {0, 1};
 * ARBD::CUDA::GPUManager::select_gpus(gpu_ids);
 *
 * // Use a specific GPU
 * ARBD::CUDA::GPUManager::use(0);
 *
 * // Synchronize all GPUs (moved to NCCLManager)
 * ARBD::CUDA::NCCLManager::sync_all();
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
   * const auto& gpu = ARBD::CUDA::GPUManager::gpus()[0];
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

    // Prevent copying, allow moving
    GPU(const GPU &) = delete;
    GPU &operator=(const GPU &) = delete;
    GPU(GPU &&) = default;
    GPU &operator=(GPU &&) = default;

    [[nodiscard]] cudaStream_t get_stream(size_t stream_id) const {
      return streams_[stream_id % NUM_STREAMS].get();
    }

    [[nodiscard]] cudaStream_t get_next_stream() {
      last_stream_ = (last_stream_ + 1) % NUM_STREAMS;
      return streams_[last_stream_].get();
    }

    [[nodiscard]] unsigned int id() const noexcept { return id_; }
    [[nodiscard]] bool may_timeout() const noexcept { return may_timeout_; }
    [[nodiscard]] const cudaDeviceProp &properties() const noexcept {
      return properties_;
    }

    // Convenience property accessors
    [[nodiscard]] size_t total_memory() const noexcept {
      return properties_.totalGlobalMem;
    }
    [[nodiscard]] int compute_capability_major() const noexcept {
      return properties_.major;
    }
    [[nodiscard]] int compute_capability_minor() const noexcept {
      return properties_.minor;
    }
    [[nodiscard]] int multiprocessor_count() const noexcept {
      return properties_.multiProcessorCount;
    }
    [[nodiscard]] int max_threads_per_block() const noexcept {
      return properties_.maxThreadsPerBlock;
    }
    [[nodiscard]] bool supports_managed_memory() const noexcept {
      return properties_.managedMemory;
    }
    [[nodiscard]] const char *name() const noexcept { return properties_.name; }

    void synchronize_all_streams();

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
  static void select_gpus(const std::vector<unsigned int> &gpu_ids) {
    gpus_.clear();
    gpus_.reserve(gpu_ids.size());

    for (unsigned int id : gpu_ids) {
      // Find the GPU with matching ID in all_gpus_
      auto it = std::find_if(all_gpus_.begin(), all_gpus_.end(),
                             [id](const GPU &gpu) { return gpu.id() == id; });

      if (it == all_gpus_.end()) {
        ARBD_Exception(ExceptionType::ValueError,
                       "Invalid GPU ID: {} (not found in available GPUs)", id);
      }

      // Copy construct the GPU (since we can't move from all_gpus_)
      // This creates a new GPU object with the same ID
      gpus_.emplace_back(id);
    }

    init_devices();
  }
  static void use(int gpu_id);
  static void sync(int gpu_id);
  static int current();
  static void prefer_safe_gpus(bool safe = true);
  static int get_safest_gpu();

  // Peer-to-peer operations
  static void enable_peer_access();
  static bool can_access_peer(int gpu1, int gpu2);

  // Cache configuration
  static void set_cache_config(cudaFuncCache config = cudaFuncCachePreferL1);

  // Advanced stream access
  static cudaStream_t get_stream(int gpu_id, size_t stream_id);

  // Legacy compatibility methods
  static int getInitialGPU() { return get_safest_gpu(); }
  static void safe(bool make_safe) { prefer_safe_gpus(make_safe); }

  [[nodiscard]] static size_t all_gpu_size() noexcept {
    return all_gpus_.size();
  }
  [[nodiscard]] static size_t allGpuSize() noexcept { return all_gpus_.size(); }
  [[nodiscard]] static bool safe() noexcept { return prefer_safe_; }
  [[nodiscard]] static cudaStream_t get_next_stream() {
    if (gpus_.empty()) {
      ARBD_Exception(ExceptionType::ValueError, "No GPUs available");
    }
    return gpus_[0].get_next_stream();
  }
  [[nodiscard]] static const std::vector<GPU> &gpus() noexcept { return gpus_; }
  [[nodiscard]] static const std::vector<GPU> &all_gpus() noexcept {
    return all_gpus_;
  }

private:
  static void init_devices();
  static void query_peer_access();

  static std::vector<GPU> all_gpus_;
  static std::vector<GPU> gpus_;
  static std::vector<GPU> safe_gpus_;
  static std::vector<std::vector<bool>> peer_access_matrix_;
  static bool prefer_safe_;
};

// ============================================================================
// Low-Level CUDA Device Utilities
// ============================================================================

/**
 * @brief Modern C++20 warp-level broadcast primitive
 *
 * Broadcasts a value from one thread (leader) to all threads in the warp.
 * This function adapts to different CUDA compute capabilities automatically.
 *
 * @param value Value to broadcast (only meaningful on leader thread)
 * @param leader Lane ID of the thread to broadcast from (0-31)
 * @return The broadcast value on all threads in the warp
 *
 * @note This function must be called by all active threads in a warp
 * @note For compute capability < 7.0, requires external shared memory setup
 *
 * @example Usage in a kernel:
 * ```cpp
 * __global__ void my_kernel() {
 *     int lane_id = threadIdx.x % 32;
 *     int value = (lane_id == 0) ? 42 : 0;
 *     int broadcast_value = ARBD::CUDA::warp_broadcast(value, 0);
 *     // Now all threads in warp have broadcast_value == 42
 * }
 * ```
 */
#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ < 300
// For very old architectures, require shared memory
extern __shared__ int warp_broadcast_shared[];
__device__ inline int warp_broadcast(int value, int leader) noexcept {
  const int tid = threadIdx.x;
  const int warp_lane = tid % 32;
  const int warp_id = tid / 32;

  if (warp_lane == leader) {
    warp_broadcast_shared[warp_id] = value;
  }
  __syncwarp();
  return warp_broadcast_shared[warp_id];
}
#elif __CUDA_ARCH__ < 700
// Pre-Volta: use legacy __shfl
__device__ constexpr int warp_broadcast(int value, int leader) noexcept {
  return __shfl(value, leader);
}
#else
// Volta and later: use __shfl_sync with mask
__device__ constexpr int warp_broadcast(int value, int leader) noexcept {
  return __shfl_sync(0xffffffff, value, leader);
}
#endif
#else
// Host-side stub (should not be called)
inline int warp_broadcast(int value, int leader) noexcept {
  return value; // Just return the input value
}
#endif

/**
 * @brief Warp-aggregated atomic increment for improved performance
 *
 * This function performs an atomic increment using warp-level aggregation,
 * which can significantly improve performance when many threads in a warp
 * need to increment the same counter.
 *
 * The algorithm:
 * 1. All active threads vote to participate
 * 2. One thread (leader) performs a single atomic add for the entire warp
 * 3. Each thread gets its unique position within the warp's contribution
 *
 * @param counter Pointer to the counter to increment
 * @param warp_lane Lane ID of calling thread (threadIdx.x % 32)
 * @return Unique value for this thread (like atomicAdd would return)
 *
 * @note Significantly faster than individual atomicAdd calls when most
 *       threads in a warp need to increment the same counter
 *
 * @example Usage in a kernel:
 * ```cpp
 * __global__ void count_kernel(int* global_counter, int* thread_ids, int n) {
 *     int tid = blockIdx.x * blockDim.x + threadIdx.x;
 *     if (tid < n) {
 *         int warp_lane = threadIdx.x % 32;
 *         int my_id = ARBD::CUDA::warp_aggregated_atomic_inc(global_counter,
 * warp_lane); thread_ids[tid] = my_id;
 *     }
 * }
 * ```
 *
 * @see
 * https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
 */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
__device__ inline int warp_aggregated_atomic_inc(int *counter,
                                                 int warp_lane) noexcept {
  // Get mask of active threads in warp
  unsigned int mask = __ballot_sync(0xffffffff, 1);

  // Find the leader (first active thread)
  int leader = __ffs(mask) - 1;

  // Count how many threads are participating
  int warp_count = __popc(mask);

  int result;
  if (warp_lane == leader) {
    // Leader performs the atomic operation for the entire warp
    result = atomicAdd(counter, warp_count);
  }

  // Broadcast the result to all threads in the warp
  result = warp_broadcast(result, leader);

  // Calculate this thread's unique position within the warp's contribution
  unsigned int rank_mask = mask & ((1u << warp_lane) - 1u);
  return result + __popc(rank_mask);
}
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
// Fallback for older architectures
__device__ inline int warp_aggregated_atomic_inc(int *counter,
                                                 int warp_lane) noexcept {
  unsigned int mask = __ballot(1);
  int leader = __ffs(mask) - 1;

  int result;
  if (warp_lane == leader) {
    result = atomicAdd(counter, __popc(mask));
  }
  result = warp_broadcast(result, leader);
  return result + __popc(mask & ((1u << warp_lane) - 1u));
}
#else
// Host-side or very old GPU stub
inline int warp_aggregated_atomic_inc(int *counter, int warp_lane) noexcept {
  return (*counter)++; // Not thread-safe, but shouldn't be called anyway
}
#endif

/**
 * @brief Template version of warp_broadcast for different types
 *
 * @tparam T Type to broadcast (must be 32-bit or smaller)
 * @param value Value to broadcast
 * @param leader Lane ID to broadcast from
 * @return Broadcast value
 */
template <typename T>
  requires(sizeof(T) <= sizeof(int))
__device__ constexpr T warp_broadcast(T value, int leader) noexcept {
  if constexpr (std::is_same_v<T, int>) {
    return warp_broadcast(value, leader);
  } else {
    // Cast to int, broadcast, then cast back
    int int_value = *reinterpret_cast<const int *>(&value);
    int result = warp_broadcast(int_value, leader);
    return *reinterpret_cast<const T *>(&result);
  }
}

} // namespace CUDA
} // namespace ARBD

// Utility macros for backward compatibility
/**
 * @brief Temporarily switch to a specific GPU and execute code
 * @param gpu_id GPU device ID to switch to
 * @param code Code block to execute on the specified GPU
 *
 * @example Usage:
 * ```cpp
 * WITH_GPU(1, {
 *     // This code runs on GPU 1
 *     cudaMalloc(&ptr, size);
 *     my_kernel<<<blocks, threads>>>();
 * });
 * // GPU context is restored after the block
 * ```
 */
#define WITH_GPU(gpu_id, code)                                                 \
  do {                                                                         \
    int _wg_current_device;                                                    \
    ARBD::CUDA::check_cuda_error(cudaGetDevice(&_wg_current_device), __FILE__, \
                                 __LINE__);                                    \
    ARBD::CUDA::check_cuda_error(cudaSetDevice(gpu_id), __FILE__, __LINE__);   \
    code;                                                                      \
    ARBD::CUDA::check_cuda_error(cudaSetDevice(_wg_current_device), __FILE__,  \
                                 __LINE__);                                    \
  } while (0)

/**
 * @brief Legacy error checking macro (for backward compatibility)
 * Use CUDA_CHECK in new code instead
 */
#define gpuErrchk(call) ARBD::CUDA::check_cuda_error(call, __FILE__, __LINE__)

#endif
