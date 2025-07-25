#pragma once

#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

// Required Manager Headers for Policies
#ifdef USE_CUDA
#include "CUDA/CUDAManager.h"
#endif
#ifdef USE_SYCL
#include "SYCL/SYCLManager.h"
#endif
#ifdef USE_METAL
#include "METAL/METALManager.h"
#endif

#include "ARBDLogger.h"
#include "Events.h"
#include "Resource.h"

namespace ARBD {

// ============================================================================
// Backend Policies
// ============================================================================
// These structs define a static interface for memory operations.
// The correct policy is chosen at compile time by wrapping the definitions
// in the appropriate preprocessor blocks.
// ============================================================================

#ifdef USE_CUDA
/**
 * @brief Policy for CUDA memory operations.
 */
struct CudaPolicy {
	static void* allocate(size_t bytes) {
		void* ptr = nullptr;
		CUDA_CHECK(cudaMalloc(&ptr, bytes));
		LOGTRACE("CudaPolicy: Allocated {} bytes.", bytes);
		return ptr;
	}

	static void deallocate(void* ptr) {
		if (ptr) {
			LOGTRACE("CudaPolicy: Deallocating pointer.");
			CUDA_CHECK(cudaFree(ptr));
		}
	}

	static void copy_to_host(void* host_dst, const void* device_src, size_t bytes) {
		LOGTRACE("CudaPolicy: Copying {} bytes from device to host.", bytes);
		CUDA_CHECK(cudaMemcpy(host_dst, device_src, bytes, cudaMemcpyDeviceToHost));
	}

	static void copy_from_host(void* device_dst, const void* host_src, size_t bytes) {
		LOGTRACE("CudaPolicy: Copying {} bytes from host to device.", bytes);
		CUDA_CHECK(cudaMemcpy(device_dst, host_src, bytes, cudaMemcpyHostToDevice));
	}

	static void copy_device_to_device(void* device_dst, const void* device_src, size_t bytes) {
		LOGTRACE("CudaPolicy: Copying {} bytes from device to device.", bytes);
		CUDA_CHECK(cudaMemcpy(device_dst, device_src, bytes, cudaMemcpyDeviceToDevice));
	}
};
#endif // USE_CUDA

#ifdef USE_SYCL
/**
 * @brief Policy for SYCL memory operations.
 */
struct SyclPolicy {
	static void* allocate(size_t bytes) {
		auto& queue = SYCL::SYCLManager::get_current_queue();
		void* ptr = sycl::malloc_device(bytes, queue.get());
		LOGTRACE("SyclPolicy: Allocated {} bytes.", bytes);
		return ptr;
	}

	static void deallocate(void* ptr) {
		if (ptr) {
			auto& queue = SYCL::SYCLManager::get_current_queue();
			LOGTRACE("SyclPolicy: Deallocating pointer.");
			sycl::free(ptr, queue.get());
		}
	}

	static void copy_to_host(void* host_dst, const void* device_src, size_t bytes) {
		auto& queue = SYCL::SYCLManager::get_current_queue();
		LOGTRACE("SyclPolicy: Copying {} bytes from device to host.", bytes);
		queue.get().memcpy(host_dst, device_src, bytes).wait();
	}

	static void copy_from_host(void* device_dst, const void* host_src, size_t bytes) {
		auto& queue = SYCL::SYCLManager::get_current_queue();
		LOGTRACE("SyclPolicy: Copying {} bytes from host to device.", bytes);
		queue.get().memcpy(device_dst, host_src, bytes).wait();
	}

	static void copy_device_to_device(void* device_dst, const void* device_src, size_t bytes) {
		auto& queue = SYCL::SYCLManager::get_current_queue();
		LOGTRACE("SyclPolicy: Copying {} bytes from device to device.", bytes);
		queue.get().memcpy(device_dst, device_src, bytes).wait();
	}
};
#endif // USE_SYCL

#ifdef USE_METAL
/**
 * @brief Policy for Metal memory operations.
 */
struct MetalPolicy {
	static void* allocate(size_t bytes) {
		LOGTRACE("MetalPolicy: Allocated {} bytes.", bytes);
		return METAL::METALManager::allocate_raw(bytes);
	}

	static void deallocate(void* ptr) {
		if (ptr) {
			LOGTRACE("MetalPolicy: Deallocating pointer.");
			METAL::METALManager::deallocate_raw(ptr);
		}
	}

	static void copy_to_host(void* host_dst, const void* device_src, size_t bytes) {
		LOGTRACE("MetalPolicy: Copying {} bytes from device to host (unified memory).", bytes);
		std::memcpy(host_dst, device_src, bytes);
	}

	static void copy_from_host(void* device_dst, const void* host_src, size_t bytes) {
		LOGTRACE("MetalPolicy: Copying {} bytes from host to device (unified memory).", bytes);
		std::memcpy(device_dst, host_src, bytes);
	}

	static void copy_device_to_device(void* device_dst, const void* device_src, size_t bytes) {
		// Metal's unified memory model simplifies this. A direct memcpy is sufficient.
		LOGTRACE("MetalPolicy: Copying {} bytes from device to device (unified memory).", bytes);
		std::memcpy(device_dst, device_src, bytes);
	}
};
#endif // USE_METAL

// ============================================================================
// The Refactored Buffer Class
// ============================================================================

/**
 * @brief A policy-based, backend-agnostic buffer for device memory.
 *
 * This class provides a clean, RAII-compliant wrapper for device-side memory.
 * It is templated on a Policy struct that defines the actual memory operations
 * for a specific backend (e.g., CudaPolicy, SyclPolicy).
 *
 * @tparam T The type of data to store in the buffer.
 * @tparam Policy A policy struct defining memory operations.
 */
template<typename T, typename Policy>
class Buffer {
  public:
	using value_type = T;

	// --- Constructors and Destructor ---

	/**
	 * @brief Constructs an empty buffer.
	 */
	Buffer() = default;

	/**
	 * @brief Allocates a buffer on the device for a given number of elements.
	 * @param count The number of elements of type T to allocate.
	 */
	explicit Buffer(size_t count) : count_(count), device_ptr_(nullptr) {
		if (count_ > 0) {
			device_ptr_ = static_cast<T*>(Policy::allocate(bytes()));
		}
	}

	/**
	 * @brief Destructor that automatically deallocates device memory.
	 */
	~Buffer() {
		Policy::deallocate(device_ptr_);
	}

	// --- Rule of Five: Movable, Not Copyable ---

	Buffer(const Buffer&) = delete;
	Buffer& operator=(const Buffer&) = delete;

	Buffer(Buffer&& other) noexcept
		: count_(std::exchange(other.count_, 0)),
		  device_ptr_(std::exchange(other.device_ptr_, nullptr)) {}

	Buffer& operator=(Buffer&& other) noexcept {
		if (this != &other) {
			// Deallocate existing memory before taking ownership of the new memory
			Policy::deallocate(device_ptr_);
			count_ = std::exchange(other.count_, 0);
			device_ptr_ = std::exchange(other.device_ptr_, nullptr);
		}
		return *this;
	}

	// --- Data Transfer Methods ---

	/**
	 * @brief Copies data from a host-side pointer to this device buffer.
	 * @param host_src Pointer to the source data on the host.
	 * @param num_elements The number of elements to copy.
	 */
	void copy_from_host(const T* host_src, size_t num_elements) {
		if (num_elements > count_) {
			throw std::runtime_error("Copy size exceeds buffer capacity.");
		}
		Policy::copy_from_host(device_ptr_, host_src, num_elements * sizeof(T));
	}

	/**
	 * @brief Copies all data from a host-side std::vector to this device buffer.
	 * @param host_vec The source vector on the host.
	 */
	void copy_from_host(const std::vector<T>& host_vec) {
		copy_from_host(host_vec.data(), host_vec.size());
	}

	/**
	 * @brief Copies data from this device buffer to a host-side pointer.
	 * @param host_dst Pointer to the destination on the host.
	 * @param num_elements The number of elements to copy.
	 */
	void copy_to_host(T* host_dst, size_t num_elements) const {
		if (num_elements > count_) {
			throw std::runtime_error("Copy size exceeds buffer size.");
		}
		Policy::copy_to_host(host_dst, device_ptr_, num_elements * sizeof(T));
	}

	/**
	 * @brief Copies all data from this device buffer to a host-side std::vector.
	 * @param host_vec The destination vector on the host. It will be resized if needed.
	 */
	void copy_to_host(std::vector<T>& host_vec) const {
		if (host_vec.size() != count_) {
			host_vec.resize(count_);
		}
		copy_to_host(host_vec.data(), count_);
	}

	// --- Accessors ---

	/**
	 * @brief Returns the raw device pointer.
	 */
	T* data() {
		return device_ptr_;
	}

	/**
	 * @brief Returns the const raw device pointer.
	 */
	const T* data() const {
		return device_ptr_;
	}

	/**
	 * @brief Returns the number of elements the buffer can hold.
	 */
	size_t size() const {
		return count_;
	}

	/**
	 * @brief Returns the total size of the buffer in bytes.
	 */
	size_t bytes() const {
		return count_ * sizeof(T);
	}

	/**
	 * @brief Checks if the buffer is empty (has zero size).
	 */
	bool empty() const {
		return count_ == 0;
	}

  private:
	size_t count_{0};
	T* device_ptr_{nullptr};
};

// ============================================================================
// Compile-Time Backend Selection
// ============================================================================

// Select the active policy based on compilation flags.
#if defined(USE_CUDA)
using BackendPolicy = CudaPolicy;
#elif defined(USE_SYCL)
using BackendPolicy = SyclPolicy;
#elif defined(USE_METAL)
using BackendPolicy = MetalPolicy;
#else
// Default or error case. Could define a 'HostPolicy' for CPU-only builds.
// For now, let's cause a compile error if no backend is chosen.
#error "No backend selected. Please define USE_CUDA, USE_SYCL, or USE_METAL."
#endif

/**
 * @brief A convenient alias for the Buffer class using the active backend policy.
 *
 * Instead of writing `Buffer<float, ActivePolicy>`, you can simply write
 * `DeviceBuffer<float>`.
 */
template<typename T>
using DeviceBuffer = Buffer<T, BackendPolicy>;

} // namespace ARBD
