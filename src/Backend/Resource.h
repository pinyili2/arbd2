#pragma once
#include "ARBDLogger.h"
#include <type_traits>

// Define HOST and DEVICE macros
#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

#ifdef USE_CUDA
#include "Backend/CUDA/CUDAManager.h"
#endif

#ifdef USE_SYCL
#include "Backend/SYCL/SYCLManager.h"
#endif

#ifdef USE_METAL
#include "Backend/METAL/METALManager.h"
#endif

/**
 * @brief Get current device ID for device computing backends
 */
inline size_t get_device_id() {
#ifdef USE_CUDA
  if (cudaGetDevice(nullptr) == cudaSuccess) {
    int device;
    cudaGetDevice(&device);
    return static_cast<size_t>(device);
  }
#endif

#ifdef USE_SYCL
  try {
    return static_cast<size_t>(
        ARBD::SYCL::SYCLManager::get_current_device().id());
  } catch (...) {
    return 0;
  }
#endif

#ifdef USE_METAL
  try {
    return static_cast<size_t>(
        ARBD::METAL::METALManager::get_current_device().get_id());
  } catch (...) {
    return 0;
  }
#endif

  return 0;
}

/**
 * @brief Resource representation for heterogeneous computing
 * Supports CUDA, SYCL, and MPI resource types
 */
namespace ARBD {

// Forward declarations
template <typename T> class UnifiedBuffer;

/**
 * @brief Memory transfer policies
 */
enum class TransferType {
  HOST_TO_DEVICE,
  DEVICE_TO_HOST,
  DEVICE_TO_DEVICE,
  HOST_TO_HOST
};

/**
 * @brief Backend capability traits for compile-time feature detection
 */
template <typename Backend> struct BackendTraits {
  static constexpr bool supports_device_memory = false;
  static constexpr bool supports_async_execution = false;
  static constexpr bool supports_peer_access = false;
  static constexpr bool requires_explicit_sync = false;

  using context_type = void;
  using event_type = void;
  using stream_type = void;
};

/**
 * @brief CUDA Backend Traits
 */
struct CUDABackend {
  static constexpr const char *name = "CUDA";
  static constexpr int resource_type = 0; // CUDA enum value
};

template <> struct BackendTraits<CUDABackend> {
  static constexpr bool supports_device_memory = true;
  static constexpr bool supports_async_execution = true;
  static constexpr bool supports_peer_access = true;
  static constexpr bool requires_explicit_sync = true;

#ifdef USE_CUDA
  using context_type = int; // CUDA device ID
  using event_type = cudaEvent_t;
  using stream_type = cudaStream_t;
#else
  using context_type = void;
  using event_type = void;
  using stream_type = void;
#endif
};

/**
 * @brief SYCL Backend Traits
 */
struct SYCLBackend {
  static constexpr const char *name = "SYCL";
  static constexpr int resource_type = 1; // SYCL enum value
};

template <> struct BackendTraits<SYCLBackend> {
  static constexpr bool supports_device_memory = true;
  static constexpr bool supports_async_execution = true;
  static constexpr bool supports_peer_access = false;
  static constexpr bool requires_explicit_sync = false;

#ifdef USE_SYCL
  using context_type = void *; // sycl::queue*
  using event_type = void *;   // sycl::event*
  using stream_type = void *;  // sycl::queue*
#else
  using context_type = void;
  using event_type = void;
  using stream_type = void;
#endif
};

/**
 * @brief METAL Backend Traits
 */
struct METALBackend {
  static constexpr const char *name = "METAL";
  static constexpr int resource_type = 2; // METAL enum value
};

template <> struct BackendTraits<METALBackend> {
  static constexpr bool supports_device_memory = true;
  static constexpr bool supports_async_execution = true;
  static constexpr bool supports_peer_access = false;
  static constexpr bool requires_explicit_sync = false;

  using context_type = void; // METAL doesn't expose contexts directly
  using event_type = void;   // METAL command buffer events
  using stream_type = void;  // METAL command queues
};

/**
 * @brief Concept to check if a type is a valid backend
 */
template <typename T>
concept ValidBackend = requires {
  typename BackendTraits<T>;
  { T::name } -> std::convertible_to<const char *>;
  { T::resource_type } -> std::convertible_to<int>;
};

/**
 * @brief Resource representation for device computing environments
 *
 * The Resource class provides a unified interface for representing and managing
 * computational resources for device computing (CUDA, SYCL, METAL).
 * For distributed computing (MPI), use MPIResource from MPIBackend.h instead.
 *
 * @details This class manages different compute devices on a single machine:
 * - CUDA GPU devices for NVIDIA GPU computing
 * - SYCL devices for cross-platform parallel computing
 * - METAL devices for Apple GPU computing
 *
 * @note MPI/distributed computing is now handled separately in MPIBackend.h
 *       since it has fundamentally different semantics (multi-machine vs
 * multi-device)
 *
 * @example Basic Usage:
 * ```cpp
 * // Create a CUDA resource for device 0
 * ARBD::Resource cuda_res(ARBD::Resource::CUDA, 0);
 *
 * // Create a SYCL resource for device 1
 * ARBD::Resource sycl_res(ARBD::Resource::SYCL, 1);
 *
 * // Check if resource is local to current execution context
 * if (cuda_res.is_local()) {
 *     // Perform local operations
 * }
 * ```
 *
 * @see MPIBackend.h for distributed computing resources
 * @see ResourceType for available device resource types
 * @see is_local() for locality checking
 * @see getTypeString() for human-readable type names
 */

struct Resource {
  enum ResourceType { CUDA, SYCL, METAL };

  // Legacy aliases for backward compatibility
  static constexpr ResourceType GPU = CUDA;

  ResourceType type;
  size_t id;
  Resource *parent;

  HOST DEVICE Resource() : type(SYCL), id(0), parent(nullptr) {}
  HOST DEVICE Resource(ResourceType t, size_t i)
      : type(t), id(i), parent(nullptr) {}
  HOST DEVICE Resource(ResourceType t, size_t i, Resource *p)
      : type(t), id(i), parent(p) {}

  HOST DEVICE constexpr const char *getTypeString() const {
    switch (type) {
    case CUDA:
      return "CUDA";
    case SYCL:
      return "SYCL";
    case METAL:
      return "METAL";
    default:
      return "Unknown";
    }
  }

  HOST DEVICE bool is_local() const {
    bool ret = false;

#ifdef USE_CUDA
    if (cudaGetDevice(nullptr) == cudaSuccess) { // Are we in a CUDA context?
      int current_device;
      cudaGetDevice(&current_device);

      switch (type) {
      case CUDA:
        ret = (current_device == static_cast<int>(id));
        break;
      case SYCL:
      case METAL:
        ret = false;
        break;
      }
      LOGWARN("Resource::is_local(): CUDA context - type %s, device %zu, "
              "current %d, returning %d",
              getTypeString(), id, current_device, ret);
    } else
#endif
#ifdef USE_SYCL
        if (type == SYCL) {
      // Check if we're in a SYCL context
      try {
        auto &current_device = SYCL::SYCLManager::get_current_device();
        ret = (current_device.id() == id);
        LOGINFO("Resource::is_local(): SYCL context - device %zu, current %u, "
                "returning %d",
                id, current_device.id(), ret);
      } catch (...) {
        ret = false;
        LOGINFO(
            "Resource::is_local(): SYCL device not available, returning false");
      }
    } else
#endif
    {
      switch (type) {
      case CUDA:
        if (parent != nullptr) {
          ret = parent->is_local();
          LOGINFO(
              "Resource::is_local(): CPU checking CUDA parent, returning %d",
              ret);
        } else {
          ret = false;
          LOGINFO(
              "Resource::is_local(): CPU with no CUDA parent, returning false");
        }
        break;
      case SYCL:
        if (parent != nullptr) {
          ret = parent->is_local();
          LOGINFO(
              "Resource::is_local(): CPU checking SYCL parent, returning %d",
              ret);
        } else {
          ret = false;
          LOGINFO(
              "Resource::is_local(): CPU with no SYCL parent, returning false");
        }
        break;

      case METAL:
        ret = false;
        break;
      }
    }
    return ret;
  }

  static Resource Local() {
#ifdef USE_CUDA
    if (cudaGetDevice(nullptr) == cudaSuccess) { // Are we in a CUDA context?
      int device;
      cudaGetDevice(&device);
      return Resource{CUDA, static_cast<size_t>(device)};
    }
#endif
#ifdef USE_SYCL
    // Check if SYCL is available and preferred
    try {
      auto &current_device = SYCL::SYCLManager::get_current_device();
      return Resource{SYCL, static_cast<size_t>(current_device.id())};
    } catch (...) {
    }
#endif
    // Default to first device for device computing backends
    return Resource{SYCL, get_device_id()};
  }

  HOST DEVICE bool operator==(const Resource &other) const {
    return type == other.type && id == other.id;
  }

  HOST std::string toString() const {
    return std::string(getTypeString()) + "[" + std::to_string(id) + "]";
  }

  /**
   * @brief Check if the resource supports asynchronous operations
   */
  HOST DEVICE bool supports_async() const {
    switch (type) {
    case CUDA:
    case SYCL:
      return true;

    default:
      return false;
    }
  }

  /**
   * @brief Get the memory space type for this resource
   */
  HOST DEVICE constexpr const char *getMemorySpace() const {
    switch (type) {
    case CUDA:
      return "device";
    case SYCL:
      return "device";

    case METAL:
      return "device";
    default:
      return "unknown";
    }
  }

  /**
   * @brief Check if this resource represents a device (GPU)
   */
  HOST DEVICE bool is_device() const {
    return type == CUDA || type == SYCL || type == METAL;
  }

  /**
   * @brief Check if this resource represents a host (CPU)
   * @note Always returns false since Resource only handles device computing.
   *       For distributed computing, use MPIResource from MPIBackend.h
   */
  HOST DEVICE bool is_host() const { return false; }
};

} // namespace ARBD
