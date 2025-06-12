#pragma once

#include "ARBDLogger.h"
#include "Resource.h"
#include <cstring>
#include <memory>
#include <type_traits>
#include <unordered_map>

#ifdef USE_CUDA
#include "CUDA/CUDAManager.h"
#endif

#ifdef USE_SYCL
#include "SYCL/SYCLManager.h"
#endif

#ifdef USE_METAL
#include "METAL/METALManager.h"
#endif

namespace ARBD {
/**
 * @brief Memory location tracking
 */
struct MemoryLocation {
  Resource resource;
  void *ptr;
  size_t size;
  bool is_valid;

  MemoryLocation() : ptr(nullptr), size(0), is_valid(false) {}
  MemoryLocation(const Resource &res, void *p, size_t s)
      : resource(res), ptr(p), size(s), is_valid(true) {}
};

/**
 * @brief Unified buffer for cross-backend memory management
 */
template <typename T> class UnifiedBuffer {
private:
  std::unordered_map<size_t, MemoryLocation> locations_;
  size_t count_;
  Resource primary_location_;

  /**
   * @brief Allocate memory using appropriate backend manager
   */
  void *allocate_memory(size_t size, const Resource &resource) {
    switch (resource.type) {
#ifdef USE_CUDA
    case Resource::CUDA: {
      void *ptr = nullptr;
      cudaMalloc(&ptr, size);
      return ptr;
    }
#endif
#ifdef USE_SYCL
    case Resource::SYCL: {
      auto &device = ARBD::SYCL::SYCLManager::get_current_device();
      auto &queue = device.get_next_queue();
      return sycl::malloc_device<char>(size, queue.get());
    }
#endif
#ifdef USE_METAL
    case Resource::METAL: {
      auto &device = ARBD::METAL::METALManager::get_current_device();
      return device.allocate<char>(size);
    }
#endif
    default:
      return std::malloc(size);
    }
  }

  /**
   * @brief Deallocate memory using appropriate backend manager
   */
  void deallocate_memory(void *ptr, const Resource &resource) {
    if (!ptr)
      return;

    switch (resource.type) {
#ifdef USE_CUDA
    case Resource::CUDA:
      cudaFree(ptr);
      break;
#endif
#ifdef USE_SYCL
    case Resource::SYCL: {
      auto &device = ARBD::SYCL::SYCLManager::get_current_device();
      auto &queue = device.get_next_queue();
      sycl::free(ptr, queue.get());
      break;
    }
#endif
#ifdef USE_METAL
    case Resource::METAL: {
      auto &device = ARBD::METAL::METALManager::get_current_device();
      device.deallocate(ptr);
      break;
    }
#endif
    default:
      std::free(ptr);
      break;
    }
  }

  /**
   * @brief Copy memory between resources
   */
  void copy_memory(void *dst, const void *src, size_t size,
                   const Resource &src_res, const Resource &dst_res) {
    // Simplified copy - just use memcpy for now
    // Full implementation would use appropriate backend copy functions
    std::memcpy(dst, src, size);
  }

public:
  /**
   * @brief Construct unified buffer
   */
  explicit UnifiedBuffer(size_t count, const Resource &primary_res = Resource())
      : count_(count), primary_location_(primary_res) {
    if (count_ > 0) {
      void *ptr = allocate_memory(sizeof(T) * count_, primary_location_);
      locations_[primary_location_.id] =
          MemoryLocation(primary_location_, ptr, sizeof(T) * count_);
    }
  }

  /**
   * @brief Construct from existing data
   */
  UnifiedBuffer(size_t count, T *existing_data, const Resource &location)
      : count_(count), primary_location_(location) {
    locations_[location.id] =
        MemoryLocation(location, existing_data, sizeof(T) * count_);
  }

  ~UnifiedBuffer() {
    for (auto &[id, loc] : locations_) {
      deallocate_memory(loc.ptr, loc.resource);
    }
  }

  // No copy constructor - use explicit clone()
  UnifiedBuffer(const UnifiedBuffer &) = delete;
  UnifiedBuffer &operator=(const UnifiedBuffer &) = delete;

  // Move constructor
  UnifiedBuffer(UnifiedBuffer &&other) noexcept
      : locations_(std::move(other.locations_)), count_(other.count_),
        primary_location_(other.primary_location_) {
    other.count_ = 0;
  }

  UnifiedBuffer &operator=(UnifiedBuffer &&other) noexcept {
    if (this != &other) {
      for (auto &[id, loc] : locations_) {
        deallocate_memory(loc.ptr, loc.resource);
      }
      locations_ = std::move(other.locations_);
      count_ = other.count_;
      primary_location_ = other.primary_location_;
      other.count_ = 0;
    }
    return *this;
  }

  /**
   * @brief Get pointer for specific resource
   */
  T *get_ptr(const Resource &resource) {
    ensure_available_at(resource);
    auto it = locations_.find(resource.id);
    return (it != locations_.end()) ? static_cast<T *>(it->second.ptr)
                                    : nullptr;
  }

  /**
   * @brief Get const pointer for specific resource
   */
  const T *get_ptr(const Resource &resource) const {
    auto it = locations_.find(resource.id);
    return (it != locations_.end()) ? static_cast<const T *>(it->second.ptr)
                                    : nullptr;
  }

  /**
   * @brief Ensure data is available at target resource
   */
  void ensure_available_at(const Resource &target) {
    if (locations_.find(target.id) != locations_.end()) {
      return; // Already available
    }

    if (count_ == 0) {
      return; // No data to migrate
    }

    // Find a source location to copy from
    MemoryLocation *source = nullptr;
    for (auto &[id, loc] : locations_) {
      if (loc.is_valid && loc.ptr) {
        source = &loc;
        break;
      }
    }

    if (!source) {
      ARBD::throw_value_error(
          "No valid source location found for data migration");
    }

    // Allocate at target
    void *target_ptr = allocate_memory(sizeof(T) * count_, target);

    // Copy data
    copy_memory(target_ptr, source->ptr, sizeof(T) * count_, source->resource,
                target);

    // Store location
    locations_[target.id] =
        MemoryLocation(target, target_ptr, sizeof(T) * count_);

    LOGINFO("UnifiedBuffer: migrated data from {} to {}",
            source->resource.getTypeString(), target.getTypeString());
  }

  /**
   * @brief Release memory at specific resource
   */
  void release_at(const Resource &resource) {
    auto it = locations_.find(resource.id);
    if (it != locations_.end()) {
      deallocate_memory(it->second.ptr, resource);
      locations_.erase(it);
    }
  }

  /**
   * @brief Get element count
   */
  size_t size() const { return count_; }

  /**
   * @brief Check if empty
   */
  bool empty() const { return count_ == 0; }

  /**
   * @brief Get primary location
   */
  const Resource &primary_location() const { return primary_location_; }

  /**
   * @brief Get all locations where data is available
   */
  std::vector<Resource> available_locations() const {
    std::vector<Resource> result;
    for (const auto &[id, loc] : locations_) {
      if (loc.is_valid) {
        result.push_back(loc.resource);
      }
    }
    return result;
  }

  /**
   * @brief Synchronize all locations
   */
  void synchronize_all() { ensure_available_at(primary_location_); }
};

} // namespace ARBD
