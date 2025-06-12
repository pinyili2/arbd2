/*********************************************************************
 * @file  UnifiedArray.h
 *
 * @brief Declaration of backend-agnostic templated Array class.
 *********************************************************************/
#pragma once
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Resource.h"
#include "Backend/UnifiedBuffer.h"
#include <cassert>
#include <memory>
#include <type_traits>

#if __cplusplus >= 202002L
#include <concepts>
// Only include span and compare if C++20 is available and supported
#if __has_include(<span>) && __has_include(<compare>)
#include <compare>
#include <span>
#endif
#endif

namespace ARBD {

// C++20 Concepts for better template constraints (if available)
#if __cplusplus >= 202002L
template <typename T>
concept HasCopyToCuda = requires(T t) { t.copy_to_cuda(); };

template <typename T>
concept HasSendChildren =
    requires(T t, const Resource &r) { t.send_children(r); };

template <typename T>
concept Arithmetic = ::std::is_arithmetic_v<T>;

template <typename T>
concept Trivial = ::std::is_trivial_v<T>;
#endif

/**
 * @brief Backend-agnostic templated array with unified memory management
 */
template <typename T> struct UnifiedArray {
  // Constructors
  HOST DEVICE constexpr UnifiedArray() noexcept : num(0) {}

  HOST explicit UnifiedArray(size_t count,
                             const Resource &location = Resource::Local())
      : num(count),
        buffer_(count > 0 ? std::make_shared<UnifiedBuffer<T>>(count, location)
                          : nullptr) {}

  HOST UnifiedArray(size_t count, const T *input,
                    const Resource &location = Resource::Local())
      : num(count),
        buffer_(count > 0 ? std::make_shared<UnifiedBuffer<T>>(count, location)
                          : nullptr) {
    if (input && buffer_) {
      T *ptr = buffer_->get_ptr(location);
      for (size_t i = 0; i < num; ++i) {
        ptr[i] = input[i];
      }
    }
  }

#if __cplusplus >= 202002L && __has_include(<span>)
  // Constructor from span (C++20 feature, if available)
  HOST explicit UnifiedArray(std::span<const T> input,
                             const Resource &location = Resource::Local())
      : num(input.size()),
        buffer_(num > 0 ? std::make_shared<UnifiedBuffer<T>>(num, location)
                        : nullptr) {
    if (buffer_) {
      T *ptr = buffer_->get_ptr(location);
      for (size_t i = 0; i < num; ++i) {
        ptr[i] = input[i];
      }
    }
  }
#endif

  // Copy constructor
  HOST UnifiedArray(const UnifiedArray<T> &other) : num(other.num) {
    if (other.buffer_) {
      buffer_ = std::make_shared<UnifiedBuffer<T>>(other.buffer_->clone());
    }
  }

  // Move constructor
  HOST constexpr UnifiedArray(UnifiedArray<T> &&other) noexcept
      : num(other.num), buffer_(std::move(other.buffer_)) {
    other.num = 0;
  }

  // Copy assignment operator
  HOST UnifiedArray<T> &operator=(const UnifiedArray<T> &other) {
    if (this != &other) {
      num = other.num;
      if (other.buffer_) {
        buffer_ = std::make_shared<UnifiedBuffer<T>>(other.buffer_->clone());
      } else {
        buffer_.reset();
      }
    }
    return *this;
  }

  // Move assignment operator
  HOST UnifiedArray<T> &operator=(UnifiedArray<T> &&other) noexcept {
    if (this != &other) {
      num = other.num;
      buffer_ = std::move(other.buffer_);
      other.num = 0;
    }
    return *this;
  }

  HOST ~UnifiedArray() = default;

  // Element access
  HOST DEVICE T &operator[](size_t i) noexcept {
    assert(i < num);
    assert(buffer_);
// For device code, assume primary location is available
#ifdef __CUDA_ARCH__
    return static_cast<T *>(buffer_->get_ptr(buffer_->primary_location()))[i];
#else
    return buffer_->get_ptr(buffer_->primary_location())[i];
#endif
  }

  HOST DEVICE const T &operator[](size_t i) const noexcept {
    assert(i < num);
    assert(buffer_);
#ifdef __CUDA_ARCH__
    return static_cast<const T *>(
        buffer_->get_ptr(buffer_->primary_location()))[i];
#else
    return buffer_->get_ptr(buffer_->primary_location())[i];
#endif
  }

  // Element access for specific resource
  HOST T &at(size_t i, const Resource &location) {
    if (i >= num) {
      ARBD::throw_out_of_range("Array index out of range");
    }
    if (!buffer_) {
      ARBD::throw_runtime_error("Array buffer is null");
    }
    return buffer_->get_ptr(location)[i];
  }

  HOST const T &at(size_t i, const Resource &location) const {
    if (i >= num) {
      ARBD::throw_out_of_range("Array index out of range");
    }
    if (!buffer_) {
      ARBD::throw_runtime_error("Array buffer is null");
    }
    return buffer_->get_ptr(location)[i];
  }

#if __cplusplus >= 202002L && __has_include(<compare>)
  // C++20 three-way comparison (if available)
  HOST constexpr auto operator<=>(const UnifiedArray<T> &other) const noexcept
    requires ::std::three_way_comparable<T>
  {
    if (auto cmp = num <=> other.num; cmp != 0)
      return cmp;

    // Compare primary location data
    if (buffer_ && other.buffer_) {
      auto loc = buffer_->primary_location();
      auto other_loc = other.buffer_->primary_location();
      T *ptr = buffer_->get_ptr(loc);
      const T *other_ptr = other.buffer_->get_ptr(other_loc);

      for (size_t i = 0; i < num; ++i) {
        if (auto cmp = ptr[i] <=> other_ptr[i]; cmp != 0)
          return cmp;
      }
    }
    return ::std::strong_ordering::equal;
  }
#endif

  HOST constexpr bool operator==(const UnifiedArray<T> &other) const noexcept {
    if (num != other.num)
      return false;

    if (buffer_ && other.buffer_) {
      auto loc = buffer_->primary_location();
      auto other_loc = other.buffer_->primary_location();
      T *ptr = buffer_->get_ptr(loc);
      const T *other_ptr = other.buffer_->get_ptr(other_loc);

      for (size_t i = 0; i < num; ++i) {
        if (ptr[i] != other_ptr[i])
          return false;
      }
    }
    return true;
  }

  // Utility methods
  HOST constexpr void clear() noexcept {
    num = 0;
    buffer_.reset();
  }

  HOST DEVICE constexpr size_t size() const noexcept { return num; }
  HOST DEVICE constexpr bool empty() const noexcept { return num == 0; }

  HOST T *data(const Resource &location = Resource::Local()) const noexcept {
    return buffer_ ? buffer_->get_ptr(location) : nullptr;
  }

#if __cplusplus >= 202002L && __has_include(<span>)
  // Modern C++20 span interface (if available)
  HOST auto span(const Resource &location = Resource::Local()) noexcept {
    return buffer_ ? std::span<T>{buffer_->get_ptr(location), num}
                   : std::span<T>{};
  }

  HOST auto span(const Resource &location = Resource::Local()) const noexcept {
    return buffer_ ? std::span<const T>{buffer_->get_ptr(location), num}
                   : std::span<const T>{};
  }
#endif

  /**
   * @brief Ensure data is available at specified location
   */
  HOST void ensure_at(const Resource &location) {
    if (buffer_) {
      buffer_->ensure_available_at(location);
    }
  }

  /**
   * @brief Get all locations where data is available
   */
  HOST std::vector<Resource> available_locations() const {
    return buffer_ ? buffer_->available_locations() : std::vector<Resource>{};
  }

  /**
   * @brief Backend-agnostic send_children using concepts
   */
#if __cplusplus >= 202002L
  template <typename U = T>
    requires(!HasSendChildren<U>)
#endif
  HOST UnifiedArray<T> send_children(const Resource &location) {
    if (num == 0) {
      return UnifiedArray<T>(0, location);
    }

    LOGINFO("UnifiedArray<{}>.send_children(...): processing {} items",
            type_name<T>(), num);

    // Create new array at target location
    UnifiedArray<T> result(num, location);

    // Copy data to target location
    T *src_ptr = buffer_->get_ptr(buffer_->primary_location());
    T *dst_ptr = result.buffer_->get_ptr(location);

    // Use backend operations for efficient copying
    auto backend_ops = get_backend_operations(location.type);
    TransferType transfer_type =
        determine_transfer_type(buffer_->primary_location(), location);

    backend_ops->copy_memory(dst_ptr, src_ptr, sizeof(T) * num, transfer_type,
                             buffer_->primary_location(), location);

    LOGINFO("UnifiedArray<{}>.send_children(...): done", type_name<T>());
    return result;
  }

#if __cplusplus >= 202002L
  template <typename U = T>
    requires HasSendChildren<U>
  HOST UnifiedArray<T> send_children(const Resource &location) {
    if (num == 0) {
      return UnifiedArray<T>(0, location);
    }

    // Create new array at target location
    UnifiedArray<T> result(num, location);
    T *dst_ptr = result.buffer_->get_ptr(location);
    T *src_ptr = buffer_->get_ptr(buffer_->primary_location());

    // Send children for each element
    for (size_t i = 0; i < num; ++i) {
      auto tmp = src_ptr[i].send_children(location);
      // Use backend operations for copying
      auto backend_ops = get_backend_operations(location.type);
      backend_ops->copy_memory(&dst_ptr[i], &tmp, sizeof(T),
                               TransferType::HOST_TO_DEVICE, Resource::Local(),
                               location);
      tmp.clear();
    }

    return result;
  }
#endif

private:
  size_t num;
  std::shared_ptr<UnifiedBuffer<T>> buffer_;

  TransferType determine_transfer_type(const Resource &src,
                                       const Resource &dst) {
    bool src_device =
        (src.type == Resource::CUDA || src.type == Resource::SYCL);
    bool dst_device =
        (dst.type == Resource::CUDA || dst.type == Resource::SYCL);

    if (!src_device && !dst_device)
      return TransferType::HOST_TO_HOST;
    if (!src_device && dst_device)
      return TransferType::HOST_TO_DEVICE;
    if (src_device && !dst_device)
      return TransferType::DEVICE_TO_HOST;
    return TransferType::DEVICE_TO_DEVICE;
  }
};

#if __cplusplus >= 202002L && __has_include(<span>)
// Deduction guides for C++20 (if available)
template <typename T> UnifiedArray(std::span<T>) -> UnifiedArray<T>;
template <typename T, size_t N> UnifiedArray(T (&)[N]) -> UnifiedArray<T>;
#endif

} // namespace ARBD
