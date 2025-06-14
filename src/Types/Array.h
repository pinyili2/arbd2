/*********************************************************************
 * @file  Array.h
 *
 * @brief Declaration of templated Array class.
 *********************************************************************/
#pragma once
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Proxy.h"
#include "Backend/Resource.h"
#include <cassert>
#if __cplusplus >= 202002L
#include <compare>
#include <span>
#endif
#include <memory>
#include <sstream>
#include <type_traits>

namespace ARBD {

// C++20 Concepts for better template constraints
template <typename T>
concept HasCopyToCuda = requires(T t) { t.copy_to_cuda(); };

template <typename T>
concept HasSendChildren =
    requires(T t, const Resource &r) { t.send_children(r); };

template <typename T>
concept Arithmetic = ::std::is_arithmetic_v<T>;

template <typename T>
concept Trivial = ::std::is_trivial_v<T>;

// Simple templated array object without resizing capabilities
template <typename T> struct Array {
  // Constructors
  HOST DEVICE constexpr Array() noexcept : num(0), values(nullptr) {}

  HOST constexpr explicit Array(size_t count) : num(count), values(nullptr) {
    host_allocate();
  }

  HOST Array(size_t count, const T *input) : num(count), values(nullptr) {
    host_allocate();
    if (input) {
      for (size_t i = 0; i < num; ++i) {
        values[i] = input[i];
      }
    }
  }

  // Constructor from span (C++20 feature)
  HOST explicit Array(std::span<const T> input)
      : num(input.size()), values(nullptr) {
    host_allocate();
    for (size_t i = 0; i < num; ++i) {
      values[i] = input[i];
    }
  }

  // Copy constructor
  HOST Array(const Array<T> &other) : num(other.num), values(nullptr) {
    host_allocate();
    for (size_t i = 0; i < num; ++i) {
      values[i] = other[i];
    }
  }

  // Move constructor
  HOST constexpr Array(Array<T> &&other) noexcept
      : num(other.num), values(other.values) {
    other.values = nullptr;
    other.num = 0;
  }

  // Copy assignment operator
  HOST DEVICE Array<T> &operator=(const Array<T> &other) {
    if (this != &other) {
      num = other.num;
#ifndef __CUDA_ARCH__
      host_allocate();
#endif
      for (size_t i = 0; i < num; ++i) {
        values[i] = other[i];
      }
    }
    return *this;
  }

  // Move assignment operator
  HOST DEVICE Array<T> &operator=(Array<T> &&other) noexcept {
    if (this != &other) {
#ifndef __CUDA_ARCH__
      host_deallocate();
#endif
      num = other.num;
      values = other.values;
      other.num = 0;
      other.values = nullptr;
    }
    return *this;
  }

  HOST ~Array() { host_deallocate(); }

  // Element access
  HOST DEVICE constexpr T &operator[](size_t i) noexcept {
    assert(i < num);
    return values[i];
  }

  HOST DEVICE constexpr const T &operator[](size_t i) const noexcept {
    assert(i < num);
    return values[i];
  }

  // C++20 three-way comparison
  HOST DEVICE constexpr auto operator<=>(const Array<T> &other) const noexcept
    requires ::std::three_way_comparable<T>
  {
    if (auto cmp = num <=> other.num; cmp != 0)
      return cmp;
    for (size_t i = 0; i < num; ++i) {
      if (auto cmp = values[i] <=> other.values[i]; cmp != 0)
        return cmp;
    }
    return ::std::strong_ordering::equal;
  }

  HOST DEVICE constexpr bool operator==(const Array<T> &other) const noexcept {
    if (num != other.num)
      return false;
    for (size_t i = 0; i < num; ++i) {
      if (values[i] != other.values[i])
        return false;
    }
    return true;
  }

  // Utility methods
  HOST constexpr void clear() noexcept {
    num = 0;
    values = nullptr;
  }

  HOST DEVICE constexpr size_t size() const noexcept { return num; }
  HOST DEVICE constexpr bool empty() const noexcept { return num == 0; }
  HOST constexpr T *data() const noexcept { return values; }

  // Modern C++20 span interface
  HOST constexpr ::std::span<T> span() noexcept { return {values, num}; }
  HOST constexpr ::std::span<const T> span() const noexcept {
    return {values, num};
  }

  // C++20 concepts-based send_children
  template <typename U = T>
    requires(!HasSendChildren<U>)
  HOST Array<T> send_children(const Resource &location) {
    T *values_d = nullptr;

    if (num > 0) {
      size_t sz = sizeof(T) * num;
      LOGINFO("  Array<{}>.send_children(...): allocating for {} items",
              type_name<T>(), num);

      switch (location.type) {
      case Resource::CUDA:
        gpuErrchk(cudaMalloc(&values_d, sz));
        for (size_t i = 0; i < num; ++i) {
          send(location, values[i], values_d + i);
        }
        break;
      case Resource::MPI:
        ARBD::throw_not_implemented(
            "Array<T>.send_children(location.type == MPI)");
        break;
      default:
        ARBD::throw_value_error("Unknown Resource type");
      }
    }

    LOGINFO("  Array<{}>.send_children(...): done", type_name<T>());
    return Array<T>{num, values_d};
  }

  template <typename U = T>
    requires HasSendChildren<U>
  HOST Array<T> send_children(const Resource &location) {
    T *values_d = nullptr;

    if (num > 0) {
      size_t sz = sizeof(T) * num;

      switch (location.type) {
      case Resource::CUDA:
        gpuErrchk(cudaMalloc(&values_d, sz));
        for (size_t i = 0; i < num; ++i) {
          auto tmp = values[i].send_children(location);
          send(location, tmp, values_d + i);
          tmp.clear();
        }
        break;
      case Resource::MPI:
        ARBD::throw_not_implemented(
            "Array<T>.send_children(location.type == MPI)");
        break;
      default:
        ARBD::throw_value_error("Unknown Resource type");
      }
    }

    return Array<T>{num, values_d};
  }

#ifdef USE_CUDA
  // C++20 concepts-based CUDA operations
  template <typename U = T>
    requires(!HasCopyToCuda<U>)
  HOST Array<T> *copy_to_cuda(Array<T> *dev_ptr = nullptr) const {
    if (dev_ptr == nullptr) {
      gpuErrchk(cudaMalloc(&dev_ptr, sizeof(Array<T>)));
    }

    T *values_d = nullptr;
    if (num > 0) {
      size_t sz = sizeof(T) * num;
      gpuErrchk(cudaMalloc(&values_d, sz));
      gpuErrchk(cudaMemcpy(values_d, values, sz, cudaMemcpyHostToDevice));
    }

    Array<T> tmp{0, values_d};
    tmp.num = num;
    gpuErrchk(
        cudaMemcpy(dev_ptr, &tmp, sizeof(Array<T>), cudaMemcpyHostToDevice));
    tmp.clear();

    return dev_ptr;
  }

  template <typename U = T>
    requires HasCopyToCuda<U>
  HOST Array<T> *copy_to_cuda(Array<T> *dev_ptr = nullptr) const {
    if (dev_ptr == nullptr) {
      gpuErrchk(cudaMalloc(&dev_ptr, sizeof(Array<T>)));
    }

    T *values_d = nullptr;
    if (num > 0) {
      size_t sz = sizeof(T) * num;
      gpuErrchk(cudaMalloc(&values_d, sz));

      for (size_t i = 0; i < num; ++i) {
        values[i].copy_to_cuda(values_d + i);
      }
    }

    Array<T> tmp{0, values_d};
    tmp.num = num;
    gpuErrchk(
        cudaMemcpy(dev_ptr, &tmp, sizeof(Array<T>), cudaMemcpyHostToDevice));
    tmp.clear();

    return dev_ptr;
  }

  template <typename U = T>
    requires(!HasCopyToCuda<U>)
  HOST static Array<T> copy_from_cuda(Array<T> *dev_ptr) {
    Array<T> tmp(0);
    if (dev_ptr != nullptr) {
      gpuErrchk(
          cudaMemcpy(&tmp, dev_ptr, sizeof(Array<T>), cudaMemcpyDeviceToHost));

      if (tmp.num > 0) {
        T *values_d = tmp.values;
        tmp.values = new T[tmp.num];
        size_t sz = sizeof(T) * tmp.num;
        gpuErrchk(cudaMemcpy(tmp.values, values_d, sz, cudaMemcpyDeviceToHost));
      } else {
        tmp.values = nullptr;
      }
    }
    return tmp;
  }

  template <typename U = T>
    requires HasCopyToCuda<U>
  HOST static Array<T> copy_from_cuda(Array<T> *dev_ptr) {
    Array<T> tmp(0);

    if (dev_ptr != nullptr) {
      gpuErrchk(
          cudaMemcpy(&tmp, dev_ptr, sizeof(Array<T>), cudaMemcpyDeviceToHost));

      if (tmp.num > 0) {
        T *values_d = tmp.values;
        tmp.values = new T[tmp.num];

        for (size_t i = 0; i < tmp.num; ++i) {
          tmp.values[i] = T::copy_from_cuda(values_d + i);
        }
      } else {
        tmp.values = nullptr;
      }
    }
    return tmp;
  }

  template <typename U = T>
    requires(!HasCopyToCuda<U>)
  HOST static void remove_from_cuda(Array<T> *dev_ptr,
                                    bool remove_self = true) {
    if (dev_ptr == nullptr)
      return;

    Array<T> tmp(0);
    gpuErrchk(
        cudaMemcpy(&tmp, dev_ptr, sizeof(Array<T>), cudaMemcpyDeviceToHost));

    if (tmp.num > 0) {
      gpuErrchk(cudaFree(tmp.values));
    }

    tmp.values = nullptr;
    gpuErrchk(cudaMemset((void *)&(dev_ptr->values), 0, sizeof(T *)));

    if (remove_self) {
      gpuErrchk(cudaFree(dev_ptr));
      dev_ptr = nullptr;
    }
  }

  template <typename U = T>
    requires HasCopyToCuda<U>
  HOST static void remove_from_cuda(Array<T> *dev_ptr,
                                    bool remove_self = true) {
    if (dev_ptr == nullptr)
      return;

    Array<T> tmp(0);
    gpuErrchk(
        cudaMemcpy(&tmp, dev_ptr, sizeof(Array<T>), cudaMemcpyDeviceToHost));

    if (tmp.num > 0) {
      for (size_t i = 0; i < tmp.num; ++i) {
        T::remove_from_cuda(tmp.values + i, false);
      }
      gpuErrchk(cudaFree(tmp.values));
    }

    tmp.values = nullptr;
    gpuErrchk(cudaMemset((void *)&(dev_ptr->values), 0, sizeof(T *)));

    if (remove_self) {
      gpuErrchk(cudaFree(dev_ptr));
      dev_ptr = nullptr;
    }
  }
#endif

private:
  // Private constructor for internal use
  HOST constexpr Array(size_t count, T *ptr) : num(count), values(ptr) {}

  HOST void host_allocate() {
    host_deallocate();
    if (num > 0) {
      values = new T[num];
    } else {
      values = nullptr;
    }
  }

  HOST void host_deallocate() {
    if (values != nullptr) {
      delete[] values;
    }
    values = nullptr;
  }

public:
  size_t num;
  T *__restrict__ values;
};

// Deduction guides for C++20
template <typename T> Array(std::span<T>) -> Array<T>;

template <typename T, size_t N> Array(T (&)[N]) -> Array<T>;

} // namespace ARBD
