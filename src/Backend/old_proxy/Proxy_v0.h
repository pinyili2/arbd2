#pragma once

#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Resource.h"
#include <cstring>
#include <future>

#include <typeinfo>

namespace ARBD {
// template<typename T, typename...Args1, typename... Args2>
// __global__ void proxy_sync_call_kernel_noreturn(T* addr,
// (T::*memberFunc(Args1...)), Args2...args);
#ifdef __CUDACC__
#include <cuda.h>
#include <cuda/std/utility>
#include <cuda_runtime.h>
template <typename T, typename RetType, typename... Args>
__global__ void proxy_sync_call_kernel(RetType *result, T *addr,
                                       RetType(T::*memberFunc(Args...)),
                                       Args... args) {
  if (blockIdx.x == 0) {
    *result = (addr->*memberFunc)(args...);
  }
}

template <typename T, typename... Args>
__global__ void constructor_kernel(T *__restrict__ devptr, Args... args) {
  if (blockIdx.x == 0) {
    devptr = new T{::cuda::std::forward<Args>(args)...};
  }
}
#endif

/**
 * @brief Concept to check if a type has a 'send_children' method that accepts a
 * Resource parameter.
 */
template <typename T>
concept has_send_children = requires(T t, Resource r) { t.send_children(r); };

/**
 * @brief Concept to check if a type has a 'no_send' member.
 */
template <typename T>
concept has_no_send = requires(T t) { t.no_send; };

// Helper alias for C++14 compatibility (still needed for Metadata_t)
template <typename...> using void_t = void;

// END traits

// Used by Proxy class
template <typename T, typename = void> struct Metadata_t {
  Metadata_t(const T &obj) {};
  Metadata_t(const Metadata_t<T> &other) {};
};
template <typename T>
struct Metadata_t<T, void_t<typename T::Metadata>> : T::Metadata {
  Metadata_t(const T &obj) : T::Metadata(obj) {};
  Metadata_t(const Metadata_t<T> &other) : T::Metadata(other) {};
};

template <typename T, typename Enable = void> struct Proxy {
  /**
   * @brief A smart proxy wrapper for distributed objects across different
   * compute resources.
   *
   * The Proxy template class provides a unified interface for managing objects
   * that may reside on different compute resources such as CPU, GPU (CUDA),
   * SYCL devices, or MPI processes. It handles automatic resource management,
   * metadata tracking, and provides transparent access to the underlying object
   * regardless of its physical location.
   *
   * Key Features:
   * - Resource-aware object management (CPU, GPU, SYCL)
   * - Automatic metadata handling for complex objects
   * - RAII semantics with proper copy/move constructors
   * - Transparent object access via operator->
   * - Asynchronous method execution support
   * - Compile-time type safety with SFINAE
   *
   * @tparam T The type of object being proxied. Must not be another Proxy type.
   * @tparam Enable SFINAE parameter for template specialization (defaults to
   * void)
   *
   * @example Basic Usage:
   * ```cpp
   * // Create a proxy on CPU (SYCL device 0)
   * Proxy<MyClass> proxy;
   *
   * // Create a proxy on a specific GPU resource
   * Resource gpu_resource{Resource::GPU, 1};
   * Proxy<MyClass> gpu_proxy(gpu_resource);
   *
   * // Wrap an existing object
   * MyClass obj;
   * Proxy<MyClass> wrapper(gpu_resource, obj);
   *
   * // Access the object transparently
   * wrapper->some_method();
   * ```
   *
   * @note This class uses metadata tracking for objects that define a Metadata
   * nested type. Objects without metadata are handled with a default empty
   * metadata wrapper.
   *
   * @warning Creating a Proxy of a Proxy is not allowed and will trigger a
   * static_assert.
   *
   * @see Resource for supported compute resource types
   * @see Metadata_t for metadata handling details
   */
  static_assert(!std::is_same<T, Proxy>::value,
                "Cannot make a Proxy of a Proxy object");

  Proxy()
      : location(Resource{Resource::SYCL, 0}), addr(nullptr),
        metadata(nullptr) {
    LOGINFO("Constructing Proxy<{}> @{}", typeid(T).name(),
            static_cast<void *>(this));
  };
  Proxy(const Resource &r) : location(r), addr(nullptr), metadata(nullptr) {
    LOGINFO("Constructing Proxy<{}> @{}", typeid(T).name(),
            static_cast<void *>(this));
  };
  Proxy(const Resource &r, T &obj, T *dest = nullptr)
      : location(r), addr(dest == nullptr ? &obj : dest) {
    if (dest == nullptr)
      metadata = nullptr;
    else
      metadata = new Metadata_t<T>(obj);
    LOGINFO("Constructing Proxy<{}> @{} wrapping @{} with metadata @{}",
            typeid(T).name(), static_cast<void *>(this),
            static_cast<void *>(&obj), static_cast<void *>(metadata));
  };
  // Copy constructor
  Proxy(const Proxy<T> &other)
      : location(other.location), addr(other.addr), metadata(nullptr) {
    LOGINFO("Copy Constructing Proxy<{}> @{}", typeid(T).name(),
            static_cast<void *>(this));
    if (other.metadata != nullptr) {
      const Metadata_t<T> &tmp = *(other.metadata);
      metadata = new Metadata_t<T>(tmp);
    }
  };
  Proxy<T> &operator=(const Proxy<T> &other) {
    if (this != &other) {
      // Free existing resources.
      if (metadata != nullptr)
        delete metadata;
      location = other.location;
      addr = other.addr;
      const Metadata_t<T> &tmp = *(other.metadata);
      metadata = new Metadata_t<T>(tmp); // copy construct!
      // std::copy(other.metadata, other.metadata + sizeof(Metadata_t<T>),
      // metadata);
    }
    return *this;
  };
  Proxy(Proxy<T> &&other) : addr(nullptr), metadata(nullptr) {
    LOGINFO("Move Constructing Proxy<{}> @{}", typeid(T).name(),
            static_cast<void *>(this));
    location = other.location;
    addr = other.addr;
    // For now we avoid std::move, but we may choose to change this behavior
    // const Metadata_t<T>& tmp = *(other.metadata);
    metadata = other.metadata;
    other.metadata = nullptr;
  };
  ~Proxy() {
    LOGINFO("Deconstructing Proxy<{}> @{} with metadata @{}", typeid(T).name(),
            static_cast<void *>(this), static_cast<void *>(metadata));
    if (metadata != nullptr)
      delete metadata;
  };

  /**
   * @brief Overloaded operator-> returns the address of the underlying object.
   * @return The address of the underlying object.
   */
  auto operator->() { return addr; };
  auto operator->() const { return addr; };

  /**
   * @brief The resource associated with the data represented by the proxy.
   */
  Resource location; ///< The device (thread/gpu) holding the data represented
                     ///< by the proxy.
  T *addr;           ///< The address of the underlying object.
  Metadata_t<T> *metadata; ///< T-specific metadata that resides in same memory
                           ///< space as Proxy<T>

  // Use two template parameter packs as suggested here:
  // https://stackoverflow.com/questions/26994969/inconsistent-parameter-pack-deduction-with-variadic-templates
  template <typename RetType, typename... Args1, typename... Args2>
  RetType callSync(RetType (T::*memberFunc)(Args1...), Args2 &&...args) {
    switch (location.type) {
    case Resource::SYCL:
#ifdef USE_SYCL
      if (location.is_local()) {
        // For SYCL, execute on device if addr is device memory
        auto &device = ARBD::SYCL::SYCLManager::get_current_device();
        auto &queue = device.get_next_queue();

        // Create host result storage
        RetType result;
        RetType *result_device = sycl::malloc_device<RetType>(1, queue.get());

        // Submit kernel to execute member function
        queue.get()
            .submit([&](sycl::handler &h) {
              h.single_task(
                  [=]() { *result_device = (addr->*memberFunc)(args...); });
            })
            .wait();

        // Copy result back to host
        queue.get().memcpy(&result, result_device, sizeof(RetType)).wait();
        sycl::free(result_device, queue.get());

        return result;
      } else {
#ifdef USE_MPI
        RetType result;
        MPI_Send(args..., location.id, MPI_COMM_WORLD);
        MPI_Recv(&result, sizeof(RetType), MPI_BYTE, location.id, MPI_ANY_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        return result;
#else
        ARBD::throw_not_implemented("Non-local SYCL calls require MPI support");
#endif
      }
#else
      ARBD::throw_not_implemented("SYCL support not enabled");
#endif
      break;
    case Resource::CUDA:
#ifdef __CUDACC__
      if (location.is_local()) {
        if (sizeof(RetType) > 0) {
          // Note: this only support basic RetType objects
          RetType *dest;
          RetType obj;
          gpuErrchk(cudaMalloc(&dest, sizeof(RetType)));
          proxy_sync_call_kernel<T, RetType, Args2...>
              <<<1, 32>>>(dest, addr, addr->*memberFunc, args...);
          // proxy_sync_call_kernel<><<<1,32>>>(dest, addr, addr->*memberFunc,
          // args...);
          gpuErrchk(
              cudaMemcpy(dest, &obj, sizeof(RetType), cudaMemcpyHostToDevice));
          gpuErrchk(cudaFree(dest));
          return obj;
        } else {
          ARBD::throw_not_implemented("Proxy::callSync() local GPU calls");
        }
      } else {
        size_t target_device = location.id;
        int current_device;
        gpuErrchk(cudaGetDevice(&current_device));
        gpuErrchk(cudaSetDevice(target_device));

        RetType *dest;
        RetType result;
        gpuErrchk(cudaMalloc(&dest, sizeof(RetType)));
        proxy_sync_call_kernel<T, RetType, Args2...>
            <<<1, 32>>>(dest, addr, memberFunc, args...);
        gpuErrchk(
            cudaMemcpy(&result, dest, sizeof(RetType), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaFree(dest));

        gpuErrchk(cudaSetDevice(current_device));
        return result;
        // Exception( NotImplementedError, "Proxy::callSync() non-local GPU
        // calls" );
      }
#else
      ARBD::throw_not_implemented("Proxy::callSync() for CUDA GPU only defined "
                                  "for files compiled with nvvc");
#endif
      break;
    case Resource::METAL:
#ifdef USE_METAL
      if (location.is_local()) {
        auto &device = ARBD::METAL::METALManager::get_current_device();

        // For METAL, execute on device (simplified for now)
        // Note: Full METAL implementation would require compute shaders
        RetType result = (addr->*memberFunc)(args...);
        return result;
      } else {
        ARBD::throw_not_implemented("Non-local METAL calls not supported");
      }
#else
      ARBD::throw_not_implemented("METAL support not enabled");
#endif
      break;

    default:
      ARBD::throw_value_error("Proxy::callSync(): Unknown resource type");
    }
    return RetType{};
  }

  // TODO generalize to handle void RetType
  template <typename RetType, typename... Args1, typename... Args2>
  std::future<RetType> callAsync(RetType (T::*memberFunc)(Args1...),
                                 Args2 &&...args) {
    switch (location.type) {
    case Resource::SYCL:
#ifdef USE_SYCL
      if (location.is_local()) {
        return std::async(std::launch::async, [this, memberFunc, args...] {
          auto &device = ARBD::SYCL::SYCLManager::get_current_device();
          auto &queue = device.get_next_queue();

          RetType result;
          RetType *result_device = sycl::malloc_device<RetType>(1, queue.get());

          queue.get()
              .submit([&](sycl::handler &h) {
                h.single_task(
                    [=]() { *result_device = (addr->*memberFunc)(args...); });
              })
              .wait();

          queue.get().memcpy(&result, result_device, sizeof(RetType)).wait();
          sycl::free(result_device, queue.get());

          return result;
        });
      } else {
        ARBD::throw_not_implemented("Proxy::callAsync() non-local SYCL calls");
      }
#else
      ARBD::throw_not_implemented("SYCL support not enabled");
#endif
      break;
    case Resource::CUDA:
#ifdef __CUDACC__
      if (location.is_local()) {
        return std::async(std::launch::async, [this, memberFunc, args...] {
          RetType *dest;
          RetType result;
          gpuErrchk(cudaMalloc(&dest, sizeof(RetType)));
          proxy_sync_call_kernel<T, RetType, Args2...>
              <<<1, 32>>>(dest, addr, memberFunc, args...);
          gpuErrchk(cudaMemcpy(&result, dest, sizeof(RetType),
                               cudaMemcpyDeviceToHost));
          gpuErrchk(cudaFree(dest));
          return result;
          // Exception( NotImplementedError, "Proxy::callAsync() local GPU
          // calls" );
        });
      } else {
        return std::async(std::launch::async, [this, memberFunc, args...] {
          int current_device;
          gpuErrchk(cudaGetDevice(&current_device));
          gpuErrchk(cudaSetDevice(location.id));

          RetType *dest;
          RetType result;
          gpuErrchk(cudaMalloc(&dest, sizeof(RetType)));
          proxy_sync_call_kernel<T, RetType, Args2...>
              <<<1, 32>>>(dest, addr, memberFunc, args...);
          gpuErrchk(cudaMemcpy(&result, dest, sizeof(RetType),
                               cudaMemcpyDeviceToHost));
          gpuErrchk(cudaFree(dest));

          gpuErrchk(cudaSetDevice(current_device));
          return result;
        });
      }
#else
      ARBD::throw_not_implemented(
          "Async for GPU only defined for files compiled with nvvc");
#endif
      break;
    case Resource::METAL:
#ifdef USE_METAL
      if (location.is_local()) {
        return std::async(std::launch::async, [this, memberFunc, args...] {
          auto &device = ARBD::METAL::METALManager::get_current_device();

          // For METAL, execute on device (simplified for now)
          RetType result = (addr->*memberFunc)(args...);
          return result;
        });
      } else {
        ARBD::throw_not_implemented("Proxy::callAsync() non-local METAL calls");
      }
#else
      ARBD::throw_not_implemented("METAL support not enabled");
#endif
      break;

    default:
      ARBD::throw_value_error("Proxy::callAsync(): Unknown resource type");
    }
    return std::async(std::launch::async, [] { return RetType{}; });
  }
};

// Specialization for bool/int/float types that do not have member functions
template <typename T>
struct Proxy<T, typename std::enable_if_t<std::is_arithmetic<T>::value>> {
  /**
   * @brief Default constructor initializes the location to a default CPU
   * resource and the address to nullptr.
   */
  Proxy() : location{Resource{Resource::SYCL, 0}}, addr{nullptr} {};
  Proxy(const Resource &r, T *obj) : location{r}, addr{obj} {};

  /**
   * @brief Overloaded operator-> returns the address of the underlying object.
   * @return The address of the underlying object.
   */
  auto operator->() { return addr; }
  auto operator->() const { return addr; }

  /**
   * @brief The resource associated with the data represented by the proxy.
   */
  Resource location; ///< The device (thread/gpu) holding the data represented
                     ///< by the proxy.
  T *addr;           ///< The address of the underlying object.
};

/**
 * @brief Template function to send data ignoring children to a specified
 * location.
 * @tparam T The type of the data to be sent.
 * @param location The destination resource for the data.
 * @param obj The data to be sent.
 * @param dest Optional parameter to provide a pre-allocated destination. If not
 * provided, memory is allocated.
 * @return A Proxy representing the data at the destination location.
 */
template <typename T>
HOST inline Proxy<T> _send_ignoring_children(const Resource &location, T &obj,
                                             T *dest = nullptr) {
  LOGTRACE("   _send_ignoring_children...");
  switch (location.type) {
  case Resource::CUDA:
    LOGINFO("   GPU...");
#ifdef USE_CUDA
    if (location.is_local()) {
      if (dest == nullptr) { // allocate if needed
        LOGTRACE("   cudaMalloc for array");
        gpuErrchk(cudaMalloc(&dest, sizeof(T)));
      }
      gpuErrchk(cudaMemcpy(dest, &obj, sizeof(T), cudaMemcpyHostToDevice));
    } else {
      ARBD::throw_not_implemented(
          "`_send_ignoring_children(...)` on non-local GPU");
    }
#else
    ARBD::throw_not_implemented("USE_CUDA is not enabled");
#endif
    break;
  case Resource::SYCL:
    LOGINFO("Using SYCL...");
#ifdef USE_SYCL
    if (location.is_local()) {
      LOGINFO("   local SYCL...");
      if (dest == nullptr) { // allocate if needed
        LOGTRACE("   SYCL allocate memory for {}", typeid(T).name());
        auto &device = ARBD::SYCL::SYCLManager::get_current_device();
        auto &queue = device.get_next_queue();
        dest = sycl::malloc_device<T>(1, queue.get());
        if (!dest) {
          ARBD::throw_value_error("SYCL allocation failed");
        }
      }
      LOGINFO("   SYCL memcpying...");
      auto &device = ARBD::SYCL::SYCLManager::get_current_device();
      auto &queue = device.get_next_queue();
      queue.get().memcpy(dest, &obj, sizeof(T)).wait();
    } else {
      ARBD::throw_not_implemented(
          "`_send_ignoring_children(...)` on non-local SYCL");
    }
#else
    ARBD::throw_not_implemented("USE_SYCL is not enabled");
#endif
    break;
  case Resource::METAL:
    LOGINFO("Using METAL...");
#ifdef USE_METAL
    if (location.is_local()) {
      LOGINFO("   local METAL...");
      auto &device = ARBD::METAL::METALManager::get_current_device();
      if (dest == nullptr) {
        dest = device.allocate<T>(1);
        if (!dest) {
          ARBD::throw_metal_error("METAL allocation failed");
        }
      }
      device.copy_to_device(&obj, dest, sizeof(T));
    } else {
      ARBD::throw_not_implemented(
          "`_send_ignoring_children(...)` on non-local METAL");
    }
#else
    ARBD::throw_not_implemented("USE_METAL is not enabled");
#endif
    break;
  default:
    // error
    ARBD::throw_value_error(
        "`_send_ignoring_children(...)` applied with unkown resource type");
  }

  LOGINFO("   creating Proxy...");
  // Proxy<T>* ret = new Proxy<T>(location, dest); // Proxies should be
  // explicitly removed LOGINFO("   ...done @{}", fmt::ptr(ret)); Proxy<T>&& ret
  // =
  return Proxy<T>(location, obj, dest); // Proxies should be explicitly removed

  // LOGINFO("   ...done @{}", fmt::ptr(&ret));
  // return ret;
  // LOGINFO("   ...done @{}", fmt::ptr(ret));
  // return *ret;
}

/**
 * @brief Template function to send simple objects to a specified location
 * without considering child objects. This version will be selected upon
 * send(location, obj) if obj.send_children does not exist (C++20 concepts)
 * @tparam T The type of the data to be sent.
 * @param location The destination resource for the data.
 * @param obj The data to be sent.
 * @param dest Optional parameter to provide a pre-allocated destination. If not
 * provided, memory is allocated.
 * @return A Proxy representing the data at the destination location.
 */
template <typename T>
  requires(!has_send_children<T>)
HOST inline Proxy<T> &send(const Resource &location, T &obj,
                           T *dest = nullptr) {
  LOGINFO("...Sending object {} @{} to device at {}", typeid(T).name(),
          static_cast<void *>(&obj), static_cast<void *>(dest));
  // Simple objects can simply be copied without worrying about contained
  // objects and arrays
  Proxy<T> &&ret = _send_ignoring_children(location, obj, dest);
  LOGTRACE("...done sending");
  // printf("...done\n");
  return ret;
}

/**
 * @brief Template function to send more complex objects to a specified
 * location. This version will be selected upon send(location, obj) if
 * obj.send_children exists (C++20 concepts)
 * @tparam T The type of the data to be sent.
 * @param location The destination resource for the data.
 * @param obj The data to be sent.
 * @param dest Optional parameter to provide a pre-allocated destination. If not
 * provided, memory is allocated on the GPU.
 * @return A Proxy representing the data at the destination location.
 */
template <typename T>
  requires has_send_children<T>
HOST inline Proxy<T> send(const Resource &location, T &obj, T *dest = nullptr) {
  // static_assert(!has_no_send<T>());
  LOGINFO("Sending complex object {} @{} to device at {}", typeid(T).name(),
          static_cast<void *>(&obj), static_cast<void *>(dest));
  auto dummy = obj.send_children(
      location); // function is expected to return an object of type obj with
                 // all pointers appropriately assigned to valid pointers on
                 // location
  Proxy<T> ret = _send_ignoring_children<T>(location, dummy, dest);
  LOGTRACE("... clearing dummy complex object");
  dummy.clear();
  LOGTRACE("... done sending");
  return ret;
}

// Utility function for constructing objects in remote memory address
// spaces, obviating the need to construct simple objects locally
// before copying. Returns a Proxy object, but in cases where the
// remote resource location is non-CPU or non-local, metadata for
// Proxy will be blank.
template <typename T, typename... Args>
Proxy<T> construct_remote(Resource location, Args &&...args) {
  // static_assert(!has_no_send<T>());
  switch (location.type) {
  case Resource::SYCL:
#ifdef USE_SYCL
    if (location.is_local()) {
      auto &device = ARBD::SYCL::SYCLManager::get_current_device();
      auto &queue = device.get_next_queue();
      T *devptr = sycl::malloc_device<T>(1, queue.get());
      if (!devptr) {
        ARBD::throw_sycl_error("SYCL allocation failed in construct_remote");
      }
      // Create object on host then copy to device
      T host_obj{std::forward<Args>(args)...};
      queue.get().memcpy(devptr, &host_obj, sizeof(T)).wait();
      return Proxy<T>(location, host_obj, devptr);
    } else {
      ARBD::throw_not_implemented("construct_remote() non-local SYCL calls");
    }
#else
    ARBD::throw_not_implemented("SYCL support not enabled");
#endif
    break;
  case Resource::CUDA:
#ifdef __CUDACC__
    if (location.is_local()) {
      T *devptr;
      LOGWARN(
          "construct_remote: TODO: switch to device associated with location");
      gpuErrchk(cudaMalloc(&devptr, sizeof(T)));
      constructor_kernel<<<1, 32>>>(devptr, std::forward<Args>(args)...);
      gpuErrchk(cudaDeviceSynchronize());
      LOGWARN("construct_remote: proxy.metadata not set");
      return Proxy<T>(location);
      // Exception( NotImplementedError, "cunstruct_remote() local GPU call" );
      // Note: this only support basic RetType objects
      // T* dest;
      // T obj;
      // gpuErrchk(cudaMalloc(&dest, sizeof(RetType)));
      // proxy_sync_call_kernel<T, RetType, Args2...><<<1,32>>>(dest, addr,
      // addr->*memberFunc, args...); 	gpuErrchk(cudaMemcpy(dest, &obj,
      // sizeof(RetType), cudaMemcpyHostToDevice));
      // gpuErrchk(cudaFree(dest));
    } else {
      ARBD::throw_not_implemented("cunstruct_remote() non-local GPU call");
    }
#else
    ARBD::throw_not_implemented(
        "construct_remote() for GPU only defined for files compiled with nvvc");
#endif
    break;
  case Resource::METAL:
#ifdef USE_METAL
    if (location.is_local()) {
      auto &device = ARBD::METAL::METALManager::get_current_device();
      T *devptr = device.allocate<T>(1);
      if (!devptr) {
        ARBD::throw_metal_error(
            "METAL allocation failed in construct_remote");
      }
      // Create object on host then copy to device
      T host_obj{std::forward<Args>(args)...};
      device.copy_to_device(&host_obj, devptr, sizeof(T));
      return Proxy<T>(location, host_obj, devptr);
    } else {
      ARBD::throw_not_implemented("construct_remote() non-local METAL calls");
    }
#else
    ARBD::throw_not_implemented("METAL support not enabled");
#endif
    break;

  default:
    ARBD::throw_value_error("construct_remote(): unknown resource type");
  }
  return Proxy<T>{};
}

}; // namespace ARBD
