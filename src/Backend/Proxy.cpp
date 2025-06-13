#include "ARBDException.h"
#include "ARBDLogger.h"
#include <cstring>
#include <future>

#ifdef USE_SYCL
#include "Backend/SYCL/SYCLManager.h"
#include <sycl/sycl.hpp>
#endif

#ifdef USE_METAL
#include "Backend/METAL/METALManager.h"
#endif
#include "Proxy.h"

namespace ARBD {
namespace ProxyImpl {

// =============================================================================
// SYCL Implementation Functions
// =============================================================================

#ifdef USE_SYCL
void *sycl_call_sync(void *addr, void *func_ptr, void *args, size_t args_size,
                     const Resource &location, size_t result_size) {
  if (location.is_local()) {
    // For SYCL, execute on device if addr is device memory
    auto &device = ARBD::SYCL::SYCLManager::get_current_device();
    auto &queue = device.get_next_queue();

    // Create host result storage
    void *result = new char[result_size];
    void *result_device = sycl::malloc_device<char>(result_size, queue.get());

    // Note: This is a simplified version. In practice, you'd need to handle
    // the member function call properly with SYCL kernels
    queue.get()
        .submit([&](sycl::handler &h) {
          h.single_task([=]() {
            // This is conceptual - actual implementation would need
            // to handle member function calls properly
            std::memcpy(result_device, addr, result_size);
          });
        })
        .wait();

    // Copy result back to host
    queue.get().memcpy(result, result_device, result_size).wait();
    sycl::free(result_device, queue.get());

    return result;
  } else {
    ARBD::throw_not_implemented(
        "Non-local SYCL calls require distributed computing support");
  }
}

std::future<void *> sycl_call_async(void *addr, void *func_ptr, void *args,
                                    size_t args_size, const Resource &location,
                                    size_t result_size) {
  if (location.is_local()) {
    return std::async(std::launch::async, [=] {
      return sycl_call_sync(addr, func_ptr, args, args_size, location,
                            result_size);
    });
  } else {
    ARBD::throw_not_implemented(
        "Proxy::callAsync() non-local SYCL calls not supported");
  }
}
#else
void *sycl_call_sync(void *addr, void *func_ptr, void *args, size_t args_size,
                     const Resource &location, size_t result_size) {
  ARBD::throw_not_implemented("SYCL support not enabled");
}

::std::future<void *> sycl_call_async(void *addr, void *func_ptr, void *args,
                                      size_t args_size,
                                      const Resource &location,
                                      size_t result_size) {
  ARBD::throw_not_implemented("SYCL support not enabled");
}
#endif

// =============================================================================
// METAL Implementation Functions
// =============================================================================

#ifdef USE_METAL
void *metal_call_sync(void *addr, void *func_ptr, void *args, size_t args_size,
                      const Resource &location, size_t result_size) {
  if (location.is_local()) {
    auto &device = ARBD::METAL::METALManager::get_current_device();

    // Create result storage
    void *result = new char[result_size];

    // For METAL, execute on device (simplified for now)
    // Note: Full METAL implementation would require compute shaders
    // This is a placeholder implementation
    std::memcpy(result, addr, std::min(result_size, sizeof(void *)));

    return result;
  } else {
    ARBD::throw_not_implemented("Non-local METAL calls not supported");
  }
}

std::future<void *> metal_call_async(void *addr, void *func_ptr, void *args,
                                     size_t args_size, const Resource &location,
                                     size_t result_size) {
  if (location.is_local()) {
    return std::async(std::launch::async, [=] {
      return metal_call_sync(addr, func_ptr, args, args_size, location,
                             result_size);
    });
  } else {
    ARBD::throw_not_implemented("Proxy::callAsync() non-local METAL calls");
  }
}
#else
void *metal_call_sync(void *addr, void *func_ptr, void *args, size_t args_size,
                      const Resource &location, size_t result_size) {
  ARBD::throw_not_implemented("METAL support not enabled");
}

::std::future<void *> metal_call_async(void *addr, void *func_ptr, void *args,
                                       size_t args_size,
                                       const Resource &location,
                                       size_t result_size) {
  ARBD::throw_not_implemented("METAL support not enabled");
}
#endif

// =============================================================================
// General Send Implementation Functions
// =============================================================================

template <typename T>
void *send_ignoring_children(const Resource &location, T &obj, T *dest) {
  switch (location.type) {
  case Resource::CUDA:
    LOGINFO("   GPU...");
#ifdef USE_CUDA
    if (location.is_local()) {
      if (dest == nullptr) {
        LOGTRACE("   cudaMalloc for array");
        cudaMalloc(&dest, sizeof(T));
      }
      cudaMemcpy(dest, &obj, sizeof(T), cudaMemcpyHostToDevice);
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
      if (dest == nullptr) {
        LOGTRACE("   SYCL allocate memory for {}", typeid(T).name());
        auto &device = ARBD::SYCL::SYCLManager::get_current_device();
        auto &queue = device.get_next_queue();
        dest = sycl::malloc_device<T>(1, queue.get());
        if (!dest) {
          ARBD::throw_sycl_error("SYCL allocation failed");
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
    ARBD::throw_notemented("USE_SYCL is not enabled");
#endif
    break;

  case Resource::METAL:
    LOGINFO("Using METAL...");
#ifdef USE_METAL
    if (location.is_local()) {
      LOGINFO("   local METAL...");
      auto &device = ARBD::METAL::METALManager::get_current_device();
      if (dest == nullptr) {
        // Note: This is conceptual - actual METAL allocation would be different
        dest = static_cast<T *>(std::malloc(sizeof(T)));
        if (!dest) {
          ARBD::throw_metal_error("METAL allocation failed");
        }
      }
      // Note: This is conceptual - actual METAL copy would use Metal APIs
      std::memcpy(dest, &obj, sizeof(T));
    } else {
      ARBD::throw_notemented(
          "`_send_ignoring_children(...)` on non-local METAL");
    }
#else
    ARBD::throw_metal_error("METAL is not enabled");
#endif
    break;

  default:
    ARBD::throw_value_error(
        "`_send_ignoring_children(...)` applied with unknown resource type");
  }

  return dest;
}

template <typename T>
void *construct_remote(const Resource &location, void *args, size_t args_size) {
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

      // For SYCL, we need to construct on host then copy
      // Note: This is simplified - proper implementation would handle args
      // properly
      T *temp_obj = new T(); // Would construct with args in practice
      queue.get().memcpy(devptr, temp_obj, sizeof(T)).wait();
      delete temp_obj;

      return devptr;
    } else {
      ARBD::throw_not_implemented("construct_remote() non-local SYCL calls");
    }
#else
    ARBD::throw_notemented("SYCL support not enabled");
#endif
    break;

  case Resource::CUDA:
    // CUDA implementation is in Proxy.cu
    ARBD::throw_not_implemented(
        "CUDA construct_remote should be handled in Proxy.cu");
    break;

  case Resource::METAL:
#ifdef USE_METAL
    if (location.is_local()) {
      // Note: This is conceptual - actual METAL allocation would be different
      T *devptr = static_cast<T *>(std::malloc(sizeof(T)));
      if (!devptr) {
        ARBD::throw_metal_error("METAL allocation failed in construct_remote");
      }

      // For METAL, we need to construct on host then copy
      // Note: This is simplified - proper implementation would handle args
      // properly
      new (devptr) T(); // Would construct with args in practice

      return devptr;
    } else {
      ARBD::throw_notemented("construct_remote() non-local METAL calls");
    }
#else
    ARBD::throw_not_implemented("METAL support not enabled");
#endif
    break;

  default:
    ARBD::throw_value_error("construct_remote(): unknown resource type");
  }

  return nullptr;
}

// Explicit template instantiations for common types
template void *send_ignoring_children<int>(const Resource &, int &, int *);
template void *send_ignoring_children<float>(const Resource &, float &,
                                             float *);
template void *send_ignoring_children<double>(const Resource &, double &,
                                              double *);
// template void* send_ignoring_children<Vector3>(const Resource&,
// Vector3&, Vector3*);

template void *construct_remote<int>(const Resource &, void *, size_t);
template void *construct_remote<float>(const Resource &, void *, size_t);
template void *construct_remote<double>(const Resource &, void *, size_t);
// template void* construct_remote<Vector3>(const Resource&, void*,
// size_t);

} // namespace ProxyImpl
} // namespace ARBD
