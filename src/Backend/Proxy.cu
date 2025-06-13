#ifdef USE_CUDA
#include "Backend/CUDA/CUDAManager.h"
#include "Proxy.h"
#include <cuda.h>
#include <cuda/std/utility>
#include <cuda_runtime.h>

namespace ARBD {

// CUDA kernel declarations
template <typename T, typename RetType, typename... Args>
__global__ void proxy_sync_call_kernel(RetType *result, T *addr,
                                       RetType (T::*memberFunc)(Args...),
                                       Args... args) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *result = (addr->*memberFunc)(args...);
  }
}

template <typename T, typename... Args>
__global__ void constructor_kernel(T *__restrict__ devptr, Args... args) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    new (devptr) T{args...};
  }
}

namespace ProxyImpl {

// =============================================================================
// CUDA Implementation Functions
// =============================================================================

void *cuda_call_sync(void *addr, void *func_ptr, void *args, size_t args_size,
                     const Resource &location, size_t result_size) {
  if (location.is_local()) {
    if (result_size > 0) {
      // Allocate result storage on device
      void *result_device = nullptr;
      void *result_host = new char[result_size];

      CUDA_CHECK(cudaMalloc(&result_device, result_size));

      // Note: This is a simplified version. In practice, you'd need to unpack
      // the arguments and member function pointer properly and call the
      // appropriate kernel template instantiation

      // For now, just copy the object as a placeholder
      CUDA_CHECK(cudaMemcpy(result_device, addr,
                            std::min(result_size, sizeof(void *)),
                            cudaMemcpyDeviceToDevice));

      CUDA_CHECK(cudaMemcpy(result_host, result_device, result_size,
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(result_device));

      return result_host;
    } else {
      ARBD::throw_not_implemented("Proxy::callSync() void return type on GPU");
    }
  } else {
    // Multi-GPU case
    size_t target_device = location.id;
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    CUDA_CHECK(cudaSetDevice(target_device));

    void *result_device = nullptr;
    void *result_host = new char[result_size];

    CUDA_CHECK(cudaMalloc(&result_device, result_size));

    // Similar simplified implementation for multi-GPU
    CUDA_CHECK(cudaMemcpy(result_device, addr,
                          std::min(result_size, sizeof(void *)),
                          cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaMemcpy(result_host, result_device, result_size,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(result_device));
    CUDA_CHECK(cudaSetDevice(current_device));

    return result_host;
  }
}

std::future<void *> cuda_call_async(void *addr, void *func_ptr, void *args,
                                    size_t args_size, const Resource &location,
                                    size_t result_size) {
  if (location.is_local()) {
    return std::async(std::launch::async, [=] {
      return cuda_call_sync(addr, func_ptr, args, args_size, location,
                            result_size);
    });
  } else {
    return std::async(std::launch::async, [=] {
      int current_device;
      CUDA_CHECK(cudaGetDevice(&current_device));
      CUDA_CHECK(cudaSetDevice(location.id));

      void *result = cuda_call_sync(addr, func_ptr, args, args_size, location,
                                    result_size);

      CUDA_CHECK(cudaSetDevice(current_device));
      return result;
    });
  }
}

// Specialized CUDA construct_remote implementation
template <typename T>
void *cuda_construct_remote(const Resource &location, void *args,
                            size_t args_size) {
  if (location.is_local()) {
    T *devptr = nullptr;
    LOGWARN(
        "construct_remote: TODO: switch to device associated with location");
    CUDA_CHECK(cudaMalloc(&devptr, sizeof(T)));

    // Note: This is simplified. In practice, you'd need to unpack args
    // and call the appropriate constructor kernel template instantiation
    constructor_kernel<T><<<1, 32>>>(devptr /* args would go here */);
    CUDA_CHECK(cudaDeviceSynchronize());

    return devptr;
  } else {
    ARBD::throw_not_implemented("construct_remote() non-local GPU call");
  }
}

// Template instantiations for common types to help with CUDA compilation
template void *cuda_construct_remote<int>(const Resource &, void *, size_t);
template void *cuda_construct_remote<float>(const Resource &, void *, size_t);
template void *cuda_construct_remote<double>(const Resource &, void *, size_t);

// Specialized CUDA send implementation (if needed)
template <typename T>
void *cuda_send_ignoring_children(const Resource &location, T &obj, T *dest) {
  if (location.is_local()) {
    if (dest == nullptr) {
      LOGTRACE("   cudaMalloc for object");
      CUDA_CHECK(cudaMalloc(&dest, sizeof(T)));
    }
    CUDA_CHECK(cudaMemcpy(dest, &obj, sizeof(T), cudaMemcpyHostToDevice));
    return dest;
  } else {
    ARBD::throw_not_implemented("CUDA send to non-local GPU");
  }
}

// Template instantiations for CUDA send operations
template void *cuda_send_ignoring_children<int>(const Resource &, int &, int *);
template void *cuda_send_ignoring_children<float>(const Resource &, float &,
                                                  float *);
template void *cuda_send_ignoring_children<double>(const Resource &, double &,
                                                   double *);

} // namespace ProxyImpl

// Explicit template instantiations for commonly used kernel templates
// This helps reduce compilation time and ensures the kernels are available

// Constructor kernel instantiations
template __global__ void constructor_kernel<int>(int *, int);
template __global__ void constructor_kernel<float>(float *, float);
template __global__ void constructor_kernel<double>(double *, double);

// Proxy call kernel instantiations for common scenarios
// Note: These would need to be expanded based on actual usage patterns
template __global__ void proxy_sync_call_kernel<int, int>(int *, int *,
                                                          int (int ::*)(), );
template __global__ void
proxy_sync_call_kernel<float, float>(float *, float *, float (float ::*)(), );

} // namespace ARBD

#endif // USE_CUDA
