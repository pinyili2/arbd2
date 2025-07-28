#ifdef USE_CUDA

#include "Backend/Kernels.h"
#include <cuda_runtime.h>

namespace ARBD {

// Template instantiation for the CUDA kernel wrapper
// This ensures the kernel is properly compiled as device code
template<typename Functor, typename... Args>
__global__ void cuda_kernel_wrapper(size_t n, Functor kernel, Args... args) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		kernel(i, args...);
	}
}

// Example for a simple lambda kernel
template __global__ void
cuda_kernel_wrapper<void (*)(size_t, float*), float*>(size_t n,
													  void (*kernel)(size_t, float*),
													  float* args);

// Example for a lambda with multiple parameters
template __global__ void
cuda_kernel_wrapper<void (*)(size_t, float*, int), float*, int>(size_t n,
																void (*kernel)(size_t, float*, int),
																float* arg1,
																int arg2);

// Explicit instantiations for test functors
template __global__ void
cuda_kernel_wrapper<CopyFunctor<float>, const float*, float*>(size_t n,
															  CopyFunctor<float> kernel,
															  const float* input,
															  float* output);

template __global__ void
cuda_kernel_wrapper<FillFunctor<float>, float*>(size_t n, FillFunctor<float> kernel, float* output);

} // namespace ARBD

#endif // USE_CUDA
