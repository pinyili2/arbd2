#include "Backend/Buffer.h"
#include "Backend/CUDA/KernelHelper.cuh"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "Kernel_for_test.h"

// This file contains the explicit template instantiations needed by CUDA.
// You tell nvcc exactly which kernels to build.
namespace ARBD {
using BufferFloat = DeviceBuffer<float>;

template Event
launch_cuda_kernel_impl<std::tuple<BufferFloat&>, std::tuple<BufferFloat&>, TransformKernel>(
	const Resource&,
	size_t,
	const std::tuple<BufferFloat&>&,
	const std::tuple<BufferFloat&>&,
	const KernelConfig&,
	TransformKernel&&);

template Event launch_cuda_kernel_impl<std::tuple<BufferFloat&, BufferFloat&>,
									   std::tuple<BufferFloat&>,
									   CombineKernel>(const Resource&,
													  size_t,
													  const std::tuple<BufferFloat&, BufferFloat&>&,
													  const std::tuple<BufferFloat&>&,
													  const KernelConfig&,
													  CombineKernel&&);

// For UniformFunctor<int>
using BufferInt = DeviceBuffer<int>;
} // namespace ARBD
