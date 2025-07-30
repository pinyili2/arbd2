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

// InitializeWalkersKernel instantiation
template Event launch_cuda_kernel_impl<std::tuple<>,
									   std::tuple<Buffer<Vector3_t<float>, CUDAPolicy>&>,
									   InitializeWalkersKernel>(
	const Resource& resource,
	size_t thread_count,
	const std::tuple<>& inputs,
	const std::tuple<Buffer<Vector3_t<float>, CUDAPolicy>&>& outputs,
	const KernelConfig& config,
	InitializeWalkersKernel&& kernel_func);

// RandomWalkKernel instantiation
template Event launch_cuda_kernel_impl<std::tuple<Buffer<Vector3_t<float>, CUDAPolicy>&>,
									   std::tuple<Buffer<Vector3_t<float>, CUDAPolicy>&>,
									   RandomWalkKernel>(
	const Resource& resource,
	size_t thread_count,
	const std::tuple<Buffer<Vector3_t<float>, CUDAPolicy>&>& inputs,
	const std::tuple<Buffer<Vector3_t<float>, CUDAPolicy>&>& outputs,
	const KernelConfig& config,
	RandomWalkKernel&& kernel_func);

// CalculateDistancesKernel instantiation
template Event launch_cuda_kernel_impl<std::tuple<Buffer<Vector3_t<float>, CUDAPolicy>&>,
									   std::tuple<Buffer<float, CUDAPolicy>&>,
									   CalculateDistancesKernel>(
	const Resource& resource,
	size_t thread_count,
	const std::tuple<Buffer<Vector3_t<float>, CUDAPolicy>&>& inputs,
	const std::tuple<Buffer<float, CUDAPolicy>&>& outputs,
	const KernelConfig& config,
	CalculateDistancesKernel&& kernel_func);

// SimpleKernel instantiation
template Event
launch_cuda_kernel_impl<std::tuple<Buffer<float, CUDAPolicy>&>,
						std::tuple<Buffer<float, CUDAPolicy>&>,
						SimpleKernel>(const Resource& resource,
									  size_t thread_count,
									  const std::tuple<Buffer<float, CUDAPolicy>&>& inputs,
									  const std::tuple<Buffer<float, CUDAPolicy>&>& outputs,
									  const KernelConfig& config,
									  SimpleKernel&& kernel_func);

// SmoothingFilterKernel instantiation
template Event launch_cuda_kernel_impl<std::tuple<Buffer<float, CUDAPolicy>&>,
									   std::tuple<Buffer<float, CUDAPolicy>&>,
									   SmoothingFilterKernel>(
	const Resource& resource,
	size_t thread_count,
	const std::tuple<Buffer<float, CUDAPolicy>&>& inputs,
	const std::tuple<Buffer<float, CUDAPolicy>&>& outputs,
	const KernelConfig& config,
	SmoothingFilterKernel&& kernel_func);

// GradientCalculationKernel instantiation
template Event launch_cuda_kernel_impl<std::tuple<Buffer<float, CUDAPolicy>&>,
									   std::tuple<Buffer<float, CUDAPolicy>&>,
									   GradientCalculationKernel>(
	const Resource& resource,
	size_t thread_count,
	const std::tuple<Buffer<float, CUDAPolicy>&>& inputs,
	const std::tuple<Buffer<float, CUDAPolicy>&>& outputs,
	const KernelConfig& config,
	GradientCalculationKernel&& kernel_func);
} // namespace ARBD
