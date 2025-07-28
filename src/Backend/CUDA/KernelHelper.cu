#include "Backend/Kernels.h"
#include "CUDAManager.h"
#include <cuda_runtime.h>

namespace ARBD {

// CUDA kernel wrapper implementation - only in .cu files
template<typename Functor, typename... Args>
__global__ void cuda_kernel_wrapper(size_t n, Functor kernel, Args... args) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		kernel(i, args...);
	}
}

// Lambda kernel wrapper for common use cases
template<typename F>
__global__ void lambda_kernel_wrapper(size_t n, F functor) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		functor(i);
	}
}

// CUDA-specific launch implementation that properly handles <<<>>> syntax
template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
Event launch_cuda_kernel_impl(const Resource& resource,
							  size_t thread_count,
							  const InputTuple& inputs,
							  const OutputTuple& outputs,
							  const KernelConfig& config,
							  Functor&& kernel_func,
							  Args&&... args) {

	KernelConfig local_config = config;
	local_config.auto_configure(thread_count);

	dim3 grid(local_config.grid_size.x, local_config.grid_size.y, local_config.grid_size.z);
	dim3 block(local_config.block_size.x, local_config.block_size.y, local_config.block_size.z);

	// Get device and stream
	auto& device =
		const_cast<CUDA::CUDAManager::Device&>(CUDA::CUDAManager::devices()[resource.id]);
	auto stream = device.get_next_stream();

	// Handle dependencies
	for (const auto& dep_event : config.dependencies.get_cuda_events()) {
		CUDA_CHECK(cudaStreamWaitEvent(stream, dep_event, 0));
	}

	cudaEvent_t event;
	CUDA_CHECK(cudaEventCreate(&event));

	// Get buffer pointers
	auto input_pointers = get_buffer_pointers(inputs);
	auto output_pointers = get_buffer_pointers(outputs);
	auto kernel_args = std::tuple_cat(input_pointers,
									  output_pointers,
									  std::make_tuple(std::forward<Args>(args)...));

	std::apply(
		[&](auto&&... unpacked_args) {
			cuda_kernel_wrapper<<<grid, block, local_config.shared_memory, stream>>>(
				thread_count,
				kernel_func,
				unpacked_args...);
		},
		kernel_args);

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaEventRecord(event, stream));

	if (!config.async) {
		cudaEventSynchronize(event);
	}

	return Event(event, resource);
}

} // namespace ARBD
