#pragma once
#include <functional>
#include <future>
#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>
#ifdef USE_CUDA
#include "CUDA/CUDAManager.h"
#include <cuda_runtime.h>
#endif

#ifdef USE_SYCL
#include "SYCL/SYCLManager.h"
#include <sycl/sycl.hpp>
#endif

#ifdef USE_METAL
#include "METAL/METALManager.h"
#include "Metal/Metal.hpp"
#endif
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Buffer.h"
#include "Events.h"
#include "Resource.h"

namespace ARBD {

struct kerneldim3 {
	size_t x = 1, y = 1, z = 1;
};

// The configuration "policy" for launching any kernel
struct KernelConfig {
	kerneldim3 grid_size = {0,
							0,
							0}; // Grid dimensions. A zero value in x signals auto-calculation.
	kerneldim3 block_size = {256, 1, 1}; // Block/Work-group dimensions.
	size_t shared_memory = 0;			 // Shared memory in bytes (primarily for CUDA).
	bool async = false;					 // If false, the host will wait for completion.
	EventList dependencies;				 // Events this kernel must wait for.

	// Auto-configure grid size for a 1D problem if not specified
	void auto_configure(size_t thread_count) {
		if (grid_size.x == 0 && grid_size.y == 0 && grid_size.z == 0) {
			if (block_size.x > 0) {
				grid_size.x = (thread_count + block_size.x - 1) / block_size.x;
			}
			grid_size.y = 1;
			grid_size.z = 1;
		}
		// NOTE: You could add logic here for 2D/3D auto-configuration if needed
	}
};

// Generic kernel function signature
template<typename... Args>
using KernelFunction = std::function<void(size_t, Args...)>;

template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
Event launch_kernel(const Resource& resource,
					size_t thread_count,
					InputTuple& inputs,
					OutputTuple& outputs,
					const KernelConfig& config,
					Functor&& kernel_func,
					Args&&... args) {
	return launch_kernel<BackendPolicy>(resource,
										thread_count,
										inputs,
										outputs,
										config,
										std::forward<Functor>(kernel_func),
										std::forward<Args>(args)...);
}

// Overload for Name-based backends (Metal)
template<typename Backend, typename... Args>
Event launch_kernel(const Resource& resource,
					size_t thread_count,
					const KernelConfig& config,
					const std::string& kernel_name, // Takes a kernel name
					Args&&... args) {
	if constexpr (std::is_same_v<Backend, METALBackend>) {
		return launch_metal_kernel(resource,
								   thread_count,
								   config,
								   kernel_name,
								   std::forward<Args>(args)...);
	} else {
		static_assert(std::is_same_v<Backend, METALBackend>,
					  "This launch_kernel overload is only for name-based backends like Metal.");
	}
}

#ifdef USE_CUDA
template<typename Functor, typename... Args>
__global__ void cuda_kernel_wrapper(size_t n, Functor kernel, Args... args) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		kernel(i, args...);
	}
}

template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
Event launch_cuda_kernel(const Resource& resource,
						 size_t thread_count,
						 InputTuple& inputs,
						 OutputTuple& outputs,
						 Functor&& kernel_func,
						 const KernelConfig& config = {},
						 Args... args) {

	KernelConfig local_config = config;
	local_config.auto_configure(thread_count);

	kerneldim3 grid(local_config.grid_size);
	kerneldim3 block(local_config.block_size);

	// Wait for dependencies
	config.dependencies.wait_all();

	cudaEvent_t event;
	cudaEventCreate(&event);

	// Get pointers from buffer objects and launch kernel
	EventList deps;
	auto input_ptr = inputs.get_read_access(deps);
	auto output_ptr = outputs.get_write_access(deps);

	cuda_kernel_wrapper<<<grid, block, local_config.shared_memory>>>(thread_count,
																	 kernel_func,
																	 input_ptr,
																	 output_ptr,
																	 args...);

	cudaEventRecord(event);

	if (!config.async) {
		cudaEventSynchronize(event);
	}

	return Event(event, resource);
}
#endif

#ifdef USE_SYCL
template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
Event launch_sycl_kernel(const Resource& resource,
						 size_t thread_count,
						 Functor&& kernel_func,
						 const KernelConfig& config = {},
						 Args... args) {

	KernelConfig local_config = config;
	local_config.auto_configure(thread_count);

	sycl::range<1> global_range(local_config.grid_size.x * local_config.block_size.x);
	sycl::range<1> local_range(local_config.block_size.x);
	sycl::nd_range<1> execution_range(global_range, local_range);

	auto& queue =
		resource.get_sycl_queue(); // Assuming you can get the queue from your resource object

	auto sycl_event = queue.submit([&](sycl::handler& h) {
		// 2. Pass dependencies to the handler instead of waiting on the host
		for (const auto& dep : config.dependencies) {
			h.depends_on(
				dep.get_sycl_event()); // Assumes your Event class can expose the sycl event
		}

		// 3. Create accessors inside the handler for automatic dependency management
		auto input_accessor =
			inputs.get_sycl_accessor(h); // Your buffer class would need this method
		auto output_accessor = outputs.get_sycl_accessor(
			h); // e.g., return sycl::accessor(my_buffer, h, sycl::read_only);

		// 4. Launch with nd_range and pass accessors to the kernel
		h.parallel_for(execution_range, [=](sycl::nd_item<1> item) {
			size_t i = item.get_global_id(0);
			if (i < thread_count) {
				kernel_func(i, input_accessor, output_accessor, args...);
			}
		});
	});

	if (!config.async) {
		sycl_event.wait();
	}

	return Event(sycl_event, resource);
}

#endif

#ifdef USE_METAL
template<typename InputTuple, typename OutputTuple, typename... Args>
Event launch_metal_kernel(const Resource& resource,
						  size_t thread_count,
						  InputTuple& inputs,
						  OutputTuple& outputs,
						  const std::string& kernel_name,
						  const KernelConfig& config = {},
						  Args... args) {

	// Wait for dependencies
	config.dependencies.wait_all();

	// Get Metal pipeline state
	MTL::ComputePipelineState* pipeline =
		METAL::METALManager::get_compute_pipeline_state(kernel_name);

	auto& device = METAL::METALManager::get_current_device();
	auto& queue = device.get_next_queue();

	// Create command buffer and encoder
	void* cmd_buffer_ptr = queue.create_command_buffer();
	MTL::CommandBuffer* cmd_buffer = static_cast<MTL::CommandBuffer*>(cmd_buffer_ptr);
	MTL::ComputeCommandEncoder* encoder = cmd_buffer->computeCommandEncoder();

	encoder->setComputePipelineState(pipeline);

	// Get pointers from buffer objects and set buffer arguments
	EventList deps;
	auto input_ptr = inputs.get_read_access(deps);
	auto output_ptr = outputs.get_write_access(deps);

	int buffer_index = 0;
	void* input_buffer_ptr = METAL::METALManager::get_metal_buffer_from_ptr(
		const_cast<void*>(static_cast<const void*>(input_ptr)));
	void* output_buffer_ptr = METAL::METALManager::get_metal_buffer_from_ptr(output_ptr);

	encoder->setBuffer(static_cast<const MTL::Buffer*>(input_buffer_ptr), 0, buffer_index++);
	encoder->setBuffer(static_cast<const MTL::Buffer*>(output_buffer_ptr), 0, buffer_index++);

	// Configure grid size
	KernelConfig local_config = config;
	local_config.auto_configure(thread_count);

	// Dispatch
	MTL::Size grid_size = MTL::Size::Make(local_config.grid_size.x,
										  local_config.grid_size.y,
										  local_config.grid_size.z);
	// 1. Get the max size from the pipeline state.
	NS::UInteger max_threads_per_group = pipeline->maxTotalThreadsPerThreadgroup();

	// 2. Ensure your desired block size is also an NS::UInteger.
	NS::UInteger desired_threads_per_group = config.block_size.x;

	// 3. Now std::min works correctly as both types match.
	NS::UInteger final_threads_per_group =
		std::min(desired_threads_per_group, max_threads_per_group);
	// Dispatch the kernel
	MTL::Size threadgroup_size = MTL::Size::Make(local_config.block_size.x,
												 local_config.block_size.y,
												 local_config.block_size.z);

	encoder->dispatchThreads(grid_size, threadgroup_size);
	encoder->endEncoding();

	ARBD::METAL::Event metal_event(cmd_buffer_ptr);
	metal_event.commit();

	if (!config.async) {
		metal_event.wait();
	}

	return Event(metal_event, resource);
}
#endif

// CPU fallback implementation (always available)
template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
Event launch_cpu_kernel(const Resource& resource,
						size_t thread_count,
						InputTuple& inputs,
						OutputTuple& outputs,
						Functor&& kernel_func,
						const KernelConfig& config = {},
						Args... args) {

	// Wait for dependencies before starting
	config.dependencies.wait_all();

	// Get raw pointers from your buffer abstractions
	EventList deps;
	auto input_ptr = inputs.get_read_access(deps);
	auto output_ptr = outputs.get_write_access(deps);

	// Determine the number of concurrent threads to use
	unsigned int num_threads = std::thread::hardware_concurrency();
	if (num_threads == 0) {
		num_threads = 1;
	} // Fallback if concurrency is not detectable

	std::vector<std::thread> threads;
	size_t chunk_size = (thread_count + num_threads - 1) / num_threads;

	// Launch threads, giving each a "chunk" of the work
	for (unsigned int t = 0; t < num_threads; ++t) {
		threads.emplace_back([=]() {
			size_t start = t * chunk_size;
			size_t end = std::min(start + chunk_size, thread_count);
			for (size_t i = start; i < end; ++i) {
				// Each thread executes the user's functor on its assigned chunk
				kernel_func(i, input_ptr, output_ptr, args...);
			}
		});
	}

	// Wait for all CPU threads to complete
	for (auto& thread : threads) {
		if (thread.joinable()) {
			thread.join();
		}
	}

	// For a CPU launch, the event is immediately considered complete
	return Event(nullptr, resource);
}

/**
 * @brief Simplified kernel launch function that works with Buffer.h types (works)
 */
template<typename KernelFunc, typename... Buffers>
Event simple_kernel(const Resource& resource,
					size_t num_elements,
					KernelFunc&& kernel,
					const KernelConfig& config,
					const std::string& kernel_name,
					Buffers&... buffers) {

	EventList deps;
	auto buffer_ptrs = std::make_tuple(buffers.get_write_access(deps)...);

	// Merge dependencies from buffers with config dependencies
	KernelConfig merged_config = config;
	for (const auto& event : deps.get_events()) {
		merged_config.dependencies.add(event);
	}

	return std::apply(
		[&](auto*... ptrs) {
			return launch_kernel_impl(resource,
									  num_elements,
									  kernel,
									  merged_config,
									  kernel_name,
									  ptrs...);
		},
		buffer_ptrs);
}

// Make KernelChain a template on the Backend type
template<typename Backend>
class KernelChain {
  private:
	const Resource& resource_;
	EventList events_;

  public:
	explicit KernelChain(const Resource& resource) : resource_(resource) {}

	// Overload for functor-based kernels (CUDA, SYCL, CPU)
	template<typename InputTuple, typename OutputTuple, typename Functor, typename... Args>
	KernelChain& then(size_t thread_count,
					  InputTuple& inputs,
					  OutputTuple& outputs,
					  Functor&& kernel,
					  const KernelConfig& config = {},
					  Args&&... args) {

		KernelConfig new_config = config;
		new_config.dependencies = events_; // Chain the events automatically
		new_config.async = true;

		// The magic happens here: call the generic, overloaded launch_kernel
		// The compiler will pick the correct version based on the Backend template parameter.
		Event completion_event = launch_kernel<Backend>(resource_,
														thread_count,
														inputs,
														outputs,
														new_config,
														std::forward<Functor>(kernel),
														std::forward<Args>(args)...);

		// The new event becomes the dependency for the next step in the chain
		events_.clear();
		events_.add(completion_event);

		return *this;
	}

	// Overload for name-based kernels (Metal)
	template<typename InputTuple, typename OutputTuple, typename... Args>
	KernelChain& then(size_t thread_count,
					  InputTuple& inputs,
					  OutputTuple& outputs,
					  const std::string& kernel_name,
					  const KernelConfig& config = {},
					  Args&&... args) {

		KernelConfig new_config = config;
		new_config.dependencies = events_;
		new_config.async = true;

		Event completion_event = launch_kernel<Backend>(resource_,
														thread_count,
														inputs,
														outputs,
														new_config,
														kernel_name,
														std::forward<Args>(args)...);

		events_.clear();
		events_.add(completion_event);

		return *this;
	}

	void wait() {
		events_.wait_all();
	}
};

// ============================================================================
// Utility Functions
// ============================================================================
template<typename T>
struct CopyFunctor {
	HOST DEVICE void operator()(size_t i, const T* src, T* dst) const {
		dst[i] = src[i];
	}
};

template<typename T>
struct FillFunctor {
	T value;
	HOST DEVICE void operator()(size_t i, T* out) const {
		out[i] = value;
	}
};

// --- Backend-Agnostic API Functions ---
template<typename Backend, typename T>
Event copy_async(const Resource& resource,
				 const DeviceBuffer<T>& source,
				 DeviceBuffer<T>& destination,
				 const KernelConfig& config = {}) {

	if (source.size() != destination.size()) {
		throw std::runtime_error("Buffer size mismatch in copy_async");
	}

	// Launch the generic copy kernel on the specified backend
	return launch_kernel<Backend>(resource,
								  source.size(),
								  std::tie(source),
								  std::tie(destination),
								  config,
								  CopyFunctor<T>{});
}

template<typename Backend, typename T>
Event fill_async(const Resource& resource,
				 DeviceBuffer<T>& buffer,
				 const T& value,
				 const KernelConfig& config = {}) {

	// Launch the generic fill kernel on the specified backend
	return launch_kernel<Backend>(resource,
								  buffer.size(),
								  std::tie(),
								  std::tie(buffer),
								  config,
								  FillFunctor<T>{value});
}

// ============================================================================
// Result Wrapper for Kernel Calls
// ============================================================================

template<typename T>
struct KernelResult {
	T result;
	Event completion_event;

	KernelResult(T&& res, Event&& event)
		: result(std::forward<T>(res)), completion_event(std::move(event)) {}

	void wait() {
		completion_event.wait();
	}
	bool is_ready() const {
		return completion_event.is_complete();
	}

	T get() {
		wait();
		return std::move(result);
	}
};

} // namespace ARBD