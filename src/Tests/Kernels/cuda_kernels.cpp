#include "../catch_boiler.h"
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
using namespace ARBD;
using namespace Catch;
using Catch::Approx;
#ifdef USE_CUDA

class CUDAKernelTestFixture {
  private:
	Resource CUDA_resource;
	bool CUDA_available = false;

  public:
	CUDAKernelTestFixture() {
		try {
			CUDA::CUDAManager::init();
			CUDA::CUDAManager::load_info();

			if (CUDA::CUDAManager::devices().empty()) {
				SKIP("No CUDA devices available. Skipping CUDA kernel tests.");
				return;
			}

			CUDA_resource = Resource(ResourceType::CUDA, 0);
			CUDA::CUDAManager::use(0);
			CUDA_available = true;

		} catch (const std::exception& e) {
			FAIL("Failed to initialize CUDAManager for kernel tests: " << e.what());
		}
	}

	~CUDAKernelTestFixture() {
		if (CUDA_available) {
			try {
				CUDA::CUDAManager::finalize();
			} catch (const std::exception& e) {
				std::cerr << "Error during CUDAManager finalization: " << e.what() << std::endl;
			}
		}
	}

	bool is_cuda_available() const {
		return CUDA_available;
	}
	const Resource& get_cuda_resource() const {
		return CUDA_resource;
	}
};

TEST_CASE_METHOD(CUDAKernelTestFixture, "CUDA Buffer Operations", "[CUDA][kernels]") {
	if (!is_cuda_available()) {
		SKIP("CUDA not available");
	}

	SECTION("Basic buffer creation and data transfer") {
		const size_t n = 1000;

		DeviceBuffer<float> buffer(n);
		REQUIRE(buffer.size() == n);
		REQUIRE(!buffer.empty());

		std::vector<float> host_data(n);
		std::iota(host_data.begin(), host_data.end(), 1.0f);

		buffer.copy_from_host(host_data);

		std::vector<float> result(n);
		buffer.copy_to_host(result);

		for (size_t i = 0; i < n; ++i) {
			REQUIRE(result[i] == host_data[i]);
		}
	}
}

TEST_CASE_METHOD(CUDAKernelTestFixture, "CUDA Kernel Launch", "[CUDA][kernels][backend]") {
	if (!is_cuda_available()) {
		SKIP("CUDA not available");
	}

	SECTION("Simple scaling kernel") {
		const size_t n = 256;

		DeviceBuffer<float> input_buf(n);
		DeviceBuffer<float> output_buf(n);

		std::vector<float> input_data(n, 2.0f);
		input_buf.copy_from_host(input_data);

		// Simple scaling kernel - multiplies input by 3
		auto scale_kernel = [](size_t i, const float* input, float* output) {
			output[i] = input[i] * 3.0f;
		};

		// Create input and output tuples
		auto inputs = std::make_tuple(std::ref(input_buf));
		auto outputs = std::make_tuple(std::ref(output_buf));

		// Launch kernel
		KernelConfig config;
		Event completion =
			launch_kernel(get_cuda_resource(), n, config, inputs, outputs, scale_kernel);

		// Wait for completion and verify results
		completion.wait();

		std::vector<float> result(n);
		output_buf.copy_to_host(result);

		for (size_t i = 0; i < n; ++i) {
			REQUIRE(result[i] == Approx(6.0f)); // 2.0f * 3.0f = 6.0f
		}
	}

	SECTION("Vector addition kernel") {
		const size_t n = 512;

		DeviceBuffer<float> a_buf(n);
		DeviceBuffer<float> b_buf(n);
		DeviceBuffer<float> c_buf(n);

		std::vector<float> a_data(n), b_data(n);
		std::iota(a_data.begin(), a_data.end(), 1.0f);
		std::iota(b_data.begin(), b_data.end(), 100.0f);

		a_buf.copy_from_host(a_data);
		b_buf.copy_from_host(b_data);

		// Vector addition kernel
		auto add_kernel = [](size_t i, const float* a, const float* b, float* c) {
			c[i] = a[i] + b[i];
		};

		auto inputs = std::make_tuple(std::ref(a_buf), std::ref(b_buf));
		auto outputs = std::make_tuple(std::ref(c_buf));

		KernelConfig config;
		Event completion =
			launch_kernel(get_cuda_resource(), n, config, inputs, outputs, add_kernel);
		completion.wait();

		std::vector<float> result(n);
		c_buf.copy_to_host(result);

		for (size_t i = 0; i < n; ++i) {
			float expected = a_data[i] + b_data[i];
			REQUIRE(result[i] == Approx(expected));
		}
	}
}
#endif
