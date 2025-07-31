// Initialize Metal

#include "../catch_boiler.h"
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Kernels.h"
#include "Backend/METAL/METALManager.h"
#include "Backend/Resource.h"

TEST_CASE("Metal Vector Addition", "[metal][kernels]") {
	// Metal manager is already initialized by the test framework
	ARBD::METAL::METALManager::load_info(); 

	try {
		// Use Metal resource instead of CPU
		ARBD::Resource metal_res(ARBD::ResourceType::METAL, 0);
		const size_t n = 100; // Smaller test size

		auto buffer_a = ARBD::DeviceBuffer<float>(n);
		auto buffer_b = ARBD::DeviceBuffer<float>(n);
		auto buffer_result = ARBD::DeviceBuffer<float>(n);

		// Initialize data
		std::vector<float> host_a(n, 1.0f);
		std::vector<float> host_b(n, 2.0f);
		buffer_a.copy_from_host(host_a.data(), n);
		buffer_b.copy_from_host(host_b.data(), n);

		ARBD::KernelConfig config;

		ARBD::Event event = ARBD::launch_kernel(
			metal_res,
			n,
			config,
			"vector_add",
			buffer_a,
			buffer_b,
			buffer_result
		);	

		
		event.wait();

		// Get results and verify
		std::vector<float> result(n);
		buffer_result.copy_to_host(result.data(), n);

		for (size_t i = 0; i < n; ++i) {
			REQUIRE(result[i] == 3.0f); // 1.0 + 2.0 = 3.0
		}

		REQUIRE(true); // Test passed

	} catch (const std::exception& e) {
		FAIL("Metal test failed with exception: " << e.what());
	}

	// Don't finalize here - let the test framework handle cleanup
	ARBD::METAL::METALManager::finalize();
}