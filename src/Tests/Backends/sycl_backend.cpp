#include "../catch_boiler.h"
#ifdef USE_SYCL
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Resource.h"
#include "Backend/SYCL/SYCLManager.h"
#include <numeric>
#include <string>
#include <vector>

using namespace ARBD;

// ============================================================================
// Test Fixture for SYCL Backend
// ============================================================================
// This fixture handles the initialization and finalization of the SYCLManager
// for each test case, ensuring a clean state.
struct SYCLTestFixture {
	Resource sycl_resource;

	SYCLTestFixture() {
		try {
			SYCL::SYCLManager::init();
			SYCL::SYCLManager::load_info();

			// Skip tests if no SYCL devices are found.
			if (SYCL::SYCLManager::all_devices().empty()) {
				WARN("No SYCL devices found. Skipping SYCL backend tests.");
				return;
			}

			// Use the first available SYCL device for all tests.
			sycl_resource = Resource(ResourceType::SYCL, 0);
			SYCL::SYCLManager::use(0);

		} catch (const std::exception& e) {
			FAIL("Failed to initialize SYCLManager in test fixture: " << e.what());
		}
	}

	~SYCLTestFixture() {
		try {
			SYCL::SYCLManager::finalize();
		} catch (const std::exception& e) {
			// It's not ideal to throw from a destructor. Log the error instead.
			std::cerr << "Error during SYCLManager finalization in test fixture: " << e.what()
					  << std::endl;
		}
	}
};

// ============================================================================
// Test Cases
// ============================================================================

TEST_CASE_METHOD(SYCLTestFixture, "SYCL Resource Creation and Properties", "[sycl][resource]") {
	if (SYCL::SYCLManager::all_devices().empty())
		return;

	SECTION("Resource type is correct") {
		CHECK(sycl_resource.type == ResourceType::SYCL);
		CHECK(sycl_resource.id == 0);
	}

	SECTION("Resource type string") {
		CHECK(std::string(sycl_resource.getTypeString()) == "SYCL");
	}

	SECTION("Resource is device") {
		CHECK(sycl_resource.is_device());
		CHECK_FALSE(sycl_resource.is_host());
	}
}

TEST_CASE_METHOD(SYCLTestFixture, "SYCL BackendPolicy", "[sycl][memory]") {
	if (SYCL::SYCLManager::all_devices().empty())
		return;
	const size_t count = 128;
	const size_t bytes = count * sizeof(float);

	SECTION("Allocate, copy, and deallocate") {
		// Allocate
		void* device_ptr = BackendPolicy::allocate(bytes);
		REQUIRE(device_ptr != nullptr);

		// Copy from host
		std::vector<float> host_data(count);
		std::iota(host_data.begin(), host_data.end(), 1.5f);
		REQUIRE_NOTHROW(BackendPolicy::copy_from_host(device_ptr, host_data.data(), bytes));

		// Copy to host
		std::vector<float> host_result(count, 0.0f);
		REQUIRE_NOTHROW(BackendPolicy::copy_to_host(host_result.data(), device_ptr, bytes));

		// Verify
		CHECK(host_result == host_data);

		// Deallocate
		REQUIRE_NOTHROW(BackendPolicy::deallocate(device_ptr));
	}
}

#else

TEST_CASE("SYCL Support Not Enabled", "[sycl]") {
	SUCCEED("SYCL support not enabled, tests are skipped.");
}

#endif // USE_SYCL