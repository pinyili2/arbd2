#ifdef USE_METAL

#include "../catch_boiler.h"
#include "ARBDException.h"
#include <vector>
#include <string>
#include <numeric>
#include "Backend/METAL/METALManager.h"

// This test case is a placeholder for when Metal is not supported.
// We need at least one test case in the file.
TEST_CASE("Metal Support Check", "[metal]") {

    // Use the METALManager to check for available devices. This is a more
    // thorough check that ensures the manager can initialize and find devices.
    REQUIRE_NOTHROW(ARBD::METAL::METALManager::init());
    REQUIRE(ARBD::METAL::METALManager::all_device_size() > 0);
    REQUIRE_NOTHROW(ARBD::METAL::METALManager::finalize());
}

TEST_CASE("Metal Manager Initialization", "[metal][manager]") {
    SECTION("Device Discovery and Properties") {
        ARBD::METAL::METALManager::init();
        auto& devices = ARBD::METAL::METALManager::all_devices();
        REQUIRE_FALSE(devices.empty());

        for (size_t i = 0; i < devices.size(); ++i) {
            const auto& device = devices[i];
            // After sorting, IDs might not be sequential from 0
            // So we just check that the name is not empty
            REQUIRE_FALSE(device.name().empty());
        }
        ARBD::METAL::METALManager::finalize();
    }
}

TEST_CASE("Metal Manager Device Properties", "[metal][manager][properties]") {
    ARBD::METAL::METALManager::load_info();
    auto& devices = ARBD::METAL::METALManager::devices();
    REQUIRE_FALSE(devices.empty());

    for (const auto& device : devices) {
        SECTION("Device Properties for " + device.name()) {
            LOGINFO("Device {}: {}", device.id(), device.name().c_str());
            REQUIRE_NOTHROW(device.metal_device());
            REQUIRE(device.max_threads_per_group() > 0);
        }
    }
    ARBD::METAL::METALManager::finalize();
}

TEST_CASE("Metal Manager Device Selection", "[metal][manager]") {
    ARBD::METAL::METALManager::init();
    SECTION("Select first device") {
        const auto& all_devices = ARBD::METAL::METALManager::all_devices();
        if (all_devices.empty()) {
            WARN("No Metal devices found, skipping test.");
            ARBD::METAL::METALManager::finalize();
            return;
        }
        
        unsigned int first_device_id = all_devices[0].id();
        std::vector<unsigned int> dev_ids = {first_device_id};
        REQUIRE_NOTHROW(ARBD::METAL::METALManager::select_devices(dev_ids));
        
        auto& selected_devices = ARBD::METAL::METALManager::devices();
        REQUIRE(selected_devices.size() == 1);
        REQUIRE(selected_devices[0].id() == first_device_id);
    }
    ARBD::METAL::METALManager::finalize();
}

TEST_CASE("Metal Queue Operations", "[metal][queue]") {
    // Setup: Initialize the manager
    ARBD::METAL::METALManager::load_info();

    // Iterate by index to get non-const device objects, which is safer
    // than using const_cast.
    for (size_t i = 0; i < ARBD::METAL::METALManager::devices().size(); ++i) {
        auto& device = ARBD::METAL::METALManager::devices()[i];

        SECTION("Queue Operations on device " + std::to_string(device.id())) {

            // Get a queue from the device
            auto& queue = const_cast<ARBD::METAL::METALManager::Device&>(device).get_next_queue();
            REQUIRE(queue.is_available());

            // Create a command buffer using the queue
            void* cmd_buffer_ptr = nullptr;
            REQUIRE_NOTHROW(cmd_buffer_ptr = queue.create_command_buffer());
            REQUIRE(cmd_buffer_ptr != nullptr);

            // The modern C++ way: Wrap the raw command buffer pointer in an Event object.
            // The Event's destructor will automatically call release() on the command buffer,
            // preventing memory leaks. This is the core of RAII.
            ARBD::METAL::Event event(cmd_buffer_ptr);

            // Commit the command buffer via the Event object
            REQUIRE_NOTHROW(event.commit());

            // Wait for the (empty) command buffer to complete
            REQUIRE_NOTHROW(event.wait());

            // Check that it's now complete
            REQUIRE(event.is_complete() == true);
        }
    }

    // Teardown: Finalize the manager
    ARBD::METAL::METALManager::finalize();
}

TEST_CASE("Metal Synchronization", "[metal][sync]") {
    ARBD::METAL::METALManager::load_info();
    const auto& devices = ARBD::METAL::METALManager::devices();
    if (devices.empty()) {
        WARN("No Metal devices found. Skipping test.");
        ARBD::METAL::METALManager::finalize();
        return;
    }

    for (const auto& device : ARBD::METAL::METALManager::devices()) {
        SECTION("Synchronization for device " + std::to_string(device.id())) {
            INFO("Device ID: " << device.id());
            REQUIRE_NOTHROW(device.synchronize_all_queues());
        }
    }
    ARBD::METAL::METALManager::finalize();
}

TEST_CASE("Metal Memory Allocation and Transfer", "[metal][memory]") {
    ARBD::METAL::METALManager::load_info();
    if (ARBD::METAL::METALManager::devices().empty()) {
        WARN("No Metal devices found. Skipping test.");
        ARBD::METAL::METALManager::finalize();
        return;
    }
    auto& device = ARBD::METAL::METALManager::get_current_device();
    
    SECTION("DeviceMemory construction and destruction") {
        ARBD::METAL::DeviceMemory<float> device_mem(device.metal_device(), 10);
        REQUIRE(device_mem.size() == 10);
    }

    SECTION("DeviceMemory move semantics") {
        ARBD::METAL::DeviceMemory<int> mem1(device.metal_device(), 100);
        REQUIRE(mem1.size() == 100);
        
        ARBD::METAL::DeviceMemory<int> mem2 = std::move(mem1);
        REQUIRE(mem1.size() == 0);
        REQUIRE(mem1.get() == nullptr);
        REQUIRE(mem2.size() == 100);
        REQUIRE(mem2.get() != nullptr);
    }

    SECTION("Data transfer Host to Device and back") {
        std::vector<float> host_data(256);
        std::iota(host_data.begin(), host_data.end(), 0.0f);

        ARBD::METAL::DeviceMemory<float> device_mem(device.metal_device(), host_data.size());
        REQUIRE_NOTHROW(device_mem.copyFromHost(host_data));

        std::vector<float> host_result(host_data.size(), 0.0f);
        REQUIRE_NOTHROW(device_mem.copyToHost(host_result));

        REQUIRE(host_data == host_result);
    }
    ARBD::METAL::METALManager::finalize();
}

#endif // USE_METAL 
 