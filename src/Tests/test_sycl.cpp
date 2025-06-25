#include <vector>
#include <numeric>
#include <iostream>
#ifdef USE_SYCL

// Use single header version which is self-contained
#include "../../extern/Catch2/extras/catch_amalgamated.hpp"

#include "Backend/SYCL/SYCLManager.h"


using namespace ARBD::SYCL;

TEST_CASE("SYCL Manager Initialization", "[sycl][manager]") {
    SECTION("Basic initialization") {
        REQUIRE_NOTHROW(SYCLManager::init());
        REQUIRE(static_cast<int>(SYCLManager::all_device_size()) > 0);
        
        INFO("Found " << SYCLManager::all_device_size() << " SYCL devices");
        
        // Print device information for debugging
        const auto& devices = SYCLManager::all_devices();
        for (size_t i = 0; i < devices.size(); ++i) {
            const auto& device = devices[i];
            std::cout << "Device " << device.id() << ": " << device.name() 
                      << " (" << device.vendor() << ")" << std::endl;
            std::cout << "  - Compute units: " << device.max_compute_units() << std::endl;
            std::cout << "  - Global memory: " << device.global_mem_size() / (1024*1024) << " MB" << std::endl;
            std::cout << "  - Type: " << (device.is_cpu() ? "CPU" : 
                                         device.is_gpu() ? "GPU" : 
                                         device.is_accelerator() ? "Accelerator" : "Unknown") << std::endl;
        }
    }
    
    SECTION("Device loading") {
        SYCLManager::init();
        REQUIRE_NOTHROW(SYCLManager::load_info());
        REQUIRE(SYCLManager::devices().size() > 0);
        REQUIRE(SYCLManager::current() == 0);
    }
}

TEST_CASE("SYCL Device Selection", "[sycl][device]") {
    SYCLManager::init();
    SYCLManager::load_info();
    SYCLManager::sync();
    
    SECTION("Current device access") {
        REQUIRE_NOTHROW(SYCLManager::get_current_device());
        REQUIRE_NOTHROW(SYCLManager::get_current_queue());
        
        auto& device = SYCLManager::get_current_device();
        REQUIRE(device.id() == static_cast<unsigned int>(SYCLManager::current()));
        SYCLManager::sync();
    }
    
    SECTION("Device switching") {
        if (SYCLManager::devices().size() > 1) {
            SYCLManager::use(1);
            REQUIRE(SYCLManager::current() == 1);
            
            auto& device = SYCLManager::get_current_device();
            REQUIRE(device.id() == 1);
            
            // Switch back
            SYCLManager::use(0);
            REQUIRE(SYCLManager::current() == 0);
        }
        SYCLManager::sync();
    }
    
    SECTION("Device synchronization") {
        REQUIRE_NOTHROW(SYCLManager::sync());
        REQUIRE_NOTHROW(SYCLManager::sync(0));
        
        auto& device = SYCLManager::get_current_device();
        REQUIRE_NOTHROW(device.synchronize_all_queues());
    }
}

TEST_CASE("SYCL Device Memory Operations", "[sycl][memory]") {
    SYCLManager::init();
    SYCLManager::load_info();
    SYCLManager::sync();
    
    auto& queue = SYCLManager::get_current_queue();
    
    SECTION("Basic memory allocation") {
        constexpr size_t SIZE = 1000;
        {
            REQUIRE_NOTHROW(DeviceMemory<float>{queue.get(), SIZE});
        }
        
        DeviceMemory<float> device_mem(queue.get(), SIZE);
        REQUIRE(device_mem.size() == SIZE);
        REQUIRE(device_mem.get() != nullptr);
        REQUIRE(device_mem.queue() == &queue.get());
        queue.synchronize();
    }
    
    SECTION("Host to device copy") {
        constexpr size_t SIZE = 100;
        std::vector<int> host_data(SIZE);
        std::iota(host_data.begin(), host_data.end(), 1); // Fill with 1, 2, 3, ...
        
        DeviceMemory<int> device_mem(queue.get(), SIZE);
        REQUIRE_NOTHROW(device_mem.copyFromHost(host_data));
        
        queue.synchronize();
    }
    
    SECTION("Device to host copy") {
        constexpr size_t SIZE = 50;
        std::vector<float> host_data(SIZE, 42.0f);
        std::vector<float> result_data(SIZE, 0.0f);
        
        DeviceMemory<float> device_mem(queue.get(), SIZE);
        device_mem.copyFromHost(host_data);
        device_mem.copyToHost(result_data);
        
        queue.synchronize();
        
        for (size_t i = 0; i < SIZE; ++i) {
            REQUIRE(result_data[i] == 42.0f);
        }
    }
    
    SECTION("Memory size validation") {
        constexpr size_t SIZE = 100;
        DeviceMemory<float> device_mem(queue.get(), SIZE);
        
        // Try to copy more data than allocated
        std::vector<float> large_data(SIZE + 1, 1.0f);
        REQUIRE_THROWS(device_mem.copyFromHost(large_data));
        
        std::vector<float> large_output(SIZE + 1);
        REQUIRE_THROWS(device_mem.copyToHost(large_output));
    }
}

TEST_CASE("SYCL Simple Kernel Execution", "[sycl][kernel]") {
    SYCLManager::init();
    SYCLManager::load_info();
    SYCLManager::sync();
    
    auto& queue = SYCLManager::get_current_queue();
    
    SECTION("Vector addition kernel") {
        constexpr size_t SIZE = 256;
        
        // Host data
        std::vector<float> a(SIZE, 1.0f);
        std::vector<float> b(SIZE, 2.0f);
        std::vector<float> c(SIZE, 0.0f);
        
        // Device memory
        DeviceMemory<float> d_a(queue.get(), SIZE);
        DeviceMemory<float> d_b(queue.get(), SIZE);
        DeviceMemory<float> d_c(queue.get(), SIZE);
        
        // Copy data to device
        d_a.copyFromHost(a);
        d_b.copyFromHost(b);
        
        // Get raw pointers for kernel
        float* ptr_a = d_a.get();
        float* ptr_b = d_b.get();
        float* ptr_c = d_c.get();
        
        // Submit kernel
        auto event = queue.submit([=](sycl::handler& h) {
            auto range = sycl::range<1>(SIZE);
            
            h.parallel_for(range, [=](sycl::id<1> idx) {
                size_t i = idx[0];
                ptr_c[i] = ptr_a[i] + ptr_b[i];
            });
        });
        
        event.wait();
        
        // Copy result back
        d_c.copyToHost(c);
        queue.synchronize();
        
        // Verify results
        for (size_t i = 0; i < SIZE; ++i) {
            REQUIRE_THAT(c[i], Catch::Matchers::WithinAbs(3.0f, 1e-6f));
        }
    }
    
    SECTION("Parallel reduction") {
        constexpr size_t SIZE = 1024;
        
        std::vector<int> data(SIZE);
        std::iota(data.begin(), data.end(), 1); // Fill with 1, 2, 3, ..., SIZE
        
        DeviceMemory<int> d_data(queue.get(), SIZE);
        DeviceMemory<int> d_result(queue.get(), 1);
        
        d_data.copyFromHost(data);
        
        // Get raw pointers for kernel
        int* ptr_data = d_data.get();
        int* ptr_result = d_result.get();
        
        // Simple reduction kernel (sum)
        auto event = queue.submit([=](sycl::handler& h) {
            // Use SYCL reduction
            auto sum_reduction = sycl::reduction(ptr_result, sycl::plus<int>());
            
            h.parallel_for(sycl::range<1>(SIZE), sum_reduction,
                          [=](sycl::id<1> idx, auto& sum) {
                              sum += ptr_data[idx[0]];
                          });
        });
        
        event.wait();
        
        std::vector<int> result(1);
        d_result.copyToHost(result);
        queue.synchronize();
        
        // Expected sum: SIZE * (SIZE + 1) / 2
        int expected = SIZE * (SIZE + 1) / 2;
        REQUIRE(result[0] == expected);
    }
}

TEST_CASE("SYCL Queue Management", "[sycl][queue]") {
    SYCLManager::init();
    SYCLManager::load_info();
    SYCLManager::sync();
    
    auto& device = SYCLManager::get_current_device();
    
    SECTION("Multiple queues") {
        auto& queue0 = device.get_queue(0);
        queue0.synchronize();
        
        auto& queue1 = device.get_queue(1);
        queue1.synchronize();
        
        // Queues should be different objects
        REQUIRE(&queue0.get() != &queue1.get());
        
        // Test queue rotation
        auto& next_queue1 = device.get_next_queue();
        next_queue1.synchronize();
        
        auto& next_queue2 = device.get_next_queue();
        next_queue2.synchronize();
        
        // Should cycle through queues
        REQUIRE(&next_queue1.get() != &next_queue2.get());
    }
    
    SECTION("Queue properties") {
        auto& queue = device.get_queue(0);
        
        // Should be able to check if queue is in-order
        REQUIRE_NOTHROW(queue.is_in_order());
        
        // Should be able to synchronize
        REQUIRE_NOTHROW(queue.synchronize());
    }
}

TEST_CASE("SYCL Device Filtering", "[sycl][device][filter]") {
    SYCLManager::init();
    
    SECTION("Get device IDs by type") {
        auto cpu_ids = SYCLManager::get_cpu_device_ids();
        auto gpu_ids = SYCLManager::get_gpu_device_ids();
        auto accel_ids = SYCLManager::get_accelerator_device_ids();
        
        // Should have at least one device of some type
        REQUIRE((cpu_ids.size() + gpu_ids.size() + accel_ids.size()) > 0);
        
        // Verify device types
        for (auto id : cpu_ids) {
            REQUIRE(SYCLManager::all_devices()[id].is_cpu());
        }
        for (auto id : gpu_ids) {
            REQUIRE(SYCLManager::all_devices()[id].is_gpu());
        }
        for (auto id : accel_ids) {
            REQUIRE(SYCLManager::all_devices()[id].is_accelerator());
        }
    }
}

TEST_CASE("SYCL Error Handling", "[sycl][error]") {
    SYCLManager::init();
    SYCLManager::load_info();
    
    SECTION("Device selection errors") {
        // Test selecting invalid device IDs
        std::vector<unsigned int> invalid_ids = {999};
        REQUIRE_THROWS(SYCLManager::select_devices(invalid_ids));
        
        // Test sync with invalid device ID
        REQUIRE_THROWS(SYCLManager::sync(999));
    }
    
    SECTION("Memory errors") {
        auto& queue = SYCLManager::get_current_queue();
        
        // Try to allocate memory with size 0
        REQUIRE_NOTHROW(DeviceMemory<float>(queue.get(), 0));
        
        // Try to copy with mismatched sizes
        DeviceMemory<float> device_mem(queue.get(), 10);
        std::vector<float> large_data(20, 1.0f);
        REQUIRE_THROWS(device_mem.copyFromHost(large_data));
        
        std::vector<float> large_output(20);
        REQUIRE_THROWS(device_mem.copyToHost(large_output));
    }
}

// Custom main function to handle SYCL cleanup properly
int main(int argc, char* argv[]) {
    // Initialize Catch2
    Catch::Session session;
    
    // Parse command line arguments
    int returnCode = session.applyCommandLine(argc, argv);
    if (returnCode != 0) {
        return returnCode;
    }
    
    // Run the tests
    int result = session.run();
    
    // Explicit cleanup to prevent mutex issues
    try {
        SYCLManager::finalize();
    } catch (...) {
        // Ignore cleanup errors
        std::cerr << "Warning: SYCL cleanup had issues (ignored)" << std::endl;
    }
    
    return result;
}

#else // PROJECT_USES_SYCL

// Fallback main when SYCL is not enabled
int main() {
    std::cout << "SYCL support not enabled, skipping SYCL tests" << std::endl;
    return 0;
}

#endif // PROJECT_USES_SYCL
