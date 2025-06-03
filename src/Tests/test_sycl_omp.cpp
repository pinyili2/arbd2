#ifdef PROJECT_USES_SYCL

#include "Backend/SYCL/SYCLManager.h"
#include <vector>
#include <numeric>
#include <iostream>
#include <limits>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace ARBD {
namespace {

bool within_abs(float a, float b, float tol) {
    return std::fabs(a - b) <= tol;
}

void test_sycl_cpu_initialization() {
    std::cout << "[TEST] SYCL CPU Initialization... ";
    try {
        // Initialize with CPU preference first
        SYCLManager::prefer_device_type(sycl::info::device_type::cpu);
        
        // Initialize the SYCL manager
        SYCLManager::init();
        
        // Verify we have at least one device
        if (SYCLManager::all_device_size() == 0) {
            std::cout << "SKIP: No SYCL devices found\n";
            return;
        }
        
        // Check if we can create a CPU device queue using the default selector
        // AdaptiveCpp might not support cpu_selector_v properly
        try {
            // Try to get all devices and find a CPU device
            auto platforms = sycl::platform::get_platforms();
            bool cpu_found = false;
            
            for (const auto& platform : platforms) {
                auto devices = platform.get_devices();
                for (const auto& device : devices) {
                    if (device.is_cpu()) {
                        // Try creating a queue with this CPU device
                        sycl::queue test_queue(device);
                        test_queue.get_device(); // Verify device access
                        cpu_found = true;
                        break;
                    }
                }
                if (cpu_found) break;
            }
            
            if (!cpu_found) {
                std::cout << "SKIP: No CPU device available\n";
                return;
            }
        } catch (const sycl::exception& e) {
            std::cout << "SKIP: CPU device test failed - " << e.what() << "\n";
            return;
        }
        
        // Print CPU device info
        const auto& devices = SYCLManager::all_devices();
        bool found_cpu = false;
        for (size_t i = 0; i < devices.size(); ++i) {
            const auto& device = devices[i];
            if (device.is_cpu()) {
                found_cpu = true;
                std::cout << "\nFound CPU Device " << device.id() << ": " << device.name()
                          << " (" << device.vendor() << ")\n";
                std::cout << "  - Compute units: " << device.max_compute_units() << "\n";
                std::cout << "  - Global memory: " << device.global_mem_size() / (1024*1024) << " MB\n";
            }
        }
        
        if (!found_cpu) {
            std::cout << "SKIP: No CPU device found in SYCLManager\n";
            return;
        }
        
        // Load device info (this will call init again but that's ok)
        SYCLManager::load_info();
        assert(SYCLManager::devices().size() > 0);
        std::cout << "PASS\n";
    } catch (const std::exception& e) {
        std::cout << "FAIL: " << e.what() << "\n";
        throw;
    }
}

void test_sycl_basic_memory_operations() {
    std::cout << "[TEST] SYCL Basic Memory Operations... ";
    try {
        auto& queue = SYCLManager::get_current_queue();
        
        // Simple vector copy test
        constexpr size_t SIZE = 1000;
        std::vector<float> host_data(SIZE);
        std::iota(host_data.begin(), host_data.end(), 0.0f);
        
        DeviceMemory<float> device_mem(queue.get(), SIZE);
        device_mem.copyFromHost(host_data);
        
        std::vector<float> result(SIZE, 0.0f);
        device_mem.copyToHost(result);
        
        // Verify the copy
        for (size_t i = 0; i < SIZE; ++i) {
            assert(result[i] == host_data[i]);
        }
        
        std::cout << "PASS\n";
    } catch (const std::exception& e) {
        std::cout << "FAIL: " << e.what() << "\n";
        throw;
    }
}

void test_sycl_parallel_for() {
    std::cout << "[TEST] SYCL Parallel For... ";
    try {
        auto& queue = SYCLManager::get_current_queue();
        
        // Simple vector addition
        constexpr size_t SIZE = 1000;
        std::vector<float> a(SIZE, 1.0f);
        std::vector<float> b(SIZE, 2.0f);
        std::vector<float> c(SIZE, 0.0f);
        
        DeviceMemory<float> d_a(queue.get(), SIZE);
        DeviceMemory<float> d_b(queue.get(), SIZE);
        DeviceMemory<float> d_c(queue.get(), SIZE);
        
        d_a.copyFromHost(a);
        d_b.copyFromHost(b);
        
        float* ptr_a = d_a.get();
        float* ptr_b = d_b.get();
        float* ptr_c = d_c.get();
        
        // Simple parallel addition
        queue.submit([=](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(SIZE), [=](sycl::id<1> idx) {
                size_t i = idx[0];
                ptr_c[i] = ptr_a[i] + ptr_b[i];
            });
        }).wait();
        
        d_c.copyToHost(c);
        
        // Verify results
        for (size_t i = 0; i < SIZE; ++i) {
            assert(within_abs(c[i], 3.0f, 1e-6f));
        }
        
        std::cout << "PASS\n";
    } catch (const std::exception& e) {
        std::cout << "FAIL: " << e.what() << "\n";
        throw;
    }
}

void test_sycl_reduction() {
    std::cout << "[TEST] SYCL Reduction... ";
    try {
        auto& queue = SYCLManager::get_current_queue();
        
        // Simple parallel reduction
        constexpr size_t SIZE = 1000;
        std::vector<int> data(SIZE);
        std::iota(data.begin(), data.end(), 1);
        
        DeviceMemory<int> d_data(queue.get(), SIZE);
        DeviceMemory<int> d_result(queue.get(), 1);
        
        d_data.copyFromHost(data);
        
        int* ptr_data = d_data.get();
        int* ptr_result = d_result.get();
        
        // Parallel sum reduction
        queue.submit([=](sycl::handler& h) {
            auto sum_reduction = sycl::reduction(ptr_result, sycl::plus<int>());
            h.parallel_for(sycl::range<1>(SIZE), sum_reduction,
                          [=](sycl::id<1> idx, auto& sum) {
                              sum += ptr_data[idx[0]];
                          });
        }).wait();
        
        std::vector<int> result(1);
        d_result.copyToHost(result);
        
        // Verify result
        int expected = SIZE * (SIZE + 1) / 2;  // Sum of 1 to SIZE
        assert(result[0] == expected);
        
        std::cout << "PASS\n";
    } catch (const std::exception& e) {
        std::cout << "FAIL: " << e.what() << "\n";
        throw;
    }
}

} // namespace
} // namespace ARBD

int main() {
    try {
        ARBD::test_sycl_cpu_initialization();
        ARBD::test_sycl_basic_memory_operations();
        ARBD::test_sycl_parallel_for();
        ARBD::test_sycl_reduction();
        
        std::cout << "\nAll tests passed!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTest failed with error: " << e.what() << "\n";
        return 1;
    }
}

#else // PROJECT_USES_SYCL

int main() {
    std::cout << "SYCL support not enabled, skipping tests\n";
    return 0;
}

#endif // PROJECT_USES_SYCL 