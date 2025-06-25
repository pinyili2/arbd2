#include <iostream>

#ifdef USE_SYCL
#include <AdaptiveCpp/sycl.hpp>
#include <vector>
#include <numeric>
#include <cassert>

using namespace acpp::sycl; // AdaptiveCpp uses acpp::sycl namespace

// Simple standalone test that doesn't rely on SYCLManager
void test_basic_sycl_functionality() {
    std::cout << "[TEST] Basic SYCL Functionality...\n";
    
    try {
        // Try to get available platforms
        auto platforms = platform::get_platforms();
        std::cout << "Found " << platforms.size() << " SYCL platforms:\n";
        
        bool found_cpu = false;
        for (const auto& platform : platforms) {
            std::cout << "  Platform: " << platform.get_info<info::platform::name>() 
                      << " (" << platform.get_info<info::platform::vendor>() << ")\n";
            
            auto devices = platform.get_devices();
            for (const auto& device : devices) {
                auto type = device.get_info<info::device::device_type>();
                std::string type_str = (type == info::device_type::cpu) ? "CPU" :
                                      (type == info::device_type::gpu) ? "GPU" :
                                      (type == info::device_type::accelerator) ? "Accelerator" : "Unknown";
                
                std::cout << "    Device: " << device.get_info<info::device::name>()
                          << " (Type: " << type_str << ")\n";
                
                if (type == info::device_type::cpu) {
                    found_cpu = true;
                }
            }
        }
        
        if (!found_cpu) {
            std::cout << "No CPU devices found, trying default selector...\n";
            try {
                queue q(default_selector_v);
                std::cout << "Default queue created successfully\n";
                
                // Simple test
                constexpr size_t N = 100;
                std::vector<int> data(N);
                std::iota(data.begin(), data.end(), 1);
                
                int* device_data = malloc_device<int>(N, q);
                q.memcpy(device_data, data.data(), N * sizeof(int)).wait();
                
                std::vector<int> result(N);
                q.memcpy(result.data(), device_data, N * sizeof(int)).wait();
                
                // Verify
                bool success = true;
                for (size_t i = 0; i < N; ++i) {
                    if (result[i] != data[i]) {
                        success = false;
                        break;
                    }
                }
                
                free(device_data, q);
                
                if (success) {
                    std::cout << "Memory copy test: PASS\n";
                } else {
                    std::cout << "Memory copy test: FAIL\n";
                }
                
            } catch (const exception& e) {
                std::cout << "Default selector failed: " << e.what() << "\n";
                std::cout << "This might indicate missing OpenMP/CPU backend support\n";
                return;
            }
        } else {
            std::cout << "Found CPU device, testing...\n";
            try {
                queue q(cpu_selector_v);
                std::cout << "CPU queue created successfully\n";
                
                // Simple parallel operation
                constexpr size_t N = 1000;
                std::vector<float> a(N, 1.0f);
                std::vector<float> b(N, 2.0f);
                std::vector<float> c(N, 0.0f);
                
                float* d_a = malloc_device<float>(N, q);
                float* d_b = malloc_device<float>(N, q);
                float* d_c = malloc_device<float>(N, q);
                
                q.memcpy(d_a, a.data(), N * sizeof(float)).wait();
                q.memcpy(d_b, b.data(), N * sizeof(float)).wait();
                
                q.submit([&](handler& h) {
                    h.parallel_for(range<1>(N), [=](id<1> idx) {
                        d_c[idx] = d_a[idx] + d_b[idx];
                    });
                }).wait();
                
                q.memcpy(c.data(), d_c, N * sizeof(float)).wait();
                
                free(d_a, q);
                free(d_b, q);
                free(d_c, q);
                
                // Verify
                bool success = true;
                for (size_t i = 0; i < N; ++i) {
                    if (std::abs(c[i] - 3.0f) > 1e-6f) {
                        success = false;
                        break;
                    }
                }
                
                if (success) {
                    std::cout << "Vector addition test: PASS\n";
                } else {
                    std::cout << "Vector addition test: FAIL\n";
                }
                
            } catch (const exception& e) {
                std::cout << "CPU test failed: " << e.what() << "\n";
            }
        }
        
        std::cout << "Basic SYCL test completed\n";
        
    } catch (const std::exception& e) {
        std::cout << "SYCL test failed with error: " << e.what() << "\n";
    }
}

#endif // PROJECT_USES_SYCL

int main() {
#ifdef PROJECT_USES_SYCL
    std::cout << "=== Standalone SYCL Test ===\n";
    test_basic_sycl_functionality();
    std::cout << "=== Test Complete ===\n";
    return 0;
#else
    std::cout << "SYCL not enabled in build\n";
    return 0;
#endif
} 