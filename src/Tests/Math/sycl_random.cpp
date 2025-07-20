#include "../catch_boiler.h"
#ifdef USE_SYCL

#include "Math/Bitmask.h"
#include "Backend/SYCL/SYCLManager.h"
#include "Backend/Resource.h"
#include "Random/Random.h"
#include <vector>
#include <memory>
#include <iostream>
#include <iomanip>

using namespace ARBD;

TEST_CASE("Random number generation", "[sycl][random]"){
    SYCL::SYCLManager::init();
    SYCL::SYCLManager::load_info();
    SYCL::SYCLManager::sync();
    LOGINFO("SYCL Random Number Generation");
    auto& queue = SYCL::SYCLManager::get_current_queue();
    auto resource = Resource(ResourceType::SYCL, 0);
    
    // Create the random generator ONCE and initialize it
    RandomSYCL<128> random_gen(resource);
    random_gen.init(111);
    
    SECTION("Uniform float generation") {
        DeviceBuffer<float> buffer(1000, resource);
        auto& sycl_device = ARBD::SYCL::SYCLManager::get_current_device();
        auto& queue = sycl_device.get_next_queue();
       
        // Use the SAME random generator instance
        auto event = random_gen.generate_uniform(buffer, 0.0f, 1.0f);
        std::vector<float> host_data(1000);
        event.wait();
        buffer.copy_to_host(host_data.data());
        
        // Print first 20 values to see what we're getting
        std::cout << "\n=== Uniform Float Generation [0.0, 1.0] ===" << std::endl;
        std::cout << "First 20 values:" << std::endl;
        for (size_t i = 0; i < 20; ++i) {
            std::cout << std::setw(3) << i << ": " << std::setprecision(6) << std::fixed 
                      << host_data[i] << std::endl;
        }
        
        // Print some statistics
        float min_val = *std::min_element(host_data.begin(), host_data.end());
        float max_val = *std::max_element(host_data.begin(), host_data.end());
        float sum = std::accumulate(host_data.begin(), host_data.end(), 0.0f);
        float mean = sum / host_data.size();
        
        std::cout << "\nStatistics:" << std::endl;
        std::cout << "  Min: " << min_val << std::endl;
        std::cout << "  Max: " << max_val << std::endl;
        std::cout << "  Mean: " << mean << " (should be ~0.5)" << std::endl;
        
        // Verify all values are in range
        for (size_t i = 0; i < 1000; ++i) {
            REQUIRE(host_data[i] >= 0.0f);
            REQUIRE(host_data[i] <= 1.0f);
        }
        
        // Check that we're getting reasonable distribution
        REQUIRE(min_val >= 0.0f);
        REQUIRE(max_val <= 1.0f);
        REQUIRE(mean > 0.4f);  // Should be around 0.5
        REQUIRE(mean < 0.6f);
        
        LOGINFO("Uniform float generation passed");
    }

    SECTION("Gaussian float generation") {    
        DeviceBuffer<float> buffer(1000, resource);
        auto& sycl_device = ARBD::SYCL::SYCLManager::get_current_device();
        
        // Use the SAME random generator instance
        auto event = random_gen.generate_gaussian(buffer, 0.0f, 1.0f);
        std::vector<float> host_data(1000);
        event.wait();
        buffer.copy_to_host(host_data.data());
        
        // Print first 20 values to see what we're getting
        std::cout << "\n=== Gaussian Float Generation (mean=0.0, stddev=1.0) ===" << std::endl;
        std::cout << "First 20 values:" << std::endl;
        for (size_t i = 0; i < 20; ++i) {
            std::cout << std::setw(3) << i << ": " << std::setprecision(6) << std::fixed 
                      << host_data[i] << std::endl;
        }
        
        // Print some statistics
        float min_val = *std::min_element(host_data.begin(), host_data.end());
        float max_val = *std::max_element(host_data.begin(), host_data.end());
        float sum = std::accumulate(host_data.begin(), host_data.end(), 0.0f);
        float mean = sum / host_data.size();
        
        // Calculate standard deviation
        float sq_sum = 0.0f;
        for (float val : host_data) {
            sq_sum += (val - mean) * (val - mean);
        }
        float stddev = std::sqrt(sq_sum / host_data.size());
        
        std::cout << "\nStatistics:" << std::endl;
        std::cout << "  Min: " << min_val << std::endl;
        std::cout << "  Max: " << max_val << std::endl;
        std::cout << "  Mean: " << mean << " (should be ~0.0)" << std::endl;
        std::cout << "  StdDev: " << stddev << " (should be ~1.0)" << std::endl;
        
        // Count values in different ranges to check distribution
        int count_within_1_sigma = 0;
        int count_within_2_sigma = 0;
        for (float val : host_data) {
            if (std::abs(val) <= 1.0f) count_within_1_sigma++;
            if (std::abs(val) <= 2.0f) count_within_2_sigma++;
        }
        
        std::cout << "  Within 1σ: " << count_within_1_sigma << "/1000 (" 
                  << (100.0f * count_within_1_sigma / 1000.0f) << "%, should be ~68%)" << std::endl;
        std::cout << "  Within 2σ: " << count_within_2_sigma << "/1000 (" 
                  << (100.0f * count_within_2_sigma / 1000.0f) << "%, should be ~95%)" << std::endl;
        
        // Verify all values are finite
        for (size_t i = 0; i < 1000; ++i) {
            REQUIRE(std::isfinite(host_data[i]));
        }
        
        // Check that we're getting reasonable Gaussian distribution
        REQUIRE(std::abs(mean) < 0.2f);  // Mean should be close to 0
        REQUIRE(stddev > 0.8f);          // Standard deviation should be close to 1
        REQUIRE(stddev < 1.2f);
        REQUIRE(count_within_1_sigma > 600);  // At least 60% within 1 sigma
        REQUIRE(count_within_2_sigma > 900);  // At least 90% within 2 sigma
        
        LOGINFO("Gaussian float generation passed");
    }
}

#endif // USE_SYCL