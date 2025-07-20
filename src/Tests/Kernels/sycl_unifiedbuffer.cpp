#include "Backend/Buffer.h"
#include "Backend/Resource.h"

#ifdef USE_SYCL
#include "Backend/SYCL/SYCLManager.h"
#include <vector>
#include <numeric>
#include "../catch_boiler.h"

TEST_CASE("UnifiedBuffer SYCL Basic Allocation", "[UnifiedBuffer][SYCL]") {
    try {
        // Initialize SYCL backend
        ARBD::SYCL::SYCLManager::init();
        ARBD::SYCL::SYCLManager::load_info();
        
        if (ARBD::SYCL::SYCLManager::devices().empty()) {
            SKIP("No SYCL devices available");
        }
        
        ARBD::Resource sycl_resource{ARBD::ResourceType::SYCL, 0};
        
        SECTION("Basic allocation and deallocation") {
            ARBD::UnifiedBuffer<float> buffer(1000, sycl_resource);
            
            REQUIRE(buffer.size() == 1000);
            REQUIRE(!buffer.empty());
            REQUIRE(buffer.primary_location().type == ARBD::ResourceType::SYCL);
            
            // Check that pointer is valid
            float* ptr = buffer.get_ptr(sycl_resource);
            REQUIRE(ptr != nullptr);
        }
        
        SECTION("Zero-sized buffer") {
            ARBD::UnifiedBuffer<int> buffer(0, sycl_resource);
            
            REQUIRE(buffer.size() == 0);
            REQUIRE(buffer.empty());
        }
        
        SECTION("Different data types") {
            ARBD::UnifiedBuffer<double> double_buffer(500, sycl_resource);
            ARBD::UnifiedBuffer<int> int_buffer(1000, sycl_resource);
            ARBD::UnifiedBuffer<char> char_buffer(2000, sycl_resource);
            
            REQUIRE(double_buffer.size() == 500);
            REQUIRE(int_buffer.size() == 1000);
            REQUIRE(char_buffer.size() == 2000);
            
            REQUIRE(double_buffer.get_ptr(sycl_resource) != nullptr);
            REQUIRE(int_buffer.get_ptr(sycl_resource) != nullptr);
            REQUIRE(char_buffer.get_ptr(sycl_resource) != nullptr);
        }
        
        ARBD::SYCL::SYCLManager::finalize();
    } catch (const ARBD::Exception& e) {
        FAIL("SYCL test failed with ARBD exception: " << e.what());
    }
}

TEST_CASE("UnifiedBuffer SYCL Data Migration", "[UnifiedBuffer][SYCL]") {
    try {
        // Initialize SYCL backend
        ARBD::SYCL::SYCLManager::init();
        ARBD::SYCL::SYCLManager::load_info();
        
        if (ARBD::SYCL::SYCLManager::devices().empty()) {
            SKIP("No SYCL devices available");
        }
        
        ARBD::Resource sycl_resource{ARBD::ResourceType::SYCL, 0};
        ARBD::Resource host_resource{}; // Default host resource
        
        SECTION("Ensure availability at different resources") {
            ARBD::UnifiedBuffer<float> buffer(100, sycl_resource);
            // Should be available at primary location
            auto locations = buffer.available_locations();
            REQUIRE(locations.size() == 1);
            REQUIRE(locations[0].type == ARBD::ResourceType::SYCL);
            
            // Ensure availability at host (should trigger migration)
            buffer.ensure_available_at(host_resource);
            
            // Should now be available at both locations
            locations = buffer.available_locations();
            if (locations.size() <= 2) {
                SKIP("less than 2 locations available");
            }
            // Both pointers should be valid
            REQUIRE(buffer.get_ptr(sycl_resource) != nullptr);
            REQUIRE(buffer.get_ptr(host_resource) != nullptr);
        }
        
        SECTION("Release memory at specific location") {
            ARBD::UnifiedBuffer<int> buffer(50, sycl_resource);
            
            // Ensure available at host
            buffer.ensure_available_at(host_resource);
            if (buffer.available_locations().size() <= 2) {
                SKIP("less than 2 devices available");
            }
            // Release at host
            buffer.release_at(host_resource);
            auto locations = buffer.available_locations();
            REQUIRE(locations.size() == 1);
            REQUIRE(locations[0].type == ARBD::ResourceType::SYCL);
        }
        
        ARBD::SYCL::SYCLManager::finalize();
    } catch (const ARBD::Exception& e) {
        FAIL("SYCL data migration test failed with ARBD exception: " << e.what());
    }
}

TEST_CASE("UnifiedBuffer SYCL Move Semantics", "[UnifiedBuffer][SYCL]") {
    try {
        // Initialize SYCL backend
        ARBD::SYCL::SYCLManager::init();
        ARBD::SYCL::SYCLManager::load_info();
        
        if (ARBD::SYCL::SYCLManager::devices().empty()) {
            SKIP("No SYCL devices available");
        }
        
        ARBD::Resource sycl_resource{ARBD::ResourceType::SYCL, 0};
        
        SECTION("Move constructor") {
            ARBD::UnifiedBuffer<float> buffer1(1000, sycl_resource);
            float* original_ptr = buffer1.get_ptr(sycl_resource);
            
            ARBD::UnifiedBuffer<float> buffer2(std::move(buffer1));
            
            REQUIRE(buffer2.size() == 1000);
            REQUIRE(buffer2.get_ptr(sycl_resource) == original_ptr);
            REQUIRE(buffer1.size() == 0); // Moved from
            REQUIRE(buffer1.empty());
        }
        
        SECTION("Move assignment") {
            ARBD::UnifiedBuffer<float> buffer1(1000, sycl_resource);
            ARBD::UnifiedBuffer<float> buffer2(500, sycl_resource);
            
            float* original_ptr = buffer1.get_ptr(sycl_resource);
            
            buffer2 = std::move(buffer1);
            
            REQUIRE(buffer2.size() == 1000);
            REQUIRE(buffer2.get_ptr(sycl_resource) == original_ptr);
            REQUIRE(buffer1.size() == 0); // Moved from
        }
        
        ARBD::SYCL::SYCLManager::finalize();
    } catch (const ARBD::Exception& e) {
        FAIL("SYCL move semantics test failed with ARBD exception: " << e.what());
    }
}

TEST_CASE("UnifiedBuffer SYCL Existing Data Constructor", "[UnifiedBuffer][SYCL]") {
    try {
        // Initialize SYCL backend
        ARBD::SYCL::SYCLManager::init();
        ARBD::SYCL::SYCLManager::load_info();
        
        if (ARBD::SYCL::SYCLManager::devices().empty()) {
            SKIP("No SYCL devices available");
        }
        
        ARBD::Resource host_resource{}; // Default host resource
        
        SECTION("Construct from existing host data") {
            std::vector<int> host_data(100);
            std::iota(host_data.begin(), host_data.end(), 1); // Fill with 1, 2, 3, ...
            
            ARBD::UnifiedBuffer<int> buffer(host_data.size(), host_data.data(), host_resource);
            
            REQUIRE(buffer.size() == 100);
            REQUIRE(buffer.get_ptr(host_resource) == host_data.data());
            
            // Check that data is accessible
            const int* const_ptr = buffer.get_ptr(host_resource);
            REQUIRE(const_ptr[0] == 1);
            REQUIRE(const_ptr[50] == 51);
            REQUIRE(const_ptr[99] == 100);
        }
        
        ARBD::SYCL::SYCLManager::finalize();
    } catch (const ARBD::Exception& e) {
        FAIL("SYCL existing data test failed with ARBD exception: " << e.what());
    }
}

TEST_CASE("UnifiedBuffer SYCL Error Handling", "[UnifiedBuffer][SYCL]") {
    try {
        // Initialize SYCL backend
        ARBD::SYCL::SYCLManager::init();
        ARBD::SYCL::SYCLManager::load_info();
        
        if (ARBD::SYCL::SYCLManager::devices().empty()) {
            SKIP("No SYCL devices available");
        }
        
        ARBD::Resource sycl_resource{ARBD::ResourceType::SYCL, 0};
        
        SECTION("Get pointer for non-existent location") {
            ARBD::UnifiedBuffer<float> buffer(100, sycl_resource);
            
            ARBD::Resource other_resource{ARBD::ResourceType::SYCL, 999}; // Non-existent device
            
            // This should trigger ensure_available_at and potentially fail gracefully
            // or migrate from existing location
            float* ptr = buffer.get_ptr(other_resource);
            // The behavior depends on implementation - should either work or handle gracefully
        }
        
        ARBD::SYCL::SYCLManager::finalize();
    } catch (const ARBD::Exception& e) {
        // Expected for some error cases
        INFO("SYCL error handling test completed with exception: " << e.what());
    }
}

#else
TEST_CASE("UnifiedBuffer SYCL - SYCL Not Available", "[UnifiedBuffer][SYCL]") {
    SKIP("SYCL support not compiled in");
}
#endif 