#include "catch_boiler.h"
#include <catch2/catch_test_macros.hpp>
#ifdef USE_SYCL

#include "Backend/Buffer.h"
#include "Backend/Kernels.h"
#include "Backend/Events.h"
#include "Backend/Resource.h"
#include "Backend/Proxy.h"
#include <vector>
#include <numeric>
#include <string>

using namespace ARBD;

TEST_CASE("SYCL Resource Creation and Properties", "[sycl][resource]") {
    Resource sycl_resource(Resource::SYCL, 0);
    
    SECTION("Resource type is correct") {
        CHECK(sycl_resource.type == Resource::SYCL);
        CHECK(sycl_resource.id == 0);
    }
    
    SECTION("Resource type string") {
        CHECK(std::string(sycl_resource.getTypeString()) == "SYCL");
    }
    
    SECTION("Resource is device") {
        CHECK(sycl_resource.is_device());
        CHECK_FALSE(sycl_resource.is_host());
    }
    
    SECTION("Resource supports async") {
        CHECK(sycl_resource.supports_async());
    }
    
    SECTION("Memory space is device") {
        CHECK(std::string(sycl_resource.getMemorySpace()) == "device");
    }
}

TEST_CASE("SYCL DeviceBuffer Basic Operations", "[sycl][buffer]") {
    Resource sycl_resource(Resource::SYCL, 0);
    
    SECTION("Empty buffer creation") {
        DeviceBuffer<int> buffer;
        CHECK(buffer.empty());
        CHECK(buffer.size() == 0);
    }
    
    SECTION("Buffer allocation") {
        const size_t count = 1000;
        DeviceBuffer<float> buffer(count, sycl_resource);
        
        CHECK(buffer.size() == count);
        CHECK_FALSE(buffer.empty());
        CHECK(buffer.get_resource().type == Resource::SYCL);
        CHECK(buffer.data() != nullptr);
    }
    
    SECTION("Buffer move semantics") {
        const size_t count = 500;
        DeviceBuffer<double> buffer1(count, sycl_resource);
        auto* original_ptr = buffer1.data();
        
        DeviceBuffer<double> buffer2 = std::move(buffer1);
        
        CHECK(buffer2.size() == count);
        CHECK(buffer2.data() == original_ptr);
        CHECK(buffer1.size() == 0);
        CHECK(buffer1.data() == nullptr);
    }
}

TEST_CASE("SYCL DeviceBuffer Data Transfer", "[sycl][buffer][transfer]") {
    Resource sycl_resource(Resource::SYCL, 0);
    const size_t count = 100;
    
    SECTION("Host to device copy") {
        DeviceBuffer<int> buffer(count, sycl_resource);
        std::vector<int> host_data(count);
        std::iota(host_data.begin(), host_data.end(), 1); // 1, 2, 3, ...
        
        CHECK_NOTHROW(buffer.copy_from_host(host_data.data(), count));
    }
    
    SECTION("Device to host copy") {
        DeviceBuffer<int> buffer(count, sycl_resource);
        std::vector<int> host_input(count);
        std::vector<int> host_output(count);
        std::iota(host_input.begin(), host_input.end(), 10); // 10, 11, 12, ...
        
        buffer.copy_from_host(host_input.data(), count);
        CHECK_NOTHROW(buffer.copy_to_host(host_output.data(), count));
        
        // Verify data integrity
        for (size_t i = 0; i < count; ++i) {
            CHECK(host_output[i] == host_input[i]);
        }
    }
    
    SECTION("Partial copy operations") {
        DeviceBuffer<int> buffer(count, sycl_resource);
        std::vector<int> host_data(count / 2);
        std::iota(host_data.begin(), host_data.end(), 100);
        
        CHECK_NOTHROW(buffer.copy_from_host(host_data.data(), count / 2));
        
        std::vector<int> readback(count / 2);
        CHECK_NOTHROW(buffer.copy_to_host(readback.data(), count / 2));
        
        for (size_t i = 0; i < count / 2; ++i) {
            CHECK(readback[i] == host_data[i]);
        }
    }
}

TEST_CASE("SYCL UnifiedBuffer Operations", "[sycl][unified_buffer]") {
    Resource sycl_resource(Resource::SYCL, 0);
    const size_t count = 200;
    
    SECTION("Unified buffer creation") {
        UnifiedBuffer<float> buffer(count, sycl_resource);
        
        CHECK(buffer.size() == count);
        CHECK_FALSE(buffer.empty());
        CHECK(buffer.primary_location().type == Resource::SYCL);
    }
    
    SECTION("Memory migration") {
        UnifiedBuffer<int> buffer(count, sycl_resource);
        
        // Ensure data is available at SYCL device
        buffer.ensure_available_at(sycl_resource);
        auto* ptr = buffer.get_ptr(sycl_resource);
        CHECK(ptr != nullptr);
        
        auto locations = buffer.available_locations();
        CHECK(locations.size() >= 1);
        CHECK(locations[0].type == Resource::SYCL);
    }
    
    SECTION("Data copy operations") {
        UnifiedBuffer<int> buffer(count, sycl_resource);
        std::vector<int> host_data(count);
        std::iota(host_data.begin(), host_data.end(), 42);
        
        CHECK_NOTHROW(buffer.copy_from_host(host_data.data(), sycl_resource, count));
        
        std::vector<int> readback(count);
        CHECK_NOTHROW(buffer.copy_to_host(readback.data(), sycl_resource, count));
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(readback[i] == host_data[i]);
        }
    }
}

TEST_CASE("SYCL Event Management", "[sycl][events]") {
    Resource sycl_resource(Resource::SYCL, 0);
    
    SECTION("Empty event is complete") {
        Event empty_event;
        CHECK(empty_event.is_complete());
        CHECK_FALSE(empty_event.is_valid());
    }
    
    SECTION("EventList operations") {
        EventList event_list;
        CHECK(event_list.empty());
        CHECK(event_list.all_complete());
        
        Event event1;
        Event event2;
        
        event_list.add(event1);
        event_list.add(event2);
        
        CHECK_FALSE(event_list.empty());
        CHECK_NOTHROW(event_list.wait_all());
    }
    
    SECTION("Event dependency tracking") {
        DeviceBuffer<int> buffer(100, sycl_resource);
        EventList deps;
        
        // This should not throw and should wait for dependencies
        CHECK_NOTHROW(buffer.get_write_access(deps));
        CHECK_NOTHROW(buffer.get_read_access(deps));
    }
}

TEST_CASE("SYCL Kernel Launch Infrastructure", "[sycl][kernels]") {
    Resource sycl_resource(Resource::SYCL, 0);
    
    SECTION("LaunchConfig creation") {
        using namespace Kernels;
        LaunchConfig config;
        CHECK(config.grid_size == 1);
        CHECK(config.block_size == 256);
        CHECK_FALSE(config.async);
    }
    
    SECTION("KernelConfig with dependencies") {
        using namespace Kernels;
        KernelConfig config;
        CHECK(config.grid_size == 0);
        CHECK(config.block_size == 256);
        CHECK_FALSE(config.async);
        CHECK(config.dependencies.empty());
    }
    
    SECTION("Simple kernel execution") {
        using namespace Kernels;
        const size_t count = 50;
        DeviceBuffer<int> buffer(count, sycl_resource);
        
        // Simple kernel that fills buffer with index values
        auto fill_kernel = [](size_t i, int* data) {
            data[i] = static_cast<int>(i);
        };
        
        Event kernel_event;
        CHECK_NOTHROW(
            kernel_event = simple_kernel(sycl_resource, count, fill_kernel, buffer)
        );
        
        // Wait for kernel completion
        kernel_event.wait();
        
        // Verify results
        std::vector<int> host_result(count);
        buffer.copy_to_host(host_result.data(), count);
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(host_result[i] == static_cast<int>(i));
        }
    }
}

TEST_CASE("SYCL Memory Operations", "[sycl][memory]") {
    Resource sycl_resource(Resource::SYCL, 0);
    
    SECTION("Memory allocation and deallocation") {
        const size_t size = 1024 * sizeof(float);
        
        void* ptr = MemoryOps::allocate(size, sycl_resource);
        CHECK(ptr != nullptr);
        
        CHECK_NOTHROW(MemoryOps::deallocate(ptr, sycl_resource));
    }
    
    SECTION("Host to device memory copy") {
        const size_t count = 100;
        const size_t size = count * sizeof(int);
        
        std::vector<int> host_data(count);
        std::iota(host_data.begin(), host_data.end(), 1);
        
        void* device_ptr = MemoryOps::allocate(size, sycl_resource);
        CHECK(device_ptr != nullptr);
        
        CHECK_NOTHROW(
            MemoryOps::copy_from_host(device_ptr, host_data.data(), size, sycl_resource)
        );
        
        std::vector<int> result(count);
        CHECK_NOTHROW(
            MemoryOps::copy_to_host(result.data(), device_ptr, size, sycl_resource)
        );
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(result[i] == host_data[i]);
        }
        
        MemoryOps::deallocate(device_ptr, sycl_resource);
    }
}

TEST_CASE("SYCL GenericAllocator", "[sycl][allocator]") {
    Resource sycl_resource(Resource::SYCL, 0);
    
    SECTION("Basic allocation") {
        const size_t count = 150;
        auto buffer = GenericAllocator<float>::allocate(count, sycl_resource);
        
        CHECK(buffer.size() == count);
        CHECK(buffer.get_resource().type == Resource::SYCL);
    }
    
    SECTION("Zero-initialized allocation") {
        const size_t count = 100;
        auto buffer = GenericAllocator<int>::allocate_zeroed(count, sycl_resource);
        
        std::vector<int> result(count);
        buffer.copy_to_host(result.data(), count);
        
        for (int val : result) {
            CHECK(val == 0);
        }
    }
    
    SECTION("Sequence allocation") {
        const size_t count = 50;
        auto buffer = GenericAllocator<int>::allocate_sequence(count, sycl_resource, 10, 2);
        
        std::vector<int> result(count);
        buffer.copy_to_host(result.data(), count);
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(result[i] == 10 + static_cast<int>(i) * 2);
        }
    }
}

TEST_CASE("SYCL MultiRef Operations", "[sycl][multiref]") {
    Resource sycl_resource(Resource::SYCL, 0);
    const size_t count = 75;
    
    SECTION("MultiRef with multiple buffers") {
        DeviceBuffer<int> buffer1(count, sycl_resource);
        DeviceBuffer<int> buffer2(count, sycl_resource);
        DeviceBuffer<int> buffer3(count, sycl_resource);
        
        auto multi_ref = make_multi_ref(buffer1, buffer2, buffer3);
        
        CHECK(multi_ref.size() == 3);
        
        EventList deps;
        auto ptrs = multi_ref.get_write_access(deps);
        
        // Verify we got valid pointers
        CHECK(std::get<0>(ptrs) != nullptr);
        CHECK(std::get<1>(ptrs) != nullptr);
        CHECK(std::get<2>(ptrs) != nullptr);
    }
    
    SECTION("Read-only MultiRef") {
        DeviceBuffer<float> buffer1(count, sycl_resource);
        DeviceBuffer<float> buffer2(count, sycl_resource);
        
        const auto& const_buffer1 = buffer1;
        const auto& const_buffer2 = buffer2;
        
        auto const_multi_ref = MultiRef<const DeviceBuffer<float>, const DeviceBuffer<float>>(
            const_buffer1, const_buffer2);
        
        EventList deps;
        auto ptrs = const_multi_ref.get_read_access(deps);
        
        CHECK(std::get<0>(ptrs) != nullptr);
        CHECK(std::get<1>(ptrs) != nullptr);
    }
}

TEST_CASE("SYCL Utility Functions", "[sycl][utilities]") {
    Resource sycl_resource(Resource::SYCL, 0);
    const size_t count = 80;
    
    SECTION("Buffer copy utility") {
        DeviceBuffer<int> source(count, sycl_resource);
        DeviceBuffer<int> dest(count, sycl_resource);
        
        // Fill source with data
        std::vector<int> host_data(count);
        std::iota(host_data.begin(), host_data.end(), 100);
        source.copy_from_host(host_data.data(), count);
        
        CHECK_NOTHROW(copy_buffer(source, dest, sycl_resource));
        
        // Verify copy worked
        std::vector<int> result(count);
        dest.copy_to_host(result.data(), count);
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(result[i] == host_data[i]);
        }
    }
    
    SECTION("Buffer fill utility") {
        DeviceBuffer<float> buffer(count, sycl_resource);
        const float fill_value = 3.14f;
        
        CHECK_NOTHROW(fill_buffer(buffer, fill_value, sycl_resource));
        
        std::vector<float> result(count);
        buffer.copy_to_host(result.data(), count);
        
        for (float val : result) {
            CHECK(val == Approx(fill_value));
        }
    }
}

TEST_CASE("SYCL Proxy Basic Operations", "[sycl][proxy]") {
    Resource sycl_resource(Resource::SYCL, 0);
    
    SECTION("Proxy construction") {
        Proxy<int> int_proxy(sycl_resource);
        
        CHECK(int_proxy.get_location().type == Resource::SYCL);
        CHECK_FALSE(int_proxy.is_valid()); // No object assigned yet
    }
    
    SECTION("Arithmetic type proxy") {
        int value = 42;
        Proxy<int> int_proxy(sycl_resource, &value);
        
        CHECK(int_proxy.is_valid());
        CHECK(int_proxy.get_address() == &value);
    }
}

TEST_CASE("SYCL Integration Test", "[sycl][integration]") {
    Resource sycl_resource(Resource::SYCL, 0);
    const size_t count = 1000;
    
    SECTION("Complete workflow: allocation -> computation -> verification") {
        // Allocate buffers
        auto input = GenericAllocator<float>::allocate_sequence(count, sycl_resource, 1.0f, 0.5f);
        auto output = GenericAllocator<float>::allocate(count, sycl_resource);
        
        // Fill input with test data
        std::vector<float> host_input(count);
        for (size_t i = 0; i < count; ++i) {
            host_input[i] = 1.0f + i * 0.5f;
        }
        input.copy_from_host(host_input.data(), count);
        
        // Simple computation kernel: output[i] = input[i] * 2.0
        using namespace Kernels;
        auto compute_kernel = [](size_t i, const float* in, float* out) {
            out[i] = in[i] * 2.0f;
        };
        
        Event compute_event = simple_kernel(sycl_resource, count, compute_kernel, input, output);
        compute_event.wait();
        
        // Verify results
        std::vector<float> host_output(count);
        output.copy_to_host(host_output.data(), count);
        
        for (size_t i = 0; i < count; ++i) {
            float expected = host_input[i] * 2.0f;
            CHECK(host_output[i] == Approx(expected));
        }
    }
}

#else

TEST_CASE("SYCL Support Not Enabled", "[sycl]") {
    CHECK(true); // Always pass when SYCL is not available
}

#endif // USE_SYCL 