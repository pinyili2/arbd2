#include "catch_boiler.h"
#include <catch2/matchers/catch_matchers_floating_point.hpp>
using Catch::Approx;

#ifdef USE_METAL

#include "Backend/Buffer.h"
#include "Backend/Kernels.h"
#include "Backend/Events.h"
#include "Backend/Resource.h"
#include "Backend/Proxy.h"
#include <vector>
#include <numeric>
#include <string>

//Only one GPU unit for METAL. 
using namespace ARBD;

TEST_CASE("Metal Resource Creation and Properties", "[metal][resource]") {
    Resource metal_resource(ResourceType::METAL, 0);
    
    SECTION("Resource type is correct") {
        CHECK(metal_resource.type == ResourceType::METAL);
        CHECK(metal_resource.id == 0);
    }
    
    SECTION("Resource type string") {
        CHECK(std::string(metal_resource.getTypeString()) == "METAL");
    }
    
    SECTION("Resource is device") {
        CHECK(metal_resource.is_device());
        CHECK_FALSE(metal_resource.is_host());
    }
    
    SECTION("Memory space is device") {
        CHECK(std::string(metal_resource.getMemorySpace()) == "device");
    }
    
    SECTION("Resource toString") {
        std::string expected = "METAL[0]";
        CHECK(metal_resource.toString() == expected);
    }
}

TEST_CASE("Metal DeviceBuffer Basic Operations", "[metal][buffer]") {
    Resource metal_resource(ResourceType::METAL, 0);
    
    SECTION("Empty buffer creation") {
        DeviceBuffer<int> buffer;
        CHECK(buffer.empty());
        CHECK(buffer.size() == 0);
    }
    
    SECTION("Buffer allocation") {
        const size_t count = 1000;
        DeviceBuffer<float> buffer(count, metal_resource);
        
        CHECK(buffer.size() == count);
        CHECK_FALSE(buffer.empty());
        CHECK(buffer.get_resource().type == ResourceType::METAL);
        CHECK(buffer.data() != nullptr);
    }
    
    SECTION("Buffer move semantics") {
        const size_t count = 500;
        DeviceBuffer<double> buffer1(count, metal_resource);
        auto* original_ptr = buffer1.data();
        
        DeviceBuffer<double> buffer2 = std::move(buffer1);
        
        CHECK(buffer2.size() == count);
        CHECK(buffer2.data() == original_ptr);
        CHECK(buffer1.size() == 0);
        CHECK(buffer1.data() == nullptr);
    }
    
    SECTION("Different data types") {
        // Test various primitive types
        const size_t count = 100;
        
        DeviceBuffer<int> int_buffer(count, metal_resource);
        CHECK(int_buffer.size() == count);
        
        DeviceBuffer<float> float_buffer(count, metal_resource);
        CHECK(float_buffer.size() == count);
        
        DeviceBuffer<uint32_t> uint_buffer(count, metal_resource);
        CHECK(uint_buffer.size() == count);
    }
}

TEST_CASE("Metal DeviceBuffer Data Transfer", "[metal][buffer][transfer]") {
    Resource metal_resource(ResourceType::METAL, 0);
    const size_t count = 100;
    
    SECTION("Host to device copy") {
        DeviceBuffer<int> buffer(count, metal_resource);
        std::vector<int> host_data(count);
        std::iota(host_data.begin(), host_data.end(), 1); // 1, 2, 3, ...
        
        CHECK_NOTHROW(buffer.copy_from_host(host_data.data(), count));
    }
    
    SECTION("Device to host copy") {
        DeviceBuffer<int> buffer(count, metal_resource);
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
        DeviceBuffer<int> buffer(count, metal_resource);
        std::vector<int> host_data(count / 2);
        std::iota(host_data.begin(), host_data.end(), 100);
        
        CHECK_NOTHROW(buffer.copy_from_host(host_data.data(), count / 2));
        
        std::vector<int> readback(count / 2);
        CHECK_NOTHROW(buffer.copy_to_host(readback.data(), count / 2));
        
        for (size_t i = 0; i < count / 2; ++i) {
            CHECK(readback[i] == host_data[i]);
        }
    }
    
    SECTION("Float data transfer") {
        DeviceBuffer<float> buffer(count, metal_resource);
        std::vector<float> host_data(count);
        for (size_t i = 0; i < count; ++i) {
            host_data[i] = static_cast<float>(i) * 0.5f + 1.0f;
        }
        
        buffer.copy_from_host(host_data.data(), count);
        
        std::vector<float> readback(count);
        buffer.copy_to_host(readback.data(), count);
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(readback[i] == Approx(host_data[i]));
        }
    }
}

TEST_CASE("Metal UnifiedBuffer Operations", "[metal][unified_buffer]") {
    Resource metal_resource(ResourceType::METAL, 0);
    const size_t count = 200;
    
    SECTION("Unified buffer creation") {
        UnifiedBuffer<float> buffer(count, metal_resource);
        
        CHECK(buffer.size() == count);
        CHECK_FALSE(buffer.empty());
        CHECK(buffer.primary_location().type == ResourceType::METAL);
    }
    
    SECTION("Memory migration") {
        UnifiedBuffer<int> buffer(count, metal_resource);
        
        // Ensure data is available at Metal device
        buffer.ensure_available_at(metal_resource);
        auto* ptr = buffer.get_ptr(metal_resource);
        CHECK(ptr != nullptr);
        
        auto locations = buffer.available_locations();
        CHECK(locations.size() >= 1);
        CHECK(locations[0].type == ResourceType::METAL);
    }
    
    SECTION("Data copy operations") {
        UnifiedBuffer<int> buffer(count, metal_resource);
        std::vector<int> host_data(count);
        std::iota(host_data.begin(), host_data.end(), 42);
        
        CHECK_NOTHROW(buffer.copy_from_host(host_data.data(), metal_resource, count));
        
        std::vector<int> readback(count);
        CHECK_NOTHROW(buffer.copy_to_host(readback.data(), metal_resource, count));
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(readback[i] == host_data[i]);
        }
    }
    
    SECTION("Multiple resource management") {
        UnifiedBuffer<float> buffer(count, metal_resource);
        
        // Fill with initial data
        std::vector<float> initial_data(count, 3.14f);
        buffer.copy_from_host(initial_data.data(), metal_resource, count);
        
        // Test synchronization
        CHECK_NOTHROW(buffer.synchronize_all());
        
        // Verify data is still correct after sync
        std::vector<float> synced_data(count);
        buffer.copy_to_host(synced_data.data(), metal_resource, count);
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(synced_data[i] == Approx(3.14f));
        }
    }
}

TEST_CASE("Metal Event Management", "[metal][events]") {
    Resource metal_resource(ResourceType::METAL, 0);
    
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
        DeviceBuffer<int> buffer(100, metal_resource);
        EventList deps;
        
        // This should not throw and should wait for dependencies
        CHECK_NOTHROW(buffer.get_write_access(deps));
        CHECK_NOTHROW(buffer.get_read_access(deps));
    }
    
    SECTION("Event completion state") {
        DeviceBuffer<float> buffer(50, metal_resource);
        Event last_event = buffer.get_last_event();
        
        // For default constructed events  
        if (last_event.is_valid()) {
            CHECK_NOTHROW(last_event.wait());
            CHECK(last_event.is_complete());
        }
    }
}

TEST_CASE("Metal Memory Operations", "[metal][memory]") {
    Resource metal_resource(ResourceType::METAL, 0);
    
    SECTION("Memory allocation and deallocation") {
        const size_t size = 1024 * sizeof(float);
        
        void* ptr = MemoryOps::allocate(size, metal_resource);
        CHECK(ptr != nullptr);
        
        CHECK_NOTHROW(MemoryOps::deallocate(ptr, metal_resource));
    }
    
    SECTION("Host to device memory copy") {
        const size_t count = 100;
        const size_t size = count * sizeof(int);
        
        std::vector<int> host_data(count);
        std::iota(host_data.begin(), host_data.end(), 1);
        
        void* device_ptr = MemoryOps::allocate(size, metal_resource);
        CHECK(device_ptr != nullptr);
        
        CHECK_NOTHROW(
            MemoryOps::copy_from_host(device_ptr, host_data.data(), size, metal_resource)
        );
        
        std::vector<int> result(count);
        CHECK_NOTHROW(
            MemoryOps::copy_to_host(result.data(), device_ptr, size, metal_resource)
        );
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(result[i] == host_data[i]);
        }
        
        MemoryOps::deallocate(device_ptr, metal_resource);
    }
    
    SECTION("Large memory allocation") {
        const size_t large_size = 1024 * 1024 * sizeof(float); // 4MB
        
        void* large_ptr = MemoryOps::allocate(large_size, metal_resource);
        CHECK(large_ptr != nullptr);
        
        MemoryOps::deallocate(large_ptr, metal_resource);
    }
}

TEST_CASE("Metal Kernel Launch Infrastructure", "[metal][kernels]") {
    Resource metal_resource(ResourceType::METAL, 0);
    
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
        DeviceBuffer<int> buffer(count, metal_resource);
        
        // Simple kernel that fills buffer with index values
        auto fill_kernel = [](size_t i, int* data) {
            data[i] = static_cast<int>(i);
        };
        
        Event kernel_event;
        CHECK_NOTHROW(
            kernel_event = simple_kernel(metal_resource, count, fill_kernel, buffer)
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
    
    SECTION("Mathematical computation kernel") {
        using namespace Kernels;
        const size_t count = 100;
        DeviceBuffer<float> input(count, metal_resource);
        DeviceBuffer<float> output(count, metal_resource);
        
        // Initialize input data
        std::vector<float> host_input(count);
        for (size_t i = 0; i < count; ++i) {
            host_input[i] = static_cast<float>(i) + 1.0f;
        }
        input.copy_from_host(host_input.data(), count);
        
        // Kernel that computes square root
        auto sqrt_kernel = [](size_t i, const float* in, float* out) {
            out[i] = std::sqrt(in[i]);
        };
        
        Event compute_event = simple_kernel(metal_resource, count, sqrt_kernel, input, output);
        compute_event.wait();
        
        // Verify results
        std::vector<float> host_output(count);
        output.copy_to_host(host_output.data(), count);
        
        for (size_t i = 0; i < count; ++i) {
            float expected = std::sqrt(host_input[i]);
            CHECK(host_output[i] == Approx(expected));
        }
    }
}

TEST_CASE("Metal GenericAllocator", "[metal][allocator]") {
    Resource metal_resource(ResourceType::METAL, 0);
    
    SECTION("Basic allocation") {
        const size_t count = 150;
        auto buffer = GenericAllocator<float>::allocate(count, metal_resource);
        
        CHECK(buffer.size() == count);
        CHECK(buffer.get_resource().type == ResourceType::METAL);
    }
    
    SECTION("Zero-initialized allocation") {
        const size_t count = 100;
        auto buffer = GenericAllocator<int>::allocate_zeroed(count, metal_resource);
        
        std::vector<int> result(count);
        buffer.copy_to_host(result.data(), count);
        
        for (int val : result) {
            CHECK(val == 0);
        }
    }
    
    SECTION("Sequence allocation") {
        const size_t count = 50;
        auto buffer = GenericAllocator<int>::allocate_sequence(count, metal_resource, 10, 2);
        
        std::vector<int> result(count);
        buffer.copy_to_host(result.data(), count);
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(result[i] == 10 + static_cast<int>(i) * 2);
        }
    }
    
    SECTION("Custom initialization") {
        const size_t count = 75;
        auto init_func = [](size_t i) { return static_cast<float>(i * i); };
        auto buffer = GenericAllocator<float>::allocate_initialized(count, metal_resource, init_func);
        
        std::vector<float> result(count);
        buffer.copy_to_host(result.data(), count);
        
        for (size_t i = 0; i < count; ++i) {
            float expected = static_cast<float>(i * i);
            CHECK(result[i] == Approx(expected));
        }
    }
}

TEST_CASE("Metal MultiRef Operations", "[metal][multiref]") {
    Resource metal_resource(ResourceType::METAL, 0);
    const size_t count = 75;
    
    SECTION("MultiRef with multiple buffers") {
        DeviceBuffer<int> buffer1(count, metal_resource);
        DeviceBuffer<int> buffer2(count, metal_resource);
        DeviceBuffer<int> buffer3(count, metal_resource);
        
        auto multi_ref = make_multi_ref(buffer1, buffer2, buffer3);
        
        CHECK(multi_ref.size() == 3);
        
        EventList deps;
        auto ptrs = multi_ref.get_write_access(deps);
        
        // Verify we got valid pointers
        CHECK(std::get<0>(ptrs) != nullptr);
        CHECK(std::get<1>(ptrs) != nullptr);
        CHECK(std::get<2>(ptrs) != nullptr);
    }
    
    SECTION("MultiRef event completion") {
        DeviceBuffer<float> buffer1(count, metal_resource);
        DeviceBuffer<float> buffer2(count, metal_resource);
        
        auto multi_ref = make_multi_ref(buffer1, buffer2);
        
        Event dummy_event;
        CHECK_NOTHROW(multi_ref.complete_event_state(dummy_event));
    }
}

TEST_CASE("Metal Utility Functions", "[metal][utilities]") {
    Resource metal_resource(ResourceType::METAL, 0);
    const size_t count = 80;
    
    SECTION("Buffer copy utility") {
        DeviceBuffer<int> source(count, metal_resource);
        DeviceBuffer<int> dest(count, metal_resource);
        
        // Fill source with data
        std::vector<int> host_data(count);
        std::iota(host_data.begin(), host_data.end(), 100);
        source.copy_from_host(host_data.data(), count);
        
        CHECK_NOTHROW(copy_buffer(source, dest, metal_resource));
        
        // Verify copy worked
        std::vector<int> result(count);
        dest.copy_to_host(result.data(), count);
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(result[i] == host_data[i]);
        }
    }
    
    SECTION("Buffer fill utility") {
        DeviceBuffer<float> buffer(count, metal_resource);
        const float fill_value = 2.71828f; // e
        
        CHECK_NOTHROW(fill_buffer(buffer, fill_value, metal_resource));
        
        std::vector<float> result(count);
        buffer.copy_to_host(result.data(), count);
        
        for (float val : result) {
            CHECK(val == Approx(fill_value));
        }
    }
}

TEST_CASE("Metal Proxy Basic Operations", "[metal][proxy]") {
    Resource metal_resource(ResourceType::METAL, 0);
    
    SECTION("Proxy construction") {
        Proxy<int> int_proxy(metal_resource, nullptr);
        
        CHECK(int_proxy.get_location().type == ResourceType::METAL);
        CHECK_FALSE(int_proxy.is_valid()); // No object assigned yet
    }
    
    SECTION("Arithmetic type proxy") {
        int value = 42;
        Proxy<int> int_proxy(metal_resource, &value);
        
        CHECK(int_proxy.is_valid());
        CHECK(int_proxy.get_address() == &value);
    }
    
    SECTION("Basic proxy functionality") {
        const size_t count = 50;
        int test_value = 42;
        
        Proxy<int> int_proxy(metal_resource, &test_value);
        
        CHECK(int_proxy.is_valid());
        CHECK(int_proxy.get_address() == &test_value);
        CHECK(int_proxy.get_location().type == ResourceType::METAL);
    }
}

TEST_CASE("Metal Integration Test", "[metal][integration]") {
    Resource metal_resource(ResourceType::METAL, 0);
    const size_t count = 1000;
    
    SECTION("Complete workflow: allocation -> computation -> verification") {
        // Allocate buffers
        auto input = GenericAllocator<float>::allocate_sequence(count, metal_resource, 1.0f, 0.5f);
        auto output = GenericAllocator<float>::allocate(count, metal_resource);
        
        // Fill input with test data
        std::vector<float> host_input(count);
        for (size_t i = 0; i < count; ++i) {
            host_input[i] = 1.0f + i * 0.5f;
        }
        input.copy_from_host(host_input.data(), count);
        
        // Simple computation kernel: output[i] = input[i] * 2.0 + 1.0
        using namespace Kernels;
        auto compute_kernel = [](size_t i, const float* in, float* out) {
            out[i] = in[i] * 2.0f + 1.0f;
        };
        
        Event compute_event = simple_kernel(metal_resource, count, compute_kernel, input, output);
        compute_event.wait();
        
        // Verify results
        std::vector<float> host_output(count);
        output.copy_to_host(host_output.data(), count);
        
        for (size_t i = 0; i < count; ++i) {
            float expected = host_input[i] * 2.0f + 1.0f;
            CHECK(host_output[i] == Approx(expected));
        }
    }
    
    SECTION("Multi-step computation pipeline") {
        const size_t pipeline_count = 500;
        
        // Create pipeline buffers
        auto buffer1 = GenericAllocator<float>::allocate(pipeline_count, metal_resource);
        auto buffer2 = GenericAllocator<float>::allocate(pipeline_count, metal_resource);
        auto buffer3 = GenericAllocator<float>::allocate(pipeline_count, metal_resource);
        
        // Initialize first buffer
        std::vector<float> initial_data(pipeline_count);
        for (size_t i = 0; i < pipeline_count; ++i) {
            initial_data[i] = static_cast<float>(i + 1);
        }
        buffer1.copy_from_host(initial_data.data(), pipeline_count);
        
        using namespace Kernels;
        
        // Step 1: Square the values
        auto square_kernel = [](size_t i, const float* in, float* out) {
            out[i] = in[i] * in[i];
        };
        
        Event step1 = simple_kernel(metal_resource, pipeline_count, square_kernel, buffer1, buffer2);
        step1.wait();
        
        // Step 2: Take square root (should get back original values)
        auto sqrt_kernel = [](size_t i, const float* in, float* out) {
            out[i] = std::sqrt(in[i]);
        };
        
        Event step2 = simple_kernel(metal_resource, pipeline_count, sqrt_kernel, buffer2, buffer3);
        step2.wait();
        
        // Verify final results
        std::vector<float> final_result(pipeline_count);
        buffer3.copy_to_host(final_result.data(), pipeline_count);
        
        for (size_t i = 0; i < pipeline_count; ++i) {
            float expected = initial_data[i];
            CHECK(final_result[i] == Approx(expected).epsilon(0.001f));
        }
    }
}

TEST_CASE("Metal Remote Kernel Call", "[metal][remote_kernel_call]") {
    Resource metal_resource(ResourceType::METAL, 0);
    
    SECTION("Remote kernel call with data modification") {
        const size_t count = 10;
        std::vector<int> data(count, 1);
        
        auto increment_kernel = [](size_t i, int* ptr) {
            ptr[i]++;
        };
        
        auto result = remote_kernel_call(metal_resource, count, increment_kernel, data);
        result.wait();
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(data[i] == 2);
        }
    }
    
    SECTION("Kernel call with host-side pointer") {
        const size_t count = 5;
        std::vector<int> data(count, 10);
        
        auto host_ptr_kernel = [](size_t i, int* vec) {
            vec[i] *= 2;
        };
        
        remote_kernel_call(metal_resource, count, host_ptr_kernel, data.data()).wait();
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(data[i] == 20);
        }
    }
    
    SECTION("Kernel with multiple arguments") {
        const size_t count = 10;
        std::vector<int> data(count, 0);
        
        auto multi_arg_kernel = [](size_t i, int* ptr, int val1, int val2) {
            ptr[i] = val1 + val2;
        };
        
        remote_kernel_call(metal_resource, count, multi_arg_kernel, data, 5, 10).wait();
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(data[i] == 15);
        }
    }
    
    SECTION("Fill operation via kernel") {
        const size_t count = 100;
        std::vector<int> data(count, 0);
        const int fill_value = 7;
        
        auto fill_kernel = [](size_t i, int* ptr, int value) {
            ptr[i] = value;
        };
        
        remote_kernel_call(metal_resource, count, fill_kernel, data, fill_value).wait();
        
        for (const auto& val : data) {
            CHECK(val == fill_value);
        }
    }
    
    SECTION("Complex kernel with multiple buffers") {
        const size_t count = 10;
        std::vector<float> input(count, 2.0f);
        std::vector<float> output(count, 0.0f);
        
        auto complex_kernel = [](size_t i, const float* in, float* out, float factor) {
            out[i] = in[i] * factor;
        };
        
        remote_kernel_call(metal_resource, count, complex_kernel, input, output, 3.0f).wait();
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(output[i] == Approx(input[i] * 3.0f));
        }
    }
}

#else

TEST_CASE("Metal Support Not Enabled", "[metal]") {
    // Always pass when Metal is not available
    CHECK(true);
}

#endif // USE_METAL 