#include "catch_boiler.h"

#ifdef USE_SYCL

#include "Backend/Buffer.h"
#include "Backend/Kernels.h"
#include "Backend/Events.h"
#include "Backend/Resource.h"
#include "Backend/SYCL/SYCLManager.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

using namespace ARBD;
using namespace ARBD::Kernels;
using namespace Catch;
template<typename KernelFunc, typename... ScalarArgs>
Event simple_kernel_with_scalars(const Resource& resource,
                                 size_t num_elements,
                                 KernelFunc&& kernel,
                                 DeviceBuffer<double>& buffer,
                                 ScalarArgs&&... scalar_args) {
    
    EventList deps;
    auto buffer_ptr = buffer.get_write_access(deps);
    
    // Direct SYCL implementation (avoiding dispatch_kernel)
    #ifdef USE_SYCL
    auto& device = SYCL::SYCLManager::get_current_device();
    auto& queue = device.get_next_queue();
    
    auto event = queue.get().parallel_for(
        sycl::range<1>(num_elements),
        [=](sycl::id<1> idx) {
            kernel(idx[0], buffer_ptr, scalar_args...);
        }
    );
    
    return Event(event, resource);
    #else
    // CPU fallback
    for (size_t i = 0; i < num_elements; ++i) {
        kernel(i, buffer_ptr, scalar_args...);
    }
    return Event(nullptr, resource);
    #endif
}

TEST_CASE("SYCL Kernel Infrastructure Setup", "[sycl][kernels][setup]") {
    // Initialize SYCL backend
    REQUIRE_NOTHROW(SYCL::SYCLManager::init());
    REQUIRE_NOTHROW(SYCL::SYCLManager::load_info());
    
    if (SYCL::SYCLManager::devices().empty()) {
        SYCL::SYCLManager::finalize();
        SKIP("No SYCL devices available");
    }
    
    Resource sycl_resource(ResourceType::SYCL, 0);
    
    SECTION("Resource properties") {
        CHECK(sycl_resource.type == ResourceType::SYCL);
        CHECK(sycl_resource.is_device());
        CHECK(sycl_resource.supports_async());
    }
    
    SECTION("KernelConfig creation") {
        KernelConfig config;
        CHECK(config.grid_size == 0);  // Auto-calculate
        CHECK(config.block_size == 256);
        CHECK(config.shared_mem == 0);
        CHECK_FALSE(config.async);
        CHECK(config.dependencies.empty());
    }
    
    SECTION("LaunchConfig creation") {
        LaunchConfig config;
        CHECK(config.grid_size == 1);
        CHECK(config.block_size == 256);
        CHECK(config.shared_memory == 0);
        CHECK_FALSE(config.async);
    }
    
    SYCL::SYCLManager::finalize();
}

TEST_CASE("SYCL TypedAllocator Operations", "[sycl][kernels][allocator]") {
    SYCL::SYCLManager::init();
    SYCL::SYCLManager::load_info();
    
    if (SYCL::SYCLManager::devices().empty()) {
        SYCL::SYCLManager::finalize();
        SKIP("No SYCL devices available");
    }
    
    Resource sycl_resource(ResourceType::SYCL, 0);
    const size_t count = 1000;
    
    SECTION("Basic allocation") {
        auto buffer = GenericAllocator<float>::allocate(count, sycl_resource);
        
        CHECK(buffer.size() == count);
        CHECK_FALSE(buffer.empty());
        CHECK(buffer.data() != nullptr);
        CHECK(buffer.get_resource().type == ResourceType::SYCL);
    }
    
    SECTION("Zero-initialized allocation") {
        auto buffer = GenericAllocator<int>::allocate_zeroed(count, sycl_resource);
        
        CHECK(buffer.size() == count);
        
        // Verify zero initialization
        std::vector<int> host_data(count);
        buffer.copy_to_host(host_data.data(), count);
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(host_data[i] == 0);
        }
    }
    
    SECTION("Initialized allocation with lambda") {
        auto buffer = GenericAllocator<float>::allocate_initialized(count, sycl_resource,
            [](size_t i) { return static_cast<float>(i * 2.5f); });
        
        CHECK(buffer.size() == count);
        
        // Verify initialization
        std::vector<float> host_data(count);
        buffer.copy_to_host(host_data.data(), count);
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(host_data[i] == Approx(static_cast<float>(i * 2.5f)));
        }
    }
    
    SYCL::SYCLManager::finalize();
}

TEST_CASE("SYCL simple_kernel Function", "[sycl][kernels][simple_kernel]") {
    SYCL::SYCLManager::init();
    SYCL::SYCLManager::load_info();
    
    if (SYCL::SYCLManager::devices().empty()) {
        SYCL::SYCLManager::finalize();
        SKIP("No SYCL devices available");
    }
    
    Resource sycl_resource(ResourceType::SYCL, 0);
    const size_t count = 512;
    
    SECTION("Simple increment kernel") {
        auto buffer = GenericAllocator<int>::allocate_zeroed(count, sycl_resource);
        
        // Define a simple increment kernel
        auto increment_kernel = [](size_t i, int* data) {
            data[i] = static_cast<int>(i + 1);
        };
        
        Event event = simple_kernel(sycl_resource, count, increment_kernel, buffer);
        event.wait();
        
        // Verify results
        std::vector<int> host_data(count);
        buffer.copy_to_host(host_data.data(), count);
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(host_data[i] == static_cast<int>(i + 1));
        }
    }
    
    SECTION("Two-buffer operation kernel") {
        auto input = GenericAllocator<float>::allocate_initialized(count, sycl_resource,
            [](size_t i) { return static_cast<float>(i); });
        auto output = GenericAllocator<float>::allocate_zeroed(count, sycl_resource);
        
        // Define a computation kernel: output[i] = input[i] * 2.0 + 1.0
        auto compute_kernel = [](size_t i, const float* in, float* out) {
            out[i] = in[i] * 2.0f + 1.0f;
        };
        
        Event event = simple_kernel(sycl_resource, count, compute_kernel, input, output);
        event.wait();
        
        // Verify results
        std::vector<float> host_output(count);
        output.copy_to_host(host_output.data(), count);
        
        for (size_t i = 0; i < count; ++i) {
            float expected = static_cast<float>(i) * 2.0f + 1.0f;
            CHECK(host_output[i] == Approx(expected));
        }
    }

    SECTION("Multi-argument kernel execution") {
        const size_t count = 256;
        DeviceBuffer<double> buffer(count, sycl_resource);
        
        // Multi-argument kernel with scalar parameters
        auto multi_arg_kernel = [](size_t i, double* data, double f1, double f2) {
            data[i] = static_cast<double>(i) * f1 + f2;
        };
        
        // Use our custom kernel launcher
        Event event = simple_kernel_with_scalars(
            sycl_resource,
            count,
            multi_arg_kernel,
            buffer,
            2.5,  // f1 value  
            10.0  // f2 value
        );
        
        event.wait();
        CHECK(event.is_complete());
        
        // Verify results
        std::vector<double> host_result(count);
        buffer.copy_to_host(host_result.data(), count);
        
        for (size_t i = 0; i < count; ++i) {
            double expected = static_cast<double>(i) * 2.5 + 10.0;
            CHECK(host_result[i] == Catch::Approx(expected));
        }
    }
    
    
    SYCL::SYCLManager::finalize();
}

TEST_CASE("SYCL copy_async Function", "[sycl][kernels][copy_async]") {
    SYCL::SYCLManager::init();
    SYCL::SYCLManager::load_info();
    
    if (SYCL::SYCLManager::devices().empty()) {
        SYCL::SYCLManager::finalize();
        SKIP("No SYCL devices available");
    }
    
    Resource sycl_resource(ResourceType::SYCL, 0);
    const size_t count = 256;
    
    SECTION("Basic async copy") {
        auto source = GenericAllocator<int>::allocate_initialized(count, sycl_resource,
            [](size_t i) { return static_cast<int>(i * 3); });
        auto destination = GenericAllocator<int>::allocate_zeroed(count, sycl_resource);
        
        Event copy_event = copy_async(source, destination, sycl_resource);
        copy_event.wait();
        
        // Verify copy
        std::vector<int> host_source(count);
        std::vector<int> host_dest(count);
        
        source.copy_to_host(host_source.data(), count);
        destination.copy_to_host(host_dest.data(), count);
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(host_dest[i] == host_source[i]);
            CHECK(host_dest[i] == static_cast<int>(i * 3));
        }
    }
    
    SECTION("Copy with different data types") {
        auto float_source = GenericAllocator<float>::allocate_initialized(count, sycl_resource,
            [](size_t i) { return static_cast<float>(i) * 0.5f; });
        auto float_dest = GenericAllocator<float>::allocate_zeroed(count, sycl_resource);
        
        Event event = copy_async(float_source, float_dest, sycl_resource);
        event.wait();
        
        std::vector<float> host_data(count);
        float_dest.copy_to_host(host_data.data(), count);
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(host_data[i] == Approx(static_cast<float>(i) * 0.5f));
        }
    }
    
    SYCL::SYCLManager::finalize();
}

TEST_CASE("SYCL fill_async Function", "[sycl][kernels][fill_async]") {
    SYCL::SYCLManager::init();
    SYCL::SYCLManager::load_info();
    
    if (SYCL::SYCLManager::devices().empty()) {
        SYCL::SYCLManager::finalize();
        SKIP("No SYCL devices available");
    }
    
    Resource sycl_resource(ResourceType::SYCL, 0);
    const size_t count = 128;
    
    SECTION("Fill with integer value") {
        auto buffer = GenericAllocator<int>::allocate_zeroed(count, sycl_resource);
        const int fill_value = 42;
        
        Event event = fill_async(buffer, fill_value, sycl_resource);
        event.wait();
        
        std::vector<int> host_data(count);
        buffer.copy_to_host(host_data.data(), count);
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(host_data[i] == fill_value);
        }
    }
    
    SECTION("Fill with floating point value") {
        auto buffer = GenericAllocator<float>::allocate_zeroed(count, sycl_resource);
        const float fill_value = 3.14159f;
        
        Event event = fill_async(buffer, fill_value, sycl_resource);
        event.wait();
        
        std::vector<float> host_data(count);
        buffer.copy_to_host(host_data.data(), count);
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(host_data[i] == Approx(fill_value));
        }
    }
    
    SYCL::SYCLManager::finalize();
}

TEST_CASE("SYCL kernel_call with MultiRef", "[sycl][kernels][kernel_call]") {
    SYCL::SYCLManager::init();
    SYCL::SYCLManager::load_info();
    
    if (SYCL::SYCLManager::devices().empty()) {
        SYCL::SYCLManager::finalize();
        SKIP("No SYCL devices available");
    }
    
    Resource sycl_resource(ResourceType::SYCL, 0);
    const size_t count = 200;
    
    SECTION("MultiRef with read and write access") {
        auto input1 = GenericAllocator<float>::allocate_initialized(count, sycl_resource,
            [](size_t i) { return static_cast<float>(i); });
        auto input2 = GenericAllocator<float>::allocate_initialized(count, sycl_resource,
            [](size_t i) { return static_cast<float>(i * 2); });
        auto output = GenericAllocator<float>::allocate_zeroed(count, sycl_resource);
        
        // Create MultiRef objects
        auto inputs = make_multi_ref(input1, input2);
        auto outputs = make_multi_ref(output);
        
        // Define kernel: output[i] = input1[i] + input2[i]
        auto add_kernel = [](size_t i, const float* in1, const float* in2, float* out) {
            out[i] = in1[i] + in2[i];
        };
        
        Event event = kernel_call(sycl_resource, inputs, outputs, count, add_kernel);
        event.wait();
        
        // Verify results
        std::vector<float> host_output(count);
        output.copy_to_host(host_output.data(), count);
        
        for (size_t i = 0; i < count; ++i) {
            float expected = static_cast<float>(i) + static_cast<float>(i * 2);
            CHECK(host_output[i] == Approx(expected));
        }
    }
    
    SYCL::SYCLManager::finalize();
}

TEST_CASE("SYCL KernelChain Sequential Execution", "[sycl][kernels][chain]") {
    SYCL::SYCLManager::init();
    SYCL::SYCLManager::load_info();
    
    if (SYCL::SYCLManager::devices().empty()) {
        SYCL::SYCLManager::finalize();
        SKIP("No SYCL devices available");
    }
    
    Resource sycl_resource(ResourceType::SYCL, 0);
    const size_t count = 100;
    
    SECTION("Multi-step computation pipeline") {
        auto buffer1 = GenericAllocator<float>::allocate_initialized(count, sycl_resource,
            [](size_t i) { return static_cast<float>(i + 1); });
        auto buffer2 = GenericAllocator<float>::allocate_zeroed(count, sycl_resource);
        auto buffer3 = GenericAllocator<float>::allocate_zeroed(count, sycl_resource);
        
        KernelChain chain(sycl_resource);
        
        // Step 1: Square the values
        auto square_kernel = [](size_t i, const float* in, float* out) {
            out[i] = in[i] * in[i];
        };
        
        // Step 2: Take square root (should get back original values)
        auto sqrt_kernel = [](size_t i, const float* in, float* out) {
            out[i] = std::sqrt(in[i]);
        };
        
        chain.then(count, square_kernel, buffer1, buffer2)
             .then(count, sqrt_kernel, buffer2, buffer3)
             .wait();
        
        // Verify final results
        std::vector<float> original_data(count);
        std::vector<float> final_data(count);
        
        buffer1.copy_to_host(original_data.data(), count);
        buffer3.copy_to_host(final_data.data(), count);
        
        for (size_t i = 0; i < count; ++i) {
            CHECK(final_data[i] == Approx(original_data[i]).epsilon(0.001f));
        }
    }
    
    SYCL::SYCLManager::finalize();
}

TEST_CASE("SYCL Performance and Stress Tests", "[sycl][kernels][performance]") {
    SYCL::SYCLManager::init();
    SYCL::SYCLManager::load_info();
    
    if (SYCL::SYCLManager::devices().empty()) {
        SYCL::SYCLManager::finalize();
        SKIP("No SYCL devices available");
    }
    
    Resource sycl_resource(ResourceType::SYCL, 0);
    
    SECTION("Large buffer operations") {
        const size_t large_count = 1024 * 1024; // 1M elements
        
        auto large_buffer = GenericAllocator<float>::allocate_zeroed(large_count, sycl_resource);
        
        auto fill_kernel = [](size_t i, float* data) {
            data[i] = static_cast<float>(i % 1000) * 0.001f;
        };
        
        Event event = simple_kernel(sycl_resource, large_count, fill_kernel, large_buffer);
        event.wait();
        
        // Spot check some values
        std::vector<float> sample_data(1000);
        large_buffer.copy_to_host(sample_data.data(), 1000);
        
        for (size_t i = 0; i < 1000; ++i) {
            float expected = static_cast<float>(i % 1000) * 0.001f;
            CHECK(sample_data[i] == Approx(expected));
        }
        
        CHECK(large_buffer.size() == large_count);
    }
    
    SECTION("Multiple concurrent kernels") {
        const size_t count = 256;
        const size_t num_buffers = 8;
        
        std::vector<DeviceBuffer<int>> buffers;
        std::vector<Event> events;
        
        // Create multiple buffers and launch kernels
        for (size_t buf = 0; buf < num_buffers; ++buf) {
            buffers.push_back(GenericAllocator<int>::allocate_zeroed(count, sycl_resource));
            
            auto kernel = [buf](size_t i, int* data) {
                data[i] = static_cast<int>(i + buf * 1000);
            };
            
            events.push_back(simple_kernel(sycl_resource, count, kernel, buffers[buf]));
        }
        
        // Wait for all kernels to complete
        for (auto& event : events) {
            event.wait();
        }
        
        // Verify all results
        for (size_t buf = 0; buf < num_buffers; ++buf) {
            std::vector<int> host_data(count);
            buffers[buf].copy_to_host(host_data.data(), count);
            
            for (size_t i = 0; i < count; ++i) {
                int expected = static_cast<int>(i + buf * 1000);
                CHECK(host_data[i] == expected);
            }
        }
    }
    
    SYCL::SYCLManager::finalize();
}

#else

TEST_CASE("SYCL Kernels - SYCL Not Available", "[sycl][kernels]") {
    SKIP("SYCL support not compiled in");
}

#endif // USE_SYCL