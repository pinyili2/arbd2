#include "../catch_boiler.h"
#include "Backend/Buffer.h"
#include "Backend/Events.h"
#include "Backend/Resource.h"
#include "Backend/Kernels.h"
#include <vector>
#include <numeric>
#include <string>
#include <random>
#include <cmath>
#include <chrono>
using namespace ARBD;

#ifdef USE_SYCL

#include "Backend/SYCL/SYCLManager.h"

// ============================================================================
// Test Fixture for SYCL Kernel Tests
// ============================================================================
struct SYCLKernelTestFixture {
    Resource sycl_resource;
    bool sycl_available = false;

    SYCLKernelTestFixture() {
        try {
            SYCL::SYCLManager::init();
            SYCL::SYCLManager::load_info();
            
            if (SYCL::SYCLManager::all_devices().empty()) {
                WARN("No SYCL devices found. Skipping SYCL kernel tests.");
                return;
            }
            
            sycl_resource = Resource(ResourceType::SYCL, 0);
            SYCL::SYCLManager::use(0);
            sycl_available = true;

        } catch (const std::exception& e) {
            FAIL("Failed to initialize SYCLManager for kernel tests: " << e.what());
        }
    }

    ~SYCLKernelTestFixture() {
        try {
            if (sycl_available) {
                SYCL::SYCLManager::finalize();
            }
        } catch (const std::exception& e) {
            std::cerr << "Error during SYCLManager finalization: " << e.what() << std::endl;
        }
    }
};

// ============================================================================
// Test Kernel Functions
// ============================================================================

/**
 * @brief Test kernel 1: Vector addition with scaling using generic dispatch with config
 * Computes: c[i] = scale * (a[i] + b[i])
 */
template<typename T>
Event test_vector_add_scale_kernel(const DeviceBuffer<T>& a,
                                 const DeviceBuffer<T>& b,
                                 DeviceBuffer<T>& c,
                                 T scale,
                                 const Resource& resource,
                                 const KernelConfig& config = {}) {
    size_t n = a.size();
    if (b.size() != n || c.size() != n) {
        throw std::runtime_error("Buffer size mismatch in vector add scale");
    }
    
    EventList deps;
    const T* a_ptr = a.get_read_access(deps);
    const T* b_ptr = b.get_read_access(deps);
    T* c_ptr = c.get_write_access(deps);
    
    // Define the kernel functor that matches launch_kernel_impl signature
    auto kernel_func = [scale](size_t i, const T* a_data, const T* b_data, T* c_data) {
        c_data[i] = scale * (a_data[i] + b_data[i]);
    };
    return launch_kernel(resource, n, kernel_func, config, a_ptr, b_ptr, c_ptr);
}

/**
 * @brief Test kernel 2: Matrix transpose using generic dispatch with config
 * Transposes a matrix stored in row-major order
 */
template<typename T>
Event test_matrix_transpose_kernel(const DeviceBuffer<T>& input,
                                 DeviceBuffer<T>& output,
                                 size_t rows,
                                 size_t cols,
                                 const Resource& resource,
                                 const KernelConfig& config = {}) {
    size_t total_elements = rows * cols;
    if (input.size() != total_elements || output.size() != total_elements) {
        throw std::runtime_error("Buffer size mismatch in matrix transpose");
    }
    
    EventList deps;
    const T* input_ptr = input.get_read_access(deps);
    T* output_ptr = output.get_write_access(deps);
    
    // Define the kernel functor that matches launch_kernel_impl signature
    auto kernel_func = [rows, cols](size_t idx, const T* input_data, T* output_data) {
        // Convert linear index to row, col
        size_t row = idx / cols;
        size_t col = idx % cols;
        
        if (row < rows && col < cols) {
            // input[row][col] -> output[col][row]
            size_t input_idx = row * cols + col;
            size_t output_idx = col * rows + row;
            output_data[output_idx] = input_data[input_idx];
        }
    };
    
    // Use launch_kernel_impl with configuration
    return launch_kernel(resource, total_elements, kernel_func, config, "", input_ptr, output_ptr);
}

// ============================================================================
// Test Cases
// ============================================================================

TEST_CASE_METHOD(SYCLKernelTestFixture, "SYCL Vector Add Scale Kernel", "[sycl][kernels]") {
    if (!sycl_available) {
        SKIP("SYCL not available");
    }

    SECTION("Float vector addition with default config") {
        const size_t n = 1000;
        const float scale = 2.5f;
        
        // Create host data
        std::vector<float> a_host(n), b_host(n), c_expected(n);
        std::iota(a_host.begin(), a_host.end(), 1.0f);  // a = [1, 2, 3, ...]
        std::iota(b_host.begin(), b_host.end(), 0.5f);  // b = [0.5, 1.5, 2.5, ...]
        
        // Calculate expected result
        for (size_t i = 0; i < n; ++i) {
            c_expected[i] = scale * (a_host[i] + b_host[i]);
        }
        
        // Create device buffers
        DeviceBuffer<float> a_buf(n, sycl_resource);
        DeviceBuffer<float> b_buf(n, sycl_resource);
        DeviceBuffer<float> c_buf(n, sycl_resource);
        
        // Copy data to device
        a_buf.copy_from_host(a_host.data(), n);
        b_buf.copy_from_host(b_host.data(), n);
        
        // Launch kernel with default config
        KernelConfig config;  // Default configuration
        auto event = test_vector_add_scale_kernel(a_buf, b_buf, c_buf, scale, sycl_resource, config);
        event.wait();
        
        // Copy result back to host
        std::vector<float> c_result(n);
        c_buf.copy_to_host(c_result.data(), n);
        
        // Verify results
        for (size_t i = 0; i < n; ++i) {
            REQUIRE(std::abs(c_result[i] - c_expected[i]) < 1e-6f);
        }
    }

    SECTION("Double precision with custom block size") {
        const size_t n = 500;
        const double scale = 3.14159;
        
        // Generate random data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-100.0, 100.0);
        
        std::vector<double> a_host(n), b_host(n), c_expected(n);
        for (size_t i = 0; i < n; ++i) {
            a_host[i] = dist(gen);
            b_host[i] = dist(gen);
            c_expected[i] = scale * (a_host[i] + b_host[i]);
        }
        
        // Create device buffers
        DeviceBuffer<double> a_buf(n, sycl_resource);
        DeviceBuffer<double> b_buf(n, sycl_resource);
        DeviceBuffer<double> c_buf(n, sycl_resource);
        
        // Copy data to device
        a_buf.copy_from_host(a_host.data(), n);
        b_buf.copy_from_host(b_host.data(), n);
        
        // Launch kernel with custom block size
        KernelConfig config;
        config.block_size = 128;  // Custom block size
        config.async = false;     // Synchronous execution
        
        auto event = test_vector_add_scale_kernel(a_buf, b_buf, c_buf, scale, sycl_resource, config);
        event.wait();
        
        // Copy result back to host
        std::vector<double> c_result(n);
        c_buf.copy_to_host(c_result.data(), n);
        
        // Verify results
        for (size_t i = 0; i < n; ++i) {
            REQUIRE(std::abs(c_result[i] - c_expected[i]) < 1e-12);
        }
    }

    SECTION("Async execution with large block size") {
        const size_t n = 2048;
        const float scale = 1.5f;
        
        std::vector<float> a_host(n), b_host(n), c_expected(n);
        std::iota(a_host.begin(), a_host.end(), 0.0f);
        std::iota(b_host.begin(), b_host.end(), 1.0f);
        
        for (size_t i = 0; i < n; ++i) {
            c_expected[i] = scale * (a_host[i] + b_host[i]);
        }
        
        DeviceBuffer<float> a_buf(n, sycl_resource);
        DeviceBuffer<float> b_buf(n, sycl_resource);
        DeviceBuffer<float> c_buf(n, sycl_resource);
        
        a_buf.copy_from_host(a_host.data(), n);
        b_buf.copy_from_host(b_host.data(), n);
        
        // Launch kernel with async config
        KernelConfig config;
        config.block_size = 512;  // Large block size
        config.async = true;      // Async execution
        config.shared_memory = 0; // No shared memory needed
        
        auto event = test_vector_add_scale_kernel(a_buf, b_buf, c_buf, scale, sycl_resource, config);
        // Test that we can check completion
        bool completed_immediately = event.is_complete();
        event.wait();  // Ensure completion before checking results
        
        std::vector<float> c_result(n);
        c_buf.copy_to_host(c_result.data(), n);
        
        for (size_t i = 0; i < n; ++i) {
            REQUIRE(std::abs(c_result[i] - c_expected[i]) < 1e-6f);
        }
    }
}

TEST_CASE_METHOD(SYCLKernelTestFixture, "SYCL Matrix Transpose Kernel", "[sycl][kernels]") {
    if (!sycl_available) {
        SKIP("SYCL not available");
    }

    SECTION("Square matrix transpose with default config") {
        const size_t rows = 16;
        const size_t cols = 16;
        const size_t total = rows * cols;
        
        // Create sequential input data
        std::vector<float> input_host(total);
        std::iota(input_host.begin(), input_host.end(), 0.0f);
        
        // Create expected transposed result
        std::vector<float> expected(total);
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                expected[c * rows + r] = input_host[r * cols + c];
            }
        }
        
        // Create device buffers
        DeviceBuffer<float> input_buf(total);
        DeviceBuffer<float> output_buf(total);
        
        // Copy data to device
        input_buf.copy_from_host(input_host.data(), total);
        
        // Launch kernel with default config
        KernelConfig config;  // Default configuration
        auto event = test_matrix_transpose_kernel(input_buf, output_buf, rows, cols, sycl_resource, config);
        event.wait();
        
        // Copy result back to host
        std::vector<float> result(total);
        output_buf.copy_to_host(result.data(), total);
        
        // Verify results
        for (size_t i = 0; i < total; ++i) {
            REQUIRE(result[i] == expected[i]);
        }
    }

    SECTION("Rectangular matrix transpose with optimized config") {
        const size_t rows = 20;
        const size_t cols = 30;
        const size_t total = rows * cols;
        
        // Create random input data
        std::random_device rd;
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<double> dist(-50.0, 50.0);
        
        std::vector<double> input_host(total);
        for (size_t i = 0; i < total; ++i) {
            input_host[i] = dist(gen);
        }
        
        // Create expected transposed result
        std::vector<double> expected(total);
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                expected[c * rows + r] = input_host[r * cols + c];
            }
        }
        
        // Create device buffers
        DeviceBuffer<double> input_buf(total);
        DeviceBuffer<double> output_buf(total);
        
        // Copy data to device
        input_buf.copy_from_host(input_host.data(), total);
        
        // Launch kernel with optimized config
        KernelConfig config;
        config.block_size = 64;   // Smaller block size for better cache usage
        config.async = true;      // Async execution
        config.grid_size = 0;     // Auto-calculate grid size
        
        auto event = test_matrix_transpose_kernel(input_buf, output_buf, rows, cols, sycl_resource, config);
        event.wait();
        
        // Copy result back to host
        std::vector<double> result(total);
        output_buf.copy_to_host(result.data(), total);
        
        // Verify results
        for (size_t i = 0; i < total; ++i) {
            REQUIRE(std::abs(result[i] - expected[i]) < 1e-12);
        }
    }

    SECTION("Large matrix with performance-oriented config") {
        const size_t rows = 32;
        const size_t cols = 64;
        const size_t total = rows * cols;
        
        std::vector<int> input_host(total);
        std::iota(input_host.begin(), input_host.end(), 1);
        
        std::vector<int> expected(total);
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                expected[c * rows + r] = input_host[r * cols + c];
            }
        }
        
        DeviceBuffer<int> input_buf(total);
        DeviceBuffer<int> output_buf(total);
        
        input_buf.copy_from_host(input_host.data(), total);
        
        // Performance-oriented configuration
        KernelConfig config;
        config.block_size = 256;      // Large block size for high throughput
        config.async = true;          // Async for potential overlap
        config.shared_memory = 1024;  // Request shared memory (if supported)
        
        auto event = test_matrix_transpose_kernel(input_buf, output_buf, rows, cols, sycl_resource, config);
        event.wait();
        
        std::vector<int> result(total);
        output_buf.copy_to_host(result.data(), total);
        
        for (size_t i = 0; i < total; ++i) {
            REQUIRE(result[i] == expected[i]);
        }
    }

    SECTION("Single element matrix with minimal config") {
        const size_t rows = 1;
        const size_t cols = 1;
        const size_t total = 1;
        
        std::vector<int> input_host = {42};
        std::vector<int> expected = {42};
        
        DeviceBuffer<int> input_buf(total);
        DeviceBuffer<int> output_buf(total);
        
        input_buf.copy_from_host(input_host.data(), total);
        
        // Minimal configuration for single element
        KernelConfig config;
        config.block_size = 1;    // Minimal block size
        config.async = false;     // Synchronous for simple case
        
        auto event = test_matrix_transpose_kernel(input_buf, output_buf, rows, cols, sycl_resource, config);
        event.wait();
        
        std::vector<int> result(total);
        output_buf.copy_to_host(result.data(), total);
        
        REQUIRE(result[0] == expected[0]);
    }
}

TEST_CASE_METHOD(SYCLKernelTestFixture, "SYCL Kernel Configuration and Chaining", "[sycl][kernels][config]") {
    if (!sycl_available) {
        SKIP("SYCL not available");
    }

    SECTION("Kernel chaining with different configurations") {
        const size_t n = 1000;
        const float scale1 = 2.0f;
        const float scale2 = 0.5f;
        
        // Create host data
        std::vector<float> a_host(n), b_host(n);
        std::vector<float> expected_intermediate(n), expected_final(n);
        
        std::iota(a_host.begin(), a_host.end(), 1.0f);
        std::iota(b_host.begin(), b_host.end(), 0.5f);
        
        // Calculate expected results for chained operations
        for (size_t i = 0; i < n; ++i) {
            expected_intermediate[i] = scale1 * (a_host[i] + b_host[i]);  // First kernel
            expected_final[i] = scale2 * (expected_intermediate[i] + a_host[i]);  // Second kernel
        }
        
        // Create device buffers
        DeviceBuffer<float> a_buf(n);
        DeviceBuffer<float> b_buf(n);
        DeviceBuffer<float> intermediate_buf(n);
        DeviceBuffer<float> final_buf(n);
        
        // Copy initial data to device
        a_buf.copy_from_host(a_host.data(), n);
        b_buf.copy_from_host(b_host.data(), n);
        
        // Use KernelChain for sequential execution with automatic dependency management
        KernelChain chain(sycl_resource);
        
        // First kernel: intermediate = scale1 * (a + b)
        chain.then(n, [scale1](size_t i, float* a, float* b, float* intermediate) {
            intermediate[i] = scale1 * (a[i] + b[i]);
        }, "", a_buf, b_buf, intermediate_buf);
        
        // Second kernel: final = scale2 * (intermediate + a)
        chain.then(n, [scale2](size_t i, float* intermediate, float* a, float* final) {
            final[i] = scale2 * (intermediate[i] + a[i]);
        }, "", intermediate_buf, a_buf, final_buf);
        
        // Wait for all kernels to complete
        chain.wait();
        
        // Verify intermediate results
        std::vector<float> intermediate_result(n);
        intermediate_buf.copy_to_host(intermediate_result.data(), n);
        
        for (size_t i = 0; i < n; ++i) {
            REQUIRE(std::abs(intermediate_result[i] - expected_intermediate[i]) < 1e-6f);
        }
        
        // Verify final results
        std::vector<float> final_result(n);
        final_buf.copy_to_host(final_result.data(), n);
        
        for (size_t i = 0; i < n; ++i) {
            REQUIRE(std::abs(final_result[i] - expected_final[i]) < 1e-6f);
        }
    }

    SECTION("Configuration validation and edge cases") {
        const size_t n = 64;
        std::vector<float> data_host(n, 1.0f);
        
        DeviceBuffer<float> data_buf(n);
        DeviceBuffer<float> result_buf(n);
        
        data_buf.copy_from_host(data_host.data(), n);
        
        // Test with minimal configuration
        KernelConfig minimal_config;
        minimal_config.block_size = 1;
        minimal_config.grid_size = n;  // Explicitly set grid size
        minimal_config.async = false;
        minimal_config.shared_memory = 0;
        
        EventList deps1;
        const float* input_ptr = data_buf.get_read_access(deps1);
        float* output_ptr = result_buf.get_write_access(deps1);
        
        auto event = launch_kernel(sycl_resource, n, minimal_config, "", input_ptr, output_ptr, [](size_t i, const float* input_data, float* output_data) {
            output_data[i] = input_data[i] * 2.0f;
        });
        
        event.wait();
        
        // Verify results
        std::vector<float> result(n);
        result_buf.copy_to_host(result.data(), n);
        
        for (size_t i = 0; i < n; ++i) {
            REQUIRE(std::abs(result[i] - 2.0f) < 1e-6f);
        }
        
        // Test with maximum reasonable configuration
        KernelConfig max_config;
        max_config.block_size = 512;
        max_config.grid_size = 0;     // Auto-calculate
        max_config.async = true;
        max_config.shared_memory = 2048;
        
        // Reset result buffer
        std::fill(result.begin(), result.end(), 0.0f);
        result_buf.copy_from_host(result.data(), n);
        
        EventList deps2;
        const float* input_ptr2 = data_buf.get_read_access(deps2);
        float* output_ptr2 = result_buf.get_write_access(deps2);
        
        auto event2 = launch_kernel_impl(sycl_resource, n, 
            [](size_t i, const float* input_data, float* output_data) {
                output_data[i] = input_data[i] * 3.0f;  // Triple the values
            }, max_config, "", input_ptr2, output_ptr2);
        
        event2.wait();
        
        // Verify results
        result_buf.copy_to_host(result.data(), n);
        for (size_t i = 0; i < n; ++i) {
            REQUIRE(std::abs(result[i] - 3.0f) < 1e-6f);
        }
    }
}

TEST_CASE_METHOD(SYCLKernelTestFixture, "SYCL Utility Functions from Kernels.h", "[sycl][kernels][utils]") {
    if (!sycl_available) {
        SKIP("SYCL not available");
    }

    SECTION("copy_async utility function") {
        const size_t n = 1000;
        std::vector<float> data_host(n);
        std::iota(data_host.begin(), data_host.end(), 1.0f);
        
        DeviceBuffer<float> source_buf(n, sycl_resource);
        DeviceBuffer<float> dest_buf(n, sycl_resource);
        
        // Copy data to source buffer
        source_buf.copy_from_host(data_host.data(), n);
        
        // Test copy_async function
        auto event = copy_async(source_buf, dest_buf, sycl_resource);
        event.wait();
        
        // Verify copy
        std::vector<float> result(n);
        dest_buf.copy_to_host(result.data(), n);
        
        for (size_t i = 0; i < n; ++i) {
            REQUIRE(result[i] == data_host[i]);
        }
    }

    SECTION("fill_async utility function") {
        const size_t n = 500;
        const double fill_value = 42.5;
        
        DeviceBuffer<double> buffer(n, sycl_resource);
        
        // Test fill_async function
        auto event = fill_async(buffer, fill_value, sycl_resource);
        event.wait();
        
        // Verify fill
        std::vector<double> result(n);
        buffer.copy_to_host(result.data(), n);
        
        for (size_t i = 0; i < n; ++i) {
            REQUIRE(std::abs(result[i] - fill_value) < 1e-12);
        }
    }

    SECTION("get_grid_size utility function") {
        // Test grid size calculation
        REQUIRE(get_grid_size(100, 32) == 4);    // 100/32 = 3.125 -> 4
        REQUIRE(get_grid_size(256, 256) == 1);   // 256/256 = 1
        REQUIRE(get_grid_size(1000, 128) == 8);  // 1000/128 = 7.8125 -> 8
        REQUIRE(get_grid_size(1, 256) == 1);     // 1/256 = 0.004 -> 1
        REQUIRE(get_grid_size(0, 256) == 0);     // 0/256 = 0
    }
}

TEST_CASE_METHOD(SYCLKernelTestFixture, "SYCL KernelResult Wrapper", "[sycl][kernels][result]") {
    if (!sycl_available) {
        SKIP("SYCL not available");
    }

    SECTION("KernelResult with async computation") {
        const size_t n = 1000;
        const float scale = 3.0f;
        
        std::vector<float> data_host(n);
        std::iota(data_host.begin(), data_host.end(), 1.0f);
        
        DeviceBuffer<float> input_buf(n, sycl_resource);
        DeviceBuffer<float> output_buf(n, sycl_resource);
        
        input_buf.copy_from_host(data_host.data(), n);
        
        // Create a KernelResult for async operation
        KernelConfig config;
        config.async = true;
        
        auto kernel_event = launch_kernel_impl(sycl_resource, n, 
            [scale](size_t i, const float* input, float* output) {
                output[i] = input[i] * scale;
            }, config, "", input_buf.data(), output_buf.data());
        
        // Wrap in KernelResult
        std::vector<float> result_data(n);
        KernelResult<std::vector<float>> result(std::move(result_data), std::move(kernel_event));
        
        // Test that we can check readiness
        bool ready_immediately = result.is_ready();
        
        // Wait for completion and get result
        result.wait();
        
        // Verify the kernel completed
        output_buf.copy_to_host(result.result.data(), n);
        for (size_t i = 0; i < n; ++i) {
            REQUIRE(std::abs(result.result[i] - data_host[i] * scale) < 1e-6f);
        }
    }
}

TEST_CASE_METHOD(SYCLKernelTestFixture, "SYCL ExecutionPolicy Configuration", "[sycl][kernels][policy]") {
    if (!sycl_available) {
        SKIP("SYCL not available");
    }

    SECTION("ExecutionPolicy struct validation") {
        ExecutionPolicy default_policy;
        REQUIRE(default_policy.type == ExecutionPolicy::Type::PARALLEL);
        REQUIRE(default_policy.preferred_block_size == 256);
        REQUIRE(default_policy.use_shared_memory == false);
        REQUIRE(default_policy.shared_memory_size == 0);
        
        // Test custom policy
        ExecutionPolicy custom_policy;
        custom_policy.type = ExecutionPolicy::Type::ASYNC;
        custom_policy.preferred_block_size = 128;
        custom_policy.use_shared_memory = true;
        custom_policy.shared_memory_size = 1024;
        
        REQUIRE(custom_policy.type == ExecutionPolicy::Type::ASYNC);
        REQUIRE(custom_policy.preferred_block_size == 128);
        REQUIRE(custom_policy.use_shared_memory == true);
        REQUIRE(custom_policy.shared_memory_size == 1024);
        
        // Test sequential policy
        ExecutionPolicy seq_policy;
        seq_policy.type = ExecutionPolicy::Type::SEQUENTIAL;
        REQUIRE(seq_policy.type == ExecutionPolicy::Type::SEQUENTIAL);
    }
}

TEST_CASE_METHOD(SYCLKernelTestFixture, "SYCL Error Handling and Edge Cases", "[sycl][kernels][errors]") {
    if (!sycl_available) {
        SKIP("SYCL not available");
    }

    SECTION("Buffer size mismatch errors") {
        DeviceBuffer<float> small_buf(100, sycl_resource);
        DeviceBuffer<float> large_buf(200, sycl_resource);
        
        // Test that copy_async detects size mismatch
        REQUIRE_THROWS_AS(
            copy_async(small_buf, large_buf, sycl_resource),
            std::runtime_error
        );
    }

    SECTION("Zero-sized buffer handling") {
        DeviceBuffer<int> empty_buf(0, sycl_resource);
        DeviceBuffer<int> normal_buf(10, sycl_resource);
        
        // Zero-sized operations should handle gracefully
        REQUIRE_NOTHROW(fill_async(empty_buf, 42, sycl_resource));
        
        // But mismatched sizes should still throw
        REQUIRE_THROWS_AS(
            copy_async(empty_buf, normal_buf, sycl_resource),
            std::runtime_error
        );
    }

    SECTION("Invalid configuration handling") {
        const size_t n = 100;
        DeviceBuffer<float> buffer(n, sycl_resource);
        
        // Test with very large block size
        KernelConfig large_config;
        large_config.block_size = 65536;  // Very large block size
        large_config.shared_memory = 1024 * 1024;  // Very large shared memory
        
        EventList deps;
        const float* input_ptr = buffer.get_read_access(deps);
        float* output_ptr = buffer.get_write_access(deps);
        
        // This should not crash, but may adjust configuration internally
        REQUIRE_NOTHROW(
            launch_kernel_impl(sycl_resource, n, 
                [](size_t i, const float* input, float* output) {
                    output[i] = input[i] + 1.0f;
                }, large_config, "", input_ptr, output_ptr)
        );
    }
}

TEST_CASE_METHOD(SYCLKernelTestFixture, "SYCL Resource Type Testing", "[sycl][kernels][resources]") {
    if (!sycl_available) {
        SKIP("SYCL not available");
    }

    SECTION("Device vs Host resource behavior") {
        const size_t n = 100;
        
        // Test with device resource
        Resource device_resource = sycl_resource;
        REQUIRE(device_resource.is_device());
        REQUIRE(!device_resource.is_host());
        
        DeviceBuffer<float> device_buf(n, device_resource);
        
        // Fill buffer using device resource
        auto device_event = fill_async(device_buf, 1.5f, device_resource);
        device_event.wait();
        
        // Verify result
        std::vector<float> result(n);
        device_buf.copy_to_host(result.data(), n);
        
        for (size_t i = 0; i < n; ++i) {
            REQUIRE(std::abs(result[i] - 1.5f) < 1e-6f);
        }
        
        // Test host resource (if supported)
        Resource host_resource(ResourceType::CPU, 0);
        REQUIRE(host_resource.is_host());
        REQUIRE(!host_resource.is_device());
    }
}

TEST_CASE_METHOD(SYCLKernelTestFixture, "SYCL Performance and Timing", "[sycl][kernels][performance]") {
    if (!sycl_available) {
        SKIP("SYCL not available");
    }

    SECTION("Async vs Sync performance comparison") {
        const size_t n = 10000;
        const float scale = 2.0f;
        
        std::vector<float> data_host(n);
        std::iota(data_host.begin(), data_host.end(), 1.0f);
        
        DeviceBuffer<float> input_buf(n, sycl_resource);
        DeviceBuffer<float> sync_result(n, sycl_resource);
        DeviceBuffer<float> async_result(n, sycl_resource);
        
        input_buf.copy_from_host(data_host.data(), n);
        
        // Synchronous execution
        KernelConfig sync_config;
        sync_config.async = false;
        sync_config.block_size = 256;
        
        auto sync_start = std::chrono::high_resolution_clock::now();
        auto sync_event = test_vector_add_scale_kernel(input_buf, input_buf, sync_result, scale, sycl_resource, sync_config);
        sync_event.wait();
        auto sync_end = std::chrono::high_resolution_clock::now();
        
        // Asynchronous execution
        KernelConfig async_config;
        async_config.async = true;
        async_config.block_size = 256;
        
        auto async_start = std::chrono::high_resolution_clock::now();
        auto async_event = test_vector_add_scale_kernel(input_buf, input_buf, async_result, scale, sycl_resource, async_config);
        async_event.wait();
        auto async_end = std::chrono::high_resolution_clock::now();
        
        // Verify both produce correct results
        std::vector<float> sync_data(n), async_data(n);
        sync_result.copy_to_host(sync_data.data(), n);
        async_result.copy_to_host(async_data.data(), n);
        
        for (size_t i = 0; i < n; ++i) {
            float expected = scale * (data_host[i] + data_host[i]);
            REQUIRE(std::abs(sync_data[i] - expected) < 1e-6f);
            REQUIRE(std::abs(async_data[i] - expected) < 1e-6f);
        }
        
        // Both should complete successfully (timing comparison is informational)
        auto sync_duration = std::chrono::duration_cast<std::chrono::microseconds>(sync_end - sync_start);
        auto async_duration = std::chrono::duration_cast<std::chrono::microseconds>(async_end - async_start);
        
        LOGINFO("Sync execution: {} μs, Async execution: {} μs", sync_duration.count(), async_duration.count());
    }

    SECTION("Block size impact on performance") {
        const size_t side = 64;  // Use 64x64 matrix 
        const size_t n = side * side;  // 4096 elements - perfect square
        
        std::vector<int> data_host(n);
        std::iota(data_host.begin(), data_host.end(), 1);
        
        DeviceBuffer<int> input_buf(n, sycl_resource);
        input_buf.copy_from_host(data_host.data(), n);
        
        std::vector<size_t> block_sizes = {32, 64, 128, 256, 512};
        
        for (size_t block_size : block_sizes) {
            DeviceBuffer<int> output_buf(n, sycl_resource);
            
            KernelConfig config;
            config.block_size = block_size;
            config.async = false;
            
            auto start = std::chrono::high_resolution_clock::now();
            auto event = test_matrix_transpose_kernel(input_buf, output_buf, side, side, sycl_resource, config);
            event.wait();
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            LOGINFO("Block size {}: {} μs", block_size, duration.count());
            
            // Verify correctness regardless of block size
            std::vector<int> result(n);
            output_buf.copy_to_host(result.data(), n);
            
            // Check a few elements to ensure transpose worked
            REQUIRE(result[0] == data_host[0]);           // Corner elements
            REQUIRE(result[side] == data_host[1]);        // First row -> first column
            REQUIRE(result[1] == data_host[side]);        // First column -> first row
        }
    }
}

#endif // USE_SYCL