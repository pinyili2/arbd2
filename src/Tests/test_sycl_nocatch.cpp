#include "Backend/SYCL/SYCLManager.h"
#include <vector>
#include <numeric>
#include <iostream>
#include <limits>
#include <cassert>
#include <cmath>
#include <stdexcept>

using namespace ARBD;

namespace {

bool within_abs(float a, float b, float tol) {
    return std::fabs(a - b) <= tol;
}

void test_sycl_manager_initialization() {
    std::cout << "[TEST] SYCL Manager Initialization... ";
    try {
        SYCLManager::init();
        assert(SYCLManager::all_device_size() > 0);
        const auto& devices = SYCLManager::all_devices();
        for (size_t i = 0; i < devices.size(); ++i) {
            const auto& device = devices[i];
            std::cout << "\nDevice " << device.id() << ": " << device.name()
                      << " (" << device.vendor() << ")\n";
            std::cout << "  - Compute units: " << device.max_compute_units() << "\n";
            std::cout << "  - Global memory: " << device.global_mem_size() / (1024*1024) << " MB\n";
            std::cout << "  - Type: " << (device.is_cpu() ? "CPU" :
                                         device.is_gpu() ? "GPU" :
                                         device.is_accelerator() ? "Accelerator" : "Unknown") << "\n";
        }
        SYCLManager::load_info();
        assert(SYCLManager::devices().size() > 0);
        assert(SYCLManager::current() == 0);
        std::cout << "PASS\n";
    } catch (const std::exception& e) {
        std::cout << "FAIL: " << e.what() << "\n";
        throw;
    }
}

void test_sycl_device_selection() {
    std::cout << "[TEST] SYCL Device Selection... ";
    try {
        SYCLManager::init();
        SYCLManager::load_info();
        SYCLManager::sync();
        // Current device access
        auto& device = SYCLManager::get_current_device();
        (void)device;
        auto& queue = SYCLManager::get_current_queue();
        (void)queue;
        assert(device.id() == static_cast<unsigned int>(SYCLManager::current()));
        SYCLManager::sync();
        // Device switching
        if (SYCLManager::devices().size() > 1) {
            SYCLManager::use(1);
            assert(SYCLManager::current() == 1);
            auto& device1 = SYCLManager::get_current_device();
            assert(device1.id() == 1);
            SYCLManager::use(0);
            assert(SYCLManager::current() == 0);
        }
        SYCLManager::sync();
        // Device synchronization
        SYCLManager::sync();
        SYCLManager::sync(0);
        auto& dev = SYCLManager::get_current_device();
        dev.synchronize_all_queues();
        std::cout << "PASS\n";
    } catch (const std::exception& e) {
        std::cout << "FAIL: " << e.what() << "\n";
        throw;
    }
}

void test_sycl_device_memory_operations() {
    std::cout << "[TEST] SYCL Device Memory Operations... ";
    try {
        SYCLManager::init();
        SYCLManager::load_info();
        SYCLManager::sync();
        auto& queue = SYCLManager::get_current_queue();
        // Basic memory allocation
        constexpr size_t SIZE = 1000;
        {
            DeviceMemory<float> device_mem(queue.get(), SIZE);
        }
        DeviceMemory<float> device_mem(queue.get(), SIZE);
        assert(device_mem.size() == SIZE);
        assert(device_mem.get() != nullptr);
        assert(device_mem.queue() == &queue.get());
        queue.synchronize();
        // Host to device copy
        {
            constexpr size_t SIZE2 = 100;
            std::vector<int> host_data(SIZE2);
            std::iota(host_data.begin(), host_data.end(), 1);
            DeviceMemory<int> device_mem2(queue.get(), SIZE2);
            device_mem2.copyFromHost(host_data);
            queue.synchronize();
        }
        // Device to host copy
        {
            constexpr size_t SIZE3 = 50;
            std::vector<float> host_data(SIZE3, 42.0f);
            std::vector<float> result_data(SIZE3, 0.0f);
            DeviceMemory<float> device_mem3(queue.get(), SIZE3);
            device_mem3.copyFromHost(host_data);
            device_mem3.copyToHost(result_data);
            queue.synchronize();
            for (size_t i = 0; i < SIZE3; ++i) {
                assert(result_data[i] == 42.0f);
            }
        }
        // Memory size validation
        {
            constexpr size_t SIZE4 = 100;
            DeviceMemory<float> device_mem4(queue.get(), SIZE4);
            std::vector<float> large_data(SIZE4 + 1, 1.0f);
            bool threw = false;
            try { device_mem4.copyFromHost(large_data); } catch (...) { threw = true; }
            assert(threw);
            std::vector<float> large_output(SIZE4 + 1);
            threw = false;
            try { device_mem4.copyToHost(large_output); } catch (...) { threw = true; }
            assert(threw);
        }
        std::cout << "PASS\n";
    } catch (const std::exception& e) {
        std::cout << "FAIL: " << e.what() << "\n";
        throw;
    }
}

void test_sycl_simple_kernel_execution() {
    std::cout << "[TEST] SYCL Simple Kernel Execution... ";
    try {
        SYCLManager::init();
        SYCLManager::load_info();
        SYCLManager::sync();
        auto& queue = SYCLManager::get_current_queue();
        // Vector addition kernel
        {
            constexpr size_t SIZE = 256;
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
            auto event = queue.submit([=](sycl::handler& h) {
                auto range = sycl::range<1>(SIZE);
                h.parallel_for(range, [=](sycl::id<1> idx) {
                    size_t i = idx[0];
                    ptr_c[i] = ptr_a[i] + ptr_b[i];
                });
            });
            event.wait();
            d_c.copyToHost(c);
            queue.synchronize();
            for (size_t i = 0; i < SIZE; ++i) {
                assert(within_abs(c[i], 3.0f, 1e-6f));
            }
        }
        // Parallel reduction
        {
            constexpr size_t SIZE = 1024;
            std::vector<int> data(SIZE);
            std::iota(data.begin(), data.end(), 1);
            DeviceMemory<int> d_data(queue.get(), SIZE);
            DeviceMemory<int> d_result(queue.get(), 1);
            d_data.copyFromHost(data);
            int* ptr_data = d_data.get();
            int* ptr_result = d_result.get();
            auto event = queue.submit([=](sycl::handler& h) {
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
            int expected = SIZE * (SIZE + 1) / 2;
            assert(result[0] == expected);
        }
        std::cout << "PASS\n";
    } catch (const std::exception& e) {
        std::cout << "FAIL: " << e.what() << "\n";
        throw;
    }
}

void test_sycl_queue_management() {
    std::cout << "[TEST] SYCL Queue Management... ";
    try {
        SYCLManager::init();
        SYCLManager::load_info();
        SYCLManager::sync();
        auto& device = SYCLManager::get_current_device();
        auto& queue0 = device.get_queue(0);
        queue0.synchronize();
        auto& queue1 = device.get_queue(1);
        queue1.synchronize();
        assert(&queue0.get() != &queue1.get());
        auto& next_queue1 = device.get_next_queue();
        next_queue1.synchronize();
        auto& next_queue2 = device.get_next_queue();
        next_queue2.synchronize();
        assert(&next_queue1.get() != &next_queue2.get());
        auto& queue = device.get_queue(0);
        queue.is_in_order();
        queue.synchronize();
        std::cout << "PASS\n";
    } catch (const std::exception& e) {
        std::cout << "FAIL: " << e.what() << "\n";
        throw;
    }
}

void test_sycl_device_filtering() {
    std::cout << "[TEST] SYCL Device Filtering... ";
    try {
        SYCLManager::init();
        auto cpu_ids = SYCLManager::get_cpu_device_ids();
        auto gpu_ids = SYCLManager::get_gpu_device_ids();
        auto accel_ids = SYCLManager::get_accelerator_device_ids();
        assert((cpu_ids.size() + gpu_ids.size() + accel_ids.size()) > 0);
        for (auto id : cpu_ids) {
            assert(SYCLManager::all_devices()[id].is_cpu());
        }
        for (auto id : gpu_ids) {
            assert(SYCLManager::all_devices()[id].is_gpu());
        }
        for (auto id : accel_ids) {
            assert(SYCLManager::all_devices()[id].is_accelerator());
        }
        std::cout << "PASS\n";
    } catch (const std::exception& e) {
        std::cout << "FAIL: " << e.what() << "\n";
        throw;
    }
}

void test_sycl_error_handling() {
    std::cout << "[TEST] SYCL Error Handling... ";
    try {
        SYCLManager::init();
        SYCLManager::load_info();
        // Device selection errors
        std::vector<unsigned int> invalid_ids = {999};
        bool threw = false;
        try { SYCLManager::select_devices(invalid_ids); } catch (...) { threw = true; }
        assert(threw);
        threw = false;
        try { SYCLManager::sync(999); } catch (...) { threw = true; }
        assert(threw);
        // Memory errors
        auto& queue = SYCLManager::get_current_queue();
        DeviceMemory<float> mem0(queue.get(), 0);
        DeviceMemory<float> device_mem(queue.get(), 10);
        std::vector<float> large_data(20, 1.0f);
        threw = false;
        try { device_mem.copyFromHost(large_data); } catch (...) { threw = true; }
        assert(threw);
        std::vector<float> large_output(20);
        threw = false;
        try { device_mem.copyToHost(large_output); } catch (...) { threw = true; }
        assert(threw);
        std::cout << "PASS\n";
    } catch (const std::exception& e) {
        std::cout << "FAIL: " << e.what() << "\n";
        throw;
    }
}

} // namespace

int main() {
    int failed = 0;
    try { test_sycl_manager_initialization(); } catch (...) { ++failed; }
    try { test_sycl_device_selection(); } catch (...) { ++failed; }
    try { test_sycl_device_memory_operations(); } catch (...) { ++failed; }
    try { test_sycl_simple_kernel_execution(); } catch (...) { ++failed; }
    try { test_sycl_queue_management(); } catch (...) { ++failed; }
    try { test_sycl_device_filtering(); } catch (...) { ++failed; }
    try { test_sycl_error_handling(); } catch (...) { ++failed; }
    // Explicitly cleanup SYCLManager before exit
    SYCLManager::finalize();
    if (failed) {
        std::cout << "\nSome tests FAILED (" << failed << ")\n";
        return 1;
    } else {
        std::cout << "\nAll tests PASSED\n";
        return 0;
    }
} 