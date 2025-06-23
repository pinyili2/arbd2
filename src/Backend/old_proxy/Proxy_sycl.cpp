#include <future>
#include <cstring>
#include "Proxy.h"

#ifdef USE_SYCL
#include "Backend/SYCL/SYCLManager.h"
#include <sycl/sycl.hpp>
#endif

namespace ARBD {
namespace ProxyImpl {

// =============================================================================
// SYCL Implementation Functions
// =============================================================================

#ifdef USE_SYCL
void* sycl_call_sync_impl(void* addr, void* func_ptr, void* args, size_t args_size, 
                         const Resource& location, size_t result_size) {
    if (location.is_local()) {
        // For SYCL, execute on device if addr is device memory
        auto &device = ARBD::SYCL::SYCLManager::get_current_device();
        auto &queue = device.get_next_queue();

        // Create host result storage
        void* result = new char[result_size];
        void* result_device = sycl::malloc_device<char>(result_size, queue.get());

        // Note: This is a simplified version. In practice, you'd need to handle
        // the member function call properly with SYCL kernels
        queue.get().submit([&](sycl::handler &h) {
            h.single_task([=]() {
                // This is conceptual - actual implementation would need
                // to handle member function calls properly
                std::memcpy(result_device, addr, result_size);
            });
        }).wait();

        // Copy result back to host
        queue.get().memcpy(result, result_device, result_size).wait();
        sycl::free(result_device, queue.get());

        return result;
    } else {
        ARBD::throw_not_implemented("Non-local SYCL calls require distributed computing support");
    }
}

std::future<void*> sycl_call_async_impl(void* addr, void* func_ptr, void* args, 
                                        size_t args_size, const Resource& location, 
                                        size_t result_size) {
    if (location.is_local()) {
        return std::async(std::launch::async, [=] {
            return sycl_call_sync_impl(addr, func_ptr, args, args_size, location, result_size);
        });
    } else {
        ARBD::throw_not_implemented("Proxy::callAsync() non-local SYCL calls not supported");
    }
}
#else
void* sycl_call_sync_impl(void* addr, void* func_ptr, void* args, size_t args_size, 
                         const Resource& location, size_t result_size) {
    ARBD::throw_not_implemented("SYCL support not enabled");
}

std::future<void*> sycl_call_async_impl(void* addr, void* func_ptr, void* args, 
                                        size_t args_size, const Resource& location, 
                                        size_t result_size) {
    ARBD::throw_not_implemented("SYCL support not enabled");
}
#endif

} // namespace ProxyImpl
} // namespace ARBD 