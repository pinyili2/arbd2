#include "SYCLManager.h"
#include "ARBDLogger.h"
#ifdef PROJECT_USES_SYCL
#include <algorithm>

namespace ARBD {

// Static member initialization
std::vector<SYCLManager::Device> SYCLManager::all_devices_;
std::vector<SYCLManager::Device> SYCLManager::devices_;
int SYCLManager::current_device_{0};
sycl::info::device_type SYCLManager::preferred_type_{sycl::info::device_type::gpu};

// Device class implementation
SYCLManager::Device::Device(const sycl::device& dev, unsigned int id) 
    : id_(id), device_(dev) {
    
    query_device_properties();
    
    LOGINFO("Device {} initialized: {} ({})", id_, name_, vendor_);
    LOGINFO("  Compute units: {}, Global memory: {:.1f}GB, Max work group: {}", 
         max_compute_units_,
         static_cast<float>(global_mem_size_) / (1024.0f * 1024.0f * 1024.0f),
         max_work_group_size_);
    
    // Create queues
    try {
        sycl::property_list props{sycl::property::queue::enable_profiling{}};
        for (auto& queue : queues_) {
            queue = Queue(device_, props);
        }
    } catch (const sycl::exception& e) {
        check_sycl_error(e, __FILE__, __LINE__);
    }
}

void SYCLManager::Device::query_device_properties() {
    try {
        name_ = device_.get_info<sycl::info::device::name>();
        vendor_ = device_.get_info<sycl::info::device::vendor>();
        version_ = device_.get_info<sycl::info::device::version>();
        max_work_group_size_ = device_.get_info<sycl::info::device::max_work_group_size>();
        max_compute_units_ = device_.get_info<sycl::info::device::max_compute_units>();
        global_mem_size_ = device_.get_info<sycl::info::device::global_mem_size>();
        local_mem_size_ = device_.get_info<sycl::info::device::local_mem_size>();
        
        auto device_type = device_.get_info<sycl::info::device::device_type>();
        is_cpu_ = (device_type == sycl::info::device_type::cpu);
        is_gpu_ = (device_type == sycl::info::device_type::gpu);
        is_accelerator_ = (device_type == sycl::info::device_type::accelerator);
        
    } catch (const sycl::exception& e) {
        check_sycl_error(e, __FILE__, __LINE__);
        // Set defaults in case of error
        name_ = "Unknown Device";
        vendor_ = "Unknown Vendor";
        version_ = "Unknown Version";
        max_work_group_size_ = 1;
        max_compute_units_ = 1;
        global_mem_size_ = 0;
        local_mem_size_ = 0;
        is_cpu_ = false;
        is_gpu_ = false;
        is_accelerator_ = false;
    }
}

void SYCLManager::Device::synchronize_all_queues() {
    for (auto& queue : queues_) {
        queue.synchronize();
    }
}

// SYCLManager static methods implementation
void SYCLManager::init() {
    LOGINFO("Initializing SYCL Manager...");
    
    all_devices_.clear();
    devices_.clear();
    current_device_ = 0;
    
    discover_devices();
    
    if (all_devices_.empty()) {
        ARBD_Exception(ExceptionType::ValueError, "No SYCL devices found");
    }
    
    LOGINFO("Found {} SYCL device(s)", all_devices_.size());
}

void SYCLManager::discover_devices() {
    try {
        // Get all platforms
        auto platforms = sycl::platform::get_platforms();
        
        unsigned int device_id = 0;
        for (const auto& platform : platforms) {
            LOGDEBUG("Platform: {} ({})", 
                platform.get_info<sycl::info::platform::name>(),
                platform.get_info<sycl::info::platform::vendor>());
            
            // Get all devices for this platform
            auto platform_devices = platform.get_devices();
            
            for (const auto& device : platform_devices) {
                try {
                    // Create our Device wrapper using emplace_back to avoid copying
                    all_devices_.emplace_back(device, device_id++);
                } catch (const sycl::exception& e) {
                    LOGWARN("Failed to initialize device {}: {}", device_id, e.what());
                    device_id++; // Still increment to maintain consistent numbering
                }
            }
        }
        
        // Sort devices by preference (GPUs first, then accelerators, then CPUs)
        std::stable_sort(all_devices_.begin(), all_devices_.end(), 
            [](const Device& a, const Device& b) {
                if (a.is_gpu() != b.is_gpu()) return a.is_gpu();
                if (a.is_accelerator() != b.is_accelerator()) return a.is_accelerator();
                return a.id() < b.id();
            });
            
        // Reassign IDs after sorting
        for (size_t i = 0; i < all_devices_.size(); ++i) {
            all_devices_[i].set_id(static_cast<unsigned int>(i));
        }
        
    } catch (const sycl::exception& e) {
        check_sycl_error(e, __FILE__, __LINE__);
    }
}

void SYCLManager::load_info() {
    init();
    devices_ = std::move(all_devices_); // Use move instead of copy
    init_devices();
}

void SYCLManager::init_devices() {
    LOGINFO("Initializing SYCL devices...");
    std::string msg;
    
    for (size_t i = 0; i < devices_.size(); i++) {
        if (i > 0) {
            if (i == devices_.size() - 1) {
                msg += " and ";
            } else {
                msg += ", ";
            }
        }
        msg += std::to_string(devices_[i].id());
        
        // Devices are already initialized in constructor
        // Just log that they're ready
        LOGDEBUG("Device {} ready: {}", devices_[i].id(), devices_[i].name());
    }
    
    LOGINFO("Initialized SYCL devices: {}", msg);
    current_device_ = 0;
}

void SYCLManager::select_devices(std::span<const unsigned int> device_ids) {
    devices_.clear();
    devices_.reserve(device_ids.size()); // Reserve space to avoid reallocations
    
    for (unsigned int id : device_ids) {
        if (id >= all_devices_.size()) {
            ARBD_Exception(ExceptionType::ValueError, 
                "Invalid device ID: {}", id);
        }
        // Create a new Device by copying the sycl::device and id
        devices_.emplace_back(all_devices_[id].sycl_device(), id);
    }
    init_devices();
}

void SYCLManager::use(int device_id) {
    if (devices_.empty()) {
        ARBD_Exception(ExceptionType::ValueError, "No devices selected");
    }
    current_device_ = device_id % static_cast<int>(devices_.size());
}

void SYCLManager::sync(int device_id) {
    if (device_id >= static_cast<int>(devices_.size())) {
        ARBD_Exception(ExceptionType::ValueError, 
            "Invalid device ID: {}", device_id);
    }
    devices_[device_id].synchronize_all_queues();
}

void SYCLManager::sync() {
    for (auto& device : devices_) {
        device.synchronize_all_queues();
    }
}

int SYCLManager::current() {
    return current_device_;
}

void SYCLManager::prefer_device_type(sycl::info::device_type type) {
    preferred_type_ = type;
    
    // Re-sort devices based on new preference
    if (!all_devices_.empty()) {
        std::sort(all_devices_.begin(), all_devices_.end(), 
            [type](const Device& a, const Device& b) {
                auto a_type = a.sycl_device().get_info<sycl::info::device::device_type>();
                auto b_type = b.sycl_device().get_info<sycl::info::device::device_type>();
                
                if ((a_type == type) != (b_type == type)) {
                    return a_type == type;
                }
                return a.id() < b.id();
            });
            
        // Reassign IDs after sorting
        for (size_t i = 0; i < all_devices_.size(); ++i) {
            const_cast<unsigned int&>(all_devices_[i].id_) = static_cast<unsigned int>(i);
        }
    }
}

std::vector<unsigned int> SYCLManager::get_gpu_device_ids() {
    std::vector<unsigned int> gpu_ids;
    for (const auto& device : all_devices_) {
        if (device.is_gpu()) {
            gpu_ids.push_back(device.id());
        }
    }
    return gpu_ids;
}

std::vector<unsigned int> SYCLManager::get_cpu_device_ids() {
    std::vector<unsigned int> cpu_ids;
    for (const auto& device : all_devices_) {
        if (device.is_cpu()) {
            cpu_ids.push_back(device.id());
        }
    }
    return cpu_ids;
}

std::vector<unsigned int> SYCLManager::get_accelerator_device_ids() {
    std::vector<unsigned int> accel_ids;
    for (const auto& device : all_devices_) {
        if (device.is_accelerator()) {
            accel_ids.push_back(device.id());
        }
    }
    return accel_ids;
}

} // namespace ARBD

#endif // PROJECT_USES_SYCL