#ifdef USE_METAL
#include "METALManager.h"
#include "ARBDLogger.h"
#include <iostream>
#include <vector>

/**
 * @brief Example demonstrating how to use the ARBD Metal backend
 * 
 * This example shows:
 * - Device initialization and selection
 * - Memory allocation and data transfer
 * - Queue management and synchronization
 * - Basic compute operations
 */

namespace ARBD {
namespace METAL {

void example_basic_usage() {
    try {
        // Initialize Metal system
        LOGINFO("Initializing Metal backend...");
        METALManager::init();
        
        // Load device information
        METALManager::load_info();
        
        // Print available devices
        LOGINFO("Available Metal devices:");
        for (const auto& device : METALManager::all_devices()) {
            LOGINFO("  Device {}: {} (Low power: {}, Unified memory: {})", 
                device.get_id(), device.name(), device.is_low_power(), device.has_unified_memory());
        }
        
        // Select first device
        if (!METALManager::devices().empty()) {
            METALManager::use(0);
            LOGINFO("Using Metal device: {}", METALManager::get_current_device().name());
        }
        
    } catch (const ARBD::Exception& e) {
        LOGERROR("ARBD Exception in Metal example: {}", e.what());
    } catch (const std::exception& e) {
        LOGERROR("Standard exception in Metal example: {}", e.what());
    }
}

void example_memory_operations() {
    try {
        if (METALManager::devices().empty()) {
            LOGWARN("No Metal devices available for memory operations example");
            return;
        }
        
        auto& device = METALManager::get_current_device();
        void* metal_device = device.metal_device();
        
        // Allocate device memory for 1000 floats
        const size_t num_elements = 1000;
        DeviceMemory<float> device_memory(metal_device, num_elements);
        
        // Create host data
        std::vector<float> host_data(num_elements);
        for (size_t i = 0; i < num_elements; ++i) {
            host_data[i] = static_cast<float>(i) * 0.5f;
        }
        
        // Copy data to device
        LOGINFO("Copying {} elements to device...", num_elements);
        device_memory.copyFromHost(host_data);
        
        // Copy data back from device
        std::vector<float> result(num_elements);
        device_memory.copyToHost(result);
        
        // Verify data integrity
        bool data_matches = true;
        for (size_t i = 0; i < num_elements; ++i) {
            if (std::abs(host_data[i] - result[i]) > 1e-6f) {
                data_matches = false;
                break;
            }
        }
        
        if (data_matches) {
            LOGINFO("Memory operations successful - data integrity verified");
        } else {
            LOGERROR("Memory operations failed - data mismatch detected");
        }
        
    } catch (const ARBD::Exception& e) {
        LOGERROR("ARBD Exception in memory operations example: {}", e.what());
    } catch (const std::exception& e) {
        LOGERROR("Standard exception in memory operations example: {}", e.what());
    }
}

void example_queue_operations() {
    try {
        if (METALManager::devices().empty()) {
            LOGWARN("No Metal devices available for queue operations example");
            return;
        }
        
        auto& device = METALManager::get_current_device();
        
        // Get multiple queues  
        auto& queue1 = const_cast<Queue&>(device.get_queue(0));
        auto& queue2 = const_cast<Queue&>(device.get_queue(1));
        auto& queue3 = device.get_next_queue();
        
        LOGINFO("Created and accessed multiple Metal command queues");
        
        // Create command buffers
        void* cmd_buffer1 = queue1.create_command_buffer();
        void* cmd_buffer2 = queue2.create_command_buffer();
        
        // In a real application, you would encode compute commands here
        // For this example, we just commit empty command buffers
        
        // Commit and wait
        queue1.commit_and_wait(cmd_buffer1);
        queue2.commit_and_wait(cmd_buffer2);
        
        // Synchronize all queues
        device.synchronize_all_queues();
        
        LOGINFO("Queue operations completed successfully");
        
    } catch (const ARBD::Exception& e) {
        LOGERROR("ARBD Exception in queue operations example: {}", e.what());
    } catch (const std::exception& e) {
        LOGERROR("Standard exception in queue operations example: {}", e.what());
    }
}

void example_multi_device() {
    try {
        if (METALManager::all_device_size() < 2) {
            LOGINFO("Multi-device example requires at least 2 Metal devices (found {})", 
                METALManager::all_device_size());
            return;
        }
        
        // Select multiple devices
        std::vector<unsigned int> device_ids = {0, 1};
        METALManager::select_devices(device_ids);
        
        LOGINFO("Selected {} Metal devices for multi-device operations", device_ids.size());
        
        // Use different devices
        for (unsigned int device_id : device_ids) {
            METALManager::use(device_id);
            auto& current_device = METALManager::get_current_device();
            LOGINFO("Operating on device {}: {}", device_id, current_device.name());
            
            // Perform operations on this device
            auto& queue = current_device.get_next_queue();
            // ... device-specific operations would go here ...
        }
        
        // Synchronize all devices
        METALManager::sync();
        LOGINFO("All Metal devices synchronized");
        
    } catch (const ARBD::Exception& e) {
        LOGERROR("ARBD Exception in multi-device example: {}", e.what());
    } catch (const std::exception& e) {
        LOGERROR("Standard exception in multi-device example: {}", e.what());
    }
}

} // namespace METAL
} // namespace ARBD

// Example main function (for testing purposes)
#ifdef METAL_EXAMPLE_MAIN
int main() {
    using namespace ARBD::METAL;
    
    std::cout << "ARBD Metal Backend Example\n";
    std::cout << "==========================\n\n";
    
    example_basic_usage();
    std::cout << "\n";
    
    example_memory_operations();
    std::cout << "\n";
    
    example_queue_operations();
    std::cout << "\n";
    
    example_multi_device();
    std::cout << "\n";
    
    // Clean up
    METALManager::finalize();
    
    std::cout << "Metal backend example completed.\n";
    return 0;
}
#endif

#endif // USE_METAL 