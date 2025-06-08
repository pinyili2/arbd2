#pragma once

#ifdef USE_METAL
#include "ARBDException.h"
#include <array>
#include <vector>
#include <string>
#include <span>
#include <optional>
#include <chrono>
#include <memory>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

//#ifdef __OBJC__
//#import <Metal/Metal.h>
//#import <Foundation/Foundation.h>
//#else
// Forward declarations for C++ when not in Objective-C++ mode
//typedef void* MTLDevice;
//typedef void* MTLCommandQueue;
//typedef void* MTLBuffer;
//typedef void* MTLCommandBuffer;
//#endif

namespace ARBD {
namespace METAL {

inline void check_metal_error(void* object, std::string_view file, int line) {
    if (object == nullptr) {
        ARBD_Exception(ExceptionType::MetalRuntimeError, 
            "Metal error at {}:{}: Object is null", 
            file, line);
    }
}

#define METAL_CHECK(call) do { auto result = call; check_metal_error(result, __FILE__, __LINE__); } while(0)

/**
 * @brief Modern RAII wrapper for Metal device memory
 * 
 * This class provides a safe and efficient way to manage Metal device memory with RAII semantics.
 * It handles memory allocation, deallocation, and data transfer between host and device memory.
 * 
 * Features:
 * - Automatic memory management (RAII)
 * - Move semantics support
 * - Safe copy operations using std::span
 * - Exception handling for Metal errors
 * 
 * @tparam T The type of data to store in device memory
 * 
 * @example Basic Usage:
 * ```cpp
 * // Create a device and allocate memory for 1000 integers
 * auto device = MTLCreateSystemDefaultDevice();
 * ARBD::DeviceMemory<int> device_mem(device, 1000);
 * 
 * // Copy data from host to device
 * std::vector<int> host_data(1000, 42);
 * device_mem.copyFromHost(host_data);
 * 
 * // Copy data back to host
 * std::vector<int> result(1000);
 * device_mem.copyToHost(result);
 * ```
 * 
 * @example Move Semantics:
 * ```cpp
 * ARBD::DeviceMemory<float> mem1(device, 1000);
 * ARBD::DeviceMemory<float> mem2 = std::move(mem1); // mem1 is now empty
 * ```
 * 
 * @note The class prevents copying to avoid accidental memory leaks.
 *       Use move semantics when transferring ownership.
 */
template<typename T>
class DeviceMemory {
private:
    void* buffer_{nullptr}; // id<MTLBuffer>
    size_t size_{0};
    void* device_{nullptr}; // id<MTLDevice>

public:
    DeviceMemory() = default;
    
    explicit DeviceMemory(void* device, size_t count);
    ~DeviceMemory();

    // Prevent copying
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
    
    // Allow moving
    DeviceMemory(DeviceMemory&& other) noexcept;
    DeviceMemory& operator=(DeviceMemory&& other) noexcept;

    // Modern copy operations using std::span
    void copyFromHost(std::span<const T> host_data);
    void copyToHost(std::span<T> host_data) const;

    // Accessors
    [[nodiscard]] T* get() noexcept;
    [[nodiscard]] const T* get() const noexcept;
    [[nodiscard]] size_t size() const noexcept { return size_; }
    [[nodiscard]] void* device() const noexcept { return device_; }
    [[nodiscard]] void* buffer() const noexcept { return buffer_; }
    
    // Conversion operators
    operator T*() noexcept { return get(); }
    operator const T*() const noexcept { return get(); }
};

/**
 * @brief Metal command queue wrapper with additional functionality
 * 
 * This class extends MTLCommandQueue with additional convenience methods
 * and RAII semantics for better integration with the ARBD framework.
 * 
 * Features:
 * - Automatic queue creation and management
 * - Synchronization utilities
 * - Exception handling integration
 * - Performance profiling support
 * 
 * @example Basic Usage:
 * ```cpp
 * // Create a queue for a specific device
 * ARBD::Queue queue(device);
 * 
 * // Create command buffer and commit
 * auto cmd_buffer = queue.create_command_buffer();
 * // ... encode commands ...
 * queue.commit_and_wait(cmd_buffer);
 * 
 * // Check if queue is available
 * bool available = queue.is_available();
 * ```
 * 
 * @note The queue is automatically synchronized when the Queue object is destroyed
 */
class Queue {
private:
    void* queue_{nullptr}; // id<MTLCommandQueue>
    void* device_{nullptr}; // id<MTLDevice>

public:
    Queue() = default;
    explicit Queue(void* device);
    ~Queue();

    // Prevent copying
    Queue(const Queue&) = delete;
    Queue& operator=(const Queue&) = delete;

    // Allow moving
    Queue(Queue&& other) noexcept;
    Queue& operator=(Queue&& other) noexcept;

    void synchronize();
    void* create_command_buffer(); // Returns id<MTLCommandBuffer>
    void commit_and_wait(void* command_buffer); // id<MTLCommandBuffer>
    
    [[nodiscard]] bool is_available() const;
    [[nodiscard]] void* get() noexcept { return queue_; }
    [[nodiscard]] const void* get() const noexcept { return queue_; }
    [[nodiscard]] void* get_device() const noexcept { return device_; }

    operator void*() noexcept { return queue_; }
    operator const void*() const noexcept { return queue_; }
};

/**
 * @brief Metal command buffer wrapper for timing and synchronization
 * 
 * This class provides a convenient wrapper around MTLCommandBuffer with
 * additional timing and synchronization utilities.
 * 
 * Features:
 * - Command buffer timing measurements
 * - Synchronization utilities
 * - Exception handling integration
 * - Profiling support
 * 
 * @example Basic Usage:
 * ```cpp
 * // Create and use a command buffer
 * ARBD::Event event = queue.create_command_buffer();
 * // ... encode commands ...
 * event.commit();
 * 
 * // Wait for completion
 * event.wait();
 * 
 * // Get execution time (if available)
 * auto duration = event.get_execution_time();
 * ```
 * 
 * @note Command buffers are automatically waited on when the Event object is destroyed
 */
class Event {
public:
    Event() = default;
    explicit Event(void* command_buffer); // id<MTLCommandBuffer>
    ~Event();
    
    // Prevent copying
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    
    // Allow moving
    Event(Event&& other) noexcept;
    Event& operator=(Event&& other) noexcept;
    
    void commit();
    void wait();
    
    [[nodiscard]] bool is_complete() const;
    [[nodiscard]] std::chrono::nanoseconds get_execution_time() const;
    
    [[nodiscard]] void* get() noexcept { return command_buffer_; }
    [[nodiscard]] const void* get() const noexcept { return command_buffer_; }
    
    operator void*() noexcept { return command_buffer_; }
    operator const void*() const noexcept { return command_buffer_; }
    
private:
    void* command_buffer_{nullptr}; // id<MTLCommandBuffer>
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool timing_available_{false};
};

/**
 * @brief Modern Metal device management system
 * 
 * This class provides a comprehensive Metal device management system with support for multiple devices,
 * queue management, and device selection. It handles device initialization, selection,
 * and provides utilities for multi-device operations.
 * 
 * Features:
 * - Multi-device support (integrated and discrete GPUs)
 * - Automatic queue management
 * - Device selection and synchronization
 * - Performance monitoring
 * - Exception handling integration
 * 
 * @example Basic Usage:
 * ```cpp
 * // Initialize Metal system
 * ARBD::METALManager::init();
 * 
 * // Select specific devices
 * std::vector<unsigned int> device_ids = {0, 1};
 * ARBD::METALManager::select_devices(device_ids);
 * 
 * // Use a specific device
 * ARBD::METALManager::use(0);
 * 
 * // Get current queue
 * auto& queue = ARBD::METALManager::get_current_queue();
 * 
 * // Synchronize all devices
 * ARBD::METALManager::sync();
 * ```
 * 
 * @example Multi-Device Operations:
 * ```cpp
 * // Get device properties
 * const auto& device = ARBD::METALManager::devices[0];
 * const auto& props = device.properties();
 * 
 * // Submit work to specific device
 * auto& queue = device.get_queue();
 * auto cmd_buffer = queue.create_command_buffer();
 * // ... encode work ...
 * queue.commit_and_wait(cmd_buffer);
 * ```
 * 
 * @note The class uses static methods for global device management.
 *       All operations are thread-safe and exception-safe.
 */
class METALManager {
public:
    static constexpr size_t NUM_QUEUES = 4; // Multiple queues per device

    /**
     * @brief Individual Metal device management class
     * 
     * This nested class represents a single Metal device and manages its resources,
     * including queues and device properties.
     * 
     * Features:
     * - Queue management
     * - Device property access
     * - Performance monitoring
     * - Safe resource cleanup
     * 
     * @example Basic Usage:
     * ```cpp
     * // Get device properties
     * const auto& device = ARBD::METALManager::devices[0];
     * const auto& props = device.properties();
     * 
     * // Get a queue
     * auto& queue = device.get_queue(0);
     * 
     * // Get next available queue
     * auto& next_queue = device.get_next_queue();
     * ```
     */
    class Device {
    public:
        explicit Device(void* device, unsigned int id); // id<MTLDevice>
        ~Device() = default;

        // Delete copy constructor and copy assignment operator
        Device(const Device&) = delete;
        Device& operator=(const Device&) = delete;

        // Enable move constructor and move assignment operator
        Device(Device&&) = default;
        Device& operator=(Device&&) = default;

        [[nodiscard]] Queue& get_queue(size_t queue_id) {
            return queues_[queue_id % NUM_QUEUES];
        }

        [[nodiscard]] const Queue& get_queue(size_t queue_id) const {
            return queues_[queue_id % NUM_QUEUES];
        }

        [[nodiscard]] Queue& get_next_queue() {
            last_queue_ = (last_queue_ + 1) % NUM_QUEUES;
            return queues_[last_queue_];
        }

        [[nodiscard]] unsigned int id() const noexcept { return id_; }
        void set_id(unsigned int new_id) noexcept { id_ = new_id; }
        [[nodiscard]] void* metal_device() const noexcept { return device_; } // id<MTLDevice>
        [[nodiscard]] const std::string& name() const noexcept { return name_; }
        [[nodiscard]] size_t max_threads_per_group() const noexcept { return max_threads_per_group_; }
        [[nodiscard]] size_t recommended_max_working_set_size() const noexcept { return recommended_max_working_set_size_; }
        [[nodiscard]] bool has_unified_memory() const noexcept { return has_unified_memory_; }
        [[nodiscard]] bool is_low_power() const noexcept { return is_low_power_; }
        [[nodiscard]] bool is_removable() const noexcept { return is_removable_; }
        [[nodiscard]] bool supports_compute() const noexcept { return supports_compute_; }

        void synchronize_all_queues();

    private:
        void query_device_properties();

        unsigned int id_;
        void* device_{nullptr}; // id<MTLDevice>
        std::array<Queue, NUM_QUEUES> queues_;
        int last_queue_{-1};
        
        // Device properties
        std::string name_;
        size_t max_threads_per_group_;
        size_t recommended_max_working_set_size_;
        bool has_unified_memory_;
        bool is_low_power_;
        bool is_removable_;
        bool supports_compute_;
        
        // Friend class to allow METALManager to access private members
        friend class METALManager;
    };

    // Static interface
    static void init();
    static void load_info();
    static void select_devices(std::span<const unsigned int> device_ids);
    static void use(int device_id);
    static void sync(int device_id);
    static void sync();
    static int current();
    static void prefer_low_power(bool prefer);
    static void finalize();
    
    [[nodiscard]] static size_t all_device_size() noexcept { return all_devices_.size(); }
    [[nodiscard]] static const std::vector<Device>& all_devices() noexcept { return all_devices_; }
    [[nodiscard]] static const std::vector<Device>& devices() noexcept { return devices_; }
    [[nodiscard]] static Queue& get_current_queue() { return devices_[current_device_].get_next_queue(); }
    [[nodiscard]] static Device& get_current_device() { return devices_[current_device_]; }
    
    // Device filtering utilities
    [[nodiscard]] static std::vector<unsigned int> get_discrete_gpu_device_ids();
    [[nodiscard]] static std::vector<unsigned int> get_integrated_gpu_device_ids();
    [[nodiscard]] static std::vector<unsigned int> get_low_power_device_ids();

private:
    static void init_devices();
    static void discover_devices();

    static std::vector<Device> all_devices_;
    static std::vector<Device> devices_;
    static int current_device_;
    static bool prefer_low_power_;
};

} // namespace METAL
} // namespace ARBD

#endif // USE_METAL 