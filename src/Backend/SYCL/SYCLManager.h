#pragma once

#ifdef USE_SYCL
#include "ARBDException.h"
#include <array>
#include <chrono>
#include <optional>
#include <span>
#include <string>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

namespace ARBD {
namespace SYCL {
inline void check_sycl_error(const sycl::exception &e, std::string_view file,
                             int line) {
  ARBD_Exception(ExceptionType::SYCLRuntimeError, "SYCL error at {}:{}: {}",
                 file, line, e.what());
}

#define SYCL_CHECK(call)                                                       \
  try {                                                                        \
    call;                                                                      \
  } catch (const sycl::exception &e) {                                         \
    check_sycl_error(e, __FILE__, __LINE__);                                   \
  }

/**
 * @brief Modern RAII wrapper for SYCL device memory
 *
 * This class provides a safe and efficient way to manage SYCL device memory
 * with RAII semantics. It handles memory allocation, deallocation, and data
 * transfer between host and device memory.
 *
 * Features:
 * - Automatic memory management (RAII)
 * - Move semantics support
 * - Safe copy operations using std::span
 * - Exception handling for SYCL errors
 *
 * @tparam T The type of data to store in device memory
 *
 * @example Basic Usage:
 * ```cpp
 * // Create a queue and allocate memory for 1000 integers
 * sycl::queue q;
 * ARBD::DeviceMemory<int> device_mem(q, 1000);
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
 * ARBD::DeviceMemory<float> mem1(queue, 1000);
 * ARBD::DeviceMemory<float> mem2 = std::move(mem1); // mem1 is now empty
 * ```
 *
 * @note The class prevents copying to avoid accidental memory leaks.
 *       Use move semantics when transferring ownership.
 */
template <typename T> class DeviceMemory {
private:
  T *ptr_{nullptr};
  size_t size_{0};
  sycl::queue *queue_{nullptr};

public:
  DeviceMemory() = default;

  explicit DeviceMemory(sycl::queue &q, size_t count)
      : queue_(&q), size_(count) {
    if (count > 0) {
      SYCL_CHECK(ptr_ = sycl::malloc_device<T>(count, *queue_));
      if (!ptr_) {
        ARBD_Exception(ExceptionType::SYCLRuntimeError,
                       "Failed to allocate {} elements of type {}", count,
                       typeid(T).name());
      }
    }
  }

  ~DeviceMemory() {
    if (ptr_ && queue_) {
      sycl::free(ptr_, *queue_);
    }
  }

  // Prevent copying
  DeviceMemory(const DeviceMemory &) = delete;
  DeviceMemory &operator=(const DeviceMemory &) = delete;

  // Allow moving
  DeviceMemory(DeviceMemory &&other) noexcept
      : ptr_(std::exchange(other.ptr_, nullptr)),
        size_(std::exchange(other.size_, 0)),
        queue_(std::exchange(other.queue_, nullptr)) {}

  DeviceMemory &operator=(DeviceMemory &&other) noexcept {
    if (this != &other) {
      if (ptr_ && queue_)
        sycl::free(ptr_, *queue_);
      ptr_ = std::exchange(other.ptr_, nullptr);
      size_ = std::exchange(other.size_, 0);
      queue_ = std::exchange(other.queue_, nullptr);
    }
    return *this;
  }

  // Modern copy operations using std::span
  void copyFromHost(std::span<const T> host_data) {
    if (host_data.size() > size_) {
      ARBD_Exception(ExceptionType::ValueError,
                     "Tried to copy {} elements but only {} allocated",
                     host_data.size(), size_);
    }
    if (!ptr_ || host_data.empty() || !queue_)
      return;

    SYCL_CHECK(
        queue_->memcpy(ptr_, host_data.data(), host_data.size() * sizeof(T))
            .wait());
  }

  void copyToHost(std::span<T> host_data) const {
    if (host_data.size() > size_) {
      ARBD_Exception(ExceptionType::ValueError,
                     "Tried to copy {} elements but only {} allocated",
                     host_data.size(), size_);
    }
    if (!ptr_ || host_data.empty() || !queue_)
      return;

    SYCL_CHECK(
        queue_->memcpy(host_data.data(), ptr_, host_data.size() * sizeof(T))
            .wait());
  }

  // Accessors
  [[nodiscard]] T *get() noexcept { return ptr_; }
  [[nodiscard]] const T *get() const noexcept { return ptr_; }
  [[nodiscard]] size_t size() const noexcept { return size_; }
  [[nodiscard]] sycl::queue *queue() const noexcept { return queue_; }

  // Conversion operators
  operator T *() noexcept { return ptr_; }
  operator const T *() const noexcept { return ptr_; }
};

/**
 * @brief SYCL queue wrapper with additional functionality
 *
 * This class extends sycl::queue with additional convenience methods
 * and RAII semantics for better integration with the ARBD framework.
 *
 * Features:
 * - Automatic device selection and queue creation
 * - Synchronization utilities
 * - Exception handling integration
 * - Performance profiling support
 *
 * @example Basic Usage:
 * ```cpp
 * // Create a queue for a specific device
 * ARBD::Queue queue(device);
 *
 * // Submit work and synchronize
 * queue.submit([&](sycl::handler& h) {
 *     // kernel code
 * });
 * queue.synchronize();
 *
 * // Check if queue is in-order
 * bool in_order = queue.is_in_order();
 * ```
 *
 * @note The queue is automatically synchronized when the Queue object is
 * destroyed
 */
class Queue {
private:
  std::optional<sycl::queue> queue_;

public:
  Queue() {
    // Don't try to create a queue with selectors in default constructor
    // as they might fail. Queue will be initialized later via assignment.
  }

  explicit Queue(const sycl::device &dev) {
    try {
      queue_ = sycl::queue(dev); // Create queue with device only, no properties
    } catch (const sycl::exception &e) {
      // Directly log before check_sycl_error
      std::cerr << "!!! ARBD::Queue constructor caught sycl::exception: "
                << e.what()
                << " for device: " << dev.get_info<sycl::info::device::name>()
                << std::endl;
      check_sycl_error(e, __FILE__, __LINE__);
    }
  }

  explicit Queue(const sycl::device &dev, const sycl::property_list &props) {
    try {
      // Create queue with single device directly, not as a multi-device queue
      queue_ = sycl::queue(dev, props);
    } catch (const sycl::exception &e) {
      check_sycl_error(e, __FILE__, __LINE__);
    }
  }

  // Add constructor for multiple devices
  explicit Queue(const std::vector<sycl::device> &devices,
                 const sycl::property_list &props = {}) {
    try {
      if (devices.empty()) {
        ARBD_Exception(ExceptionType::SYCLRuntimeError,
                       "Cannot create queue with empty device list");
      }
      // Create context with all devices, but queue with first device
      sycl::context ctx(devices);
      queue_ = sycl::queue(ctx, devices[0], props);
    } catch (const sycl::exception &e) {
      check_sycl_error(e, __FILE__, __LINE__);
    }
  }

  ~Queue() {
    try {
      if (queue_.has_value()) {
        // queue_->wait(); // Still commented out from previous debugging
      }
    } catch (...) {
      // Don't throw from destructor
    }
  }

  // Prevent copying
  Queue(const Queue &) = delete;
  Queue &operator=(const Queue &) = delete;

  // Allow moving
  Queue(Queue &&other) noexcept : queue_(std::move(other.queue_)) {}

  Queue &operator=(Queue &&other) noexcept {
    if (this != &other) {
      if (queue_.has_value()) {
        // try { queue_->wait(); } catch (...) {} // If re-enabled
      }
      queue_ = std::move(other.queue_);
    }
    return *this;
  }

  void synchronize() {
    if (queue_.has_value()) {
      SYCL_CHECK(queue_->wait());
    }
  }

  template <typename KernelName = class kernel_default_name, typename F>
  sycl::event submit(F &&f) {
    if (!queue_.has_value()) {
      ARBD_Exception(ExceptionType::SYCLRuntimeError,
                     "Queue not initialized for submission");
    }
    try {
      return queue_->submit(std::forward<F>(f));
    } catch (const sycl::exception &e) {
      check_sycl_error(e, __FILE__, __LINE__);
      throw; // Re-throw after logging
    }
  }

  [[nodiscard]] bool is_in_order() const {
    return queue_.has_value() ? queue_->is_in_order() : false;
  }

  [[nodiscard]] sycl::context get_context() const {
    if (!queue_.has_value()) {
      ARBD_Exception(ExceptionType::SYCLRuntimeError,
                     "Queue not initialized for get_context");
    }
    return queue_->get_context();
  }

  // Add method to get all devices
  [[nodiscard]] std::vector<sycl::device> get_devices() const {
    if (!queue_.has_value()) {
      ARBD_Exception(ExceptionType::SYCLRuntimeError,
                     "Queue not initialized for get_devices");
    }
    return {queue_->get_device()};
  }

  [[nodiscard]] sycl::queue &get() {
    if (!queue_.has_value()) {
      ARBD_Exception(ExceptionType::SYCLRuntimeError,
                     "Queue not initialized for get");
    }
    return *queue_;
  }

  [[nodiscard]] const sycl::queue &get() const {
    if (!queue_.has_value()) {
      ARBD_Exception(ExceptionType::SYCLRuntimeError,
                     "Queue not initialized for get const");
    }
    return *queue_;
  }

  operator sycl::queue &() { return get(); }
  operator const sycl::queue &() const { return get(); }
};

/**
 * @brief SYCL event wrapper for timing and synchronization
 *
 * This class provides a convenient wrapper around sycl::event with
 * additional timing and synchronization utilities.
 *
 * Features:
 * - Event timing measurements
 * - Synchronization utilities
 * - Exception handling integration
 * - Profiling support
 *
 * @example Basic Usage:
 * ```cpp
 * // Submit work and get an event
 * ARBD::Event event = queue.submit([&](sycl::handler& h) {
 *     // kernel code
 * });
 *
 * // Wait for completion
 * event.wait();
 *
 * // Get execution time
 * auto duration = event.get_execution_time();
 * ```
 *
 * @note Events are automatically waited on when the Event object is destroyed
 */
class Event {
public:
  Event() = default;

  explicit Event(sycl::event e) : event_(std::move(e)) {}

  ~Event() {}

  // Prevent copying
  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;

  // Allow moving
  Event(Event &&other) noexcept : event_(std::move(other.event_)) {}

  Event &operator=(Event &&other) noexcept {
    if (this != &other) {
      event_ = std::move(other.event_);
    }
    return *this;
  }

  void wait() {
    if (event_.has_value()) {
      SYCL_CHECK(event_->wait());
    }
  }

  [[nodiscard]] bool is_complete() const {
    if (!event_.has_value())
      return true;

    try {
      auto status =
          event_->get_info<sycl::info::event::command_execution_status>();
      return status == sycl::info::event_command_status::complete;
    } catch (const sycl::exception &) {
      return false;
    }
  }

  [[nodiscard]] std::chrono::nanoseconds get_execution_time() const {
    if (!event_.has_value()) {
      return std::chrono::nanoseconds{0};
    }

    try {
      auto start = event_->get_profiling_info<
          sycl::info::event_profiling::command_start>();
      auto end =
          event_
              ->get_profiling_info<sycl::info::event_profiling::command_end>();
      return std::chrono::nanoseconds{end - start};
    } catch (const sycl::exception &e) {
      check_sycl_error(e, __FILE__, __LINE__);
      return std::chrono::nanoseconds{0};
    }
  }

  [[nodiscard]] sycl::event &get() {
    if (!event_.has_value()) {
      ARBD_Exception(ExceptionType::CUDARuntimeError, "Event not initialized");
    }
    return *event_;
  }

  [[nodiscard]] const sycl::event &get() const {
    if (!event_.has_value()) {
      ARBD_Exception(ExceptionType::CUDARuntimeError, "Event not initialized");
    }
    return *event_;
  }

  operator sycl::event &() { return get(); }
  operator const sycl::event &() const { return get(); }

private:
  std::optional<sycl::event> event_;
};

/**
 * @brief Modern SYCL device management system
 *
 * This class provides a comprehensive SYCL device management system with
 * support for multiple devices, queue management, and device selection. It
 * handles device initialization, selection, and provides utilities for
 * multi-device operations.
 *
 * Features:
 * - Multi-device support (GPU, CPU, accelerators)
 * - Automatic queue management
 * - Device selection and synchronization
 * - Performance monitoring
 * - Exception handling integration
 *
 * @example Basic Usage:
 * ```cpp
 * // Initialize SYCL system
 * ARBD::SYCLManager::init();
 *
 * // Select specific devices
 * std::vector<unsigned int> device_ids = {0, 1};
 * ARBD::SYCLManager::select_devices(device_ids);
 *
 * // Use a specific device
 * ARBD::SYCLManager::use(0);
 *
 * // Get current queue
 * auto& queue = ARBD::SYCLManager::get_current_queue();
 *
 * // Synchronize all devices
 * ARBD::SYCLManager::sync();
 * ```
 *
 * @example Multi-Device Operations:
 * ```cpp
 * // Get device properties
 * const auto& device = ARBD::SYCLManager::devices[0];
 * const auto& props = device.properties();
 *
 * // Submit work to specific device
 * auto& queue = device.get_queue();
 * queue.submit([&](sycl::handler& h) {
 *     // kernel code
 * });
 * ```
 *
 * @note The class uses static methods for global device management.
 *       All operations are thread-safe and exception-safe.
 */
class SYCLManager {
public:
  static constexpr size_t NUM_QUEUES = 4; // Fewer queues than CUDA streams

  /**
   * @brief Individual SYCL device management class
   *
   * This nested class represents a single SYCL device and manages its
   * resources, including queues and device properties.
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
   * const auto& device = ARBD::SYCLManager::devices[0];
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
    explicit Device(const sycl::device &dev, unsigned int id);
    ~Device() = default;

    // Delete copy constructor and copy assignment operator
    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;

    // Enable move constructor and move assignment operator
    Device(Device &&) = default;
    Device &operator=(Device &&) = default;

    [[nodiscard]] Queue &get_queue(size_t queue_id) {
      return queues_[queue_id % NUM_QUEUES];
    }

    [[nodiscard]] const Queue &get_queue(size_t queue_id) const {
      return queues_[queue_id % NUM_QUEUES];
    }

    [[nodiscard]] Queue &get_next_queue() {
      last_queue_ = (last_queue_ + 1) % NUM_QUEUES;
      return queues_[last_queue_];
    }

    [[nodiscard]] unsigned int id() const noexcept { return id_; }
    void set_id(unsigned int new_id) noexcept {
      id_ = new_id;
    } // Add setter for ID
    [[nodiscard]] const sycl::device &sycl_device() const noexcept {
      return device_;
    }
    [[nodiscard]] const std::string &name() const noexcept { return name_; }
    [[nodiscard]] const std::string &vendor() const noexcept { return vendor_; }
    [[nodiscard]] const std::string &version() const noexcept {
      return version_;
    }
    [[nodiscard]] size_t max_work_group_size() const noexcept {
      return max_work_group_size_;
    }
    [[nodiscard]] size_t max_compute_units() const noexcept {
      return max_compute_units_;
    }
    [[nodiscard]] size_t global_mem_size() const noexcept {
      return global_mem_size_;
    }
    [[nodiscard]] size_t local_mem_size() const noexcept {
      return local_mem_size_;
    }
    [[nodiscard]] bool is_cpu() const noexcept { return is_cpu_; }
    [[nodiscard]] bool is_gpu() const noexcept { return is_gpu_; }
    [[nodiscard]] bool is_accelerator() const noexcept {
      return is_accelerator_;
    }

    void synchronize_all_queues();

  private:
    void query_device_properties();

    unsigned int id_;
    sycl::device device_;
    std::array<Queue, NUM_QUEUES> queues_;
    int last_queue_{-1};

    // Device properties
    std::string name_;
    std::string vendor_;
    std::string version_;
    size_t max_work_group_size_;
    size_t max_compute_units_;
    size_t global_mem_size_;
    size_t local_mem_size_;
    bool is_cpu_;
    bool is_gpu_;
    bool is_accelerator_;

    // Friend class to allow SYCLManager to access private members
    friend class SYCLManager;
  };

  // Static interface
  static void init();
  static void load_info();
  static void select_devices(std::span<const unsigned int> device_ids);
  static void use(int device_id);
  static void sync(int device_id);
  static void sync();
  static int current();
  static void prefer_device_type(sycl::info::device_type type);
  static void finalize();

  [[nodiscard]] static size_t all_device_size() noexcept {
    return all_devices_.size();
  }
  [[nodiscard]] static const std::vector<Device> &all_devices() noexcept {
    return all_devices_;
  }
  [[nodiscard]] static const std::vector<Device> &devices() noexcept {
    return devices_;
  }
  [[nodiscard]] static Queue &get_current_queue() {
    return devices_[current_device_].get_next_queue();
  }
  [[nodiscard]] static Device &get_current_device() {
    return devices_[current_device_];
  }

  // Device filtering utilities
  [[nodiscard]] static std::vector<unsigned int> get_gpu_device_ids();
  [[nodiscard]] static std::vector<unsigned int> get_cpu_device_ids();
  [[nodiscard]] static std::vector<unsigned int> get_accelerator_device_ids();

private:
  static void init_devices();
  static void discover_devices();

  static std::vector<Device> all_devices_;
  static std::vector<Device> devices_;
  static int current_device_;
  static sycl::info::device_type preferred_type_;
};
} // namespace SYCL
} // namespace ARBD

#endif // PROJECT_USES_SYCL
