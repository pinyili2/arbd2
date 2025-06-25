#pragma once

/**
 * @file ARBDLogger.h
 * @brief Unified logging system combining ARBDLogger and Debug.h functionality
 *
 * This header provides a comprehensive logging solution that merges:
 * - Modern C++20 ARBDLogger with printf-style formatting
 * - Classic Debug.h numeric level system (0-10 scale)
 * - Device-specific logging for CUDA and SYCL
 * - Universal compatibility across compilers
 *
 * @section Usage Examples
 *
 * @subsection original_logger Original ARBDLogger functionality:
 * @code
 * #include "ARBDLogger.h"
 *
 * // Basic logging with different levels
 * LOGTRACE("This is a trace message: %d", 42);
 * LOGDEBUG("This is a debug message: %s", "hello world");
 * LOGINFO("This is an info message: %d + %d = %d", 3, 4, 7);
 * LOGWARN("This is a warning message");
 * LOGERROR("This is an error message");
 * LOGCRITICAL("This is a critical message");
 *
 * // Set log level to filter messages
 * ARBD::Logger::set_level(ARBD::LogLevel::WARN);
 * @endcode
 *
 * @subsection debug_compat Debug.h compatibility layer:
 * @code
 * // Enable debug messages (define before including header or in build system)
 * #define DEBUGMSG
 * #include "ARBDLogger.h"
 *
 * // Numeric debug levels (0-10 scale)
 * DebugMessage(0, "Plain debug message (level 0)");           // stdout
 * DebugMessage(1, "Low severity debug (level 1)");           // stdout
 * DebugMessage(4, "Important message (level 4)");            // stdout
 * DebugMessage(5, "Warning level (level 5 - goes to stderr)"); // stderr
 * DebugMessage(10, "CRASH BANG BOOM error (level 10 - stderr)");   // stderr
 *
 * // Format arguments (printf style)
 * DebugMsg(2, "Formatted debug: value = %d", 123);
 * DebugMsg(3, "Multiple args: %s and %s make %s", "foo", "bar", "foobar");
 *
 * // Conditional compilation macro
 * Debug(std::cout << "This appears only when DEBUGMSG is defined" <<
 * std::endl);
 * @endcode
 *
 * @subsection device_logging Device-specific logging:
 * @code
 * // CUDA kernel code
 * __global__ void my_kernel() {
 *     LOGINFO("Running on GPU thread %d", threadIdx.x);  // Uses printf
 * }
 *
 * // SYCL kernel code
 * queue.submit([&](handler& h) {
 *     h.parallel_for(range, [=](id<1> idx) {
 *         LOGDEBUG("SYCL work-item %d", idx[0]);  // Uses printf
 *     });
 * });
 * @endcode
 *
 * @note When DEBUGMSG is not defined, all Debug.h macros become no-ops
 * @note Device code automatically uses printf-based logging
 * @note Host code uses iostreams with timestamps for compatibility
 */

#include "ARBDException.h" // For SourceLocation
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <type_traits>

// Debug level configuration (from Debug.h)
#ifndef MIN_DEBUG_LEVEL
#define MIN_DEBUG_LEVEL 0
#endif
#ifndef MAX_DEBUG_LEVEL
#define MAX_DEBUG_LEVEL 10
#endif
#ifndef STDERR_LEVEL
/* anything >= this error level goes to stderr */
#define STDERR_LEVEL 5
#endif

namespace ARBD {

enum class LogLevel {
  TRACE = 0,
  DEBUG = 1,
  INFO = 2,
  WARN = 3,
  ERROR = 4,
  CRITICAL = 5
};

class Logger {
private:
  // Helper function to convert std::string to const char* for printf compatibility
  template<typename T>
  static constexpr auto convert_for_printf(T&& arg) {
    if constexpr (std::is_same_v<std::decay_t<T>, std::string>) {
      return arg.c_str();
    } else {
      return std::forward<T>(arg);
    }
  }

public:
  static LogLevel current_level;

  // Printf-style variadic template logging with std::string support
  template <typename... Args>
  static void log(LogLevel level, const SourceLocation &loc, const char *fmt,
                  Args &&...args) {
    if (level < current_level)
      return;

    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);

    auto &stream = (level >= LogLevel::ERROR) ? std::cerr : std::cout;

    stream << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S") << "] "
           << "[" << level_to_string(level) << "] ";

    // Convert std::string arguments to c_str() for printf compatibility
    char buffer[1024];
    std::snprintf(buffer, sizeof(buffer), fmt, convert_for_printf(args)...);

    stream << buffer << " (" << loc.file_name << ":" << loc.line << ")"
           << std::endl;
  }

  // Simple string version
  static void log(LogLevel level, const SourceLocation &loc,
                  const std::string &message) {
    if (level < current_level)
      return;

    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);

    auto &stream = (level >= LogLevel::ERROR) ? std::cerr : std::cout;

    stream << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S") << "] "
           << "[" << level_to_string(level) << "] " << message << " ("
           << loc.file_name << ":" << loc.line << ")" << std::endl;
  }

  /**
   * @brief Debug message function with numeric levels (from Debug.h)
   */
  template <typename... Args>
  static void debug_message(int level, const char *fmt,
                            const SourceLocation &loc, Args &&...args) {
    if ((level >= MIN_DEBUG_LEVEL) && (level <= MAX_DEBUG_LEVEL)) {
      auto &stream = (level >= STDERR_LEVEL) ? std::cerr : std::cout;

      if (level >= STDERR_LEVEL) {
        stream << "[ERROR " << level << "] ";
      } else if (level > 0) {
        stream << "[Debug " << level << "] ";
      }

      char buffer[1024];
      std::snprintf(buffer, sizeof(buffer), fmt, convert_for_printf(args)...);

      stream << loc.file_name << ":" << loc.line << " " << buffer << std::endl;
    }
  }

  // Simple debug message with string
  static void debug_message_simple(int level, const std::string &message,
                                   const SourceLocation &loc) {
    if ((level >= MIN_DEBUG_LEVEL) && (level <= MAX_DEBUG_LEVEL)) {
      auto &stream = (level >= STDERR_LEVEL) ? std::cerr : std::cout;

      if (level >= STDERR_LEVEL) {
        stream << "[ERROR " << level << "] ";
      } else if (level > 0) {
        stream << "[Debug " << level << "] ";
      }

      stream << loc.file_name << ":" << loc.line << " " << message << std::endl;
    }
  }

  static void set_level(LogLevel level) { current_level = level; }

private:
  static constexpr const char *level_to_string(LogLevel level) {
    switch (level) {
    case LogLevel::TRACE:
      return "TRACE";
    case LogLevel::DEBUG:
      return "DEBUG";
    case LogLevel::INFO:
      return "INFO";
    case LogLevel::WARN:
      return "WARN";
    case LogLevel::ERROR:
      return "ERROR";
    case LogLevel::CRITICAL:
      return "CRITICAL";
    default:
      return "UNKNOWN";
    }
  }
};

// Initialize static member
inline LogLevel Logger::current_level = LogLevel::INFO;

} // namespace ARBD

// Debug.h compatibility layer
#ifdef DEBUGMSG
#define Debug(x) (x)

// Macro wrapper to automatically capture source location
#define DebugMsg(level, format, ...)                                           \
  ARBD::Logger::debug_message(level, format, ARBD::SourceLocation(),           \
                              ##__VA_ARGS__)

// Simple version for string compatibility
#define DebugMessage(level, message)                                           \
  ARBD::Logger::debug_message_simple(level, std::string(message),              \
                                     ARBD::SourceLocation())

#else
  // Make void functions when DEBUGMSG is not defined
#define Debug(x) static_cast<void>(0)
#define DebugMsg(level, format, ...) static_cast<void>(0)
#define DebugMessage(level, message) static_cast<void>(0)
#endif /* DEBUGMSG */

// Device-specific logging for CUDA and SYCL
#ifdef __CUDACC__ //(USE_CUDA)
  // CUDA device code - use printf
#define DEVICE_LOCATION_STRINGIFY(line) #line
#define DEVICE_LOCATION_TO_STRING(line) DEVICE_LOCATION_STRINGIFY(line)
#define DEVICE_CODE_LOCATION                                                   \
  __FILE__ "(" DEVICE_LOCATION_TO_STRING(__LINE__) ")"
#define LOGHELPER(TYPE, FMT, ...)                                              \
  printf("[%s] [%s]: " FMT "\n", TYPE, DEVICE_CODE_LOCATION, ##__VA_ARGS__)
#define LOGTRACE(...) LOGHELPER("TRACE", __VA_ARGS__)
#define LOGDEBUG(...) LOGHELPER("DEBUG", __VA_ARGS__)
#define LOGINFO(...) LOGHELPER("INFO", __VA_ARGS__)
#define LOGWARN(...) LOGHELPER("WARN", __VA_ARGS__)
#define LOGERROR(...) LOGHELPER("ERROR", __VA_ARGS__)
#define LOGCRITICAL(...) LOGHELPER("CRITICAL", __VA_ARGS__)
#elif defined(USE_SYCL) && defined(__SYCL_DEVICE_ONLY__)
  // SYCL device code - use printf (similar to CUDA)
#define DEVICE_LOCATION_STRINGIFY(line) #line
#define DEVICE_LOCATION_TO_STRING(line) DEVICE_LOCATION_STRINGIFY(line)
#define DEVICE_CODE_LOCATION                                                   \
  __FILE__ "(" DEVICE_LOCATION_TO_STRING(__LINE__) ")"
#define LOGHELPER(TYPE, FMT, ...)                                              \
  printf("[%s] [%s]: " FMT "\n", TYPE, DEVICE_CODE_LOCATION, ##__VA_ARGS__)
#define LOGTRACE(...) LOGHELPER("TRACE", __VA_ARGS__)
#define LOGDEBUG(...) LOGHELPER("DEBUG", __VA_ARGS__)
#define LOGINFO(...) LOGHELPER("INFO", __VA_ARGS__)
#define LOGWARN(...) LOGHELPER("WARN", __VA_ARGS__)
#define LOGERROR(...) LOGHELPER("ERROR", __VA_ARGS__)
#define LOGCRITICAL(...) LOGHELPER("CRITICAL", __VA_ARGS__)
#else
  // Host code macros - use improved logger with std::string support
#define LOGTRACE(...)                                                          \
  ARBD::Logger::log(ARBD::LogLevel::TRACE, ARBD::SourceLocation(), __VA_ARGS__)
#define LOGDEBUG(...)                                                          \
  ARBD::Logger::log(ARBD::LogLevel::DEBUG, ARBD::SourceLocation(), __VA_ARGS__)
#define LOGINFO(...)                                                           \
  ARBD::Logger::log(ARBD::LogLevel::INFO, ARBD::SourceLocation(), __VA_ARGS__)
#define LOGWARN(...)                                                           \
  ARBD::Logger::log(ARBD::LogLevel::WARN, ARBD::SourceLocation(), __VA_ARGS__)
#define LOGERROR(...)                                                          \
  ARBD::Logger::log(ARBD::LogLevel::ERROR, ARBD::SourceLocation(), __VA_ARGS__)
#define LOGCRITICAL(...)                                                       \
  ARBD::Logger::log(ARBD::LogLevel::CRITICAL, ARBD::SourceLocation(),          \
                    __VA_ARGS__)
#endif
