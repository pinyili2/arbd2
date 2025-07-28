#pragma once

/**
 * @file ARBDLogger.h
 * @brief Unified logging system combining ARBDLogger and Debug.h functionality
 *
 * This header provides a comprehensive logging solution that merges:
 * - Modern C++20 ARBDLogger with fmt-style formatting and string_view
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
 * LOGTRACE("This is a trace message: {}", 42);
 * LOGDEBUG("This is a debug message: {}", "hello world");
 * LOGINFO("This is an info message: {} + {} = {}", 3, 4, 7);
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
 * // Format arguments (fmt style)
 * DebugMsg(2, "Formatted debug: value = {}", 123);
 * DebugMsg(3, "Multiple args: {} and {} make {}", "foo", "bar", "foobar");
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
 *     LOGINFO("Running on GPU thread {}", threadIdx.x);  // Uses printf
 * }
 *
 * // SYCL kernel code
 * queue.submit([&](handler& h) {
 *     h.parallel_for(range, [=](id<1> idx) {
 *         LOGDEBUG("SYCL work-item {}", idx[0]);  // Uses printf
 *     });
 * });
 * @endcode
 *
 * @note When DEBUGMSG is not defined, all Debug.h macros become no-ops
 * @note Device code automatically uses printf-based logging
 * @note Host code uses iostreams with timestamps for compatibility
 */

#include "ARBDException.h"
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>   // For std::forward
#define FMT_HEADER_ONLY
#include "../extern/fmt/include/fmt/format.h"

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
public:
  static LogLevel current_level;

  // Handles simple messages (no format args)
  static void log(LogLevel level, const SourceLocation &loc,
                  std::string_view message) {
    if (level < current_level)
      return;

    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto &stream = (level >= LogLevel::ERROR) ? std::cerr : std::cout;

    stream << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S") << "] "
           << "[" << level_to_string(level) << "] " << message << " ("
           << loc.file_name << ":" << loc.line << ")" << std::endl;
  }

  // Handles fmt-style format strings with arguments
  template <typename... Args>
  static void log(LogLevel level, const SourceLocation &loc,
                  const std::string_view fmt_str,
                  Args &&...args) requires(sizeof...(Args) > 0) {
    if (level < current_level)
      return;

    std::string formatted_message;
    try {
      // Use fmt::format with string conversion to avoid compile-time format string validation
      std::string fmt_str_copy(fmt_str);
      formatted_message = fmt::format(fmt::runtime(fmt_str_copy), std::forward<Args>(args)...);
    } catch (const fmt::format_error& e) {
      // Fallback to show the format string and error
      formatted_message = fmt::format("FORMAT_ERROR: {} ({})", fmt_str, e.what());
    }
    log(level, loc, std::string_view(formatted_message));
  }

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

public:
  /**
   * @brief Debug message function with numeric levels (from Debug.h)
   */
  static void debug_message_simple(int level, std::string_view message,
                                   const SourceLocation &loc) {
    if ((level >= MIN_DEBUG_LEVEL) && (level <= MAX_DEBUG_LEVEL)) {
      auto &stream = (level >= STDERR_LEVEL) ? std::cerr : std::cout;
      if (level >= STDERR_LEVEL) {
        stream << "[ERROR " << level << "] ";
      } else if (level > 0) {
        stream << "[Debug " << level << "] ";
      }
      stream << loc.file_name << ":" << loc.line << " " << message
             << std::endl;
    }
  }

  template <typename... Args>
  static void debug_message(int level, std::string_view fmt_str,
                            const SourceLocation &loc,
                            Args &&...args) requires(sizeof...(Args) > 0) {
    if ((level >= MIN_DEBUG_LEVEL) && (level <= MAX_DEBUG_LEVEL)) {
      auto &stream = (level >= STDERR_LEVEL) ? std::cerr : std::cout;
      if (level >= STDERR_LEVEL) {
        stream << "[ERROR " << level << "] ";
      } else if (level > 0) {
        stream << "[Debug " << level << "] ";
      }
      std::string formatted_message;
      try {
        // Use fmt::format with string conversion to avoid compile-time format string validation
        std::string fmt_str_copy(fmt_str);
        formatted_message = fmt::format(fmt::runtime(fmt_str_copy), std::forward<Args>(args)...);
      } catch (const fmt::format_error& e) {
        // Fallback to show the format string and error
        formatted_message = fmt::format("FORMAT_ERROR: {} ({})", fmt_str, e.what());
      }
      stream << loc.file_name << ":" << loc.line << " " << formatted_message
             << std::endl;
    }
  }

  static void set_level(LogLevel level) { current_level = level; }

private:
  /**
   * @brief Debug message function with numeric levels (from Debug.h)
   */
  static void debug_message(int level, const SourceLocation &loc,
                            std::string_view message) {
    debug_message_simple(level, message, loc);
  }
};

// Initialize static member
inline LogLevel Logger::current_level = LogLevel::INFO;

} // namespace ARBD

// Debug.h compatibility layer
#ifdef DEBUGMSG
#define Debug(x) (x)

// Macro wrapper to automatically capture source location
#define DebugMsg(level, ...)                                                   \
  ARBD::Logger::debug_message(level, ARBD::SourceLocation(), __VA_ARGS__)

// Simple version for string compatibility
#define DebugMessage(level, message)                                           \
  ARBD::Logger::debug_message_simple(level, std::string_view(message),         \
                                     ARBD::SourceLocation())

#else
  // Make void functions when DEBUGMSG is not defined
#define Debug(x) static_cast<void>(0)
#define DebugMsg(level, ...) static_cast<void>(0)
#define DebugMessage(level, message) static_cast<void>(0)
#endif /* DEBUGMSG */

#if defined(__CUDACC__) || (defined(USE_SYCL) && defined(__SYCL_DEVICE_ONLY__))
// Device-specific logging for CUDA and SYCL
#ifdef __CUDACC__
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace ARBD {
namespace detail {

// Overloaded helper functions to print a single value using the correct
// printf format specifier. You can add more overloads for types like
// 'long', 'unsigned int', 'const void*', etc., as needed.
DEVICE inline void print_log_value(const char* val) { printf("%s", val); }
DEVICE inline void print_log_value(char* val) { printf("%s", val); }
DEVICE inline void print_log_value(int val) { printf("%d", val); }
DEVICE inline void print_log_value(long val) { printf("%ld", val); }
DEVICE inline void print_log_value(long long val) { printf("%lld", val); }
DEVICE inline void print_log_value(unsigned val) { printf("%u", val); }
DEVICE inline void print_log_value(unsigned long val) { printf("%lu", val); }
DEVICE inline void print_log_value(unsigned long long val) { printf("%llu", val); }
DEVICE inline void print_log_value(float val) { printf("%f", val); }
DEVICE inline void print_log_value(double val) { printf("%lf", val); }
DEVICE inline void print_log_value(void* val) { printf("%p", val); }
DEVICE inline void print_log_value(std::string_view val) {
  // Use the "%.*s" printf format specifier to print a string of a given length.
  // This is safe for string_view which may not be null-terminated.
  printf("%.*s", static_cast<int>(val.length()), val.data());
}

DEVICE inline void print_log_args() {}

template <typename T, typename... Rest>
DEVICE inline void print_log_args(const T& first, const Rest&... rest) {
print_log_value(first);
// Add a space only if there are more arguments to print
if constexpr (sizeof...(rest) > 0) {
  printf(" ");
}
print_log_args(rest...);
}



} // namespace detail
} // namespace ARBD

// Define file/line location macros for device code
#define DEVICE_LOCATION_STRINGIFY(line) #line
#define DEVICE_LOCATION_TO_STRING(line) DEVICE_LOCATION_STRINGIFY(line)
#define DEVICE_CODE_LOCATION __FILE__ "(" DEVICE_LOCATION_TO_STRING(__LINE__) ")"

// The core device-side LOGHELPER.
// It prints the log metadata, then unpacks and prints all arguments using the helpers.
// The original format string is ignored for argument parsing but can be printed for context.
#define LOGHELPER(TYPE, FMT, ...)                                     \
  do {                                                                \
    printf("[%s] [%s]: ", TYPE, DEVICE_CODE_LOCATION);                \
    ARBD::detail::print_log_args(__VA_ARGS__);                        \
    printf("\n");                                                     \
  } while (0)

// Define the user-facing macros for device code
#define LOGTRACE(...) LOGHELPER("TRACE", __VA_ARGS__)
#define LOGDEBUG(...) LOGHELPER("DEBUG", __VA_ARGS__)
#define LOGINFO(...) LOGHELPER("INFO", __VA_ARGS__)
#define LOGWARN(...) LOGHELPER("WARN", __VA_ARGS__)
#define LOGERROR(...) LOGHELPER("ERROR", __VA_ARGS__)
#define LOGCRITICAL(...) LOGHELPER("CRITICAL", __VA_ARGS__)
#else
  // Host code macros - use improved logger with fmt formatting
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
