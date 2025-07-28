#pragma once

#include "ARBDException.h"
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#define FMT_HEADER_ONLY
#include "../extern/fmt/include/fmt/format.h"

#ifndef MIN_DEBUG_LEVEL
#define MIN_DEBUG_LEVEL 0
#endif
#ifndef MAX_DEBUG_LEVEL
#define MAX_DEBUG_LEVEL 10
#endif
#ifndef STDERR_LEVEL
#define STDERR_LEVEL 5
#endif

namespace ARBD {

enum class LogLevel { TRACE = 0, DEBUG = 1, INFO = 2, WARN = 3, ERROR = 4, CRITICAL = 5 };

class Logger {
  public:
	static LogLevel current_level;

	// Host-side logging methods
	static void log(LogLevel level, const SourceLocation& loc, std::string_view message) {
		if (level < current_level)
			return;

		auto now = std::chrono::system_clock::now();
		auto time_t = std::chrono::system_clock::to_time_t(now);
		auto& stream = (level >= LogLevel::ERROR) ? std::cerr : std::cout;

		stream << "[" << get_level_name(level) << "] " << loc.file_name << ":" << loc.line << " "
			   << message << std::endl;
	}

	template<typename... Args>
	static void
	log(LogLevel level, const SourceLocation& loc, std::string_view fmt_str, Args&&... args) {
		if (level < current_level)
			return;

		try {
			std::string formatted =
				fmt::format(fmt::runtime(std::string(fmt_str)), std::forward<Args>(args)...);
			log(level, loc, formatted);
		} catch (const fmt::format_error& e) {
			log(LogLevel::ERROR, loc, fmt::format("FORMAT_ERROR: {} ({})", fmt_str, e.what()));
		}
	}

	static void set_level(LogLevel level) {
		current_level = level;
	}

  private:
	static const char* get_level_name(LogLevel level) {
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

inline LogLevel Logger::current_level = LogLevel::INFO;

} // namespace ARBD

// Debug.h compatibility layer
#ifdef DEBUGMSG
#define Debug(x) (x)
#define DebugMsg(level, ...)                                           \
	ARBD::Logger::log(static_cast<ARBD::LogLevel>(std::min(level, 5)), \
					  ARBD::SourceLocation(),                          \
					  __VA_ARGS__)
#define DebugMessage(level, message)                                   \
	ARBD::Logger::log(static_cast<ARBD::LogLevel>(std::min(level, 5)), \
					  ARBD::SourceLocation(),                          \
					  std::string_view(message))
#else
#define Debug(x) static_cast<void>(0)
#define DebugMsg(level, ...) static_cast<void>(0)
#define DebugMessage(level, message) static_cast<void>(0)
#endif

// Device/Host separation
#if defined(__CUDACC__) || (defined(USE_SYCL) && defined(__SYCL_DEVICE_ONLY__))
// Device-specific logging
#ifdef __CUDACC__
#define DEVICE __device__
#define HOST_DEVICE __host__ __device__
#else
#define DEVICE
#define HOST_DEVICE
#endif

namespace ARBD {
namespace detail {

// Device-only print functions
DEVICE inline void print_log_value(const char* val) {
	printf("%s", val);
}
DEVICE inline void print_log_value(char* val) {
	printf("%s", val);
}
DEVICE inline void print_log_value(int val) {
	printf("%d", val);
}
DEVICE inline void print_log_value(long val) {
	printf("%ld", val);
}
DEVICE inline void print_log_value(long long val) {
	printf("%lld", val);
}
DEVICE inline void print_log_value(unsigned val) {
	printf("%u", val);
}
DEVICE inline void print_log_value(unsigned long val) {
	printf("%lu", val);
}
DEVICE inline void print_log_value(unsigned long long val) {
	printf("%llu", val);
}
DEVICE inline void print_log_value(float val) {
	printf("%f", val);
}
DEVICE inline void print_log_value(double val) {
	printf("%lf", val);
}
DEVICE inline void print_log_value(void* val) {
	printf("%p", val);
}

DEVICE inline void print_log_args() {}

template<typename T, typename... Rest>
DEVICE inline void print_log_args(const T& first, const Rest&... rest) {
	print_log_value(first);
	if constexpr (sizeof...(rest) > 0) {
		printf(" ");
	}
	print_log_args(rest...);
}

} // namespace detail
} // namespace ARBD

#define DEVICE_LOCATION_STRINGIFY(line) #line
#define DEVICE_LOCATION_TO_STRING(line) DEVICE_LOCATION_STRINGIFY(line)
#define DEVICE_CODE_LOCATION __FILE__ "(" DEVICE_LOCATION_TO_STRING(__LINE__) ")"

// Device-side logging macros
#define LOGHELPER(TYPE, FMT, ...)                          \
	do {                                                   \
		printf("[%s] [%s]: ", TYPE, DEVICE_CODE_LOCATION); \
		ARBD::detail::print_log_args(__VA_ARGS__);         \
		printf("\n");                                      \
	} while (0)

#define LOGTRACE(...) LOGHELPER("TRACE", __VA_ARGS__)
#define LOGDEBUG(...) LOGHELPER("DEBUG", __VA_ARGS__)
#define LOGINFO(...) LOGHELPER("INFO", __VA_ARGS__)
#define LOGWARN(...) LOGHELPER("WARN", __VA_ARGS__)
#define LOGERROR(...) LOGHELPER("ERROR", __VA_ARGS__)
#define LOGCRITICAL(...) LOGHELPER("CRITICAL", __VA_ARGS__)

#else
// Host-side logging macros - only active when NOT compiling device code
#ifndef __CUDA_ARCH__
#define LOGTRACE(...) ARBD::Logger::log(ARBD::LogLevel::TRACE, ARBD::SourceLocation(), __VA_ARGS__)
#define LOGDEBUG(...) ARBD::Logger::log(ARBD::LogLevel::DEBUG, ARBD::SourceLocation(), __VA_ARGS__)
#define LOGINFO(...) ARBD::Logger::log(ARBD::LogLevel::INFO, ARBD::SourceLocation(), __VA_ARGS__)
#define LOGWARN(...) ARBD::Logger::log(ARBD::LogLevel::WARN, ARBD::SourceLocation(), __VA_ARGS__)
#define LOGERROR(...) ARBD::Logger::log(ARBD::LogLevel::ERROR, ARBD::SourceLocation(), __VA_ARGS__)
#define LOGCRITICAL(...) \
	ARBD::Logger::log(ARBD::LogLevel::CRITICAL, ARBD::SourceLocation(), __VA_ARGS__)
#else
// Device-side: use simplified printf-based logging
#define LOGTRACE(...)                \
	printf("[TRACE]: " __VA_ARGS__); \
	printf("\n")
#define LOGDEBUG(...)                \
	printf("[DEBUG]: " __VA_ARGS__); \
	printf("\n")
#define LOGINFO(...)                \
	printf("[INFO]: " __VA_ARGS__); \
	printf("\n")
#define LOGWARN(...)                \
	printf("[WARN]: " __VA_ARGS__); \
	printf("\n")
#define LOGERROR(...)                \
	printf("[ERROR]: " __VA_ARGS__); \
	printf("\n")
#define LOGCRITICAL(...)                \
	printf("[CRITICAL]: " __VA_ARGS__); \
	printf("\n")
#endif

#endif
