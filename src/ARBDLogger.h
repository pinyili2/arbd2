// ARBDLogger.h
#pragma once

#include <format>
#include <iostream>
#include <source_location>
#include <string_view>
#include <chrono>
#include <iomanip>

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
    
    template<typename... Args>
    static void log(LogLevel level, 
                   const std::source_location& loc,
                   std::format_string<Args...> fmt, 
                   Args&&... args) {
        if (level < current_level) return;
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        auto& stream = (level >= LogLevel::ERROR) ? std::cerr : std::cout;
        
        stream << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S") << "] "
               << "[" << level_to_string(level) << "] "
               << std::format(fmt, std::forward<Args>(args)...)
               << " (" << loc.file_name() << ":" << loc.line() << ")"
               << std::endl;
    }
    
    static void set_level(LogLevel level) {
        current_level = level;
    }
    
private:
    static constexpr std::string_view level_to_string(LogLevel level) {
        switch (level) {
            case LogLevel::TRACE: return "TRACE";
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARN: return "WARN";
            case LogLevel::ERROR: return "ERROR";
            case LogLevel::CRITICAL: return "CRITICAL";
            default: return "UNKNOWN";
        }
    }
};

// Initialize static member
inline LogLevel Logger::current_level = LogLevel::INFO;

} // namespace ARBD

// Device-specific logging for CUDA and SYCL
#ifdef __CUDA_ARCH__
    // CUDA device code - use printf
    #define DEVICE_LOCATION_STRINGIFY(line) #line 
    #define DEVICE_LOCATION_TO_STRING(line) DEVICE_LOCATION_STRINGIFY(line)
    #define DEVICE_CODE_LOCATION __FILE__ "(" DEVICE_LOCATION_TO_STRING(__LINE__) ")"
    #define LOGHELPER(TYPE, FMT, ...) printf("[%s] [%s]: " FMT "\n", TYPE, DEVICE_CODE_LOCATION, ##__VA_ARGS__)
    #define LOGTRACE(...) LOGHELPER("TRACE", __VA_ARGS__)
    #define LOGDEBUG(...) LOGHELPER("DEBUG", __VA_ARGS__)
    #define LOGINFO(...) LOGHELPER("INFO", __VA_ARGS__)
    #define LOGWARN(...) LOGHELPER("WARN", __VA_ARGS__)
    #define LOGERROR(...) LOGHELPER("ERROR", __VA_ARGS__)
    #define LOGCRITICAL(...) LOGHELPER("CRITICAL", __VA_ARGS__)
#elif defined(__SYCL_DEVICE_ONLY__)
    // SYCL device code - use printf (similar to CUDA)
    #define DEVICE_LOCATION_STRINGIFY(line) #line 
    #define DEVICE_LOCATION_TO_STRING(line) DEVICE_LOCATION_STRINGIFY(line)
    #define DEVICE_CODE_LOCATION __FILE__ "(" DEVICE_LOCATION_TO_STRING(__LINE__) ")"
    #define LOGHELPER(TYPE, FMT, ...) printf("[%s] [%s]: " FMT "\n", TYPE, DEVICE_CODE_LOCATION, ##__VA_ARGS__)
    #define LOGTRACE(...) LOGHELPER("TRACE", __VA_ARGS__)
    #define LOGDEBUG(...) LOGHELPER("DEBUG", __VA_ARGS__)
    #define LOGINFO(...) LOGHELPER("INFO", __VA_ARGS__)
    #define LOGWARN(...) LOGHELPER("WARN", __VA_ARGS__)
    #define LOGERROR(...) LOGHELPER("ERROR", __VA_ARGS__)
    #define LOGCRITICAL(...) LOGHELPER("CRITICAL", __VA_ARGS__)
#else
    // Host code - use our C++20 logger
    #define LOGTRACE(...) ARBD::Logger::log(ARBD::LogLevel::TRACE, std::source_location::current(), __VA_ARGS__)
    #define LOGDEBUG(...) ARBD::Logger::log(ARBD::LogLevel::DEBUG, std::source_location::current(), __VA_ARGS__)
    #define LOGINFO(...) ARBD::Logger::log(ARBD::LogLevel::INFO, std::source_location::current(), __VA_ARGS__)
    #define LOGWARN(...) ARBD::Logger::log(ARBD::LogLevel::WARN, std::source_location::current(), __VA_ARGS__)
    #define LOGERROR(...) ARBD::Logger::log(ARBD::LogLevel::ERROR, std::source_location::current(), __VA_ARGS__)
    #define LOGCRITICAL(...) ARBD::Logger::log(ARBD::LogLevel::CRITICAL, std::source_location::current(), __VA_ARGS__)
#endif