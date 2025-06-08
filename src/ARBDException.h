#pragma once

/**
 * @file ARBDException.h
 * @brief Advanced exception handling system for the ARBD library with source location tracking and formatted messages.
 * 
 * This header provides a robust exception handling system that includes:
 * - Automatic source location tracking (file, line, function)
 * - Modern string formatting support (C++20)
 * - Type-safe exception categories
 * - Convenient factory functions for common exceptions
 * 
 * @example Basic Usage:
 * ```cpp
 * void your_function() {
 *    // Simplest way - just throw with a message
 *    throw ARBD::Exception(ARBD::ExceptionType::ValueError, "Something went wrong");
 *    
 *    // Or use the convenient macro (recommended)
 *    ARBD_Exception(ARBD::ExceptionType::ValueError, "Invalid value: {}", 42);
 * }
 * ```
 * @example Using Convenience Functions:
 * ```cpp
 * void your_function() {
 *    // Using convenience functions
 *    if (value < 0) {
 *        ARBD::throw_value_error("Expected positive value, got {}", negative_value);
 *    }
 *    if (!feature_implemented) {
 *        ARBD::throw_not_implemented("Feature {} is not yet implemented", feature_name);
 *    }
 * }
 * ```
 * @example Using the Macro:
 * ```cpp
 * void your_function() {
 *    // Using the macro with formatted message
 *    ARBD_Exception(ARBD::ExceptionType::ValueError, "Invalid value: {}", 42);
 * }
 * ```
 * 
 * @note The exception system automatically captures:
 * - Exception type and message
 * - Source file name
 * - Line number
 * - Function name
 * 
 * Available Exception Types:
 * - UnspecifiedError (0): Generic error type
 * - NotImplementedError (1): Feature or function not yet implemented
 * - ValueError (2): Invalid value or parameter
 * - DivideByZeroError (3): Mathematical division by zero
 * - CUDARuntimeError (4): CUDA-specific runtime errors
 * - SYCLRuntimeError (5): SYCL-specific runtime errors
 * - MetalRuntimeError (6): Metal-specific runtime errors
 * - FileIoError (7): General file I/O errors
 * - FileOpenError (8): Specific file opening errors
 * 
 * Advanced Usage with Formatting:
 * ```cpp
 * void process_data(int value, const std::string& name) {
 *     if (value < 0) {
 *         // Using C++20 formatting with multiple arguments
 *         ARBD_Exception(ARBD::ExceptionType::ValueError,
 *             "Invalid value {} for parameter '{}'. Must be non-negative.",
 *             value, name);
 *     }
 *     
 *     if (!feature_implemented) {
 *         // Using convenience function with formatting
 *         ARBD::throw_not_implemented("Feature '{}' planned for version {}.{}", 
 *             feature_name, major_version, minor_version);
 *     }
 * }
 * ```
 */

#include <string>
#include <exception>
#include <source_location>
#include <sstream>
#include <iostream>
#include <format> // C++20 feature

namespace ARBD {
enum class ExceptionType { 
    UnspecifiedError = 0,
    NotImplementedError = 1,
    ValueError = 2,
    DivideByZeroError = 3,
    CUDARuntimeError = 4,
    SYCLRuntimeError = 5,
    MetalRuntimeError = 6,
    FileIoError = 7,
    FileOpenError = 8
};

class Exception : public std::exception {
private:
    std::string _error_message;
    ExceptionType _type;
    std::source_location _location;
    
    static std::string type_to_str(ExceptionType type);

public:
    // Modern variadic template constructor
    template<typename... Args>
    Exception(
        ExceptionType type,
        const std::source_location& location,
        std::format_string<Args...> fmt,
        Args&&... args
    ) : _type(type), _location(location) {
        std::ostringstream oss;
        oss << type_to_str(type) << ": ";
        
        try {
            oss << std::format(fmt, std::forward<Args>(args)...);
        } catch (const std::format_error& e) {
            oss << "[Format error: " << e.what() << "]";
        }
        
        oss << " [" << _location.file_name()
            << "(" << _location.line()
            << ") '" << _location.function_name() << "']";
            
        _error_message = oss.str();
    }
    
    // Convenience constructor for simple messages
    Exception(
        ExceptionType type,
        std::string_view message,
        const std::source_location& location = std::source_location::current()
    ) : _type(type), _location(location) {
        std::ostringstream oss;
        oss << type_to_str(type) << ": " << message << " [" << _location.file_name()
            << "(" << _location.line()<< ") '" << _location.function_name() << "']";
        _error_message = oss.str();
    }

    virtual const char* what() const noexcept override { 
        return _error_message.c_str(); 
    }
    
    ExceptionType type() const noexcept { return _type; }
    const std::source_location& where() const noexcept { return _location; }
};

// Factory functions for common exceptions
template<typename... Args>
[[noreturn]] inline void throw_not_implemented(
    std::format_string<Args...> fmt, 
    Args&&... args,
    const std::source_location& loc = std::source_location::current()
) {
    throw Exception(ExceptionType::NotImplementedError, loc, fmt, std::forward<Args>(args)...);
}

template<typename... Args>
[[noreturn]] inline void throw_value_error(
    std::format_string<Args...> fmt, 
    Args&&... args,
    const std::source_location& loc = std::source_location::current()
) {
    throw Exception(ExceptionType::ValueError, loc, fmt, std::forward<Args>(args)...);
}

template<typename... Args>
[[noreturn]] inline void throw_cuda_error(
    std::format_string<Args...> fmt, 
    Args&&... args,
    const std::source_location& loc = std::source_location::current()
) {
    throw Exception(ExceptionType::CUDARuntimeError, loc, fmt, std::forward<Args>(args)...);
}

template<typename... Args>
[[noreturn]] inline void throw_metal_error(
    std::format_string<Args...> fmt, 
    Args&&... args,
    const std::source_location& loc = std::source_location::current()
) {
    throw Exception(ExceptionType::MetalRuntimeError, loc, fmt, std::forward<Args>(args)...);
}

} // namespace ARBD

#define ARBD_Exception(type, ...) \
    throw ARBD::Exception(type, std::source_location::current(), __VA_ARGS__)
