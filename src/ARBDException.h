#pragma once

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
    FileIoError = 5,
    FileOpenError = 6
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

} // namespace ARBD

#define ARBD_Exception(type, ...) \
    throw ARBD::Exception(type, std::source_location::current(), __VA_ARGS__)
