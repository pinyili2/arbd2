/*
 * ARBD exception handler
 * to handle the run-time exception.
 */
#include "ARBDException.h"
#include <limits>

std::string _ARBDException::sformat(const std::string &fmt, va_list &ap) {
    // Start with reasonable buffer size
    int size = 512;
    const int max_size = std::numeric_limits<int>::max() / 2;
    std::string str;
    
    while(size < max_size) {
        str.resize(size);
        va_list ap_copy;
        va_copy(ap_copy, ap);
        int n = vsnprintf(&str[0], size, fmt.c_str(), ap_copy);
        va_end(ap_copy);
        
        if (n < 0) {
            // Formatting error occurred
            return "Error formatting string";
        }
        
        if (n < size) {
            str.resize(n);
            return str;
        }
        
        size = n + 1;  // Exactly the size we need
    }
    
    return "Error: formatted string too long";
}

_ARBDException::_ARBDException(const std::string& location, ExceptionType type, const std::string &ss, ...) {

    _error = _ARBDException::type_to_str(type) + ": "; 
    va_list ap;
    va_start(ap, ss);
    _error += sformat(ss, ap); 
    va_end(ap);
    _error += " [" + location + "]";
}

const char* _ARBDException::what() const noexcept {
    return _error.c_str();
}

std::string _ARBDException::type_to_str(ExceptionType type) {
    switch (type) {
        case ExceptionType::Unspecified:     return "Unspecified Error";
        case ExceptionType::NotImplemented:   return "Not Implemented Error";
        case ExceptionType::Value:           return "Value Error";
        case ExceptionType::DivideByZero:    return "Divide By Zero Error";
        case ExceptionType::CudaRuntime:     return "CUDA Runtime Error";
        case ExceptionType::FileIo:          return "File IO Error";
        case ExceptionType::FileOpen:        return "File Open Error";
        default:
            return "Unknown Error Code (" + std::to_string(static_cast<int>(type)) + ")";
    }
}