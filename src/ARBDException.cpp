#include "ARBDException.h"
#include <limits>    
#include <vector>  
#include <sstream>  
#include <cstdarg>  
#include <cstdio>    
#include <source_location> 

namespace ARBD {

std::string Exception::sformat(const std::string &fmt_str, va_list &ap) {
    int size = 512;
    const int max_size = 32768; // Practical limit
    std::string str;
    
    while(size < max_size) {
        str.resize(size);
        va_list ap_copy;
        va_copy(ap_copy, ap);
        // Use &str[0] in C++11 and later as string data is contiguous
        int n = vsnprintf(&str[0], size, fmt_str.c_str(), ap_copy);
        va_end(ap_copy);
        
        if (n < 0) {
            // vsnprintf error
            return "[sformat error]";
        }
        if (n < size) { // Content fit
            str.resize(n);
            return str;
        }
        // Buffer too small, new size is n + 1 (for null terminator)
        size = n + 1;  
    }
    return "[Error: formatted string too long in sformat]";
}

std::string Exception::type_to_str(ExceptionType type) {
    switch (type) {
        case ExceptionType::UnspecifiedError:    return "Unspecified Error";
        case ExceptionType::NotImplementedError: return "Not Implemented Error";
        case ExceptionType::ValueError:          return "Value Error";
        case ExceptionType::DivideByZeroError:   return "Divide By Zero Error";
        case ExceptionType::CUDARuntimeError:    return "CUDA Runtime Error";
        case ExceptionType::FileIoError:         return "File IO Error";
        case ExceptionType::FileOpenError:       return "File Open Error";
        default:
            return "Unknown Error Code (" + std::to_string(static_cast<int>(type)) + ")";
    }
}

Exception::Exception(
    ExceptionType type,
    const char* message_format, 
    const std::source_location& location,
    ...
) : _error_message() { 
    std::ostringstream location_info;
    location_info << location.file_name()
                  << "(" << location.line()
                  << ") '" << location.function_name() << "'";

    std::string user_message;
    if (message_format) {
        va_list ap;
        va_start(ap, location); 
        user_message = sformat(message_format, ap);
        va_end(ap);
    }
    
    _error_message = type_to_str(type) + ": " + user_message + " [" + location_info.str() + "]";
}

const char* Exception::what() const noexcept {
    return _error_message.c_str();
}

} // namespace arbd