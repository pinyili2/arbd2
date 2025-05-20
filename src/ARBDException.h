#pragma once

#include <string>
#include <cstdarg>
#include <exception>
#include <source_location> // C++20 requirement

enum class ExceptionType { 
    UnspecifiedError = 0,
    NotImplementedError = 1,
    ValueError = 2,
    DivideByZeroError = 3,
    CUDARuntimeError = 4,
    FileIoError = 5,
    FileOpenError = 6
};
//usage: if (some_error_condition) {
//    ARBD_Exception(ExceptionType::ValueError, "The input value %d is invalid.", input_value);}

class _ARBDException : public std::exception {
  private:
    std::string _error_message; // Changed from _error for clarity
    std::string sformat(const std::string &fmt, va_list &ap);
    static std::string type_to_str(ExceptionType type);

  public:

    _ARBDException(
        ExceptionType type,
        const char* message_format, 
        const std::source_location& location = std::source_location::current(),
        ...
    );

    virtual const char* what() const noexcept override; 
};


#define ARBD_Exception(type, fmt_str, ...) \
    throw _ARBDException(type, fmt_str, std::source_location::current(), ##__VA_ARGS__)
