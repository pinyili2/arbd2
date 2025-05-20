/*
 * ARBDException class handles the 
 * run-time exception.
 * Han-Yi Chou
 */

#pragma once

#include <string>
#include <cstdarg>
#include <exception>
#include "SignalManager.h"

//struct ExceptionType {
//public:
//    explicit constexpr ExceptionType(int value) noexcept : value{value} {};
//    constexpr operator int() const { return value; };
//private:
//    const int value;
//};

enum class ExceptionType {
    Unspecified = 0, // Renamed for clarity as an enum member
    NotImplemented = 1,
    Value = 2,
    DivideByZero = 3,
    CudaRuntime = 4, // Renamed to avoid conflict with potential CUDARuntimeError class name
    FileIo = 5,
    FileOpen = 6
};

class _ARBDException : public std::exception 
{
  private:
    std::string _error;
    std::string sformat(const std::string &fmt, va_list &ap); // Definition in .cpp
    static std::string type_to_str(ExceptionType type);      

  public:
    // UPDATED: Constructor signature uses the new enum class
    _ARBDException(const std::string& location, ExceptionType type, const std::string &ss, ...); 
    virtual const char* what() const noexcept; 
};

// #include "common_macros.h"
#ifdef CUDACC
#define Exception(...) throw _ARBDException(LOCATION, __VA_ARGS__)
#else
#define Exception(EXCPT_TYPE_ENUM_MEMBER,...) \
    printf("Runtime Exception at %s: ", LOCATION); \
    printf("%s " __VA_ARGS__ "\n", _ARBDException::type_to_str(EXCPT_TYPE_ENUM_MEMBER).c_str() );
#endif

// Use illegal instruction to abort; used in functions defined both in __host__ and __device__
#if 0
#define CudaException(...) \
    printf("Run-time exception occurs at %s: ", LOCATION); \
    printf(__VA_ARGS__); \
    asm("trap;");  /* Trigger hardware trap to abort execution */
#endif
