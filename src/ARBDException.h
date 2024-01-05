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

enum ExceptionType {
    UnspeficiedError,
    NotImplementedError,
    ValueError,
    DivideByZeroError,
    CUDARuntimeError,
    FileIoError,
    FileOpenError
};

class _ARBDException : public std::exception 
{
    
  private:
    std::string _error;
    std::string sformat(const std::string &fmt, va_list &ap);
    static std::string type_to_str(ExceptionType type) {
	switch (type) {
	case UnspeficiedError:
	    return "Error";
	case NotImplementedError:
	    return "NotImplementedError";
	}
	return "UnkownError";
    }
    
  public:
    _ARBDException(const std::string& location, const ExceptionType type, const std::string &ss, ...);
    // compile with c++11 !!!
    virtual const char* what() const noexcept;
};

// #include "common_macros.h"
#define S1(x) #x
#define S2(x) S1(x)
#define LOCATION __FILE__ "(" S2(__LINE__)")"

#define Exception(...) throw _ARBDException(LOCATION, __VA_ARGS__)
//use illegal instruction to abort; used in functions defined both in __host__ and __device__
#if 0
#define CudaException(...) \
printf("Run-time exception occurs at %s: ", LOCATION); \
printf(__VA_ARGS__);
//TODO I want to add asm("trap;") but the compiling does not work
asm("trap;");
#endif
