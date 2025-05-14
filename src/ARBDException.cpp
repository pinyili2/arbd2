/*
 * ARBD exception handler
 * to handle the run-time exception.
 */
#include "ARBDException.h"

std::string _ARBDException::sformat(const std::string &fmt, va_list &ap) 
{
    int size = 512;
    std::string str;
    while(1) 
    {
        str.resize(size);
        va_list ap_copy;
        va_copy(ap_copy, ap);
        int n = vsnprintf((char*)str.c_str(), size, fmt.c_str(), ap_copy);
        va_end(ap_copy);
        if (n > -1 && n < size) 
        {
            str.resize(n);
            return str;
        }
        if(n > -1)
            size = n + 1;
        else
            size *= 2;
    }
    return str;
}

_ARBDException::_ARBDException(const std::string& location, const ExceptionType type, const std::string &ss, ...)
{
        _error = _ARBDException::type_to_str(type) + ": ";
	va_list ap;
	va_start(ap, ss);
	_error += sformat(ss, ap);
	va_end(ap);
        _error += " ["+location+"]";
}

const char* _ARBDException::what() const noexcept
{
    return _error.c_str();
}
