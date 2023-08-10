#pragma once

#include "GPUManager.h"
#include <stdarg.h>  // For va_start, etc.
#include <memory>    // For std::unique_ptr
#include <cstring>

#include "type_name.h"

// Utility function used by types to return std::string using format syntax
inline std::string string_format(const std::string fmt_str, ...) {
    // from: https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf/8098080#8098080
    int final_n, n = ((int)fmt_str.size()) * 2; /* Reserve two times as much as the length of the fmt_str */
    std::unique_ptr<char[]> formatted;
    va_list ap;
    while(1) {
        formatted.reset(new char[n]); /* Wrap the plain char array into the unique_ptr */
        strcpy(&formatted[0], fmt_str.c_str());
        va_start(ap, fmt_str);
        final_n = vsnprintf(&formatted[0], n, fmt_str.c_str(), ap);
        va_end(ap);
        if (final_n < 0 || final_n >= n)
            n += abs(final_n - n + 1);
        else
            break;
    }
    return std::string(formatted.get());
}

#include "Types/Vector3.h"
#include "Types/Matrix3.h"

using Vector3 = Vector3_t<float>;
using Matrix3 = Matrix3_t<float,false>;

#include "Types/Bitmask.h"

#include "Types/Array.h"
using VectorArr = Array<Vector3>;
