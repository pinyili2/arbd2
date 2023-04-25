#pragma once

#ifdef __CUDACC__
    #define HOST __host__
    #define DEVICE __device__
#else
    #define HOST
    #define DEVICE
#endif

#if defined(__CUDACC__) // NVCC
   #define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
  #define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
  #define MY_ALIGN(n) __declspec(align(n))
#else
  #error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#else
struct MY_ALIGN(16) float4 {
    float4() : x(0), y(0), z(0), w(0) {};
    float4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {};
    float4 operator+(const float4&& o) {
	return float4(x+o.x,y+o.y,z+o.z,w+o.w);
    };
    float4 operator*(const float&& s) {
	return float4(x*s,y*s,z*s,w*s);
    };
    
    float x,y,z,w;
};
#endif

#include <stdarg.h>  // For va_start, etc.
#include <memory>    // For std::unique_ptr
#include <cstring>

// from: https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf/8098080#8098080
inline std::string string_format(const std::string fmt_str, ...) {
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
