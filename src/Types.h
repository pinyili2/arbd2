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
	// float4 r;
	// r.x = x+o.x; r.y = y+o.y; r.z = z+o.z; r.w = w+o.w;
	// return r;
    };
    float4 operator*(const float&& s) {
	return float4(x*s,y*s,z*s,w*s);
    };
    
    float x,y,z,w;
};
#endif


#include "Types/Vector3.h"
#include "Types/Vector3.h"

using Vector3 = Vector3_t<float>;
using Matrix3 = Vector3_t<float>; /* TODO: FIX */
