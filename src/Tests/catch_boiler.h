#include <iostream>
#include <cstdio>

#include "../SignalManager.h"
/* #include "../Types.h" */
#include "../GPUManager.h"
#include <cuda.h>
#include <nvfunctional>

#include "type_name.h"

/* #include <catch2/catch_tostring.hpp> */
/* namespace Catch { */
/*     template<typename T, bool b1, bool b2> */
/*     struct StringMaker<Matrix3_t<T,b1,b2>> { */
/*         static std::string convert( Matrix3_t<T,b1,b2> const& value ) { */
/*             return value.to_string(); */
/*         } */
/*     }; */
/* } */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace Tests {
    template<typename Op_t, typename R, typename ...T>
    __global__ void op_kernel(R* result, T...in) {
	if (blockIdx.x == 0) {
	    *result = Op_t::op(in...);
	}
    }
}

#define DEF_RUN_TRIAL \
namespace Tests {\
template<typename Op_t, typename R, typename ...T>\
    void run_trial( std::string name, R expected_result, T...args) {\
	R *gpu_result_d, gpu_result, cpu_result;\
	cpu_result = Op_t::op(args...);\
	cudaMalloc((void **)&gpu_result_d, sizeof(R));\
	\
	op_kernel<Op_t, R, T...><<<1,1>>>(gpu_result_d, args...);\
	cudaMemcpy(&gpu_result, gpu_result_d, sizeof(R), cudaMemcpyDeviceToHost);\
	cudaDeviceSynchronize();\
\
	INFO( name );\
\
	CAPTURE( cpu_result );\
	CAPTURE( expected_result );\
	\
	REQUIRE( cpu_result == expected_result );\
	CHECK( cpu_result == gpu_result );\
    }\
}

namespace Tests::Unary {
    template<typename R, typename T>
    struct NegateOp { HOST DEVICE static R op(T in) { return static_cast<R>(-in); } };

    template<typename R, typename T>
    struct NormalizedOp { HOST DEVICE static R op(T in) { return static_cast<R>(in.normalized()); } };

}

namespace Tests::Binary {
    // R is return type, T and U are types of operands
    template<typename R, typename T, typename U> 
    struct AddOp { HOST DEVICE static R op(T a, U b) { return static_cast<R>(a+b); } };
    template<typename R, typename T, typename U> 
    struct SubOp { HOST DEVICE static R op(T a, U b) { return static_cast<R>(a-b); } };
    template<typename R, typename T, typename U> 
    struct MultOp { HOST DEVICE static R op(T a, U b) { return static_cast<R>(a*b); } };
    template<typename R, typename T, typename U> 
    struct DivOp { HOST DEVICE static R op(T a, U b) { return static_cast<R>(a/b); } };

}
