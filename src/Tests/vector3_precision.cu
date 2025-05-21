#include <float.h>
#include <iostream>
#include <cstdio>

// #include "useful.h"
#include "SignalManager.h"
#include "../Types/Types.h"
#include <cuda.h>
#include <nvfunctional>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../type_name.h"

namespace Tests::Vector3 {
    enum BinaryOp_t { ADD, CROSS, DOT, SUB, FINAL };
    BinaryOp_t& operator++(BinaryOp_t& op) { return op = static_cast<BinaryOp_t>( 1+static_cast<int>(op) ); }

    std::string get_binary_op_name( BinaryOp_t op ) {
	switch (op) {
	case ADD:
	    return "add";
	case SUB:
	    return "subtract";
	case CROSS:
	    return "cross";
	case DOT:
	    return "dot";
	}
	return std::string(""); // (static_cast<int>(op)));
    }

    template<typename R, typename T, typename U>
    __host__ __device__ nvstd::function<R(T,U)> get_binary_op_func( BinaryOp_t op) {
	switch (op) {
	case ADD:
	    return [] (T a, U b) {return static_cast<R>(b+a);};
	case SUB:
	    return [] (T a, U b) {return static_cast<R>(b-a);};
	case CROSS:
	    return [] (T a, U b) {return static_cast<R>(b.cross(a));};
	case DOT:
	    return [] (T a, U b) {return static_cast<R>(b.dot(a));};
	default:
	    assert(false);
	}
	return [] (T a, U b) {return static_cast<R>(b+a);};
    }

    template<typename R, typename T, typename U>
    __global__ void binary_op_test_kernel( BinaryOp_t op, R* result, T in1, U in2 ) {
	nvstd::function<R(T,U)> fn = get_binary_op_func<R,T,U>(op);
	if (blockIdx.x == 0) {
	    *result = fn(in1,in2);
	}
    }

    template<typename T, typename U>
    void check_vectors_equal( T&& cpu, U&& gpu) {
	CHECK( type_name<decltype(cpu)>() == type_name<decltype(gpu)>() ); // should be unneccesary
	CHECK( cpu.x == gpu.x );
	CHECK( cpu.y == gpu.y );
	CHECK( cpu.z == gpu.z );
	CHECK( cpu.w == gpu.w );
    }

    template<typename A, typename B>
    void run_tests() {
	using T = Vector3_t<A>;
	using U = Vector3_t<B>;
	using R = std::common_type_t<T,U>;
    
	T v1(1,1.005,0);
	U v2(0,2,0);
	R *gpu_result_d, gpu_result, cpu_result;
	cudaMalloc((void **)&gpu_result_d, sizeof(R));

	for (BinaryOp_t op = ADD; op < FINAL; ++op) {
	    INFO( get_binary_op_name( op ) );
	    binary_op_test_kernel<R,T,U><<<1,1>>>(op, gpu_result_d, v1, v2);
	    cudaMemcpy(&gpu_result, gpu_result_d, sizeof(R), cudaMemcpyDeviceToHost);
	    cudaDeviceSynchronize();
	
	    // Get cpu_result
	    cpu_result = (get_binary_op_func<R,T,U>(op))(v1,v2);

	    // Check consistency
	    check_vectors_equal(cpu_result, gpu_result);
	}
	cudaFree(gpu_result_d);
    }

    TEST_CASE( "Check that Vector3_t binary operations are identical on GPU and CPU", "[Vector3]" ) {
	// INFO("Test case start");
    
	run_tests<int,int>();
	run_tests<float,float>();
	run_tests<double,double>();

	run_tests<float,double>();
	run_tests<double,float>();
	run_tests<int,double>();
    }
}
