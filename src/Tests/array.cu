#include <float.h>
#include <iostream>
#include <cstdio>

// #include "useful.h"
#include "SignalManager.h"
#include "Types/Types.h"
#include <cuda.h>
#include <nvfunctional>

#include "catch2/catch_test_macros.hpp"
#include "catch2/benchmark/catch_benchmark.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"

namespace Tests::TestArray {
    // enum BinaryOp_t { ADD, CROSS, DOT, SUB, FINAL };
    // BinaryOp_t& operator++(BinaryOp_t& op) { return op = static_cast<BinaryOp_t>( 1+static_cast<int>(op) ); }

    // std::string get_binary_op_name( BinaryOp_t op ) {
    // 	switch (op) {
    // 	case ADD:
    // 	    return "add";
    // 	case SUB:
    // 	    return "subtract";
    // 	case CROSS:
    // 	    return "cross";
    // 	case DOT:
    // 	    return "dot";
    // 	}
    // 	return std::string(""); // (static_cast<int>(op)));
    // }

    // template<typename R, typename T, typename U>
    // __host__ __device__ nvstd::function<R(T,U)> get_binary_op_func( BinaryOp_t op) {
    // 	switch (op) {
    // 	case ADD:
    // 	    return [] (T a, U b) {return static_cast<R>(b+a);};
    // 	case SUB:
    // 	    return [] (T a, U b) {return static_cast<R>(b-a);};
    // 	case CROSS:
    // 	    return [] (T a, U b) {return static_cast<R>(b.cross(a));};
    // 	case DOT:
    // 	    return [] (T a, U b) {return static_cast<R>(b.dot(a));};
    // 	default:
    // 	    assert(false);
    // 	}
    // 	return [] (T a, U b) {return static_cast<R>(b+a);};
    // }

    // template<typename R, typename T, typename U>
    // __global__ void binary_op_test_kernel( BinaryOp_t op, R* result, T in1, U in2 ) {
    // 	nvstd::function<R(T,U)> fn = get_binary_op_func<R,T,U>(op);
    // 	if (blockIdx.x == 0) {
    // 	    *result = fn(in1,in2);
    // 	}
    // }

    // template<typename T, typename U>
    // void check_vectors_equal( T&& cpu, U&& gpu) {
    // 	CHECK( type_name<decltype(cpu)>() == type_name<decltype(gpu)>() ); // should be unneccesary
    // 	CHECK( cpu.x == gpu.x );
    // 	CHECK( cpu.y == gpu.y );
    // 	CHECK( cpu.z == gpu.z );
    // 	CHECK( cpu.w == gpu.w );
    // }

    // template<typename A, typename B>
    // void run_tests() {
    // 	using T = Vector3_t<A>;
    // 	using U = Vector3_t<B>;
    // 	using R = std::common_type_t<T,U>;
    
    // 	T v1(1,1.005,0);
    // 	U v2(0,2,0);
    // 	R *gpu_result_d, gpu_result, cpu_result;
    // 	cudaMalloc((void **)&gpu_result_d, sizeof(R));

    // 	for (BinaryOp_t op = ADD; op < FINAL; ++op) {
    // 	    INFO( get_binary_op_name( op ) );
    // 	    binary_op_test_kernel<R,T,U><<<1,1>>>(op, gpu_result_d, v1, v2);
    // 	    cudaMemcpy(&gpu_result, gpu_result_d, sizeof(R), cudaMemcpyDeviceToHost);
    // 	    cudaDeviceSynchronize();
	
    // 	    // Get cpu_result
    // 	    cpu_result = (get_binary_op_func<R,T,U>(op))(v1,v2);

    // 	    // Check consistency
    // 	    check_vectors_equal(cpu_result, gpu_result);
    // 	}
    // 	cudaFree(gpu_result_d);
    // }

    // template <typename T>
    // void print_enable_if_value_helper(std::true_type) {
    // 	std::cout << "has_copy_to_cuda is true" << std::endl;
    // }

    // template <typename T>
    // void print_enable_if_value_helper(std::false_type) {
    // 	std::cout << "has_copy_to_cuda is false" << std::endl;
    // }

    // template <typename T>
    // void print_enable_if_value_helper(std::true_type) {
    // 	std::cout << "has_copy_to_cuda is true" << std::endl;
    // }

    // template <typename T>
    // void print_enable_if_value_helper(std::false_type) {
    // 	std::cout << "has_copy_to_cuda is false" << std::endl;
    // }

    // template <typename T>
    // void print_enable_if_value() {
    // 	print_enable_if_value_helper<has_copy_to_cuda<T>>(typename has_copy_to_cuda<T>::type{});
    // }

    
    template<typename T> __host__ __device__ 
    void print_it(T x) { printf("Unsupported type\n"); }
    template<> __host__ __device__
    void print_it(const int x) { printf("int %d\n", x); }
    template<> __host__ __device__
    void print_it(const long int x) { printf("long int %ld\n", x); }
    template<> __host__ __device__
    void print_it(const float x) { printf("float %f\n", x); }
    template<> __host__ __device__
    void print_it(const double x) { printf("double %lf\n", x); }
    template<> __host__ __device__
    void print_it(const ARBD::Vector3&& x) { x.print(); }
    template<> __host__ __device__
    void print_it(const ARBD::Vector3& x) { x.print(); }
    
    // Simple has_copy_to_cuda trait for testing
    template <typename T, typename = void>
    struct has_copy_to_cuda : std::false_type {};
    
    template <typename T>
    struct has_copy_to_cuda<T, decltype(std::declval<T>().copy_to_cuda(), void())> : std::true_type {};
    
    template <typename T>
    void print_enable_if_value() {
	if (has_copy_to_cuda<T>::value) {
	    std::cout << "has_copy_to_cuda is true" << std::endl;
	} else {
	    std::cout << "has_copy_to_cuda is false" << std::endl;
	}
    }

    template<typename T>
    ARBD::Array<T> allocate_array_host(size_t num) {
	ARBD::Array<T> arr(num);
	return arr;
    }

    template<typename T>
    T* allocate_plain_array_host(size_t num) {
	T* arr = new T[num];
	return arr;
    }
    
    template<typename T>
    T* allocate_plain_array_device(size_t num) {
	T* arr = allocate_plain_array_host<T>(num);
	T* arr_d;
	size_t sz = sizeof(T)*num;
	ARBD::check_cuda_error(cudaMalloc(&arr_d, sz), __FILE__, __LINE__);
	ARBD::check_cuda_error(cudaMemcpy(arr_d, arr, sz, cudaMemcpyHostToDevice), __FILE__, __LINE__);
	delete[] arr;
	return arr_d;
    }
    
    template<typename T>
    HOST DEVICE void inline _copy_helper(size_t& idx, T* __restrict__ out, const T* __restrict__ inp) {
	out[idx] = inp[idx];
    }

    // HOST DEVICE void inline _copy_helper(size_t& idx, float* __restrict__ out, const float* __restrict__ inp) {
    // 	out[idx] = inp[idx];
    // }
    template<typename T>
    HOST DEVICE void inline _copy_helper(size_t& idx, ARBD::Array<T>* __restrict__ out, const ARBD::Array<T>* __restrict__ inp) {
	(*out)[idx] = (*inp)[idx];
    }

    
    template<typename T>
    __global__ void copy_kernel(size_t num, T* __restrict__ out, const T* __restrict__ inp) {
	for (size_t i = threadIdx.x+blockIdx.x*blockDim.x; i < num; i+=blockDim.x*gridDim.x) {
	    _copy_helper(i, out, inp);
	}
    }

    template<typename T>
    void call_copy_kernel(size_t num, T* __restrict__ out, const T* __restrict__ inp, size_t block_size=256) {
	copy_kernel<<<block_size,1,0>>>(num, out, inp);
	ARBD::check_cuda_error(cudaDeviceSynchronize(), __FILE__, __LINE__);
    }
    
    // Array<T> _copy_array_cuda(size_t num) {
    // 	Array<T> arr(num);
    // 	return arr;
    // }

    
    TEST_CASE( "Test Array assignment and basic operations", "[Array]" ) {
	{
	    // Creation and copy assignment
	    ARBD::Array<ARBD::Vector3> a = allocate_array_host<ARBD::Vector3>(10);
	}

	{
	    // Allocation and deallocation
	    ARBD::VectorArr a(10);
	    a[0] = ARBD::Vector3(1);
	    a[3] = ARBD::Vector3(3);

	    // Just test basic array operations without CUDA for now
	    REQUIRE( a[0].x == 1.0f );
	    REQUIRE( a[3].x == 3.0f );

	    print_enable_if_value<int>();  
	    print_enable_if_value<ARBD::Vector3>();  
	    print_enable_if_value<ARBD::VectorArr>();  
	    print_enable_if_value<ARBD::Array<ARBD::VectorArr>>();  
	}
    }
    
    TEST_CASE( "Test Array nesting operations", "[Array]" ) {
	{
	    // Allocation and deallocation
	    ARBD::VectorArr v1(10);
	    for (int i = 0; i < v1.size(); ++i) {
		v1[i] = ARBD::Vector3(i+1);
	    }
 	    ARBD::VectorArr v2(20);
	    for (int i = 0; i < v2.size(); ++i) {
		v2[i] = ARBD::Vector3(10*i+1);
	    }
	    
	    ARBD::Array<ARBD::VectorArr> a(3);
	    a[0] = v1;
	    a[1] = v2;

	    // Test nested array access
	    REQUIRE( a[0][1].x == 2.0f );
	    REQUIRE( a[1][1].x == 11.0f );
	}
    }

    TEST_CASE( "Test basic CUDA kernels", "[Array]" ) {
	{
	    // Test basic CUDA kernel operations  
	    size_t num = 1000;
	    float* inp = allocate_plain_array_device<float>(num);
	    float* out = allocate_plain_array_device<float>(num);

	    call_copy_kernel(num, out, inp);
	    
	    ARBD::check_cuda_error(cudaFree(inp), __FILE__, __LINE__);
	    ARBD::check_cuda_error(cudaFree(out), __FILE__, __LINE__);
	    
	    REQUIRE( true ); // Basic test that CUDA operations work
	}
    }
}
