#include <float.h>
#include <iostream>
#include <cstdio>

// #include "useful.h"
#include "SignalManager.h"
#include "Types.h"
#include <cuda.h>
#include <nvfunctional>

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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
    void print_it(const Vector3&& x) { x.print(); }
    template<> __host__ __device__
    void print_it(const Vector3& x) { x.print(); }
    
    template <typename T>
    void print_enable_if_value() {
	if (has_copy_to_cuda<T>::value) {
	    std::cout << "has_copy_to_cuda is true" << std::endl;
	} else {
	    std::cout << "has_copy_to_cuda is false" << std::endl;
	}
    }

    template<typename T>
    Array<T> allocate_array_host(size_t num) {
	Array<T> arr(num);
	return arr;
    }

    template<typename T>
    Array<T>* allocate_array_device(size_t num) {
	Array<T> arr(num);
	return arr.copy_to_cuda();
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
	gpuErrchk(cudaMalloc(&arr_d, sz));
	gpuErrchk(cudaMemcpy(arr_d, arr, sz, cudaMemcpyHostToDevice));
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
    HOST DEVICE void inline _copy_helper(size_t& idx, Array<T>* __restrict__ out, const Array<T>* __restrict__ inp) {
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
	gpuErrchk( cudaDeviceSynchronize() );
    }
    
    // Array<T> _copy_array_cuda(size_t num) {
    // 	Array<T> arr(num);
    // 	return arr;
    // }

    
    TEST_CASE( "Test Array assignment and copy_to_cuda", "[Array]" ) {
	{
	    // Creation and copy assignment
	    Array<Vector3> a = allocate_array_host<Vector3>(10);
	}

	{
	    // Allocation and deallocation
	    VectorArr a(10);
	    a[0] = Vector3(1);
	    // a[0].print();
	    // a[1].print();
	    a[3] = Vector3(3);
	    // a[3].print();

	    VectorArr* a_d = a.copy_to_cuda();
	    VectorArr b(0);
	    VectorArr* b_d = b.copy_to_cuda();
	    VectorArr a_d_h = a_d->copy_from_cuda(a_d);
	    VectorArr b_d_h = b_d->copy_from_cuda(b_d);
		    
	    // a_d_h[0].print();
	    // a_d_h[1].print();
	    // a_d_h[3].print();

	    REQUIRE( a[1] == a_d_h[1] );
	    REQUIRE( a[3] == a_d_h[3] );

	    VectorArr::remove_from_cuda(a_d);
	    VectorArr::remove_from_cuda(b_d);

	    print_enable_if_value<int>();  // Replace VectorArr with your actual type
	    print_enable_if_value<Vector3>();  // Replace VectorArr with your actual type
	    print_enable_if_value<VectorArr>();  // Replace VectorArr with your actual type
	    print_enable_if_value<Array<VectorArr>>();  // Replace VectorArr with your actual type
	    
	    // b_d_h[0].print();
	}
    }
    TEST_CASE( "Test Assigment and copying of Arrays of Arrays and copy_to_cuda", "[Array]" ) {
	{
	    // Allocation and deallocation
	    // printf("Creating v1(10)\n");
	    VectorArr v1(10);
	    for (int i = 0; i < v1.size(); ++i) {
		v1[i] = Vector3(i+1);
	    }
 	    // printf("Creating v2(20)\n");
	    VectorArr v2(20);
	    for (int i = 0; i < v2.size(); ++i) {
		v2[i] = Vector3(10*i+1);
	    }
	    
	    // printf("Creating a(2)\n");
	    Array<VectorArr> a(3);
	    a[0] = v1;
	    a[1] = v2;
	    // a[1] = std::move(v2);

	    Array<VectorArr>* a_d = a.copy_to_cuda();
	    Array<VectorArr> a_d_h = a_d->copy_from_cuda(a_d);
	    
	    
	    REQUIRE( a[0][1] == a_d_h[0][1] );
	    // REQUIRE( a[0][5] == a_d_h[0][5] );

	    a_d->remove_from_cuda(a_d);
	}
    }

    TEST_CASE( "Test sending Arrays", "[Array]" ) {
	{
	    // Allocation and deallocation
	    // printf("Creating v1(10)\n");
	    Resource loc = Resource{Resource::GPU,0};
	    
	    VectorArr v1(10);
	    for (int i = 0; i < v1.size(); ++i) {
		v1[i] = Vector3(i+1);
	    }
	    VectorArr v2(20);
	    for (int i = 0; i < v2.size(); ++i) {
		v2[i] = Vector3(10*i+1);
	    }
	    
	    Array<VectorArr> a(3);
	    a[0] = v1;
	    a[1] = v2;
	    // a[1] = std::move(v2);

	    Proxy<Array<VectorArr>> a_d = send(loc, a);
	    // Array<VectorArr> a_d_h = a_d->copy_from_cuda(a_d);
	    
	    // REQUIRE( a[0][1] == a_d_h[0][1] );
	    // REQUIRE( a[0][5] == a_d_h[0][5] );
	    printf("Removing...\n");
	    a.remove_from_cuda(a_d.addr); // TODO: generalize

	}
    }
    
    TEST_CASE( "Test Assigment and copying of Arrays of Arrays of Arrays", "[Array]" ) {
	{
	    // Allocation and deallocation
	    // printf("Creating v1(10)\n");
	    VectorArr v1(10);
	    for (int i = 0; i < v1.size(); ++i) {
		v1[i] = Vector3(i+1);
	    }
 	    // printf("Creating v2(20)\n");
	    VectorArr v2(20);
	    for (int i = 0; i < v2.size(); ++i) {
		v2[i] = Vector3(10*i+1);
	    }
	    
	    // printf("Creating a(3)\n");
	    Array<VectorArr> a(3);
	    a[0] = v1;
	    a[1] = v2;

	    Array<Array<VectorArr>> b(3);
	    b[0] = a;
	    b[2] = std::move(a);

	    Array<Array<VectorArr>>* b_d = b.copy_to_cuda();
	    Array<Array<VectorArr>> b_d_h = b_d->copy_from_cuda(b_d);
	    	    
	    REQUIRE( b[0][0][0] == b_d_h[0][0][0] );
	    b_d->remove_from_cuda(b_d);
	}
    }

    //Benchmark showing that Array<Vector3> performs similarly to plain array for device copy, at least 
    /*
    TEST_CASE( "Test performance copying Array vs plain arrays", "[Array]" ) {
	size_t num = 100000;
	float* inp3 = allocate_plain_array_device<float>(3*num);
	float* out3 = allocate_plain_array_device<float>(3*num);

	float* inp4 = allocate_plain_array_device<float>(4*num);
	float* out4 = allocate_plain_array_device<float>(4*num);

	float4* inpF4 = allocate_plain_array_device<float4>(num);
	float4* outF4 = allocate_plain_array_device<float4>(num);

	Array<Vector3>* inpV = allocate_array_device<Vector3>(num);
	Array<Vector3>* outV = allocate_array_device<Vector3>(num);

	// call_copy_kernel(3*num, out3, inp3);
	// call_copy_kernel(4*num, out4, inp4);
	// call_copy_kernel(num, outV, inpV);
	BENCHMARK("Call 3x num float copy") {
	    call_copy_kernel(3*num, out3, inp3);
	};
	BENCHMARK("Call num Vector3 copy") {
	    call_copy_kernel(num, outV, inpV);
	};
	BENCHMARK("Call num float4 copy") {
	    call_copy_kernel(num, outF4, inpF4);
	};
	BENCHMARK("Call 3x num float copy (repeat)") {
	    call_copy_kernel(3*num, out3, inp3);
	};
	BENCHMARK("Call 4x num float copy") {
	    call_copy_kernel(4*num, out4, inp4);
	};
	BENCHMARK("Call num Vector3 copy (repeat)") {
	    call_copy_kernel(num, outV, inpV);
	};
	BENCHMARK("Call num float4 copy (repeat)") {
	    call_copy_kernel(num, outF4, inpF4);
	};
    }
    // */
}
