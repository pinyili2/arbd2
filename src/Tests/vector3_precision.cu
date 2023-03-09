#include <float.h>
#include <iostream>
#include <cstdio>

// #include "useful.h"
#include "../SignalManager.h"
#include "../Types.h"
#include <cuda.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "type_name.h"


// Test kernel function
// namespace TestVectorOps {

// ChatGPT("Add doxygen to each function, class and struct, where R represents a return parameter, T and U")

template<typename R, typename T, typename U>
HOST DEVICE R cross_vectors(T&& v1, U&& v2) {
    return v1.cross(v2);
}

template<typename F, typename R, typename T, typename U>
__global__ void binary_op_test_kernel( F fn, R* result, T in1, U in2 ) {
    if (blockIdx.x == 0) {
	*result = (*fn)(in1,in2);
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

template<typename R, typename T, typename U>
struct TestOp {
    std::string name;
    typename R (*)(T,U) func;
};


TEST_CASE( "Check that Vector3_t binary operations are identical on GPU and CPU", "[Vector3Dcross]" ) {
    // INFO("Test case start");
    using T = Vector3_t<float>;
    using U = Vector3_t<double>;
    using R = std::common_type_t<T,U>;
    
    T v1(1,0,0);
    U v2(0,2,0);
    R *gpu_result_d, gpu_result, cpu_result;

    
    std::vector<TestOp> ops;
    ops.push_back(TestOp("add", [] __host__ __device__ (T a, U b){ return static_cast<R>(b+a); }; ));
    ops.push_back(TestOp("cross", [] __host__ __device__ (T a, U b){ return a.cross(b); }; ));
    // auto sub = [] __host__ __device__ (T& a, U& b){ return a.cross(b); };
    // auto dot = [] __host__ __device__ (T& a, U& b){ return a.cross(b); };
    auto cross = [] __host__ __device__ (T a, U b){ return a.cross(b); };

    using L = decltype(add);
    std::vector<R(*)(T,U)> lambdas;
    lambdas.push_back(add);
    lambdas.push_back(cross);
    
    // Get_result
    cudaMalloc((void **)&gpu_result_d, sizeof(R));

    for (auto& op: ops) {
	INFO(op.name);
	binary_op_test_kernel<<<1,1>>>(op.func, gpu_result_d, v1, v2);
	cudaMemcpy(&gpu_result, gpu_result_d, sizeof(R), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	// Get cpu_result
	cpu_result = fn(v1,v2); // rvalue version failed on cpu, but works on gpu

	// Check consistency
	check_vectors_equal(cpu_result, gpu_result);
    }
}

// int main(int argc, char* argv[]) {
//     return 0;
// }

// int main(int argc, char* argv[]) {
//     SignalManager::manage_segfault();
//     Vector3_t<float>  v1(1,0,0);
//     Vector3_t<double> v2(0,2,0);
//     v2.y = 6.62606957e-44;
    
//     std::cout << "test_vector3.cpp" << std::endl;
//     // // std::cout << v1.to_char() << std::endl;
    
//     // std::cout << v1.to_char() << " x " << v2.to_char() <<
//     // 	" = " << (v1.cross(v2)).to_char() << std::endl;
//     // std::cout << (-v2).to_char() << " x " << v1.to_char() <<
//     // 	" = " << (-v2.cross(v1)).to_char() << std::endl;

//     // std::cout.precision(17);
//     // std::cout << sizeof(v2.cross(v1).z) << " " << sizeof(v1.cross(v2).z) << std::endl;
//     // std::cout << (v2.cross(v1).z) << " " << (v1.cross(v2).z) << std::endl;
//     // std::cout << float(6.62606957e-44) << std::endl;
//     // std::cout << " ... done" << std::endl;

//     // test_kernel<<<1,1>>>(v1,v2);
//     // cudaDeviceSynchronize();
//     return 0;
// }

// #include <iostream>

// // Since C++ 11
// // template<typename T>
// // using func_t = T (*) (T, T);

// template<typename R, typename T, typename U>
// using func_t = R (*) (T, U);

// template <typename R, typename T, typename U> 
// __host__ __device__ R add_func (T x, U y)
// {
//     return x + y;
// }

// template <typename T> 
// __host__ __device__ T mul_func (T x, T y)
// {
//     return x * y;
// }

// // Required for functional pointer argument in kernel function
// // Static pointers to device functions
// template <typename R, typename T, typename U> 
// __device__ func_t<R,T,U> p_add_func = add_func<R,T,U>;
// // template <typename T> 
// // __device__ func_t<T> p_mul_func = mul_func<T>;


// template <typename R, typename T, typename U> 
// __global__ void kernel(func_t<R,T,U> op, T * d_x, U * d_y, R * result)
// {
//     *result = (*op)(*d_x, *d_y);
// }

// template <typename T, typename U> 
// void mytest(T x, U y)
// {
//     using R = std::common_type_t<T,U>;
//     func_t<R,T,U> h_add_func;
// //    func_t<T> h_mul_func;

//     T * d_x;
//     U * d_y;
//     cudaMalloc(&d_x, sizeof(T));
//     cudaMalloc(&d_y, sizeof(U));
//     cudaMemcpy(d_x, &x, sizeof(T), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_y, &y, sizeof(U), cudaMemcpyHostToDevice);

//     R result;
//     R * d_result, * h_result;
//     cudaMalloc(&d_result, sizeof(R));
//     h_result = &result;

//     // Copy device function pointer to host side
//     cudaMemcpyFromSymbol(&h_add_func, p_add_func<R,T,U>, sizeof(func_t<R,T,U>));
//     // cudaMemcpyFromSymbol(&h_mul_func, p_mul_func<T>, sizeof(func_t<T>));

//     kernel<R,T,U><<<1,1>>>(h_add_func, d_x, d_y, d_result);
//     cudaDeviceSynchronize();
//     cudaMemcpy(h_result, d_result, sizeof(R), cudaMemcpyDeviceToHost);
//     CHECK( result == add_func<R,T,U>(x,y) );
    
//     // kernel<T><<<1,1>>>(h_mul_func, d_x, d_y, d_result);
//     // cudaDeviceSynchronize();
//     // cudaMemcpy(h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
//     // CHECK( result == mul_func(x,y) );
// }

// TEST_CASE( "Check that Vector3_t binary operations are identical on GPU and CPU", "[Vector3Dcross]" ) {
//     INFO("TEST START");
//     mytest<int,float>(2.05, 10.00);
//     // mytest<float>(2.05, 10.00);
//     // mytest<double>(2.05, 10.00);
// }

// */
