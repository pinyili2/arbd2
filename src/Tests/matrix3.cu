#include <iostream>
#include <cstdio>

#include "../SignalManager.h"
#include "../Types.h"
#include <cuda.h>
#include <nvfunctional>

#include "type_name.h"

#include <catch2/catch_tostring.hpp>
namespace Catch {
    template<typename T, bool b1, bool b2>
    struct StringMaker<Matrix3_t<T,b1,b2>> {
        static std::string convert( Matrix3_t<T,b1,b2> const& value ) {
            return value.to_string();
        }
    };
}
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>



namespace Tests {

    template<typename Op_t, typename R, typename ...T>
    __global__ void op_kernel(R* result, T...in) {
	if (blockIdx.x == 0) {
	    *result = Op_t::op(in...);
	}
    }


    // In C++14, how can I unpack a tuple so its elements are passed to a function as arguments?
    template <typename F, typename Tuple, size_t... I>
    decltype(auto) apply_impl(F&& f, Tuple&& t, std::index_sequence<I...>)
    {
	return std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))...);
    }
    template <typename F, typename Tuple>
    decltype(auto) apply(F&& f, Tuple&& t)
    {
	using Indices = std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>;
	return apply_impl(std::forward<F>(f), std::forward<Tuple>(t), Indices{});
    }

    template <typename F, typename Tuple, size_t... I>
    decltype(auto) apply_each_impl(F&& f, Tuple&& t, std::index_sequence<I...>)
    {
	return std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))...);
    }
    template <typename F, typename Tuple>
    decltype(auto) apply_each(F&& f, Tuple&& t)
    {
	using Indices = std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>;
	return apply_each_impl(std::forward<F>(f), std::forward<Tuple>(t), Indices{});
    }

    template<typename T>
    void call_info(T t) {
	// INFO( type_name(t) );
	// UNSCOPED_INFO( type_name<T>() << " " << t );
    }
    // template<>
    // void call_info(double t) {
    // 	UNSCOPED_INFO( "double " << t );
    // }
    // template<>
    // void call_info(float t) {
    // 	UNSCOPED_INFO( "double " << t );
    // 	// INFO( type_name(t) );
    // 	INFO( "float" );
    // }
    // template<>
    // void call_info(int t) {
    // 	// INFO( type_name(t) );
    // 	INFO( "int" );
    // }
    
    template<typename Op_t, typename R, typename ...T>
    void run_trial( std::string name, R expected_result, T...args) {
	R *gpu_result_d, gpu_result, cpu_result;
	cpu_result = Op_t::op(args...);
	cudaMalloc((void **)&gpu_result_d, sizeof(R));
	
	op_kernel<Op_t, R, T...><<<1,1>>>(gpu_result_d, args...);
	cudaMemcpy(&gpu_result, gpu_result_d, sizeof(R), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	INFO( name );
	// INFO( type_name<T...>() );
	
	// auto tmp = std::make_tuple(args...);
	// for (int i = 0; i < sizeof(T...); ++i) {
	//     auto& arg = std::get<i>(tmp);
	//     CAPTURE( arg );
	// }
	// auto fn = [](auto a) { CAPTURE( a ); };

	// auto fn = [](auto a) { INFO( a->to_string() ); };
	// using expander = int[];
	// (void)expander{0, (void(call_info<T>(std::forward<T>(args))), 0)...};

	CAPTURE( cpu_result );
	CAPTURE( expected_result );
	
	REQUIRE( cpu_result == expected_result );
	CHECK( cpu_result == gpu_result );
    }
}

namespace Tests::Unary {
    template<typename R, typename T>
    struct NegateOp { HOST DEVICE static R op(T in) { return static_cast<R>(-in); } };

    template<typename R, typename T>
    struct NormalizedOp { HOST DEVICE static R op(T in) { return static_cast<R>(in.normalized()); } };

    namespace Matrix3 {
	template<typename R, typename T>
	struct DeterminantOp { HOST DEVICE static R op(T in) { return static_cast<R>(in.det()); } };

	template<typename R, typename T>
	struct NormalizeDetOp { HOST DEVICE static R op(T in) { return static_cast<R>(in.normalized().det()); } };

	// template<typename R, typename T>
	// struct NormalizeOp { HOST DEVICE static R op(T in) { return static_cast<R>(in.normalized()); } };


	
	template<typename R, typename T>
	struct InverseOp { HOST DEVICE static R op(T in) { return static_cast<R>(in.inverse()); } };

	template<typename R, typename T>
	struct TransposeOp { HOST DEVICE static R op(T in) { return static_cast<R>(in.transpose()); } };
    }
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

    namespace Matrix3 {
	template<typename R, typename T, typename U> 
	struct CrossOp { HOST DEVICE static R op(T a, U b) { return static_cast<R>(a.cross(b)); } };
    }
}

namespace Tests::Unary::Matrix3 {

    template<typename A, bool is_diag=false, bool check_diag=false>
    void run_tests() {
	using T = Matrix3_t<A, is_diag, check_diag>;
	// std::ostream test_info;
	INFO( "Testing " << type_name<T>() << " unary operators that" );
	{
	    using R = A;
	    INFO(  "    return " << type_name<R>() );
	    run_trial<DeterminantOp<R,T>,R,T>( "Testing determinant", R(6), T(1,2,3) );
	    run_trial<NormalizeDetOp<R,T>,R,T>( "Testing that normalized matrix has determinant == 1", R(1), T(1,1,1) );
	    // run_trial<NormalizeDetOp<R,T>,R,T>( "Testing that normalized matrix has determinant == 1", R(1), T(2,2,2) );
	    // run_trial<NormalizeDetOp<R,T>,R,T>( "Testing that normalized matrix has determinant == 1", R(1), T(1,2,3) );
	}

	{ // Test operators that return Matrix
	    using R = T;
	    INFO( "    return " << type_name<R>() );
	    run_trial<TransposeOp<R,T>,R,T>( "Testing transpose",
					     R(1,4,7,
					       2,5,8,
					       3,6,9),
					     T(1,2,3,
					       4,5,6,
					       7,8,9) );
	    run_trial<NegateOp<R,T>,R,T>( "Testing negation", R(-1,-2,-3), T(1,2,3) );
	    run_trial<InverseOp<R,T>,R,T>( "Testing inversion",
					   R(3.0/16, 0.25, -5.0/16,
					     0.25, 0, 0.25, -5.0/16,
					     0.25, 3.0/16),
					   T(1,2,-1,
					     2,1,2,
					     -1,2,1) );
	    run_trial<NormalizedOp<R,T>,R,T>( "Testing Matrix3_t<>.normalized()", R(1,1,1), T(1,1,1) );
	    run_trial<NormalizedOp<R,T>,R,T>( "Testing Matrix3_t<>.normalized()", R(1,1,1), T(2,2,2) );
	    run_trial<NormalizedOp<R,T>,R,T>( "Testing Matrix3_t<>.normalized()", R(A(1.0/6),A(2.0/6),A(3.0/6)), T(1,2,3) );
	}
    }
    TEST_CASE( "Check that Matrix3_t unary operations are identical on GPU and CPU", "[Tests::Unary::Matrix3]" ) {
	// INFO("Test case start");
	const bool is_diag = false;
	const bool check_diag = true;
	run_tests<double, is_diag, check_diag>();
	run_tests<float, is_diag, check_diag>();
	run_tests<int, is_diag, check_diag>();
    }
}

namespace Tests::Binary::Matrix3 {

    template<typename A, typename B, bool is_diag=false, bool check_diag=false>
    void run_tests() {
	using T = Matrix3_t<A, is_diag, check_diag>;

	{ // Test binary operators that return Matrix where U b is a scalar
	    using U = B;
	    using R = T;
	    run_trial<MultOp<R,T,U>,R,T,U>( "Scale", R(2,4,6), T(1,2,3), U(2) );
	}

	{ // Test binary operators that return Vector where U b is Vector
	    using U = Vector3_t<B>;
	    using R = U;
	    run_trial<MultOp<R,T,U>,R,T,U>( "Matrix.transform(Vector)",
					    R(1+4+9,
					      4+10+18,
					      7+16+27),
					    T(1,2,3,
					      4,5,6,
					      7,8,9),
					    U(1,2,3) );
	}

	{ // Test binary operators that return Vector and U b is Vector
	    using U = Vector3_t<B>;
	    using R = U;
	    run_trial<MultOp<R,T,U>,R,T,U>( "Matrix element multiplication",
					    R(1+4+9,
					      4+10+18,
					      7+16+27),
					    T(1,2,3,
					      4,5,6,
					      7,8,9),
					    U(1,2,3) );
	}

	{ // Test binary operators that return Matrix and U b is Matrix
	    using U = Matrix3_t<B,is_diag,check_diag>;
	    using R = std::common_type_t<T,U>;
	    run_trial<AddOp<R,T,U>,R,T,U>( "Matrix addition",
					    R(2,4,6,
					      8,10,12,
					      14,16,18),
					    T(1,2,3,
					      4,5,6,
					      7,8,9),
					    T(1,2,3,
					      4,5,6,
					      7,8,9));

	    run_trial<SubOp<R,T,U>,R,T,U>( "Matrix subtraction",
					    R(0,0,0),
					    T(1,2,3,
					      4,5,6,
					      7,8,9),
					    T(1,2,3,
					      4,5,6,
					      7,8,9) );

	    run_trial<MultOp<R,T,U>,R,T,U>( "Matrix transformation",
					    R(1+8+21, 2+10+24, 3+12+27,
					      4+20+42, 8+25+48, 12+30+54,
					      7+32+63, 14+40+72, 21+48+81),
					    T(1,2,3,
					      4,5,6,
					      7,8,9),
					    T(1,2,3,
					      4,5,6,
					      7,8,9) );
	}	
    }
    TEST_CASE( "Check that Matrix3_t binary operations are identical on GPU and CPU", "[Tests::Binary::Matrix3]" ) {
	// INFO("Test case start");
	const bool is_diag = false;
	const bool check_diag = true;
	run_tests<double,double, is_diag, check_diag>();
	// run_tests<float,double, is_diag, check_diag>();
	//run_tests<int,double, is_diag, check_diag>();

	//run_tests<double,float, is_diag, check_diag>();
	run_tests<float,float, is_diag, check_diag>();
	// run_tests<int,float, is_diag, check_diag>();

	//run_tests<double,int, is_diag, check_diag>();
	//run_tests<float,int, is_diag, check_diag>();

	// run_tests<int,int, is_diag, check_diag>();

    }
}
