#include "catch_boiler.h"
#include "Types/Types.h"

#include <catch2/catch_tostring.hpp>
namespace Catch {
    template <>
    struct StringMaker<Bitmask> {
        static std::string convert( Bitmask const& value ) {
            return value.to_string();
        }
    };
}

DEF_RUN_TRIAL

namespace Tests::Bitmask {

    void run_tests() {
	using T = ::Bitmask;
	// std::ostream test_info;
	INFO( "Testing " << type_name<T>() << " functionally" );
	{
	    using R = T;
	    int i = 10;
	    T b = T(i);	// bitmask of length i
	    for (int j: {0,3,10,19}) {
		if (j < i) b.set_mask(j,1);
	    }

	    REQUIRE( b.to_string() == "1001000000" );

	    T* b_d = b.copy_to_cuda();
	    cudaDeviceSynchronize();

	    T b2 = b.copy_from_cuda(b_d);
	    cudaDeviceSynchronize();
	    REQUIRE( b == b2 );

	    b.remove_from_cuda(b_d);
	    cudaDeviceSynchronize();

	}
    }

    TEST_CASE( "Testing Bitmask for GPU/CPU consistency", "[Tests::Bitmask]" ) {
	const bool check_diag = true;
	run_tests();
    }

}
