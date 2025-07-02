#include "catch_boiler.h"
#include "Types/Types.h"

#include "catch2/catch_test_macros.hpp"

namespace Catch {
    template <>
    struct StringMaker<ARBD::Bitmask> {
        static std::string convert( ARBD::Bitmask const& value ) {
            return value.to_string();
        }
    };
}

namespace Tests::Bitmask {

    void run_tests() {
	using T = ARBD::Bitmask;
	INFO( "Testing " << ARBD::type_name<T>() << " functionally" );
	{
	    int i = 10;
	    T b = T(i);	// bitmask of length i
	    for (int j: {0,3}) {  // Remove invalid indices
		if (j < i) b.set_mask(j,1);
	    }

	    REQUIRE( b.to_string() == "1001000000" );

	    // Skip CUDA operations for now since Bitmask doesn't have copy_to_cuda
	    // TODO: Implement CUDA support for Bitmask if needed
	    
	    // Test equality
	    T b2(i);
	    b2.set_mask(0, 1);
	    b2.set_mask(3, 1);
	    REQUIRE( b == b2 );
	}
    }

    TEST_CASE( "Testing Bitmask for CPU consistency", "[Tests::Bitmask]" ) {
	run_tests();
    }

}
