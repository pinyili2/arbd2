#find_package(Catch2 3 REQUIRED)
Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.4.0 # or a later release
)

FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch) # contains some cmake macros that will help with discovery of tests
include(CTest)


# Common libraries and include directories for all tests
set(COMMON_TEST_LIBS "lib${PROJECT_NAME}" Catch2::Catch2WithMain)
if(USE_CUDA)
    list(APPEND COMMON_TEST_LIBS curand)
endif()
if(USE_NCCL)
    list(APPEND COMMON_TEST_LIBS nccl)
endif()

# Metal Tests
if(USE_METAL AND APPLE)
    add_executable(arbd_test_metal test_metal.mm unifiedbuffer_metal.mm)
    target_include_directories(arbd_test_metal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../)
    target_link_libraries(arbd_test_metal PRIVATE ${COMMON_TEST_LIBS} ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    catch_discover_tests(arbd_test_metal)
    install(TARGETS arbd_test_metal)
    message(STATUS "Added Metal test executable: arbd_test_metal")
elseif(USE_METAL)
    message(WARNING "USE_METAL is ON but not on Apple platform - Metal tests will be excluded")
endif()

# SYCL Tests
if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
    # Test for test_sycl.cpp
    add_executable(arbd_test_sycl test_sycl.cpp unifiedbuffer_sycl.cpp)
    target_include_directories(arbd_test_sycl PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(arbd_test_sycl PRIVATE ${COMMON_TEST_LIBS})
    
    # Fix C++ standard library linking for AdaptiveCpp
    if(USE_SYCL_ACPP)
        target_link_libraries(arbd_test_sycl PRIVATE stdc++)
        # Also link Boost libraries for AdaptiveCpp
        target_link_directories(arbd_test_sycl PRIVATE "${BOOST_LIB_DIR}")
    endif()
    
    # Instead of catch_discover_tests which causes bus errors on ACPP,
    # we'll add a single test that runs the entire executable
    add_test(NAME SYCL_Tests COMMAND arbd_test_sycl)
    
    # Set environment variables for SYCL test execution
    set_tests_properties(SYCL_Tests PROPERTIES
        ENVIRONMENT "ACPP_TARGETS=omp;SYCL_DEVICE_FILTER=cpu;ACPP_DEBUG=1"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src/Tests
    )
    
    install(TARGETS arbd_test_sycl)
    message(STATUS "Added SYCL test executable: arbd_test_sycl")

    # Test for test_sycl_omp.cpp
    add_executable(arbd_test_sycl_omp test_sycl_omp.cpp)
    target_include_directories(arbd_test_sycl_omp PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(arbd_test_sycl_omp PRIVATE ${COMMON_TEST_LIBS})
    
    # Fix C++ standard library linking for AdaptiveCpp
    if(USE_SYCL_ACPP)
        target_link_libraries(arbd_test_sycl_omp PRIVATE stdc++)
        target_link_directories(arbd_test_sycl_omp PRIVATE "${BOOST_LIB_DIR}")
    endif()
    
    # Add OMP test with environment variables
    add_test(NAME SYCL_OMP_Tests COMMAND arbd_test_sycl_omp)
    set_tests_properties(SYCL_OMP_Tests PROPERTIES
        ENVIRONMENT "ACPP_TARGETS=omp;SYCL_DEVICE_FILTER=cpu;ACPP_DEBUG=1"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src/Tests
    )
    
    install(TARGETS arbd_test_sycl_omp)
    message(STATUS "Added SYCL OMP test executable: arbd_test_sycl_omp")
endif()

# CUDA Tests
if(USE_CUDA)
    add_executable(arbd_test_cuda 
        matrix3.cu
        vector3_precision.cu
        bitmask.cu
        array.cu
    )
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
    target_include_directories(arbd_test_cuda PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(arbd_test_cuda PRIVATE ${COMMON_TEST_LIBS})
    catch_discover_tests(arbd_test_cuda)
    install(TARGETS arbd_test_cuda)
    message(STATUS "Added CUDA test executable: arbd_test_cuda")
endif()

if(USE_MPI)
    add_executable(arbd_test_mpi mpitest.cpp)
    target_include_directories(arbd_test_mpi PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(arbd_test_mpi PRIVATE ${COMMON_TEST_LIBS})
    
    # Ensure MPI test uses only CXX, not CUDA compiler
    set_target_properties(arbd_test_mpi PROPERTIES
        LINKER_LANGUAGE CXX
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )
    
    # Explicitly set this as a CXX-only target to avoid CUDA flag inheritance
    set_source_files_properties(mpitest.cpp PROPERTIES LANGUAGE CXX)
    
    # Fix C++ standard library linking issues (same as main executable)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_link_libraries(arbd_test_mpi PRIVATE -Wl,--whole-archive stdc++ -Wl,--no-whole-archive)
        target_link_libraries(arbd_test_mpi PRIVATE gcc_s gcc m pthread dl)
        target_link_options(arbd_test_mpi PRIVATE -Wl,--no-as-needed)
    endif()
    
    catch_discover_tests(arbd_test_mpi)
    install(TARGETS arbd_test_mpi)
    message(STATUS "Added MPI test executable: arbd_test_mpi")
endif()