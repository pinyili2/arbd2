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

## TODO, add a macro to add these projects, build with CUDA
add_executable("arbd_tests"
test.cpp)

# Conditionally add Metal test file
if(USE_METAL AND APPLE)
    target_sources(arbd_tests PRIVATE
        test_metal.mm
    )
    target_include_directories(arbd_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../
    )
    message(STATUS "Added Metal tests (test_metal.mm) to arbd_tests")
    # Link against the main library to get access to Metal manager
    target_link_libraries(arbd_tests PRIVATE libarbd)
    message(STATUS "Added Metal framework support to arbd_tests")
elseif(USE_METAL)
    message(WARNING "USE_METAL is ON but not on Apple platform - Metal tests will be excluded")
endif()

# Conditionally add SYCL test file
if(USE_SYCL)
    target_sources("arbd_tests" PRIVATE test_sycl.cpp)
    message(STATUS "Added SYCL tests to arbd_tests")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

target_include_directories(arbd_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries("arbd_tests" PRIVATE "lib${PROJECT_NAME}" Catch2::Catch2WithMain)

## Add optional libraries
if(USE_CUDA)
  target_link_libraries(arbd_tests PUBLIC curand)
  target_sources("arbd_tests" PRIVATE
    matrix3.cu
    vector3_precision.cu
    bitmask.cu
    array.cu)
endif()
if(USE_NCCL)
  target_link_libraries(arbd_tests PUBLIC nccl)
endif()
if(USE_NVTX)
  target_link_libraries(arbd_tests PUBLIC nvToolsExt)
endif()

if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
  # SYCL runtime linking is usually handled automatically by the SYCL compiler
  # but we ensure the SYCL definitions are available
  target_compile_definitions(arbd_tests PRIVATE PROJECT_USES_SYCL)
  if(USE_SYCL_ACPP)
    target_compile_definitions(arbd_tests PRIVATE PROJECT_USES_SYCL_ACPP)
    # Link OpenMP for AdaptiveCpp
    if(OpenMP_CXX_FOUND)
      target_link_libraries(arbd_tests PRIVATE OpenMP_Custom)
      message(STATUS "Linked custom OpenMP to arbd_tests for AdaptiveCpp backend")
    endif()
  endif()
  if(USE_SYCL_ICPX)
    target_compile_definitions(arbd_tests PRIVATE PROJECT_USES_SYCL_ICPX)
  endif()
  message(STATUS "Added SYCL support to arbd_tests")
endif()

if(USE_METAL)
  # Metal runtime linking is handled automatically on macOS
  # but we ensure the Metal definitions are available
  target_compile_definitions(arbd_tests PRIVATE USE_METAL)
  find_library(METAL_FRAMEWORK Metal)
  find_library(FOUNDATION_FRAMEWORK Foundation)
  if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
    target_link_libraries(arbd_tests PRIVATE ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    message(STATUS "Added Metal framework support to arbd_tests")
  else()
    message(WARNING "Metal or Foundation framework not found")
  endif()
endif()

## catch_discover_tests("${PROJECT_NAME}_TESTS")

catch_discover_tests(arbd_tests)
install(TARGETS arbd_tests)
