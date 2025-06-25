cmake_minimum_required(VERSION 3.25.5 FATAL_ERROR)
set(PROJECT_NAME_BASE "arbd")
if(DEFINED PROJECT_NAME_SUFFIX)
    set(PROJECT_NAME "${PROJECT_NAME_BASE}${PROJECT_NAME_SUFFIX}")
else()
    set(PROJECT_NAME "${PROJECT_NAME_BASE}")
endif()

# Single project() call
if(USE_CUDA)
    if(USE_METAL)
        project("${PROJECT_NAME}" VERSION 1.2 LANGUAGES CXX CUDA OBJCXX)
    else()
        project("${PROJECT_NAME}" VERSION 1.2 LANGUAGES CXX CUDA)
    endif()
else()
    if(USE_METAL)
        project("${PROJECT_NAME}" VERSION 1.2 LANGUAGES CXX OBJCXX)
    else()
        project("${PROJECT_NAME}" VERSION 1.2 LANGUAGES CXX)
    endif()
endif()

# --- Project Options ---
option(USE_CUDA "Use CUDA" OFF)
option(USE_SYCL_ACPP "Use SYCL with AdaptiveCpp (acpp)" OFF)
option(USE_SYCL_ICPX "Use SYCL with Intel DPC++ (icpx)" OFF)
option(DEBUG_BUILD "Build with debug flags (overrides CMAKE_BUILD_TYPE)" OFF)
option(USE_NCCL "Use NCCL for single node GPU peer communication" OFF)
option(USE_NVTX "Build with NVTX profiling ranges" OFF)
option(USE_PYBIND "Build Python bindings" OFF)
option(USE_METAL "Use Apple Silicon Metal" OFF)

# --- Basic CMake Setup ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate compile_commands.json for clangd and other tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(USE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

# Set a default build type if none was specified
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
    STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# Compiler specific flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_FLAGS_RELEASE)
    string(REPLACE "-O2" "-O3" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  endif()
if(USE_CUDA AND CMAKE_CUDA_FLAGS_RELEASE)
    string(REPLACE "-O2" "-O3" CMAKE_CUDA_FLAGS_RELEASE ${CMAKE_CUDA_FLAGS_RELEASE})
  endif()
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    # string(REPLACE "-O2" "-O3" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE}) # If desired for Intel
endif()

# Metal specific setup
if(USE_METAL AND APPLE)
    # Enable Objective-C++ for .mm files
    set(CMAKE_OBJCXX_STANDARD 20)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
    set(CMAKE_OBJCXX_EXTENSIONS OFF)

    # Enable ARC (Automatic Reference Counting) for Objective-C++
    set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -fobjc-arc")
endif()

# --- SYCL Configuration (Explicit ACPP or ICPX) ---
if(USE_SYCL_ACPP AND USE_SYCL_ICPX)
    message(FATAL_ERROR "Both USE_SYCL_ACPP and USE_SYCL_ICPX are ON. Please choose only one SYCL implementation for this build.")
endif()

if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
    message(STATUS "SYCL support is requested.")
    add_definitions(-DUSE_SYCL) # Generic definition for SYCL API usage in C++

    if(NOT CMAKE_CXX_COMPILER)
        message(FATAL_ERROR "A SYCL option (USE_SYCL_ACPP or USE_SYCL_ICPX) is ON, but CMAKE_CXX_COMPILER is not set.
Please specify your SYCL compiler via -D CMAKE_CXX_COMPILER=<path_to_sycl_compiler>
Example for USE_SYCL_ICPX on Linux:      cmake -D CMAKE_CXX_COMPILER=icpx -D USE_SYCL_ICPX=ON ..
Example for USE_SYCL_ACPP on macOS:    cmake -D CMAKE_CXX_COMPILER=/opt/homebrew/bin/acpp -D USE_SYCL_ACPP=ON ..")
    endif()

    set(SYCL_COMPILER_CONFIGURED_CORRECTLY FALSE) # Flag to check if config matches expectation

    if(USE_SYCL_ACPP)
        message(STATUS "Configuring for SYCL with AdaptiveCpp (acpp).")
        add_definitions(-DPROJECT_USES_SYCL_ACPP) # Specific definition for acpp

        if(CMAKE_CXX_COMPILER MATCHES "acpp")
            message(STATUS "Verified AdaptiveCpp (acpp) compiler: ${CMAKE_CXX_COMPILER}")
            set(ACPP_TARGET_FLAGS "")
            if(CMAKE_SYSTEM_NAME STREQUAL "Darwin") # macOS
                message(STATUS "Setting AdaptiveCpp default target flags for macOS (CPU via OpenMP).")
                set(ACPP_TARGET_FLAGS "--acpp-targets=omp")
                # For M-series GPU using OpenCL: set(ACPP_TARGET_FLAGS "--acpp-targets=omp,opencl")
                # Ensure OpenCL dev headers/libs are available if using OpenCL backend.
            elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
                message(STATUS "Setting AdaptiveCpp default target flags for Linux (CPU via OpenMP).")
                set(ACPP_TARGET_FLAGS "--acpp-targets=omp")
                # For specific GPU backends on Linux:
                # set(ACPP_TARGET_FLAGS "--acpp-targets=omp,hip") # For AMD GPUs (if acpp built with ROCm/HIP backend)
                # set(ACPP_TARGET_FLAGS "--acpp-targets=omp,cuda") # For NVIDIA GPUs (if acpp built with CUDA backend)
            else() # Other OS
                message(WARNING "AdaptiveCpp on unknown OS ${CMAKE_SYSTEM_NAME}. Defaulting to OpenMP target.")
                set(ACPP_TARGET_FLAGS "--acpp-targets=omp")
            endif()

            if(NOT ACPP_TARGET_FLAGS STREQUAL "")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ACPP_TARGET_FLAGS}")
                message(STATUS "Applied AdaptiveCpp flags: ${ACPP_TARGET_FLAGS}")
            endif()
            # acpp generally enables SYCL mode automatically. No -fsycl needed.
            set(SYCL_COMPILER_CONFIGURED_CORRECTLY TRUE)
        else()
            message(WARNING "USE_SYCL_ACPP is ON, but CMAKE_CXX_COMPILER (${CMAKE_CXX_COMPILER}) does not appear to be 'acpp'.
Ensure CMAKE_CXX_COMPILER is set to your AdaptiveCpp executable path (e.g., /opt/homebrew/bin/acpp for macOS).")
        endif()
    endif()

    if(USE_SYCL_ICPX)
        message(STATUS "Configuring for SYCL with Intel DPC++ (icpx).")
        add_definitions(-DPROJECT_USES_SYCL_ICPX) # Specific definition for icpx

        # CMAKE_CXX_COMPILER_ID for icpx is typically "IntelLLVM".
        if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM" OR CMAKE_CXX_COMPILER MATCHES "icpx")
            message(STATUS "Verified Intel DPC++ (icpx) compiler: ${CMAKE_CXX_COMPILER}")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
            # Add any other DPC++ specific flags here if needed for icpx.
            # Example: set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-sycl-strict")
            set(SYCL_COMPILER_CONFIGURED_CORRECTLY TRUE)
            if(CMAKE_SYSTEM_NAME STREQUAL "Darwin") # macOS
                message(WARNING "Intel DPC++ (icpx) is not officially supported on macOS. Compilation might fail or be unstable.")
            endif()
        else()
            message(WARNING "USE_SYCL_ICPX is ON, but CMAKE_CXX_COMPILER (${CMAKE_CXX_COMPILER}, ID: ${CMAKE_CXX_COMPILER_ID})
does not appear to be Intel DPC++ (icpx). Ensure CMAKE_CXX_COMPILER is set to 'icpx'.")
        endif()
    endif()

    if(NOT SYCL_COMPILER_CONFIGURED_CORRECTLY) # If neither acpp nor icpx matched the selected option
         message(WARNING "A SYCL option was enabled, but the specified CMAKE_CXX_COMPILER does not match the expected SYCL compiler for that option.
Please review your -D CMAKE_CXX_COMPILER=... setting and the USE_SYCL_ACPP / USE_SYCL_ICPX CMake options.
Compilation will proceed, but may not use SYCL correctly.")
    endif()

endif()
# --- End SYCL Configuration ---

# --- Metal Configuration ---
if(USE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "Metal is only available on Apple platforms (macOS, iOS, tvOS, watchOS)")
    endif()

    message(STATUS "Metal support is enabled for Apple platforms.")
    add_definitions(-DUSE_METAL)

    # Find Metal framework
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

    if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
        message(STATUS "Found Metal framework: ${METAL_FRAMEWORK}")
        message(STATUS "Found Foundation framework: ${FOUNDATION_FRAMEWORK}")
    else()
        message(FATAL_ERROR "Metal or Foundation framework not found")
    endif()
endif()
# --- End Metal Configuration ---

## Set flags and definitions after project()
if(USE_CUDA)
  add_definitions(-DUSE_CUDA) # Kept for CUDA-specific sections in C++
endif()

if(DEBUG_BUILD)
  set(CMAKE_BUILD_TYPE Debug)
  message(STATUS "DEBUG_BUILD is ON: Build type set to Debug.")
endif()

if(USE_NVTX)
  add_definitions(-DUSE_NVTX)
endif()

if(USE_NCCL AND USE_CUDA)
  add_definitions(-DUSE_NCCL)
endif()


if(USE_CUDA)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
endif()
if(USE_CUDA)
    set(CMAKE_BUILD_RPATH "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")
endif()

add_subdirectory(src)
add_subdirectory(src/Tests)

add_executable("${PROJECT_NAME}" src/arbd.cpp)

if(USE_CUDA)
    target_sources("${PROJECT_NAME}" PRIVATE
        # src/SimManager.cu
        # src/useful.cu
    )
endif()

if(USE_PYBIND)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11/CMakeLists.txt")
    add_subdirectory(extern/pybind11)
    add_subdirectory(src/pybindings)
  else()
    message(WARNING "pybind11 directory or CMakeLists.txt not found at extern/pybind11.")
  endif()
endif()

target_link_libraries("${PROJECT_NAME}" PUBLIC "lib${PROJECT_NAME}")

if(USE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${PROJECT_NAME}
        PRIVATE
        CUDA::cuda_driver
        CUDA::cudart
        CUDA::curand
        CUDA::cublas)
endif()

if(USE_NCCL AND USE_CUDA)
  target_link_libraries("${PROJECT_NAME}" PUBLIC nccl)
endif()

if(USE_NVTX)
  if(USE_CUDA AND TARGET CUDA::nvToolsExt)
    target_link_libraries("${PROJECT_NAME}" PUBLIC CUDA::nvToolsExt)
  else()
    message(STATUS "NVTX linking might require manual setup if not using CUDA or CUDA::nvToolsExt target.")
  endif()
endif()

if(USE_METAL)
    target_link_libraries("${PROJECT_NAME}"
        PRIVATE
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
    )
    message(STATUS "Linked Metal and Foundation frameworks")
endif()
