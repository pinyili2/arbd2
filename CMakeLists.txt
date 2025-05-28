
cmake_minimum_required(VERSION 3.25 FATAL_ERROR) 

option(USE_CUDA "Use CUDA" ON)
option(USE_SYCL "Use SYCL (with DPC++)" OFF)


# Use target_compile_features instead of setting standards directly
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_FLAGS_RELEASE)
    string(REPLACE "-O2" "-O3" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  endif()
  if(CMAKE_CUDA_FLAGS_RELEASE) # Only if USE_CUDA is ON
    string(REPLACE "-O2" "-O3" CMAKE_CUDA_FLAGS_RELEASE ${CMAKE_CUDA_FLAGS_RELEASE})
  endif()
endif()

## OPTIONS
option(DEBUG_BUILD "Build with debug flags (overrides CMAKE_BUILD_TYPE)" OFF) # Renamed from DEBUG to avoid conflict
option(USE_NCCL "Use NCCL for single node GPU peer communication" OFF)
option(USE_NVTX "Build with NVTX profiling ranges" OFF)
option(USE_LOGGER "Build with SPDLog" ON)
option(USE_PYBIND "Build Python bindings" OFF) # Added option for pybind


# --- SYCL Configuration ---
if(USE_SYCL)
    # Check if the current C++ compiler is likely DPC++
    # For DPC++ (icpx or dpcpp), CMAKE_CXX_COMPILER_ID is "IntelLLVM".
    # For older dpcpp, it might also be "Intel".
    if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        message(STATUS "Configuring SYCL with Intel DPC++ compiler: ${CMAKE_CXX_COMPILER}")
        # Add -fsycl flag for SYCL compilation.
        # This is typically added to CMAKE_CXX_FLAGS if most .cpp files might include SYCL headers,
        # or to specific targets using target_compile_options.
        # For simplicity in a "minimal" setup, we add it globally here if USE_SYCL is ON.
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
        add_definitions(-DUSE_SYCL)

        # DPC++ usually links the SYCL runtime library automatically when -fsycl is used.
        # If explicit linking is needed for the host part, it would be like:
        # find_package(SYCL) or target_link_libraries(target PRIVATE sycl)
        # For now, rely on -fsycl.
    else()
        message(WARNING "USE_SYCL is ON, but the C++ compiler (${CMAKE_CXX_COMPILER_ID}, ${CMAKE_CXX_COMPILER}) does not appear to be Intel DPC++. SYCL compilation might fail or require different flags. Please ensure you are using icpx or dpcpp as your CMAKE_CXX_COMPILER.")
    endif()
    # Ensure C++ standard is compatible (already C++20, which is good)
    if (CMAKE_CXX_STANDARD LESS 17)
        message(WARNING "SYCL typically requires C++17 or newer. Current CMAKE_CXX_STANDARD is ${CMAKE_CXX_STANDARD}.")
    endif()
endif()
# --- End SYCL Configuration ---

set(PROJECT_NAME "arbd${PROJECT_NAME_SUFFIX}")

if(USE_CUDA)
    project("${PROJECT_NAME}" VERSION 1.2 LANGUAGES CXX CUDA)
else()
    project("${PROJECT_NAME}" VERSION 1.2 LANGUAGES CXX)
endif()

## Set flags and definitions after project() to ensure languages are enabled
if(USE_CUDA)
  add_definitions(-DUSE_CUDA)
  # CUDA_INCLUDE_DIRS check might not be necessary with modern CMake and FindCUDAToolkit
  # include_directories(${CUDA_INCLUDE_DIRS}) # Usually handled by FindCUDAToolkit
endif()



if(DEBUG)
  set(CMAKE_BUILD_TYPE Debug)
  set(PROJECT_NAME arbd_dbg)
else(DEBUG)
  set(PROJECT_NAME arbd)
endif()
if(USE_NVTX)
  add_definitions(-DUSE_NVTX)
endif()
if(USE_NCCL)
  add_definitions(-DUSE_NCCL)
  target_link_libraries("${PROJECT_NAME}" PRIVATE nccl)
endif()
if(USE_LOGGER)
  set(spdlog_DIR "extern/spdlog/include")
  set(CMAKE_spdlog_DIR "${spdlog_DIR}")
  add_subdirectory(extern/spdlog)
  include_directories(${spdlog_DIR})
  set(SPDLOG_LEVEL SPDLOG_LEVEL_DEBUG)	
endif()

## Two lines below needed?
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(USE_CUDA)
  project("${PROJECT_NAME}" VERSION 1.2 LANGUAGES CXX CUDA)
else(USE_CUDA)
  project("${PROJECT_NAME}" VERSION 1.2 LANGUAGES CXX)
endif()


## Print all variables by uncommenting block below 
# get_cmake_property(_variableNames VARIABLES)
# list (SORT _variableNames)
# foreach (_variableName ${_variableNames})
#     message(STATUS "${_variableName}=${${_variableName}}")
# endforeach()

## Set rpath
set(CMAKE_MACOSX_RPATH 1)	# Unsure if this works for CMAKE_BUIlD_RPATH, or just CMAKE_INSTALL_RPATH
set(CMAKE_BUILD_RPATH "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")

## Set up doctest
# option(ENABLE_DOCTESTS "Include tests in the library. Setting this to OFF will remove all doctest related code.
#                     Tests in tests/*.cpp will still be enabled." ON)
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
# include(doctest)

## Add subdirectories
add_subdirectory(src)
add_subdirectory(src/Tests)

# set(CMAKE_VERBOSE_MAKEFILE True)
add_executable("${PROJECT_NAME}" src/arbd.cpp
#src/SimManager.cu
#src/useful.cu
)

if(USE_PYBIND)
  add_subdirectory(extern/pybind11)
  add_subdirectory(src/pybindings)
endif()

target_link_libraries("${PROJECT_NAME}" PUBLIC "lib${PROJECT_NAME}")

if(USE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) # If USE_CUDA
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(CMAKE_BUILD_RPATH "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")
  target_link_libraries(${PROJECT_NAME}
        PRIVATE
        CUDA::cuda_driver
        CUDA::cudart
        CUDA::curand
        CUDA::cublas)
endif()

if(USE_NCCL)
  target_link_libraries("${PROJECT_NAME}" PUBLIC nccl)
endif()

if(USE_NVTX)
  target_link_libraries("${PROJECT_NAME}" PUBLIC nvToolsExt)
endif()

if(USE_LOGGER)
  add_definitions(-DUSE_LOGGER)
  target_include_directories("${PROJECT_NAME}" PRIVATE ${spdlog_DIR})
  target_link_libraries("${PROJECT_NAME}" PRIVATE spdlog::spdlog_header_only)
  add_compile_definitions(SPDLOG_ACTIVE_LEVEL=${SPDLOG_LEVEL})
endif()
       
