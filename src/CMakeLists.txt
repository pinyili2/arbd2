## Specify the project
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 75)
endif()

# option(USE_CUDA "Use CUDA" ON)
set(USE_CUDA ON)

if(USE_CUDA)
  project(arbd VERSION 1.2 LANGUAGES CXX CUDA)
else(USE_CUDA)
  project(arbd VERSION 1.2 LANGUAGES CXX)
endif()

## specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set a default build type if none was specified
set(default_build_type "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
    STRING "Choose the type of build." FORCE)
  ## Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()



if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  ## Some CMake versions use -O2 for release; replace with -O3
  if(CMAKE_CXX_FLAGS_RELEASE)
    string(REPLACE "-O2" "-O3" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  endif()
  if(CMAKE_CUDA_FLAGS_RELEASE)
    string(REPLACE "-O2" "-O3" CMAKE_CUDA_FLAGS_RELEASE ${CMAKE_CUDA_FLAGS_RELEASE})
  endif()
endif()


## OPTIONS
option(USE_NCCL "Use NCCL for single node GPU peer communication" False)
option(USE_NVTX "Build with NVTX profiling ranges" False)
option(DEBUG "Build with debug flags" False)

# (not yet optional) message(STATUS "USE_CUDA: ${USE_CUDA}")
message(STATUS "USE_NCCL: ${USE_NCCL}")
message(STATUS "USE_NVTX: ${USE_NVTX}")
message(STATUS "DEBUG: ${DEBUG}")


## Set flags before adding executable 
# configure_file(TutorialConfig.h.in TutorialConfig.h)
if(USE_CUDA)
  add_definitions(-DUSE_CUDA)
  ## CUDA_INCLUDE_DIRS wasn't getting set on my system with cmake 3.14.1, so check if in env
  if(DEFINED ENV{CUDA_INCLUDE_DIRS})
    set(CUDA_INCLUDE_DIRS $ENV{CUDA_INCLUDE_DIRS})
  endif()
  # message(STATUS "CUDA_INC: ${CUDA_INCLUDE_DIRS}")
  include_directories(${CUDA_INCLUDE_DIRS})
endif()

if(USE_NVTX)
  add_definitions(-DUSE_NVTX)
endif()
if(USE_NCCL)
  add_definitions(-DUSE_NCCL)
  target_link_libraries(arbd PRIVATE nccl)
endif()

## Two lines below needed?
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

## Print all variables by uncommenting block below 
# get_cmake_property(_variableNames VARIABLES)
# list (SORT _variableNames)
# foreach (_variableName ${_variableNames})
#     message(STATUS "${_variableName}=${${_variableName}}")
# endforeach()

## Set rpath
set(CMAKE_MACOSX_RPATH 1)	# Unsure if this works for CMAKE_BUIlD_RPATH, or just CMAKE_INSTALL_RPATH
set(CMAKE_BUILD_RPATH "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")

# set(CMAKE_VERBOSE_MAKEFILE True)
add_executable(arbd arbd.cpp
  Configuration.cpp
  FlowForce.cpp
  GPUManager.cpp
  Scatter.cpp
  SignalManager.cpp
  WKFUtils.cpp
  Angle.cu
  BaseGrid.cu
  BrownianParticle.cu
  BrownianParticleType.cpp
  RigidBodyController.cu
  RigidBody.cu
  RigidBodyGrid.cu
  RigidBodyType.cu
  CellDecomposition.cu
  ComputeForce.cuh
  ComputeForce.cu
  ComputeGridGrid.cuh
  ComputeGridGrid.cu
  CudaUtil.cu
  CudaUtil.cuh
  Dihedral.cu
  Exclude.cu
  GrandBrownTown.cu
  GrandBrownTown.cuh
  imd.cpp
  vmdsock.cpp
  JamesBond.cu
  RandomCUDA.cu
  Reservoir.cu
  TabulatedAngle.cu
  TabulatedDihedral.cu
  TabulatedMethods.cuh
  TabulatedPotential.cu
  useful.cu
  )

## Add optional libraries
if(USE_CUDA)
  target_link_libraries(arbd PRIVATE curand)
endif()
if(USE_NCCL)
  target_link_libraries(arbd PRIVATE nccl)
endif()
if(USE_NVTX)
  target_link_libraries(arbd PRIVATE nvToolsExt)
endif()

install(TARGETS arbd)
