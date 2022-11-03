cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

# option(USE_CUDA "Use CUDA" ON)
set(USE_CUDA ON)

if(USE_CUDA)
# set the project name and version
project(arbd VERSION 1.2 LANGUAGES CXX CUDA)
else(USE_CUDA)
project(arbd VERSION 1.2 LANGUAGES CXX)
endif()

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_VERBOSE_MAKEFILE True)

## OPTIONS
option(USE_NCCL "Use NCCL for single node GPU peer communication" ON)

# configure_file(TutorialConfig.h.in TutorialConfig.h)
message(STATUS "USE_CUDA: ${USE_CUDA}")
if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
    ## CUDA_INCLUDE_DIRS wasn't getting set on my system with cmake 3.14.1, so check if in env
    if(DEFINED ENV{CUDA_INCLUDE_DIRS})
    set(CUDA_INCLUDE_DIRS $ENV{CUDA_INCLUDE_DIRS})
    endif()    
    message(STATUS "CUDA_INC: ${CUDA_INCLUDE_DIRS}")
    include_directories(${CUDA_INCLUDE_DIRS})
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

target_link_libraries(arbd PRIVATE curand)

message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
## target_link_libraries(ARBD PUBLIC) MPI::MPI_CXX)

install(TARGETS arbd)
