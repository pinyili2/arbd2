
cmake_minimum_required(VERSION 3.25 FATAL_ERROR) 

# option(USE_CUDA "Use CUDA" ON)
set(USE_CUDA ON)


set(USE_MPI OFF)

## specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

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
option(DEBUG "Build with debug flags" False)
option(USE_NCCL "Use NCCL for single node GPU peer communication" False)
option(USE_NVTX "Build with NVTX profiling ranges" False)
option(USE_LOGGER "Build with SPDLog" True)

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

  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    # set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()
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
  # set(spdlog_DIR "extern/spdlog/include/spdlog")
  # set(CMAKE_spdlog_DIR "${spdlog_DIR}")
  # find_package(spdlog REQUIRED)
  set(spdlog_DIR "extern/spdlog/include")
  set(CMAKE_spdlog_DIR "${spdlog_DIR}")
  add_subdirectory(extern/spdlog)
  # find_package(spdlog REQUIRED)
  include_directories(${spdlog_DIR})
  set(SPDLOG_LEVEL SPDLOG_LEVEL_DEBUG)

  # target_include_directories("lib${PROJECT_NAME}" PRIVATE ${spdlog_DIR})
  # target_include_directories("${PROJECT_NAME}" PRIVATE ${spdlog_DIR})

  # if(DEFINED ENV{CUDA_INCLUDE_DIRS})
  #   set(CUDA_INCLUDE_DIRS $ENV{CUDA_INCLUDE_DIRS})
  # endif()
  # target_link_libraries("${PROJECT_NAME}" PRIVATE spdlog::spdlog_header_only)
  # target_link_libraries("lib${PROJECT_NAME}" PRIVATE spdlog::spdlog_header_only)	
  # target_link_libraries("lib${PROJECT_NAME}" spdlog)	
  # target_link_libraries("${PROJECT_NAME}" spdlog)	
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

## Add optional libraries
if(USE_CUDA)
  target_link_libraries("${PROJECT_NAME}" PUBLIC curand)
endif()
if(USE_NCCL)
  target_link_libraries("${PROJECT_NAME}" PUBLIC nccl)
endif()
if(USE_NVTX)
  target_link_libraries("${PROJECT_NAME}" PUBLIC nvToolsExt)
endif()
if(USE_LOGGER)
  add_definitions(-DUSE_LOGGER)
  # include_directories(${spdlog_DIR})
  target_include_directories("${PROJECT_NAME}" PRIVATE ${spdlog_DIR})
  target_link_libraries("${PROJECT_NAME}" PRIVATE spdlog::spdlog_header_only)
  # find_package(fmt)
  # target_link_libraries("${PROJECT_NAME}" PRIVATE fmt::fmt)
  add_compile_definitions(SPDLOG_ACTIVE_LEVEL=${SPDLOG_LEVEL})
endif()

install(TARGETS "${PROJECT_NAME}")
message(STATUS "---------------------------------------------------------")
message(STATUS "Project Configuration Summary for ${PROJECT_NAME}:")
message(STATUS "  Build Type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:       ${CMAKE_CXX_STANDARD}")
if(USE_CUDA)
  message(STATUS "  CUDA Enabled:       YES")
  message(STATUS "  CUDA Standard:      ${CMAKE_CUDA_STANDARD}")
  message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "  CUDA Enabled:       NO")
endif()
message(STATUS "  Logger (spdlog):    ${USE_LOGGER} (Active Level: ${ARBD_SPDLOG_LEVEL})")
message(STATUS "  NVTX Profiling:     ${USE_NVTX}")
message(STATUS "  NCCL Support:       ${USE_NCCL}")
message(STATUS "  Python Bindings:    ${USE_PYBIND}")
message(STATUS "---------------------------------------------------------")
