cmake_minimum_required(VERSION 3.25.5 FATAL_ERROR)
project(arbd_mpi_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable MPI support
set(USE_MPI ON)
add_definitions(-DUSE_MPI)

# Generate compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set build type
set(default_build_type "Debug")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
    STRING "Choose the type of build." FORCE)
endif()

# Note: Using MPI compiler (mpicxx) handles MPI linking automatically

# Core library sources (based on actual src/CMakeLists.txt)
set(LIB_SOURCES
    src/ARBDException.cpp
    src/SignalManager.cpp
)

# Create the core library
add_library(libarbd ${LIB_SOURCES})

# Include directories (based on actual project structure)
target_include_directories(libarbd
    PUBLIC
        src  # This allows #include "Backend/MPIBackend.h", etc.
)

# Add C++ standard library linking for GCC (from original CMakeLists.txt)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(libarbd PRIVATE stdc++ gcc_s gcc m pthread dl)
    target_link_options(libarbd PRIVATE -Wl,--no-as-needed)
endif()

# Fetch Catch2 for testing
include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.4.0
)
FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
include(CTest)

# Create the MPI test executable
add_executable(arbd_test_mpi src/Tests/mpitest.cpp)
target_include_directories(arbd_test_mpi PRIVATE src)
target_link_libraries(arbd_test_mpi PRIVATE 
    libarbd 
    Catch2::Catch2WithMain
)

# Ensure proper C++ standard library linking for GCC (same as main executable)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(arbd_test_mpi PRIVATE stdc++ gcc_s gcc m pthread dl)
    target_link_options(arbd_test_mpi PRIVATE -Wl,--no-as-needed)
endif()

# Enable test discovery
catch_discover_tests(arbd_test_mpi)

message(STATUS "MPI-only build configured successfully")
message(STATUS "Library sources: ${LIB_SOURCES}")
message(STATUS "Use: cmake --build . --target arbd_test_mpi")
message(STATUS "Run with: mpirun -np 8 ./arbd_test_mpi") 