set(LIB_SOURCES
    ARBDException.cpp
    SignalManager.cpp
)

# Conditionally add SYCL specific source files to the library
# We check if either SYCL ACPP or ICPX option is enabled from the top-level CMake.
if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
    message(STATUS "src: Adding SYCL specific sources to lib${PROJECT_NAME}")
    list(APPEND LIB_SOURCES
        Backend/SYCL/SYCLManager.cpp
        Random/Random_SYCL.cpp
        # Add any other SYCL-specific .cpp files here
    )
    # If your SYCL implementation uses a different file extension for device code
    # (though typically it's .cpp for SYCL), list them here too.
endif()

# Conditionally add CUDA specific source files (.cpp or .cu) to the library
if(USE_CUDA)

    message(STATUS "src: Adding CUDA specific sources to lib${PROJECT_NAME}")
    list(APPEND LIB_SOURCES
        Backend/CUDA/CUDAManager.cpp         # Can be a .cpp file managing CUDA operations
        #Random/Random.cu
        #Simulation/Integrator/IntegratorCUDA.cu # Note .cu extension
        #Simulation/Interactions/ExampleForceCUDA.cu # Note .cu extension
    )
    # Ensure the CUDA language is enabled in your top-level project() call
    # when USE_CUDA is ON, so CMake knows how to compile .cu files.
    message(STATUS "src: CUDA is enabled for lib${PROJECT_NAME}. CUDA runtime linking may be handled by compiler or main executable.")
endif()

# Conditionally add Metal specific source files (.mm) to the library
if(USE_METAL)
    message(STATUS "src: Adding Metal specific sources to lib${PROJECT_NAME}")
    list(APPEND LIB_SOURCES
        Backend/METAL/METALManager.mm         # Objective-C++ implementation file
        #Simulation/Integrator/IntegratorMetal.mm # Future Metal integrator
        #Simulation/Interactions/ExampleForceMetal.mm # Future Metal force calculations
    )
    # Ensure the OBJCXX language is enabled in your top-level project() call
    # when USE_METAL is ON, so CMake knows how to compile .mm files.
    message(STATUS "src: Metal is enabled for lib${PROJECT_NAME}. Metal framework linking handled by main executable.")
endif()

# Define the library with all collected sources.
add_library("lib${PROJECT_NAME}" ${LIB_SOURCES})

# --- Include Directories ---
# CMAKE_CURRENT_SOURCE_DIR in this file refers to the "src" directory.
# This allows includes like #include "Backend/GPUManager.h" from within any source
# file of this library, and also allows targets linking to this library to
# include public headers from "src/" (e.g., #include "ARBDException.h").
target_include_directories("lib${PROJECT_NAME}"
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}  # Adds the "src/" directory itself
        # If you have other specific subdirectories that should be directly on the include path
        # for this library or its users, you can add them here.
        # For example, if you wanted to do #include "GPUManager.h" instead of #include "Backend/GPUManager.h",
        # you could add ${CMAKE_CURRENT_SOURCE_DIR}/Backend, but generally keeping some
        # prefix like "Backend/" is good for organization.
)

# --- SYCL specific linking or include directories for the library ---
# This is where you might link to SYCL runtime if not handled by the compiler,
# or add SYCL specific include dirs for this target if necessary.
if(USE_SYCL_ACPP OR USE_SYCL_ICPX)
    # For Intel DPC++ (icpx), linking is usually handled by -fsycl.
    message(STATUS "src: SYCL is enabled for lib${PROJECT_NAME}.")

    # Explicit linking for AdaptiveCpp runtime libraries
    if(USE_SYCL_ACPP AND CMAKE_CXX_COMPILER MATCHES "acpp")
        # Find ACPP installation directory from the compiler path
        get_filename_component(ACPP_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
        get_filename_component(ACPP_INSTALL_DIR "${ACPP_BIN_DIR}" DIRECTORY)
        set(ACPP_LIB_DIR "${ACPP_INSTALL_DIR}/lib")

        # Determine the appropriate library extension for the platform
        if(APPLE)
            set(ACPP_LIB_EXT "dylib")
        else()
            set(ACPP_LIB_EXT "so")
        endif()

        # Find and link the essential ACPP runtime libraries
        find_library(ACPP_RT_LIB NAMES "acpp-rt" PATHS "${ACPP_LIB_DIR}" NO_DEFAULT_PATH)
        find_library(ACPP_COMMON_LIB NAMES "acpp-common" PATHS "${ACPP_LIB_DIR}" NO_DEFAULT_PATH)

        if(ACPP_RT_LIB AND ACPP_COMMON_LIB)
            target_link_libraries("lib${PROJECT_NAME}" PRIVATE
                ${ACPP_RT_LIB}
                ${ACPP_COMMON_LIB}
            )
            message(STATUS "src: Successfully linked ACPP runtime libraries:")
            message(STATUS "  - ACPP Runtime: ${ACPP_RT_LIB}")
            message(STATUS "  - ACPP Common: ${ACPP_COMMON_LIB}")
        else()
            message(WARNING "Could not find required ACPP runtime libraries in ${ACPP_LIB_DIR}")
            message(WARNING "  - ACPP_RT_LIB: ${ACPP_RT_LIB}")
            message(WARNING "  - ACPP_COMMON_LIB: ${ACPP_COMMON_LIB}")
        endif()

        # Optional LLVM/Clang libraries - try to find them but don't fail if missing
        find_library(ACPP_CLANG_LIB NAMES "acpp-clang" PATHS "${ACPP_LIB_DIR}" NO_DEFAULT_PATH)
        if(ACPP_CLANG_LIB)
            target_link_libraries("lib${PROJECT_NAME}" PRIVATE ${ACPP_CLANG_LIB})
            message(STATUS "  - ACPP Clang: ${ACPP_CLANG_LIB}")
        endif()

        message(STATUS "Configured lib${PROJECT_NAME} to avoid C++ standard flag conflicts with AdaptiveCpp")
    endif()

    # Link OpenMP for AdaptiveCpp
    if(USE_SYCL_ACPP AND OpenMP_CXX_FOUND)
        target_link_libraries("lib${PROJECT_NAME}" PUBLIC OpenMP_Custom)
        message(STATUS "src: Linked custom OpenMP to lib${PROJECT_NAME} for AdaptiveCpp backend")
    endif()
endif()

# --- CUDA specific linking for the library ---
if(USE_CUDA)
    # Explicitly link the C++ standard library for CUDA targets
    # This ensures that CUDA code can access std library symbols from the host compiler
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_link_libraries("lib${PROJECT_NAME}" PRIVATE stdc++ gcc_s gcc m pthread dl)
        # Set additional linker flags for proper symbol resolution
        target_link_options("lib${PROJECT_NAME}" PRIVATE -Wl,--no-as-needed)
    else()
        target_link_libraries("lib${PROJECT_NAME}" PRIVATE stdc++)
    endif()

    message(STATUS "src: CUDA is enabled for lib${PROJECT_NAME}. Added explicit C++ standard library linking.")
endif()

# --- Metal specific linking and configuration for the library ---
if(USE_METAL)
    # Link Metal, Foundation, and QuartzCore frameworks to the library
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)

    if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK AND QUARTZCORE_FRAMEWORK)
        target_link_libraries("lib${PROJECT_NAME}"
            PUBLIC
            ${METAL_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
            ${QUARTZCORE_FRAMEWORK}
        )
        message(STATUS "src: Linked Metal frameworks to lib${PROJECT_NAME}")
    else()
        message(FATAL_ERROR "Required frameworks not found for lib${PROJECT_NAME}")
    endif()

    # Add metal-cpp include directories
    target_include_directories("lib${PROJECT_NAME}"
        PUBLIC
        ${CMAKE_SOURCE_DIR}/extern/metal-cpp-cmake/metal-cmake/metal-cpp
        ${CMAKE_SOURCE_DIR}/extern/metal-cpp-cmake/metal-cmake/metal-cpp-extensions
    )

    # Add Metal-specific compile definitions
    target_compile_definitions("lib${PROJECT_NAME}" PUBLIC USE_METAL)

    # Ensure ARC is enabled for Objective-C++ files
    target_compile_options("lib${PROJECT_NAME}" PRIVATE
        $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
    )

    message(STATUS "src: Metal is enabled for lib${PROJECT_NAME}. Framework linking and ARC configured.")
endif()
