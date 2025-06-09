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
        #Simulation/Integrator/IntegratorSYCL.cpp
        #Simulation/Interactions/ExampleForceSYCL.cpp
        # Add any other SYCL-specific .cpp files here
    )
    # If your SYCL implementation uses a different file extension for device code
    # (though typically it's .cpp for SYCL), list them here too.
endif()

# Conditionally add CUDA specific source files (.cpp or .cu) to the library
if(USE_CUDA)
    message(STATUS "src: Adding CUDA specific sources to lib${PROJECT_NAME}")
    list(APPEND LIB_SOURCES
        #Backend/CUDA/CUDAManager.cpp         # Can be a .cpp file managing CUDA operations
        #Simulation/Integrator/IntegratorCUDA.cu # Note .cu extension
        #Simulation/Interactions/ExampleForceCUDA.cu # Note .cu extension
    )
    # Ensure the CUDA language is enabled in your top-level project() call
    # when USE_CUDA is ON, so CMake knows how to compile .cu files.
    message(STATUS "src: CUDA is enabled for lib${PROJECT_NAME}. CUDA runtime linking may be handled by compiler or main executable.")
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
    # For AdaptiveCpp (acpp), linking is also typically automatic when using acpp as the compiler.
    # If explicit linking to a SYCL runtime library target provided by acpp or icpx is needed,
    # it would go here. For example:
    # target_link_libraries("lib${PROJECT_NAME}" PRIVATE SYCL::sycl) # If such an imported target exists
    message(STATUS "src: SYCL is enabled for lib${PROJECT_NAME}. Ensure SYCL runtime is linked (usually automatic).")
    
    # Link OpenMP for AdaptiveCpp
    if(USE_SYCL_ACPP AND OpenMP_CXX_FOUND)
        target_link_libraries("lib${PROJECT_NAME}" PUBLIC OpenMP_Custom)
        message(STATUS "src: Linked custom OpenMP to lib${PROJECT_NAME} for AdaptiveCpp backend")
    endif()
    
    # Special handling for AdaptiveCpp to avoid duplicate C++ standard flags
    if(USE_SYCL_ACPP AND CMAKE_CXX_COMPILER MATCHES "acpp")
        message(STATUS "Configured lib${PROJECT_NAME} to avoid C++ standard flag conflicts with AdaptiveCpp")
    endif()
endif()
