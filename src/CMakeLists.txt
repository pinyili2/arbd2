# This CMakeLists.txt is for the library built from sources within the 'src' directory.

# Define the library that will contain most of your project's code.
# The executable in the parent scope will link against this.
add_library(arbd_lib) # Using a fixed name "arbd_lib" for clarity

# List source files relative to the current directory (src/).
# These paths reflect your TARGET capitalized directory structure.
target_sources(arbd_lib PRIVATE
    # Core C++ files for initial modernization
    "ARBDException.h"
    "ARBDException.cpp"
    "Core/Common.h"
    "Core/SignalManager.h" 
    "Core/SignalManager.cpp"
    "Core/TypeName.h" 

    #"Core/Types.h"
    #"Core/Types/Array.h"
    #"Core/Types/Bitmask.h"
    #"Core/Types/Matrix3.h"
    #"Core/Types/Vector3.h"
    #"Core/MemoryManager.h"
    #"Core/Proxy.h"
    #"Core/Backend/GPU/CudaMemory.h"
    #"Core/Backend/Resource.h"
    #"Core/Backend/GPU/GPUManager.h"

)

# --- Include Directories for the Library ---
target_include_directories(arbd_lib
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
    # PRIVATE
        # Example: "${CMAKE_CURRENT_SOURCE_DIR}/Backend/Gpu/internal_details"
)

# --- Compile Definitions & Options for the Library ---
if(USE_LOGGER) # Option from parent scope
    target_compile_definitions(arbd_lib PRIVATE USE_LOGGER)
    # SPDLOG_ACTIVE_LEVEL is also set at the executable level,
    # but if libarbd uses it directly, it might need it here too.
    # target_compile_definitions(arbd_lib PRIVATE SPDLOG_ACTIVE_LEVEL=${ARBD_SPDLOG_LEVEL})
endif()

if(USE_NVTX) # Option from parent scope
    target_compile_definitions(arbd_lib PRIVATE USE_NVTX)
endif()

if(USE_NCCL) # Option from parent scope
    target_compile_definitions(arbd_lib PRIVATE USE_NCCL)
endif()

# --- Linking for the Library ---
# Link libraries that arbd_lib *itself* needs for its implementation.
# If these are only needed by the final executable, they are linked there.
if(USE_CUDA)
    # If libarbd uses CUDA runtime functions directly in its implementation
    # (not just in headers exposed to the executable).
    # target_link_libraries(arbd_lib PRIVATE CUDA::cudart)
    # target_link_libraries(arbd_lib PRIVATE CUDA::curand) # If random generation is part of libarbd
endif()

if(USE_LOGGER AND ARBD_SPDLOG_TARGET) # ARBD_SPDLOG_TARGET from parent
    # If libarbd's implementation uses spdlog.
    # If spdlog is header-only, linking might just be about include directories,
    # which target_link_libraries can also handle for INTERFACE libraries.
    target_link_libraries(arbd_lib PRIVATE ${ARBD_SPDLOG_TARGET})
    # If spdlog is not an imported target and is header only, ensure include path
    if(NOT TARGET ${ARBD_SPDLOG_TARGET})
        target_include_directories(arbd_lib PRIVATE "${CMAKE_SOURCE_DIR}/extern/spdlog/include")
    endif()
endif()

# if(USE_NVTX)
#   target_link_libraries(arbd_lib PRIVATE CUDA::nvToolsExt)
# endif()

# if(USE_NCCL AND NCCL_FOUND) # NCCL_FOUND from parent
#   target_link_libraries(arbd_lib PRIVATE NCCL::nccl)
# elif(USE_NCCL)
#   target_link_libraries(arbd_lib PRIVATE nccl) # Fallback
# endif()

message(STATUS "Configuring library: arbd_lib")