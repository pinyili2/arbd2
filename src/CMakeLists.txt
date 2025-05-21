# This CMakeLists.txt is for the library built from sources within the 'src' directory.

# Define the library that will contain most of your project's code.
# The executable in the parent scope will link against this.
add_library(arbd_lib)

# Set modern C++ requirements
target_compile_features(arbd_lib PUBLIC cxx_std_20)

target_sources(arbd_lib 
    PRIVATE
        ARBDException.cpp
        SignalManager.cpp
    PUBLIC
        ARBDException.h
        SignalManager.h
        Common.h
        TypeName.h
)

# --- Include Directories for the Library ---

target_include_directories(arbd_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/internal
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

install(TARGETS arbd_lib
    EXPORT arbd-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)
message(STATUS "Configuring library: arbd_lib")