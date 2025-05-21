
add_library("lib${PROJECT_NAME}"
  ${CMAKE_CURRENT_SOURCE_DIR}          # /path/to/arbd2_cpp20/src
  ${CMAKE_CURRENT_SOURCE_DIR}/Backend
  ARBDException.cpp
  SignalManager.cpp
)

if(USE_LOGGER)
  add_definitions(-DUSE_LOGGER)
  target_include_directories("lib${PROJECT_NAME}" PRIVATE ${spdlog_DIR})
  target_link_libraries("lib${PROJECT_NAME}" PRIVATE spdlog::spdlog_header_only)	
  add_compile_definitions(SPDLOG_ACTIVE_LEVEL=${SPDLOG_LEVEL})
endif()
