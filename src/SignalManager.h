#pragma once

/**
 * @file SignalManager.h
 * @brief Header file for signal handling and logging functionality in the ARBD namespace.
 * 
 * This header provides signal handling capabilities, particularly for segmentation faults,
 * along with a comprehensive logging system that works both on host and device (CUDA) code.
 * The logging system can be enabled/disabled using the USE_LOGGER macro.
 */

#include <csignal> 

#ifdef USE_LOGGER
    #ifdef __CUDA_ARCH__
        // Using standard __FILE__ and __LINE__ for device code location, and ensuring FMT is format string
        #define DEVICE_LOCATION_STRINGIFY_DETAIL_SM(line) #line 
        #define DEVICE_LOCATION_TO_STRING_DETAIL_SM(line) DEVICE_LOCATION_STRINGIFY_DETAIL_SM(line)
        #define DEVICE_CODE_LOCATION_DETAIL_SM __FILE__ "(" DEVICE_LOCATION_TO_STRING_DETAIL_SM(__LINE__) ")"
        #define LOGHELPER(TYPE, FMT, ...) printf("[%s] [%s]: " FMT "\n", TYPE, DEVICE_CODE_LOCATION_DETAIL_SM, ##__VA_ARGS__)
        #define LOGTRACE(...) LOGHELPER("trace",  __VA_ARGS__)
        #define LOGDEBUG(...) LOGHELPER("debug",__VA_ARGS__)
        #define LOGINFO(...) LOGHELPER("info",__VA_ARGS__)
        #define LOGWARN(...) LOGHELPER("warn",__VA_ARGS__)
        #define LOGERROR(...) LOGHELPER("error",__VA_ARGS__)
        #define LOGCRITICAL(...) LOGHELPER("critical",__VA_ARGS__)
    #else // Host side logger
        #define FMT_HEADER_ONLY
        #include <spdlog/fmt/bundled/core.h>   
        #include <spdlog/fmt/bundled/format.h> 
        #include <spdlog/spdlog.h>             
        #ifndef SPDLOG_ACTIVE_LEVEL
            #define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
        #endif
        #define LOGTRACE(...) SPDLOG_TRACE(__VA_ARGS__)
        #define LOGDEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
        #define LOGINFO(...) SPDLOG_INFO(__VA_ARGS__)
        #define LOGWARN(...) SPDLOG_WARN(__VA_ARGS__)
        #define LOGERROR(...) SPDLOG_ERROR(__VA_ARGS__)
        #define LOGCRITICAL(...) SPDLOG_CRITICAL(__VA_ARGS__)
    #endif
#else // USE_LOGGER not defined
    #define LOGTRACE(...) (void)0
    #define LOGDEBUG(...) (void)0
    #define LOGINFO(...) (void)0
    #define LOGWARN(...) (void)0
    #define LOGERROR(...) (void)0
    #define LOGCRITICAL(...) (void)0
#endif


#ifdef SIGNAL
// These are POSIX specific, might need guards for other OS if portability is a concern
extern "C" { 
    #include <execinfo.h> 
    #ifndef __USE_GNU
        #define __USE_GNU
    #endif
    #include <ucontext.h>
}

#if __WORDSIZE == 64 
#define MY_REG_RIP REG_RIP
#else
#define MY_REG_RIP REG_EIP
#endif

#endif // SIGNAL

namespace ARBD {
/**
 * @namespace SignalManager
 * @brief Namespace containing signal handling and management functionality.
 * 
 * This namespace provides utilities for handling system signals, particularly
 * segmentation faults, and managing program shutdown requests.
 */
namespace SignalManager {
    
    /**
     * @brief Handles segmentation fault signals.
     * 
     * This function is called when a segmentation fault occurs. It prints
     * detailed information about the fault location and stack trace.
     * 
     * @param sig The signal number
     * @param info Pointer to siginfo_t structure containing signal information
     * @param secret Pointer to ucontext_t structure containing signal context
     */
    void segfault_handler(int sig, siginfo_t *info, void *secret);

    /**
     * @brief Sets up signal handling for segmentation faults.
     * 
     * This function configures the system to use the custom segfault_handler
     * for handling segmentation faults. It should be called during program
     * initialization.
     */
    void manage_segfault();

    /**
     * @brief Global flag indicating if a shutdown has been requested.
     * 
     * This atomic variable is used to coordinate program shutdown across
     * different threads. When set to non-zero, it indicates that the program
     * should begin shutdown procedures.
     */
    extern volatile sig_atomic_t shutdown_requested; 
    
    /**
     * @brief Checks if a shutdown has been requested.
     * 
     * @return true if shutdown has been requested, false otherwise
     */
    inline bool is_shutdown_requested() { return shutdown_requested != 0; }

} 
}
