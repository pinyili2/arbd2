#pragma once


#include <csignal> 

// Remove __FILE__ / __LINE__ macros as now uses std::source_location

// #define S1(x) #x
// #define S2(x) S1(x)
// #define LOCATION __FILE__ "(" S2(__LINE__)")" 

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


namespace SignalManager {
    
    void segfault_handler(int sig, siginfo_t *info, void *secret);
    void manage_segfault();

    extern volatile sig_atomic_t shutdown_requested; 
    
    inline bool is_shutdown_requested() { return shutdown_requested != 0; }

} 
