/**
 * @file SignalManager.cpp
 * @brief Implementation of segmentation fault signal handling and stack trace capture
 * 
 * This file implements a signal handler for segmentation faults (SIGSEGV) that captures
 * detailed diagnostic information when a segmentation fault occurs. The handler provides:
 * - Faulty memory address identification
 * - Instruction pointer at time of fault
 * - Stack backtrace with up to 16 frames
 * 
 * @warning Signal Handler Safety Considerations:
 * - This implementation uses non-async-signal-safe functions including fmt::print,
 *   backtrace(), and memory allocation which may be unsafe in a signal context
 * - The handler terminates the program via exit() which may not perform proper cleanup
 * 
 * @note Platform Dependencies:
 * - Requires MY_REG_RIP register access which may be platform-specific
 * - Uses POSIX signal handling interfaces (sigaction)
 * - Conditional compilation requires SIGNAL macro to be defined
 * 
 * Usage Example:
 * @code
 * SignalManager::manage_segfault(); // Call early in program initialization
 * @endcode
 * 
 * @note Memory Management:
 * The implementation uses std::unique_ptr with a custom deleter to ensure proper
 * cleanup of resources allocated by backtrace_symbols().
 * 
 * @dependencies
 * - fmt library for formatted output
 * - POSIX signal handling support
 * - backtrace facilities (typically provided by glibc)
 */

#include "SignalManager.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <unistd.h>
#ifdef SIGNAL
#include "Common.h"

struct BacktraceSymbolsDeleter {
    void operator()(char** p) const { if (p) std::free(p); }
};
// Initialize the shutdown flag
volatile sig_atomic_t SignalManager::shutdown_requested = 0;

/**
 * @brief Handles segmentation fault signals by capturing and displaying diagnostic information
 * 
 * @param sig The signal number (expected to be SIGSEGV)
 * @param info Signal information structure containing fault details
 * @param secret Context information at the time of the fault (cast to ucontext_t*)
 * 
 * This handler performs the following:
 * 1. Captures the fault address and instruction pointer
 * 2. Generates a stack backtrace
 * 3. Prints diagnostic information to stdout
 * 4. Sets shutdown flag for graceful termination
 * 
 * @note This function is not async-signal-safe due to use of fmt::print and memory allocation
 */
void SignalManager::segfault_handler(int sig, siginfo_t *info, void *secret) {
    // Set shutdown flag immediately
    shutdown_requested = 1;

    void *trace[16];
    int trace_size = 0;
    ucontext_t *uc = (ucontext_t *)secret;

    // Use write() instead of fmt::print as it's signal-safe
    const char msg[] = "Segmentation fault detected. Initiating shutdown...\n";
    write(STDERR_FILENO, msg, sizeof(msg) - 1);

    trace_size = backtrace(trace, 16);
    trace[1] = (void *)uc->uc_mcontext.gregs[MY_REG_RIP];

    std::unique_ptr<char*, BacktraceSymbolsDeleter> messages_ptr;

    char** raw_messages = backtrace_symbols(trace, trace_size);
    if (raw_messages) {
        messages_ptr.reset(raw_messages);
    }

    // Write basic info using signal-safe write()
    const char trace_msg[] = "Stack trace:\n";
    write(STDERR_FILENO, trace_msg, sizeof(trace_msg) - 1);

    if (messages_ptr) {
        for (int i = 1; i < trace_size; ++i) {
            write(STDERR_FILENO, messages_ptr.get()[i], strlen(messages_ptr.get()[i]));
            write(STDERR_FILENO, "\n", 1);
        }
    }
}

/**
 * @brief Sets up the segmentation fault signal handler
 * 
 * Configures a signal handler for SIGSEGV using sigaction with SA_SIGINFO flag
 * to receive extended signal information. Should be called during program initialization.
 * 
 * @note When USE_LOGGER is defined, sets spdlog level to trace
 */
void SignalManager::manage_segfault() 
{
#ifdef USE_LOGGER
    spdlog::set_level(spdlog::level::trace);
#endif
	struct sigaction sa;

	sa.sa_sigaction = segfault_handler;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = SA_RESTART | SA_SIGINFO;

	sigaction(SIGSEGV, &sa, NULL);
}
