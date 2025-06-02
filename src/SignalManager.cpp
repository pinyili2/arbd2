#include "SignalManager.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <unistd.h>
#include <execinfo.h>

namespace ARBD::SignalManager {
volatile sig_atomic_t shutdown_requested = 0;

struct BacktraceSymbolsDeleter {
    void operator()(char** p) const { if (p) std::free(p); }
};
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
void segfault_handler(int sig, siginfo_t *info, void *secret) {
    // Set shutdown flag immediately
    shutdown_requested = 1;

    void *trace[16];
    int trace_size = 0;
    ucontext_t *uc = (ucontext_t *)secret;

    // Use write() instead of fmt::print as it's signal-safe
    const char msg[] = "Segmentation fault detected. Initiating shutdown...\n";
    write(STDERR_FILENO, msg, sizeof(msg) - 1);

    trace_size = backtrace(trace, 16);
#ifdef __APPLE__
    trace[1] = (void *)uc->uc_mcontext->__ss.__rip;
#else
    trace[1] = (void *)uc->uc_mcontext.gregs[REG_RIP];
#endif

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
void manage_segfault() 
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
}