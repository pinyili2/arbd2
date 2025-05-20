#include "SignalManager.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#ifdef SIGNAL
#include "Common.h"
#include "ARBDException.h"

void SignalManager::segfault_handler(int sig, siginfo_t *info, void *secret) 
{
    //comment out for now
    //if(sig != SIGSEGV) 
        //throw ARBDException("segfault_handler should handle segmentation faults only, got %d", sig);
    void *trace[16];
    int trace_size = 0;
    ucontext_t *uc = (ucontext_t *) secret;

    // Using fmt for modern string formatting
    fmt::print(stdout, "Segmentation fault identified, faulty address is {}, from {}\n", 
              (void*)info->si_addr, (void*)uc->uc_mcontext.gregs[MY_REG_RIP]);

    trace_size = backtrace(trace, 16);
    /* overwrite sigaction with caller's address */
    trace[1] = (void *) uc->uc_mcontext.gregs[MY_REG_RIP];

    // Use smart pointer with custom deleter for messages
    char **messages = nullptr;
    messages = (char **) backtrace_symbols(trace, trace_size);

    fmt::print(stdout, "Execution path:\n");
    for (int i = 1; i < trace_size; ++i) {
        fmt::print(stdout, "\t[bt] {}\n", messages[i]);
    }
    //throw here or just exit ?
    exit(0);
}

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

#else
void SignalManager::segfault_handler(int sig, siginfo_t *info, void *secret) {}
void SignalManager::manage_segfault() {
#ifdef USE_LOGGER
    // Commented when moving to logger.h
    // spdlog::set_level(spdlog::level::trace);
#endif
}

#endif
