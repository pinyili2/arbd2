/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

#pragma once

#include <source_location>
#include <string_view>

#ifndef MIN_DEBUG_LEVEL
  #define MIN_DEBUG_LEVEL 0
#endif
#ifndef MAX_DEBUG_LEVEL
  #define MAX_DEBUG_LEVEL 10
#endif
#ifndef STDERR_LEVEL
  /* anything >= this error level goes to stderr */
  #define STDERR_LEVEL 5
#endif


/**
 * @brief Displays a debug message with a specified severity level.
 *
 * Messages have different levels. The low numbers are low severity,
 * while the high numbers are really important. Very high numbers
 * are sent to stderr rather than stdout.
 *
 * The default severity scale is from 0 to 10:
 *   - 0: plain message
 *   - 4: important message
 *   - 5: warning (stderr)
 *   - 10: CRASH BANG BOOM error (stderr)
 *
 * The remaining arguments are like printf: a format string and some args.
 * This function can be turned off by compiling without the DEBUGM flag.
 *
 * @note No parameters to this function should have a side effect!
 * @note No functions should be passed as parameters! (including inline)
 */

#ifdef DEBUGMSG


  #define Debug(x) (x)
  
  // C++20 version using std::source_location
  template<typename... Args>
  void DebugMessage(int level, std::string_view format, 
                   const std::source_location& location = std::source_location::current()) {
    if ((level >= MIN_DEBUG_LEVEL) && (level <= MAX_DEBUG_LEVEL)) {
      infostream Dout;
      if (level >= STDERR_LEVEL) {
        Dout << "[ERROR " << level << "] ";
      } else if (level > 0) {
        Dout << "[Debug " << level << "] ";
      }
      Dout << iPE << ' ' << location.file_name() << ":" << location.line();
      Dout << format << endi;
    }
  }
  
  // Macro wrapper to automatically capture source location
  #define DebugMsg(level, format) DebugMessage(level, format)

#else
  // Make void functions
  #define Debug(x) static_cast<void>(0)
  #define DebugMsg(level, format) static_cast<void>(0)
  template<typename... Args>
  constexpr void DebugMessage(int, std::string_view, 
                             const std::source_location& = std::source_location::current()) {}
#endif /* DEBUGM */

