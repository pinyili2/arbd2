/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

/*
   common definitions for namd.
*/

#ifndef COMMON_H
#define COMMON_H

#include <sys/stat.h>

#if !defined(WIN32) || defined(__CYGWIN__)
#include <unistd.h>
#endif
#include <stdio.h>
#include <limits.h>

#if ( INT_MAX == 2147483647L )
typedef	int	int32;
#elif ( SHRT_MAX == 2147483647L )
typedef	short	int32;
#endif

#ifdef _MSC_VER
typedef __int64 int64;
#else
#if ( INT_MAX == 9223372036854775807LL )
typedef int int64;
#elif ( LONG_MAX == 9223372036854775807LL )
typedef long int64;
#else
typedef long long int64;
#endif
#endif
// USe std::int32_t now

#if defined(PLACEMENT_NEW)
void * ::operator new (size_t, void *p) { return p; }
#elif defined(PLACEMENT_NEW_GLOBAL)
void * operator new (size_t, void *p) { return p; }
#endif
//use new instead


constexpr float COLOUMB 332.0636f
constexpr float BOLTZMAN 0.001987191f
constexpr float TIMEFACTOR 48.88821f
constexpr float PRESSUREFACTOR 6.95E4f
constexpr float PDBVELFACTOR 20.45482706f
constexpr float PDBVELINVFACTOR (1.0f/PDBVELFACTOR)
constexpr float PNPERKCALMOL 69.479f

#ifndef PI
constexpr float PI	3.141592653589793f
#endif

#ifndef TWOPI
constexpr float TWOPI	2.0f * PI
#endif

#ifndef ONE
constexpr float ONE	1.000000000000000f
#endif

#ifndef ZERO
constexpr float ZERO	0.000000000000000f
#endif

#ifndef SMALLRAD
constexpr float SMALLRAD      0.0005f
#endif

#ifndef SMALLRAD2
constexpr float SMALLRAD2     SMALLRAD*SMALLRAD
#endif

/* Define the size for Real and BigReal.  Real is usually mapped to float */
/* and BigReal to double.  To get BigReal mapped to float, use the 	  */
/* -DSHORTREALS compile time option					  */
using Real = float;
using BigReal = float;

#ifndef FALSE
#define FALSE 0
#define TRUE 1
#endif

#ifndef NO
#define NO 0
#define YES 1
#endif

typedef int Bool;


#ifndef STRINGNULL
#define STRINGNULL '\0'
#endif

#define MAX_NEIGHBORS 27

typedef int Bool;

class Communicate;

// global functions

namespace NAMDUtils{


}
void NAMD_quit(const char *);
void NAMD_die(const char *);
void NAMD_err(const char *);  // also prints strerror(errno)
void NAMD_bug(const char *);
void NAMD_backup_file(const char *filename, const char *extension = 0);
// void NAMD_write(int fd, const void *buf, size_t count); // NAMD_die on error
char *NAMD_stringdup(const char *);

class FileHandle {
  FILE* m_file = nullptr;
public:
  FileHandle(const char* filename, const char* mode) : m_file(std::fopen(filename, mode)) {
      if (!m_file) { /* throw or handle error */ }
  }
  ~FileHandle() {
      if (m_file) std::fclose(m_file);
  }
  // Delete copy constructor/assignment
  FileHandle(const FileHandle&) = delete;
  FileHandle& operator=(const FileHandle&) = delete;
  // Allow move
  FileHandle(FileHandle&& other) noexcept : m_file(other.m_file) { other.m_file = nullptr; }
  FileHandle& operator=(FileHandle&& other) noexcept { /* ... */ }

  FILE* get() const { return m_file; }
  // operator FILE*() const { return m_file; } // If implicit conversion is desired
};
// Usage: FileHandle my_file("data.txt", "r"); // Automatically closes



enum class MessageTag : int {
  SimParams = 100,
  StaticParams = 101,
  Molecule = 102,
  Full=104,
  FullForce=105,
  Dpm=106,
    // ...
};
// message tags
//#define SIMPARAMSTAG	100	//  Tag for SimParameters class
//#define STATICPARAMSTAG 101	//  Tag for Parameters class
//#define MOLECULETAG	102	//  Tag for Molecule class
//#define FULLTAG	104
//#define FULLFORCETAG 105
//#define DPMTATAG 106

#define CYCLE_BARRIER   0
#define PME_BARRIER     0
#define STEP_BARRIER    0

#define USE_BARRIER   (CYCLE_BARRIER || PME_BARRIER || STEP_BARRIER)

#define NAMD_SeparateWaters    0

#define NAMD_ComputeNonbonded_SortAtoms                   0
#define NAMD_ComputeNonbonded_SortAtoms_LessBranches    1

// plf -- alternate water models
#define WAT_TIP3 0
#define WAT_TIP4 1

#endif

