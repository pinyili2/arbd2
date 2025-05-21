
#pragma once

// Any necessary includes would go here
#include <cstdint>  
#include <limits>  
// Use std::int32_t now

//use new instead
/**
 * @file Constants.h
 * @brief Physical and mathematical constants for ARBD simulations
 * @details Contains all constant values used throughout the ARBD project
 */

namespace arbd::constants {
    // Mathematical constants
    constexpr float PI = 3.141592653589793f;
    constexpr float TWOPI = 2.0f * PI;
    constexpr float HALFPI = 0.5f * PI;
    
    // Physical constants
    constexpr float COULOMB = 332.0636f;
    constexpr float BOLTZMANN = 0.001987191f;
    constexpr float TIMEFACTOR = 48.88821f;
    constexpr float PRESSUREFACTOR = 6.95E4f;
    
    // Simulation constants
    constexpr float PDBVELFACTOR = 20.45482706f;
    constexpr float PDBVELINVFACTOR = 1.0f/PDBVELFACTOR;
    constexpr float PNPERKCALMOL = 69.479f;
    constexpr float SMALLRAD = 0.0005f;
    constexpr float SMALLRAD2 = SMALLRAD * SMALLRAD;
}

// Use inline variables for configuration constants
inline constexpr int MAX_NEIGHBORS = 27;
/* Define the size for Real and BigReal.  Real is usually mapped to float */
/* and BigReal to double.  To get BigReal mapped to float, use the 	  */
/* -DSHORTREALS compile time option					  */
using Real = float;
using BigReal = float;

using Bool = bool;

#ifndef FALSE
    #define FALSE false 
#endif
#ifndef TRUE
    #define TRUE true  
#endif

#ifndef NO
    #define NO false   
#endif
#ifndef YES
    #define YES true  
#endif

constexpr char STRING_NULL_CHAR = '\0';


class Communicate;

// global functions

namespace NAMD{
  void quit(const char *);
  void die(const char *);
  void err(const char *);  // also prints strerror(errno)
  void bug(const char *);
  void backup_file(const char *filename, const char *extension = 0);
  char *stringdup(const char *);

  constexpr int SeparateWaters = 0;

  constexpr int ComputeNonbonded_SortAtoms = 0;
  constexpr int ComputeNonbonded_SortAtoms_LessBranches = 1;
  constexpr int WAT_TIP3 = 0;
  constexpr int WAT_TIP4 = 1;
  constexpr int CYCLE_BARRIER = 0;
  constexpr int PME_BARRIER = 0;
  constexpr int STEP_BARRIER = 0;

  constexpr int USE_BARRIER = (CYCLE_BARRIER || PME_BARRIER || STEP_BARRIER);

}



enum class MessageTag : std::uint32_t {
  SIMPARAMS=100,	//  Tag for SimParameters class
  STATICPARAMS=101,	//  Tag for Parameters class
  MOLECULE=102, //  Tag for Molecule class
  FULL=104,
  FULLFORCE=105,
  DPMTA=106
};
// message tags

