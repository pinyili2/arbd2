#pragma once
#include <iostream>


#include "System/SimSystem.h"
#include "ARBDLogger.h"
#include "ARBDException.h"

// Q: what is our parallel heirarchy?
// A: depends!

// Serial/openMP, MPI-only, Single-GPU, or NVSHMEM

// 1 Patch per MPI rank or GPU
// Patches should work independently with syncronization mediated by SimManager
// Patch to Patch data exchange should not require explicit scheduling by SimManager

// Load balancing?
namespace ARBD {
class LoadBalancer {
    // nothing for now
};

class SimManager {

public:
    SimManager() {}; //: load_balancer() {}

private:    
    LoadBalancer load_balancer;
    SimSystem sys;	// make it a list for replicas
    Decomposer decomp;
    //std::vector<SymbolicOp> sym_ops;
    //std::vector<PatchOp>  ops;
    
public:

    void run();    
};
}