/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

#ifndef RIGIDBODYCONTROLLER_H
#define RIGIDBODYCONTROLLER_H

#include <fstream>

#include <set>
#include <vector>
#include "ResizeArray.h"

#include "Vector.h"
#include "Tensor.h"
#include "SimParameters.h"
#include "NamdTypes.h"
#include "MStream.h"
#include "charm++.h"
#include "ReductionMgr.h"

#include "NamdState.h"
#include "Molecule.h"

#include "RigidBody.h"

class ComputeMgr;
class Random;

class RigidBodyController {
public:
    RigidBodyController(const NamdState *s, int reductionTag, SimParameters *sp);
    ~RigidBodyController();
    void integrate(int step);
    void print(int step);
    
private:
    void printLegend(std::ofstream &file);
    void printData(int step, std::ofstream &file);

    SimParameters *simParams;
    const NamdState * state;		
    
    std::ofstream trajFile;

    Random *random;
    ComputeMgr *computeMgr;
    RequireReduction *gridReduction;

    ResizeArray<Vector> trans; // would have made these static, but
    ResizeArray<Tensor> rot;	// there are errors on rigidBody->integrate
    std::vector<RigidBody*> rigidBodyList;
};

#endif

