/*********************************************************************
 * @file  PatchOp.h
 * 
 * @brief Declaration of BasePatchOp class. 
 *
 * Includes headers of derived classes for convenient access to
 * factory methods. Virtual method `BasePatchOp::compute(Patch*
 * patch)` called by `patch->compute()` by any Patch to which the
 * PatchOp has been added.
 *********************************************************************/
#pragma once

class Patch;

/// Abstract base class that operates on Patch data.
class BasePatchOp {
public:
    virtual void compute(Patch* patch) = 0;
    virtual int num_patches() const = 0;
private:
    void* compute_data;
};

#include "Integrator.h"
#include "Interaction.h"
