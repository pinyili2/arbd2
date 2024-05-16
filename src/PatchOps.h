#pragma once

class Patch;

class BaseCompute {
    // Low level class that operates on Patch data
public:
    virtual void compute(Patch* patch) = 0;
    virtual int num_patches() const = 0;
private:
    void* compute_data;
};

#include "Integrator.h"
