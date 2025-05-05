#pragma once
#include "../Integrator.h"

#ifdef USE_CUDA
class BDIntegrateCUDA : public Integrator {
public:
    void compute(Patch* patch);
    int num_patches() const { return 1; };
};
#endif
