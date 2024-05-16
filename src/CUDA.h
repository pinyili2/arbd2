#pragma once

class BDIntegrateCUDA : public Integrator {
public:
    void compute(Patch* patch);
    int num_patches() const { return 1; };
};
