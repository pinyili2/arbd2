#pragma once
#include "../GPUManager.h"
#include "../ParticlePatch.h"

class PatchCUDA : public Patch {
public:
    
    void compute();
    // void assign_gpu(GPU);
    
private:

    GPU my_gpu;
    
    // Device arrays
    size_t* groupSiteData_d;
    Vector3* pos_force_d;
    Vector3* momentum_d;
    Vector3* rb_pos_d;
    Matrix3* rb_orient_d;
    Vector3* rb_mom_d;
    Vector3* rb_ang_mom_d;
    size_t* type_d;
    size_t* rb_type_d;

};
