#pragma once

#ifdef __CUDACC__
    #define HOST __host__
    #define DEVICE __device__
#else
    #define HOST
    #define DEVICE
#endif

#include <vector> // std::vector
#include <memory> // std::make_unique

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#endif

#include "SimSystem.h"
#include "useful.h"

#include "PatchOps.h"
//class BaseCompute;

class BasePatch {
public:
    // BasePatch(size_t num, short thread_id, short gpu_id, SimSystem& sys);
    // BasePatch(size_t num, short thread_id, short gpu_id);
    // BasePatch();
    // ~BasePatch();

private:
    size_t capacity;
    size_t num;
    short thread_id;		// MPI
    short gpu_id;		// -1 if GPU unavailable

    int patch_idx;		// Unique ID across ranks

    Vector3 lower_bound;
    Vector3 upper_bound;
};

class PatchProxy : public BasePatch {
public:
    // ???
private:
    //
};

class Patch : public BasePatch {
public:
    Patch(size_t num, short thread_id, short gpu_id) {};
    // void deleteParticles(IndexList& p);
    // void addParticles(int n, int typ);
    // template<class T>
    // void add_compute(std::unique_ptr<T>&& p) {
    // 	std::unique_ptr<BaseCompute> base_p = static_cast<std::unique_ptr<BaseCompute>>(p);
    // 	local_computes.emplace_back(p);
    // };
    void add_compute(std::unique_ptr<BaseCompute>&& p) {
	local_computes.emplace_back(std::move(p));
    };

    void compute();
    
private:
    // std::vector<PatchProxy> neighbors;    
    std::vector<std::unique_ptr<BaseCompute>> local_computes; // Operations that will be performed on this patch each timestep
    std::vector<std::unique_ptr<BaseCompute>> nonlocal_computes; // Operations that will be performed on this patch each timestep
    
    static int patch_idx;		// Unique ID across ranks

    // CPU particle arrays
    Vector3* pos;
    Vector3* momentum;

    Vector3* rb_pos;
    Matrix3* rb_orient;
    Vector3* rb_mom;
    Vector3* rb_amom;

    int* type;	     // particle types: 0, 1, ... -> num * numReplicas

    int num_rb_attached_particles;
    int num_group_sites;
    int* groupSiteData_d;

    // Device arrays
    Vector3* pos_d;
    Vector3* momentum_d;
    Vector3* rb_pos_d;
    Matrix3* rb_orient_d;
    Vector3* rb_mom_d;
    Vector3* rb_amom_d;
    int* type_d;
};

// // Patch::Patch(size_t num, short thread_id, short gpu_id) {};
// #ifndef USE_CUDA
// void Patch::compute() {
//     for (auto& c_p: local_computes) {
// 	c_p->compute(this);
//     }
// };
// #endif
