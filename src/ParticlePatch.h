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
// #include "useful.h"
#include "Types.h"

#include "PatchOp.h"

class BasePatch {
public:
    // BasePatch(size_t num, short thread_id, short gpu_id, SimSystem& sys);
    // BasePatch(size_t num, short thread_id, short gpu_id);
    BasePatch() : num(0), capacity(0) {}
    // ~BasePatch();
    
private:
    size_t capacity;
    size_t num;

    // short thread_id;		// MPI
    // short gpu_id;		// -1 if GPU unavailable

    static size_t patch_idx;		// Unique ID across ranks

    Vector3 lower_bound;
    Vector3 upper_bound;
};

class PatchProxy {
    // 
public:
    // ???
private:
    //
};

class Patch : public BasePatch {
public:
    Patch();

    // void deleteParticles(IndexList& p);
    // void addParticles(size_t n, size_t typ);
    // template<class T>
    // void add_compute(std::unique_ptr<T>&& p) {
    // 	std::unique_ptr<BasePatchOp> base_p = static_cast<std::unique_ptr<BasePatchOp>>(p);
    // 	local_computes.emplace_back(p);
    // };
    
    void add_compute(std::unique_ptr<BasePatchOp>&& p) {
	local_computes.emplace_back(std::move(p));
    };

    void add_point_particles(size_t num_added);
    void add_point_particles(size_t num_added, Vector3* positions, Vector3* momenta = nullptr);
    
    // TODO? emplace_point_particles
    void compute();
    
private:

    // void randomize_positions(size_t start = 0, size_t num = -1);

    // std::vector<PatchProxy> neighbors;    
    std::vector<std::unique_ptr<BasePatchOp>> local_computes; // Operations that will be performed on this patch each timestep
    std::vector<std::unique_ptr<BasePatchOp>> nonlocal_computes; // Operations that will be performed on this patch each timestep
    
    // CPU particle arrays
    Vector3* pos_force;
    Vector3* momentum;

    // size_t num_rb;
    // Vector3* rb_pos;
    // Matrix3* rb_orient;
    // Vector3* rb_mom;
    // Vector3* rb_ang_mom;

    // size_t* type;	     // particle types: 0, 1, ... -> num * numReplicas
    // size_t* rb_type;	     // rigid body types: 0, 1, ... -> num * numReplicas

    // size_t num_rb_attached_particles;

    // TODO: add a rb_attached bitmask
    
    size_t num_group_sites;

};

// // Patch::Patch(size_t num, short thread_id, short gpu_id) {};
// #ifndef USE_CUDA
// void Patch::compute() {
//     for (auto& c_p: local_computes) {
// 	c_p->compute(this);
//     }
// };
// #endif
