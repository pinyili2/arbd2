#pragma once

#include <cassert>
#include <iostream>
#include <map>
#include "PatchOps.h"

#ifdef __CUDACC__
    #define HOST __host__
    #define DEVICE __device__
#else
    #define HOST
    #define DEVICE
#endif

namespace IntegratorKernels {
    HOST DEVICE  void __inline__ BDIntegrate() {
	// std::cout << "Computes::BDIntegrate_inline" << std::endl;
	printf("Integrator::BDIntegrate\n");
    };
}

class Integrator : public BaseCompute {
public:
    virtual void compute(Patch* patch) = 0;
    int num_patches() const { return 1; };

    // Following relates to lazy initialized factory method
    struct Conf {
	enum Object   { Particle, RigidBody };
	enum Algorithm { BD, MD };
	enum Backend   { Default, CUDA, CPU };    

	Object object_type;
	Algorithm algorithm;
	Backend backend;

	explicit operator int() const {return object_type*16 + algorithm*4 + backend;};
    };
        
    static Integrator* GetIntegrator(Conf& conf);
	    	
protected:
    static std::map<Conf, Integrator*> _integrators;

};

#include "Integrator/CUDA.h"
#include "Integrator/CPU.h"
