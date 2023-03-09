#pragma once

#include <cassert>
#include <iostream>
#include <map>
#include "PatchOp.h"

class LocalInteraction : public BasePatchOp {
public:
    virtual void compute(Patch* patch) = 0;
    int num_patches() const { return 1; };

    // Following relates to lazy initialized factory method
    struct Conf {
	enum Object    {Particle, RigidBody };
	enum DoF {Bond, Angle, Dihedral, Bonded, NeighborhoodPair};
	enum Form {Harmonic, Tabulated, LennardJones, Coulomb, LJ};
	enum Backend   { Default, CUDA, CPU };
	
	Object object_type;
	DoF dof;
	Form form;
	Backend backend;

	explicit operator int() const {return object_type*64 + dof*16 + form*4 + backend;};
    };

    static LocalInteraction* GetInteraction(Conf& conf);

private:
    size_t num_interactions;
    
protected:
    static std::map<Conf, LocalInteraction*> _interactions;

};

// class LocalNonbondedInteraction : public BasePatchOp {
// public:
//     virtual void compute(Patch* patch) = 0;
//     int num_patches() const { return 1; };

//     // Following relates to lazy initialized factory method
//     struct Conf {
// 	enum Object    {Particle, RigidBody };
// 	enum Electrostatics {None, Coulomb, DebyeHuckel}
// 	enum Form      {Tabulated, LennardJones, Coulomb, LJ, DebyeHueckel};
// 	enum Backend   { Default, CUDA, CPU };
	
// 	Object object_type;
	
// 	std::string tabulated_file = "";
// 	Algorithm algorithm;
// 	Backend backend;

// 	explicit operator int() const {return object_type*16 + algorithm*4 + backend;};
//     };

//     static Integrator* GetIntegrator(Conf& conf);

// private:
//     size_t num_interactions;
    
// protected:
//     static std::map<Conf, Integrator*> _integrators;

// };

// class LocalBonded : public LocalInteraction {

// private:
//     size_t bondlist;		// Encode bond, angle, dihedral
    
// };

// class LocalBonds : public LocalInteraction {

// private:
//     // static const char* type = "Bond";
//     size_t bondlist;
    
// };


#include "Interaction/kernels.h"
#include "Interaction/CUDA.h"
#include "Interaction/CPU.h"
