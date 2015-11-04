/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

#ifndef RIGIDBODY_H
#define RIGIDBODY_H

#include <set>
#include <vector>

#include "Vector.h"
#include "Tensor.h"
#include "SimParameters.h"
#include "NamdTypes.h"

class RigidBody {
    /*
      For details underlying this class, see Appendix A of:
      Dullweber, Leimkuhler and McLaclan. "Symplectic splitting methods for rigid body molecular dynamics". J Chem Phys. (1997)
    */

public:
    RigidBody(SimParameters *simParams, RigidBodyParams *rbParams);
    ~RigidBody();


    void addForce(Force f); 
    void addTorque(Force t);
    void addLangevin(Vector w1, Vector w2);

    inline void clearForce() { force = Force(0); }
    inline void clearTorque() { torque = Force(0); }

    void integrate(Vector *p_trans, Tensor *p_rot, int startFinishAll);

    inline char* getKey() { return key; }
    inline Vector getPosition() { return position; }
    inline Tensor getOrientation() { return orientation; }
    inline BigReal getMass() { return mass; }
    inline Vector getVelocity() { return momentum/mass; }
    inline Vector getAngularVelocity() { 
	return Vector( angularMomentum.x / inertia.x,
		       angularMomentum.y / inertia.y,
		       angularMomentum.z / inertia.z );
    }
    bool langevin;
    
private:
    char* key;
    static const SimParameters * simParams;
    BigReal mass;

    Vector position;
    Tensor orientation;

    Vector inertia; // diagonal elements of inertia tensor
    Vector momentum;
    Vector angularMomentum; // angular momentum along corresponding principal axes
    
    // Langevin
    Vector langevinTransFriction;
    Vector langevinRotFriction;
    BigReal Temp;

    Vector transDampingCoeff;
    Vector transForceCoeff;
    Vector rotDampingCoeff;
    Vector rotTorqueCoeff;    

    // integration
    int timestep;
    Vector force;  // lab frame
    Vector torque; // lab frame (except in integrate())

    bool isFirstStep; 
    // units "kcal_mol/AA * fs"  "(AA/fs) * amu"
    const BigReal impulse_to_momentum;

    inline Tensor Rx(BigReal t);
    inline Tensor Ry(BigReal t);
    inline Tensor Rz(BigReal t);
    inline Tensor eulerToMatrix(const Vector e);
};

#endif

