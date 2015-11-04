/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

#ifndef MIN_DEBUG_LEVEL
#define MIN_DEBUG_LEVEL 5
#endif
#define DEBUGM
#include "Debug.h"

#include <iostream>
#include <typeinfo>

#include "RigidBody.h"
#include "Vector.h"
#include "RigidBodyParams.h"


RigidBody::RigidBody(SimParameters *simParams, RigidBodyParams *rbParams) 
    : impulse_to_momentum(0.0004184) {

    int len = strlen(rbParams->rigidBodyKey);
    key = new char[len+1];
    strncpy(key,rbParams->rigidBodyKey,len+1);

    DebugM(4, "Initializing Rigid Body " << key << "\n" << endi);
    
    timestep = simParams->dt;
    Temp = simParams->langevinTemp;

    mass = rbParams->mass;
    inertia = rbParams->inertia;

    position = rbParams->position;
    orientation = rbParams->orientation; // orientation a matrix that brings a vector from the RB frame to the lab frame
	
    momentum = rbParams->velocity * mass; // lab frame
    DebugM(4, "velocity " << rbParams->velocity << "\n" << endi);
    DebugM(4, "momentum " << momentum << "\n" << endi);

    angularMomentum = rbParams->orientationalVelocity; // rigid body frame
    angularMomentum.x *= rbParams->inertia.x;
    angularMomentum.y *= rbParams->inertia.y;
    angularMomentum.z *= rbParams->inertia.z;

    isFirstStep = true; // this might not work flawlessly...

    clearForce();
    clearTorque();
    
    DebugM(4, "RigidBody initial Force: " << force << "\n" << endi);

    // setup for langevin
    langevin = rbParams->langevin;
    if (langevin) {
	// DiffCoeff = kT / dampingCoeff mass
	// rbParams->DampingCoeff has units of (1/ps)	
	// f = - dampingCoeff * momentum
	// units "(1/ps) * (amu AA/fs)" "kcal_mol/AA" * 2.3900574
	transDampingCoeff = 2.3900574 * rbParams->transDampingCoeff;

	// < f(t) f(t') > = 2 kT dampingCoeff mass delta(t-t')
	// units "sqrt( k K (1/ps) amu / fs )" "kcal_mol/AA" * 0.068916889
	transForceCoeff = 0.068916889 * element_sqrt( 2*Temp*mass*rbParams->transDampingCoeff/timestep );
     
	// T = - dampingCoeff * angularMomentum
	rotDampingCoeff = 2.3900574 * rbParams->rotDampingCoeff;
	// < f(t) f(t') > = 2 kT dampingCoeff inertia delta(t-t')
	rotTorqueCoeff = 0.068916889 * element_sqrt( 2*Temp*element_mult(inertia, rbParams->rotDampingCoeff) / timestep );
    }
}

void RigidBody::addForce(Force f) { 
    DebugM(1, "RigidBody "<<key<<" adding f ("<<f<<") to Force " << force << "\n" << endi);    
    force += f; 
} 
void RigidBody::addTorque(Force t) {
    DebugM(1, "RigidBody adding t ("<<t<<") to torque " << torque << "\n" << endi);   
    torque += t; 
}
RigidBody::~RigidBody() {}

void RigidBody::addLangevin(Vector w1, Vector w2) {
    // w1 and w2 should be standard normal distributions

    /***************************************************************** 
      Following "Algorithm for rigid-body Brownian dynamics"
       Dan Gordon, Matthew Hoyles, and Shin-Ho Chung 
       http://langevin.anu.edu.au/publications/PhysRevE_80_066703.pdf
     *****************************************************************/
    // BUT: assume diagonal friction tensor 
    //   and no Wiener process / stochastic calculus
    //  then this is just the same as for translation
    // < T_i(t) T_i(t) > = 2 kT friction inertia 
    // friction / kt  = Diff

    // in RB frame     
    Force f = element_mult(transForceCoeff,w1) -
    	element_mult(transDampingCoeff, transpose(orientation)*momentum); 
    
    Force t = element_mult(rotTorqueCoeff,w2) -
    	element_mult(rotDampingCoeff, angularMomentum);

    f = orientation * f; // return to lab frame
    t = orientation * t;
    
    addForce(f);
    addTorque(t);
}

void RigidBody::integrate(Vector *p_trans, Tensor *p_rot, int startFinishAll) {
// from: Dullweber, Leimkuhler, Maclachlan. Symplectic splitting methods for rigid body molecular dynamics. JCP 107. (1997)
// http://jcp.aip.org/resource/1/jcpsa6/v107/i15/p5840_s1
    Vector trans; // = *p_trans;
    Tensor rot = Tensor::identity(); // = *p_rot;

#ifdef DEBUGM
    switch (startFinishAll) {
    case 0: // start
	DebugM(2, "Rigid Body integrating start of cycle" << "\n" << endi);
    case 1: // finish
	DebugM(2, "Rigid Body integrating finish of cycle" << "\n" << endi);
    case 2: // finish and start
	DebugM(2, "Rigid Body integrating finishing last cycle, starting this one" << "\n" << endi);
    }    
#endif

    if ( isnan(force.x) || isnan(torque.x) ) // NaN check
	NAMD_die("Rigid Body force was NaN!\n");

    // torque = Vector(0,0,10); // debug
    Force tmpTorque = transpose(orientation)*torque; // bring to rigid body frame

    DebugM(3, "integrate" <<": force "<<force <<": velocity "<<getVelocity() << "\n" << endi);
    DebugM(3, "integrate" <<": torque "<<tmpTorque <<": orientationalVelocity "<<getAngularVelocity() << "\n" << endi);

    if (startFinishAll == 0 || startFinishAll == 1) {
	// propogate momenta by half step
	momentum += 0.5 * timestep * force * impulse_to_momentum;
	angularMomentum += 0.5 * timestep * tmpTorque * impulse_to_momentum;
    } else {
	// propogate momenta by a full timestep
	momentum += timestep * force * impulse_to_momentum;
	angularMomentum += timestep * tmpTorque * impulse_to_momentum;
    }

    DebugM(3, "  position before: " << position << "\n" << endi);

    if (startFinishAll == 0 || startFinishAll == 2) {
	// update positions
	// trans = Vector(0); if (false) {
	trans = timestep * momentum / mass;
	position += trans; // update CoM a full timestep
	// }

	// update orientations a full timestep
	Tensor R; // represents a rotation about a principle axis
	R = Rx(0.5*timestep * angularMomentum.x / inertia.x ); // R1
	angularMomentum = R * angularMomentum;
	rot = transpose(R);
	DebugM(1, "R: " << R << "\n" << endi);
	DebugM(1, "Rot 1: " << rot << "\n" << endi);

	R = Ry(0.5*timestep * angularMomentum.y / inertia.y ); // R2
	angularMomentum = R * angularMomentum;
	rot = rot * transpose(R);
	DebugM(1, "R: " << R << "\n" << endi);
	DebugM(1, "Rot 2: " << rot << "\n" << endi);

	R = Rz(    timestep * angularMomentum.z / inertia.z ); // R3
	angularMomentum = R * angularMomentum;
	rot = rot * transpose(R);
	DebugM(1, "R: " << R << "\n" << endi);
	DebugM(1, "Rot 3: " << rot << "\n" << endi);

	R = Ry(0.5*timestep * angularMomentum.y / inertia.y ); // R4
	angularMomentum = R * angularMomentum;
	rot = rot * transpose(R);
	DebugM(1, "R: " << R << "\n" << endi);
	DebugM(1, "Rot 4: " << rot << "\n" << endi);

	R = Rx(0.5*timestep * angularMomentum.x / inertia.x ); // R5
	angularMomentum = R * angularMomentum;
	rot = rot * transpose(R);
	DebugM(1, "R: " << R << "\n" << endi);
	DebugM(1, "Rot 5: " << rot << "\n" << endi);

	// DebugM(3,"TEST: " << Ry(0.01) <<"\n" << endi); // DEBUG
 
	// update actual orientation
	Tensor newOrientation = orientation*rot; // not 100% sure; rot could be in rb frame
	orientation = newOrientation;
	rot = transpose(rot);

	DebugM(2, "trans during: " << trans
	       << "\n" << endi);
	DebugM(2, "rot during: " << rot
	       << "\n" << endi);
    
	clearForce();
	clearTorque();
	
	*p_trans = trans;
	*p_rot = rot;
    }
    DebugM(3, "  position after: " << position << "\n" << endi);
}    

// Rotations about axes

// for very small angles 10^-8, cos^2+sin^2 != 1 
// concerned about the accumulation of errors in non-unitary transformations!
Tensor RigidBody::Rx(BigReal t) {
    BigReal qt = 0.25*t*t;  // for approximate calculations of sin(t) and cos(t)
    BigReal cos = (1-qt)/(1+qt);
    BigReal sin = t/(1+qt);

    Tensor tmp;
    tmp.xx = 1; tmp.xy =   0; tmp.xz =    0;
    tmp.yx = 0; tmp.yy = cos; tmp.yz = -sin;
    tmp.zx = 0; tmp.zy = sin; tmp.zz =  cos;
    return tmp;
}
Tensor RigidBody::Ry(BigReal t) {
    BigReal qt = 0.25*t*t;  // for approximate calculations of sin(t) and cos(t)
    BigReal cos = (1-qt)/(1+qt);
    BigReal sin = t/(1+qt);

    Tensor tmp;
    tmp.xx =  cos; tmp.xy = 0; tmp.xz = sin;
    tmp.yx =    0; tmp.yy = 1; tmp.yz =   0;
    tmp.zx = -sin; tmp.zy = 0; tmp.zz = cos;
    return tmp;
}
Tensor RigidBody::Rz(BigReal t) {
    BigReal qt = 0.25*t*t;  // for approximate calculations of sin(t) and cos(t)
    BigReal cos = (1-qt)/(1+qt);
    BigReal sin = t/(1+qt);

    Tensor tmp;
    tmp.xx = cos; tmp.xy = -sin; tmp.xz = 0;
    tmp.yx = sin; tmp.yy =  cos; tmp.yz = 0;
    tmp.zx =   0; tmp.zy =    0; tmp.zz = 1;
    return tmp;
}
Tensor RigidBody::eulerToMatrix(const Vector e) {
    // convert euler angle input to rotation matrix
    // http://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Conversion_formulae_between_formalisms
    return Rz(e.z) * Ry(e.y) * Rx(e.x);
}
