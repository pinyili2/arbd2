/* #ifndef MIN_DEBUG_LEVEL */
/* #define MIN_DEBUG_LEVEL 5 */
/* #endif */
/* #include "Debug.h" */

#include <iostream>
#include <typeinfo>
#include "RigidBody.h"
#include "Configuration.h"

#include "Debug.h"


RigidBody::RigidBody(String name, const Configuration& cref, RigidBodyType& tref)
	: name(name), c(&cref), t(&tref), impulse_to_momentum(4.184e8f) {
	// units "(kcal_mol/AA) * ns" "amu AA/ns" * 4.184e+08
	
	timestep = c->timestep;
	// RBTODO: fix this
	Temp = 295;
	// tempgrid = c->temperatureGrid;

	position = t->initPos; // Vector3();

	// Orientation matrix that brings vector from the RB frame to the lab frame
	orientation = t->initRot; //Matrix3(1.0f);
	
	momentum = Vector3() * t->mass; // lab frame
	/* DebugM(4, "velocity " << rbParams->velocity << "\n" << endi); */
	DebugM(4, "momentum " << momentum << "\n" << endi);

	angularMomentum = Vector3(); // rigid body frame
	angularMomentum.x *= t->inertia.x;
	angularMomentum.y *= t->inertia.y;
	angularMomentum.z *= t->inertia.z;

	/* isFirstStep = true; // this might not work flawlessly... */

	/* clearForce(); */
	/* clearTorque(); */
    
	/* DebugM(4, "RigidBody initial Force: " << force << "\n" << endi); */
}

void RigidBody::addForce(Force f) { 
	// DebugM(1, "RigidBody "<<key<<" adding f ("<<f<<") to Force " << force << "\n" << endi);    
	force += f; 
} 
void RigidBody::addTorque(Force torq) {
	// DebugM(1, "RigidBody adding t ("<<t<<") to torque " << torque << "\n" << endi);   
	torque += torq; 
}
RigidBody::~RigidBody() {}

	/*===========================================================================\
	| Following "Algorithm for rigid-body Brownian dynamics" Dan Gordon, Matthew |
	|   Hoyles, and Shin-Ho Chung                                                |
	|   http://langevin.anu.edu.au/publications/PhysRevE_80_066703.pdf           |
	|                                                                            |
	|                                                                            |
	| BUT: assume diagonal friction tensor and no Wiener process / stochastic    |
	|   calculus then this is just the same as for translation                   |
	|                                                                            |
	|   < T_i(t) T_i(t) > = 2 kT friction inertia                                |
	|                                                                            |
	|   friction / kt = Diff                                                     |
	\===========================================================================*/
void RigidBody::addLangevin(Vector3 w1, Vector3 w2) {
	// w1 and w2 should be standard normal distributions

	// in RB frame     
	Force f = Vector3::element_mult(t->transForceCoeff,w1) -
		Vector3::element_mult(t->transDamping, orientation.transpose()*momentum); 
    
	Force torq = Vector3::element_mult(t->rotTorqueCoeff,w2) -
		Vector3::element_mult(t->rotDamping, angularMomentum);

	f = orientation * f; // return to lab frame
	torq = orientation * torq;

	// printf("LANGTORQUE: %f %f %f\n",torq.x,torq.y,torq.z);
	
	addForce(f);
	addTorque(torq);
}

  /*==========================================================================\
	| from: Dullweber, Leimkuhler, Maclachlan. Symplectic splitting methods for |
	| rigid body molecular dynamics. JCP 107. (1997)                            |
	| http://jcp.aip.org/resource/1/jcpsa6/v107/i15/p5840_s1                    |
	\==========================================================================*/
// void RigidBody::integrate(Vector3& old_trans, Matrix3& old_rot, int startFinishAll) {}
void RigidBody::integrate(int startFinishAll) {
	Vector3 trans; // = *p_trans;
	Matrix3 rot = Matrix3(1); // = *p_rot;

	/* printf("Rigid Body force\n"); */
	
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

	if ( isnan(force.x) || isnan(torque.x) ) { // NaN check
		printf("Rigid Body force was NaN!\n");
		exit(-1);
	}
	// torque = Vector3(0,0,10); // + orientation.transpose()*Vector3(1,0,0);

	if (startFinishAll == 0 || startFinishAll == 1) {
		// propogate momenta by half step
		momentum += 0.5 * timestep * force * impulse_to_momentum;
		angularMomentum += 0.5 * timestep * torque * impulse_to_momentum;
	} else {
		// propogate momenta by a full timestep
		momentum += timestep * force * impulse_to_momentum;
		angularMomentum += timestep * torque * impulse_to_momentum;
	}

	DebugM(3, "  position before: " << position << "\n" << endi);

	if (startFinishAll == 0 || startFinishAll == 2) {
		// update positions
		// trans = Vector(0); if (false) {
		trans = timestep * momentum / t->mass;
		position += trans; // update CoM a full timestep
		// }

		// update orientations a full timestep
		Matrix3 R; // represents a rotation about a principle axis
		R = Rx(0.5*timestep * angularMomentum.x / t->inertia.x ); // R1
		angularMomentum = R * angularMomentum;
		orientation = R * orientation;
		DebugM(1, "R: " << R << "\n" << endi);
		DebugM(1, "Rot 1: " << rot << "\n" << endi);

		R = Ry(0.5*timestep * angularMomentum.y / t->inertia.y ); // R2
		angularMomentum = R * angularMomentum;
		orientation = R * orientation;
		DebugM(1, "R: " << R << "\n" << endi);
		DebugM(1, "Rot 2: " << rot << "\n" << endi);

		R = Rz(    timestep * angularMomentum.z / t->inertia.z ); // R3
		angularMomentum = R * angularMomentum;
		orientation = R * orientation;
		DebugM(1, "R: " << R << "\n" << endi);
		DebugM(1, "Rot 3: " << rot << "\n" << endi);

		R = Ry(0.5*timestep * angularMomentum.y / t->inertia.y ); // R4
		angularMomentum = R * angularMomentum;
		orientation = R * orientation;
		DebugM(1, "R: " << R << "\n" << endi);
		DebugM(1, "Rot 4: " << rot << "\n" << endi);

		R = Rx(0.5*timestep * angularMomentum.x / t->inertia.x ); // R5
		angularMomentum = R * angularMomentum;
		orientation = R * orientation;
		DebugM(1, "R: " << R << "\n" << endi);
		DebugM(1, "Rot 5: " << rot << "\n" << endi);
	}
	DebugM(3, "  position after: " << position << "\n" << endi);
}    

// Rotations about axes
// for very small angles 10^-8, cos^2+sin^2 != 1 
// concerned about the accumulation of errors in non-unitary transformations!
Matrix3 RigidBody::Rx(BigReal t) {
	BigReal qt = 0.25*t*t;  // for approximate calculations of sin(t) and cos(t)
	BigReal cos = (1-qt)/(1+qt);
	BigReal sin = t/(1+qt);

	Matrix3 tmp;
	tmp.exx = 1; tmp.exy =   0; tmp.exz =    0;
	tmp.eyx = 0; tmp.eyy = cos; tmp.eyz = -sin;
	tmp.ezx = 0; tmp.ezy = sin; tmp.ezz =  cos;
	return tmp;
}
Matrix3 RigidBody::Ry(BigReal t) {
	BigReal qt = 0.25*t*t;  // for approximate calculations of sin(t) and cos(t)
	BigReal cos = (1-qt)/(1+qt);
	BigReal sin = t/(1+qt);

	Matrix3 tmp;
	tmp.exx =  cos; tmp.exy = 0; tmp.exz = sin;
	tmp.eyx =    0; tmp.eyy = 1; tmp.eyz =   0;
	tmp.ezx = -sin; tmp.ezy = 0; tmp.ezz = cos;
	return tmp;
}
Matrix3 RigidBody::Rz(BigReal t) {
	BigReal qt = 0.25*t*t;  // for approximate calculations of sin(t) and cos(t)
	BigReal cos = (1-qt)/(1+qt);
	BigReal sin = t/(1+qt);

	Matrix3 tmp;
	tmp.exx = cos; tmp.exy = -sin; tmp.exz = 0;
	tmp.eyx = sin; tmp.eyy =  cos; tmp.eyz = 0;
	tmp.ezx =   0; tmp.ezy =    0; tmp.ezz = 1;
	return tmp;
}
Matrix3 RigidBody::eulerToMatrix(const Vector3 e) {
	// convert euler angle input to rotation matrix
	// http://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Conversion_formulae_between_formalisms
	return Rz(e.z) * Ry(e.y) * Rx(e.x);
}
