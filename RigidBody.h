/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

#pragma once

/* #include <set> */
/* #include <vector> */

#include "useful.h"
/* #include "Vector.h" */
/* #include "Tensor.h" */


/* #include "strlib.h" */
/* #include "common.h" */
/* #include "Vector.h" */
/* #include "Tensor.h" */
/* #include "InfoStream.h" */
/* #include "MStream.h" */


/* #include "SimParameters.h" */
/* #include "NamdTypes.h" */

typedef BigReal float;					/* strip this out later */
typedef Force Vector3;

#define DEVICE __device__

/* class ComputeMgr; */
/* class Random; */

class RigidBody {
	/*=====================================================================\
		|   See Appendix A of: Dullweber, Leimkuhler and McLaclan. "Symplectic |
		|   splitting methods for rigid body molecular dynamics". J Chem       |
		|   Phys. (1997)                                                       |
		\=====================================================================*/
public:
	DEVICE RigidBody(SimParameters *simParams, RigidBodyParams *rbParams);
	DEVICE ~RigidBody();

	DEVICE void addForce(Force f); 
	DEVICE void addTorque(Force t);
	DEVICE void addLangevin(Vector3 w1, Vector3 w2);

	DEVICE inline void clearForce() { force = Force(0); }
	DEVICE inline void clearTorque() { torque = Force(0); }

	DEVICE void integrate(Vector3 *p_trans, Matrix3 *p_rot, int startFinishAll);

	DEVICE inline char* getKey() { return key; }
	DEVICE inline Vector3 getPosition() { return position; }
	DEVICE inline Matrix3 getOrientation() { return orientation; }
	DEVICE inline BigReal getMass() { return mass; }
	DEVICE inline Vector3 getVelocity() { return momentum/mass; }
	DEVICE inline Vector3 getAngularVelocity() { 
		return Vector3( angularMomentum.x / inertia.x,
									 angularMomentum.y / inertia.y,
									 angularMomentum.z / inertia.z );
	}
	bool langevin;
    
private:
	String key;
	/* static const SimParameters * simParams; */
	BigReal mass;

	Vector3 position;
	Matrix3 orientation;

	Vector3 inertia; // diagonal elements of inertia tensor
	Vector3 momentum;
	Vector3 angularMomentum; // angular momentum along corresponding principal axes
    
	// Langevin
	Vector3 langevinTransFriction;
	Vector3 langevinRotFriction;
	BigReal Temp;

	Vector3 transDampingCoeff;
	Vector3 transForceCoeff;
	Vector3 rotDampingCoeff;
	Vector3 rotTorqueCoeff;    

	// integration
	int timestep;
	Vector3 force;  // lab frame
	Vector3 torque; // lab frame (except in integrate())

	bool isFirstStep; 

	// units "kcal_mol/AA * fs"  "(AA/fs) * amu"
	const BigReal impulse_to_momentum;

	DEVICE inline Matrix3 Rx(BigReal t);
	DEVICE inline Matrix3 Ry(BigReal t);
	DEVICE inline Matrix3 Rz(BigReal t);
	DEVICE inline Matrix3 eulerToMatrix(const Vector3 e);
};

class RigidBodyController {
public:
	/* DEVICE RigidBodyController(const NamdState *s, int reductionTag, SimParameters *sp); */
	DEVICE RigidBodyController(const NamdState *s, int reductionTag, SimParameters *sp);
	
	DEVICE ~RigidBodyController();
	DEVICE void integrate(int step);
	DEVICE void print(int step);
    
private:
	/* void printLegend(std::ofstream &file); */
	/* void printData(int step, std::ofstream &file); */

	/* SimParameters* simParams; */
	/* const NamdState* state;		 */    
	/* std::ofstream trajFile; */

	Random* random;
	/* RequireReduction *gridReduction; */
	
	Vector3* trans; // would have made these static, but
	Matrix3* rot;  	// there are errors on rigidBody->integrate
	RigidBody* rigidBodyList;
	
};

class RigidBodyParams {
public:
	RigidBodyParams() {
		rigidBodyKey = 0;
		mass = 0;
		inertia = Vector3();
		langevin = FALSE;
		temperature = 0;
		transDampingCoeff = Vector3();
		rotDampingCoeff = Vector3();
		gridList;
		position = Vector3();
		velocity = Vector3();
		orientation = Matrix3();
		orientationalVelocity = Vector3();	   
	}    
	int typeID;

	String *rigidBodyKey;
	BigReal mass;
	Vector3 inertia;
	Bool langevin;
	BigReal temperature;
	Vector3 transDampingCoeff;
	Vector3 rotDampingCoeff;
	String *gridList;
	
    
	Vector3 position;
	Vector3 velocity;
	Matrix3 orientation;
	Vector3 orientationalVelocity;

	RigidBodyParams *next;

	const void print();
};


/* class RigidBodyParamsList { */
/* public: */
/*   RigidBodyParamsList() { */
/*     clear(); */
/*   } */
  
/*   ~RigidBodyParamsList()  */
/*   { */
/*     RBElem* cur; */
/*     while (head != NULL) { */
/*       cur = head; */
/*       head = cur->nxt; */
/*       delete cur; */
/*     } */
/*     clear(); */
/*   } */
/*   const void print(char *s); */
/*   const void print(); */

/*   // The SimParameters bit copy overwrites these values with illegal pointers, */
/*   // So thise throws away the garbage and lets everything be reinitialized */
/*   // from scratch */
/*   void clear() { */
/*     head = tail = NULL; */
/*     n_elements = 0; */
/*   } */
  
/*   RigidBodyParams* find_key(const char* key);   */
/*   int index_for_key(const char* key); */
/*   RigidBodyParams* add(const char* key); */
  
/*   RigidBodyParams *get_first() { */
/*     if (head == NULL) { */
/*       return NULL; */
/*     } else return &(head->elem); */
/*   } */
  
/*   void pack_data(MOStream *msg);   */
/*   void unpack_data(MIStream *msg); */
  
/*   // convert from a string to Bool; returns 1(TRUE) 0(FALSE) or -1(if unknown) */
/*   static int atoBool(const char *s) */
/*   { */
/*     if (!strcasecmp(s, "on")) return 1; */
/*     if (!strcasecmp(s, "off")) return 0; */
/*     if (!strcasecmp(s, "true")) return 1; */
/*     if (!strcasecmp(s, "false")) return 0; */
/*     if (!strcasecmp(s, "yes")) return 1; */
/*     if (!strcasecmp(s, "no")) return 0; */
/*     if (!strcasecmp(s, "1")) return 1; */
/*     if (!strcasecmp(s, "0")) return 0; */
/*     return -1; */
/*   } */


/* private: */
/*   class RBElem { */
/*   public: */
/*     RigidBodyParams elem; */
/*     RBElem* nxt; */
/*   }; */
/*   RBElem* head; */
/*   RBElem* tail; */
/*   int n_elements; */

/* }; */
