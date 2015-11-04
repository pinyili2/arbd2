/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

#pragma once

#include <set>
#include <vector>

#include "Vector.h"
#include "Tensor.h"


/* #include "strlib.h" */
/* #include "common.h" */
/* #include "Vector.h" */
/* #include "Tensor.h" */
/* #include "InfoStream.h" */
/* #include "MStream.h" */


/* #include "SimParameters.h" */
/* #include "NamdTypes.h" */

typedef BigReal float;					/* strip this out later */

class ComputeMgr;
class Random;

class RigidBody {
	/*=====================================================================\
	|   See Appendix A of: Dullweber, Leimkuhler and McLaclan. "Symplectic |
	|   splitting methods for rigid body molecular dynamics". J Chem       |
	|   Phys. (1997)                                                       |
	\=====================================================================*/
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

class RigidBodyController {
public:
	RigidBodyController(const NamdState *s, int reductionTag, SimParameters *sp);
	~RigidBodyController();
	void integrate(int step);
	void print(int step);
    
private:
	void printLegend(std::ofstream &file);
	void printData(int step, std::ofstream &file);

	SimParameters *simParams;
	const NamdState * state;		
    
	std::ofstream trajFile;

	Random *random;
	ComputeMgr *computeMgr;
	RequireReduction *gridReduction;

	ResizeArray<Vector> trans; // would have made these static, but
	ResizeArray<Tensor> rot;	// there are errors on rigidBody->integrate
	std::vector<RigidBody*> rigidBodyList;
};

class RigidBodyParams {
public:
    RigidBodyParams() {
	rigidBodyKey = 0;
	mass = 0;
	inertia = Vector(0);
	langevin = FALSE;
	temperature = 0;
	transDampingCoeff = Vector(0);
	rotDampingCoeff = Vector(0);
	gridList;
	position = Vector(0);
	velocity = Vector(0);
	orientation = Tensor();
	orientationalVelocity = Vector(0);	   
    }    
    char *rigidBodyKey;
    BigReal mass;
    zVector inertia;
    Bool langevin;
    BigReal temperature;
    zVector transDampingCoeff;
    zVector rotDampingCoeff;
    std::vector<std::string> gridList;
    
    zVector position;
    zVector velocity;
    Tensor orientation;
    zVector orientationalVelocity;

    RigidBodyParams *next;

    const void print();
};


class RigidBodyParamsList {
public:
  RigidBodyParamsList() {
    clear();
  }
  
  ~RigidBodyParamsList() 
  {
    RBElem* cur;
    while (head != NULL) {
      cur = head;
      head = cur->nxt;
      delete cur;
    }
    clear();
  }
  const void print(char *s);
  const void print();

  // The SimParameters bit copy overwrites these values with illegal pointers,
  // So thise throws away the garbage and lets everything be reinitialized
  // from scratch
  void clear() {
    head = tail = NULL;
    n_elements = 0;
  }
  
  RigidBodyParams* find_key(const char* key);  
  int index_for_key(const char* key);
  RigidBodyParams* add(const char* key);
  
  RigidBodyParams *get_first() {
    if (head == NULL) {
      return NULL;
    } else return &(head->elem);
  }
  
  void pack_data(MOStream *msg);  
  void unpack_data(MIStream *msg);
  
  // convert from a string to Bool; returns 1(TRUE) 0(FALSE) or -1(if unknown)
  static int atoBool(const char *s)
  {
    if (!strcasecmp(s, "on")) return 1;
    if (!strcasecmp(s, "off")) return 0;
    if (!strcasecmp(s, "true")) return 1;
    if (!strcasecmp(s, "false")) return 0;
    if (!strcasecmp(s, "yes")) return 1;
    if (!strcasecmp(s, "no")) return 0;
    if (!strcasecmp(s, "1")) return 1;
    if (!strcasecmp(s, "0")) return 0;
    return -1;
  }


private:
  class RBElem {
  public:
    RigidBodyParams elem;
    RBElem* nxt;
  };
  RBElem* head;
  RBElem* tail;
  int n_elements;

};
