/*===========================\
| RigidBody Class for device |
\===========================*/
#pragma once

#include "useful.h"
#include "RigidBodyType.h"


#ifdef __CUDACC__
    #define HOST __host__
    #define DEVICE __device__
#else
    #define HOST 
    #define DEVICE 
#endif

class Configuration;						/* forward decleration */

typedef float BigReal;					/* strip this out later */
typedef Vector3 Force;


class RigidBody { // host side representation of rigid bodies
	/*––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––.
	| See Appendix A of: Dullweber, Leimkuhler and McLaclan. "Symplectic        |
	| splitting methods for rigid body molecular dynamics". J Chem Phys. (1997) |
	`––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––*/
		public:
	HOST DEVICE RigidBody(const Configuration& c, RigidBodyType& t);
	/* HOST DEVICE RigidBody(RigidBodyType t); */
	HOST DEVICE ~RigidBody();

	HOST DEVICE void addForce(Force f); 
	HOST DEVICE void addTorque(Force t);
	HOST DEVICE void addLangevin(Vector3 w1, Vector3 w2);
	
	HOST DEVICE inline void clearForce() { force = Force(0.0f); }
	HOST DEVICE inline void clearTorque() { torque = Force(0.0f); }

	HOST DEVICE void integrate(Vector3& old_trans, Matrix3& old_rot, int startFinishAll);
	
	HOST DEVICE inline String getKey() { return key; }

	HOST DEVICE inline Vector3 getPosition() const { return position; }
	HOST DEVICE inline Matrix3 getOrientation() const { return orientation; }
	HOST DEVICE inline Matrix3 getBasis() const { return orientation; }
	HOST DEVICE inline BigReal getMass() const { return t->mass; }
	HOST DEVICE inline Vector3 getVelocity() const { return momentum/t->mass; }
	HOST DEVICE inline Vector3 getAngularVelocity() const { 
		return Vector3( angularMomentum.x / t->inertia.x,
									 angularMomentum.y / t->inertia.y,
									 angularMomentum.z / t->inertia.z );
	}
	bool langevin;
    
private:
	String key;
	/* static const SimParameters * simParams; */
	Vector3 position;
	Matrix3 orientation;

	Vector3 momentum;
	Vector3 angularMomentum; // angular momentum along corresponding principal axes
    
	// Langevin
	Vector3 langevinTransFriction; /* RBTODO: make this work with a grid */
	Vector3 langevinRotFriction;
	BigReal Temp;

	/* Vector3 transDampingCoeff; */
	/* Vector3 transForceCoeff; */
	/* Vector3 rotDampingCoeff; */
	/* Vector3 rotTorqueCoeff;     */

	// integration
	const Configuration* c;
	RigidBodyType* t;					/* RBTODO: const? */
	int timestep;
	Vector3 force;  // lab frame
	Vector3 torque; // lab frame (except in integrate())

	bool isFirstStep; 

	/*–––––––––––––––––––––––––––––––––––––––––.
	| units "kcal_mol/AA * fs" "(AA/fs) * amu" |
	`–––––––––––––––––––––––––––––––––––––––––*/
	BigReal impulse_to_momentum; /* should be const, but copy constructor failed */


	HOST DEVICE inline Matrix3 Rx(BigReal t);
	HOST DEVICE inline Matrix3 Ry(BigReal t);
	HOST DEVICE inline Matrix3 Rz(BigReal t);
	HOST DEVICE inline Matrix3 eulerToMatrix(const Vector3 e);
};

