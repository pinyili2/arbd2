/*===========================\
| RigidBody Class for device |
\===========================*/
#pragma once

#include "useful.h"
#include "RandomCPU.h"		/* for BD integration; RBTODO: fix this */

#ifdef __CUDACC__
    #define HOST __host__
    #define DEVICE __device__
#else
    #define HOST 
    #define DEVICE 
#endif

#include "RigidBodyType.h"
#include "RigidBodyController.h"

class Configuration;


typedef float BigReal;					/* strip this out later */
typedef Vector3 Force;


class RigidBody { // host side representation of rigid bodies
	friend class RigidBodyController;
	/*––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––.
	| See Appendix A of: Dullweber, Leimkuhler and McLaclan. "Symplectic        |
	| splitting methods for rigid body molecular dynamics". J Chem Phys. (1997) |
	`––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––*/
		public:
	RigidBody(String name, const Configuration& c, const RigidBodyType& t, RigidBodyController* RBC);
    RigidBody(const RigidBody& rb);
    // RigidBody(const RigidBody& rb) : RigidBody(rb.name, *rb.c, *rb.t) {};
	void init();
	/* HOST DEVICE RigidBody(RigidBodyType t); */
	~RigidBody();

	HOST DEVICE void addForce(Force f); 
	HOST DEVICE void addTorque(Force t);
	HOST DEVICE void addLangevin(Vector3 w1, Vector3 w2);
	
	HOST DEVICE inline void clearForce() { force = Force(0.0f); }
	HOST DEVICE inline void clearTorque() { torque = Force(0.0f); }

	// HOST DEVICE void integrate(Vector3& old_trans, Matrix3& old_rot, int startFinishAll);
	// HOST DEVICE void integrate(Vector3& old_trans, Matrix3& old_rot, int startFinishAll);
	void integrateDLM(int startFinishAll);
	void integrate(int startFinishAll);	

	// HOST DEVICE inline String getKey() const { return key; }
	// HOST DEVICE inline String getKey() const { return t->name; }
	HOST DEVICE inline String getKey() const { return name; }
	
	HOST DEVICE inline Vector3 transformBodyToLab(Vector3 v) const { return orientation*v + position; }
	HOST DEVICE inline Vector3 getPosition() const { return position; }
	HOST DEVICE inline Matrix3 getOrientation() const { return orientation; }
	// HOST DEVICE inline Matrix3 getBasis() const { return orientation; }
	HOST DEVICE inline BigReal getMass() const { return t->mass; }
	//HOST DEVICE inline Vector3 getVelocity() const { return momentum/t->mass; }
	HOST DEVICE inline Vector3 getVelocity() const { return momentum; }
	//HOST DEVICE inline Vector3 getAngularVelocity() const { 
	//	return Vector3( angularMomentum.x / t->inertia.x,
	//								 angularMomentum.y / t->inertia.y,
									 //angularMomentum.z / t->inertia.z );
	//}
	HOST DEVICE inline Vector3 getAngularVelocity() const { 
              return Vector3( angularMomentum.x, angularMomentum.y, angularMomentum.z);
        }

	void updateParticleList(Vector3* pos_d);
	void callGridParticleForceKernel(Vector3* pos_d, Vector3* force_d, int s);
	void retrieveGridParticleForces();
	
	bool langevin;
	Vector3 torque; // lab frame (except in integrate())
        
private:
	
	RigidBodyController* RBC;
	inline Vector3 getRandomGaussVector() { 
	    return RBC->getRandomGaussVector();
	}

	// String key;
	String name;
	/* static const SimParameters * simParams; */
	Vector3 position;		  /* position of center of mass */
	// Q = orientation.transpose(); in Dullweber et al
	Matrix3 orientation;					/* rotation that brings RB coordinates into the lab frame */

	Vector3 momentum;		 /* in lab frame */
	Vector3 angularMomentum; // angular momentum along corresponding principal axes
        Vector3 W1,W2;
 
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
	const RigidBodyType* t;
	float timestep;					
	Vector3 force;  // lab frame

	bool isFirstStep; 
	
	int* numParticles;		  /* particles affected by potential grids */
	int** particles_d;		 	
	Vector3** particleForces;
	Vector3** particleTorques;
	Vector3** particleForces_d;
	Vector3** particleTorques_d;
	
	
	/*–––––––––––––––––––––––––––––––––––––––––.
	| units "kcal_mol/AA * ns" "(AA/ns) * amu" |
	`–––––––––––––––––––––––––––––––––––––––––*/
	BigReal impulse_to_momentum; /* should be const, but copy constructor failed */

	HOST DEVICE inline void applyRotation(const Matrix3& R);
	HOST DEVICE inline Matrix3 Rx(BigReal t);
	HOST DEVICE inline Matrix3 Ry(BigReal t);
	HOST DEVICE inline Matrix3 Rz(BigReal t);
	HOST DEVICE inline Matrix3 eulerToMatrix(const Vector3 e);
        float Temperature();
        void  Boltzmann();
};

