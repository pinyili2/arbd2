#include <vector>
#include "RigidBody.h"
/* #include "RandomCUDA.h" */
#include <cuda.h>
#include <cuda_runtime.h>

#include "ComputeGridGrid.cuh"

class Configuration;

struct gridInteractionList {
	
};

class RigidBodyController {
public:
	/* DEVICE RigidBodyController(const NamdState *s, int reductionTag, SimParameters *sp); */
	RigidBodyController();
  ~RigidBodyController();
	RigidBodyController(const Configuration& c);

	DEVICE void integrate(int step);
	DEVICE void print(int step);
    
private:
	void updateForces();

	/* void printLegend(std::ofstream &file); */
	/* void printData(int step, std::ofstream &file); */

	/* std::ofstream trajFile; */

	const Configuration& conf;
	
	/* Random* random; */
	/* RequireReduction *gridReduction; */
	
	Vector3* trans; // would have made these static, but
	Matrix3* rot;  	// there are errors on rigidBody->integrate

	std::vector< std::vector<RigidBody> > rigidBodyByType;
	/* RigidBody* rigidBodyList; */

	
	
};

