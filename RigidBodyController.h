#include <vector>
#include "RigidBody.h"
/* #include "RandomCUDA.h" */
#include <cuda.h>
#include <cuda_runtime.h>

#define NUMTHREADS 256

class Configuration;

class RigidBodyForcePair  {
	friend class RigidBodyController;

public:
	RigidBodyForcePair(RigidBodyType* t1, RigidBodyType* t2,
										 RigidBody* rb1, RigidBody* rb2,
										 std::vector<int> gridKeyId1, std::vector<int> gridKeyId2) :
		type1(t1), type2(t2), rb1(rb1), rb2(rb2),
		gridKeyId1(gridKeyId1), gridKeyId2(gridKeyId2)
		{
			printf("    Constructing RB force pair...\n");
			initialize();
			// printf("    done constructing RB force pair\n");
		}
	RigidBodyForcePair(const RigidBodyForcePair& o) :
		type1(o.type1), type2(o.type2), rb1(o.rb1), rb2(o.rb2),
		gridKeyId1(o.gridKeyId1), gridKeyId2(o.gridKeyId2) {
		printf("    Copying RB force pair...\n");
		initialize();
	}
	RigidBodyForcePair& operator=(RigidBodyForcePair& o) {
		printf("    Copying assigning RB force pair...\n");
		swap(*this,o);
		return *this;
	}
	
	~RigidBodyForcePair();

private:
	void initialize();
	void swap(RigidBodyForcePair& a, RigidBodyForcePair& b);
	
	static const int numThreads = NUMTHREADS;

	RigidBodyType* type1;
	RigidBodyType* type2;
	RigidBody* rb1;
	RigidBody* rb2;
	
	std::vector<int> gridKeyId1;
	std::vector<int> gridKeyId2;
	std::vector<int> numBlocks;

	std::vector<Vector3*> forces;
	std::vector<Vector3*> forces_d;
	std::vector<Vector3*> torques;
	std::vector<Vector3*> torques_d;
	
	void updateForces();
};

class RigidBodyController {
public:
	/* DEVICE RigidBodyController(const NamdState *s, int reductionTag, SimParameters *sp); */
	RigidBodyController();
  ~RigidBodyController();
	RigidBodyController(const Configuration& c);

	void integrate(int step);
	DEVICE void print(int step);

	void updateForces();
    
private:
	void copyGridsToDevice();
	void initializeForcePairs();
	
	/* void printLegend(std::ofstream &file); */
	/* void printData(int step, std::ofstream &file); */
public:
		RigidBodyType** rbType_d;
	
private:
	/* std::ofstream trajFile; */

	const Configuration& conf;
	
	/* Random* random; */
	/* RequireReduction *gridReduction; */
	
	Vector3* trans; // would have made these static, but
	Matrix3* rot;  	// there are errors on rigidBody->integrate
	std::vector< std::vector<RigidBody> > rigidBodyByType;
	
	std::vector< RigidBodyForcePair > forcePairs;
};
