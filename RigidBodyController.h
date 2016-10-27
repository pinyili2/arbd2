#pragma once

#include <vector>
#include <fstream>
/* #include "RandomCUDA.h" */
#include <cuda.h>
#include <cuda_runtime.h>
#include "useful.h"

// #define NUMTHREADS 128					/* try with 64, every 32+ */
#define NUMTHREADS 96
#define NUMSTREAMS 8

// #include "RigidBody.h"

class RigidBodyType;
class RigidBody;
class Configuration;
class RandomCPU;


// TODO: performance: create RigidBodyGridPair so pairlistdist check is done per grid pair, not per RB pair
class RigidBodyForcePair  {
	friend class RigidBodyController;

public:
	RigidBodyForcePair(RigidBodyType* t1, RigidBodyType* t2,
										 RigidBody* rb1, RigidBody* rb2,
										 std::vector<int> gridKeyId1, std::vector<int> gridKeyId2, bool isPmf) :
		type1(t1), type2(t2), rb1(rb1), rb2(rb2),
		gridKeyId1(gridKeyId1), gridKeyId2(gridKeyId2), isPmf(isPmf)
		{
			printf("    Constructing RB force pair...\n");
			/* initialize(); */
			// printf("    done constructing RB force pair\n");
		}
	RigidBodyForcePair(const RigidBodyForcePair& o) :
		type1(o.type1), type2(o.type2), rb1(o.rb1), rb2(o.rb2),
		gridKeyId1(o.gridKeyId1), gridKeyId2(o.gridKeyId2), isPmf(o.isPmf) {
		printf("    Copying RB force pair...\n");
		/* initialize(); */
	}
	RigidBodyForcePair& operator=(RigidBodyForcePair& o) {
		printf("    Copying assigning RB force pair...\n");
		swap(*this,o);
		return *this;
	}	
	~RigidBodyForcePair();

	bool isWithinPairlistDist() const;	

private:
	int initialize();
	void swap(RigidBodyForcePair& a, RigidBodyForcePair& b);
	
	static const int numThreads = NUMTHREADS;
	
	RigidBodyType* type1;
	RigidBodyType* type2;
	RigidBody* rb1;
	RigidBody* rb2;
	
	std::vector<int> gridKeyId1;
	std::vector<int> gridKeyId2;
	std::vector<int> numBlocks;

	bool isPmf;
	
	std::vector<Vector3*> forces;
	std::vector<Vector3*> forces_d;
	std::vector<Vector3*> torques;
	std::vector<Vector3*> torques_d;

	static int nextStreamID; 
	std::vector<int> streamID;
	static cudaStream_t* stream;
	static void createStreams();

	static int lastStreamID;
	static RigidBodyForcePair* lastRbForcePair;
	static int lastRbGridID;

	void callGridForceKernel(int pairId, int s);
	void retrieveForcesForGrid(const int i);
	void processGPUForces();
	Matrix3 getBasis1(const int i);
	Matrix3 getBasis2(const int i);
	Vector3 getOrigin1(const int i);
	Vector3 getOrigin2(const int i);
};

class RigidBodyController {
public:
	/* DEVICE RigidBodyController(const NamdState *s, int reductionTag, SimParameters *sp); */
	RigidBodyController();
  ~RigidBodyController();
	RigidBodyController(const Configuration& c, const char* outArg);

	void integrate(int step);
	void updateForces(Vector3* pos_d, Vector3* force_d, int s);
	void updateParticleLists(Vector3* pos_d);
    
private:
	bool loadRBCoordinates(const char* fileName);
	void initializeForcePairs();

	void print(int step);
	void printLegend(std::ofstream &file);
	void printData(int step, std::ofstream &file);
public:
		RigidBodyType** rbType_d;
	
private:
	std::ofstream trajFile;
	
	const Configuration& conf;
	const char* outArg;
	
	RandomCPU* random;
	/* RequireReduction *gridReduction; */
	
	Vector3* trans; // would have made these static, but
	Matrix3* rot;  	// there are errors on rigidBody->integrate
	std::vector< std::vector<RigidBody> > rigidBodyByType;
	
	std::vector< RigidBodyForcePair > forcePairs;

	
	
};
