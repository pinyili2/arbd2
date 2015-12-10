/* #ifndef MIN_DEBUG_LEVEL */
/* #define MIN_DEBUG_LEVEL 5 */
/* #endif */
/* #define DEBUGM */
/* #include "Debug.h" */

/* #include "RigidBody.h" */
#include "RigidBodyController.h"
#include "Configuration.h"
#include "RigidBodyType.h"
#include "ComputeGridGrid.cuh"

// #include <vector>

/* #include "Random.h" */

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, String file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), __FILE__, line);
      if (abort) exit(code);
   }
}

/* #include <cuda.h> */
/* #include <cuda_runtime.h> */
/* #include <curand_kernel.h> */

RigidBodyController::RigidBodyController(const Configuration& c) :
conf(c) {

	if (conf.numRigidTypes > 0) {
		copyGridsToDevice();
	}

	int numRB = 0;
	// grow list of rbs
	for (int i = 0; i < conf.numRigidTypes; i++) {			
		numRB += conf.rigidBody[i].num;
		std::vector<RigidBody> tmp;
		for (int j = 0; j < conf.rigidBody[i].num; j++) {
			RigidBody r(conf, conf.rigidBody[i]);
			tmp.push_back( r );
		}
		rigidBodyByType.push_back(tmp);
	}

	initializeForcePairs();
}
RigidBodyController::~RigidBodyController() {
	for (int i = 0; i < rigidBodyByType.size(); i++)
		rigidBodyByType[i].clear();
	rigidBodyByType.clear();
}

void RigidBodyController::initializeForcePairs() {
	// Loop over all pairs of rigid body types
	//   the references here make the code more readable, but they may incur a performance loss
	printf("Initializing force pairs\n");
	for (int ti = 0; ti < conf.numRigidTypes; ti++) {
		RigidBodyType& t1 = conf.rigidBody[ti];
		for (int tj = ti; tj < conf.numRigidTypes; tj++) {
			RigidBodyType& t2 = conf.rigidBody[tj];


			const std::vector<String>& keys1 = t1.densityGridKeys; 
			const std::vector<String>& keys2 = t2.potentialGridKeys;

			printf("  Working on type pair ");
			t1.name.printInline(); printf(":"); t2.name.print();
			
			// Loop over all pairs of grid keys (e.g. "Elec")
			std::vector<int> gridKeyId1;
			std::vector<int> gridKeyId2;
			
			printf("  Grid keys %d:%d\n",keys1.size(),keys2.size());

			bool paired = false;
			for(int k1 = 0; k1 < keys1.size(); k1++) {
				for(int k2 = 0; k2 < keys2.size(); k2++) {
					printf("    checking grid keys ");
					keys1[k1].printInline(); printf(":"); keys2[k2].print();
					
					if ( keys1[k1] == keys2[k2] ) {
						gridKeyId1.push_back(k1);
						gridKeyId2.push_back(k2);
						paired = true;
					}
				}
			}
			
			if (paired) {
				// found matching keys => calculate force between all grid pairs
				std::vector<RigidBody>& rbs1 = rigidBodyByType[ti];
				std::vector<RigidBody>& rbs2 = rigidBodyByType[tj];

				// Loop over rigid bodies of these types
				for (int i = 0; i < rbs1.size(); i++) {
					for (int j = (ti==tj ? i+1 : 0); j < rbs2.size(); j++) {
						RigidBody* rb1 = &(rbs1[i]);
						RigidBody* rb2 = &(rbs2[j]);

						printf("    pushing RB force pair for %d:%d\n",i,j);
						RigidBodyForcePair fp = RigidBodyForcePair(&(t1),&(t2),rb1,rb2,gridKeyId1,gridKeyId2);
						gpuErrchk(cudaDeviceSynchronize()); /* RBTODO: this should be extraneous */
						forcePairs.push_back( fp ); 
						printf("    done pushing RB force pair for %d:%d\n",i,j);
					}
				}
			}
		}
	}
}
	
void RigidBodyController::updateForces() {
	/*––{ RBTODO }–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––.
	| probably coalesce kernel calls, or move this to a device kernel caller   |
	|                                                                          |
	| - consider removing references (unless they are optimized out!) ---      |
	| - caclulate numthreads && numblocks                                      |
	|                                                                          |
	| all threads in a block should: ----------------------------------------  |
	|   (1) apply the same transformations to get the data point position in a |
	|   destination grid ----------------------------------------------------- |
	|   (2) reduce forces and torques to the same location ------------------- |
	|   (3) ???  ------------------------------------------------------------- |
	|                                                                          |
	| Opportunities for memory bandwidth savings:                              |
	`–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––*/
	// int numBlocks = (num * numReplicas) / NUM_THREADS + (num * numReplicas % NUM_THREADS == 0 ? 0 : 1);
	// int numBlocks = 1;
	/* int numThreads = 256; */

	for (int i=0; i < forcePairs.size(); i++) {
		forcePairs[i].updateForces();
	}	
	// get 3rd law forces and torques
		
	// RBTODO: see if there is a better way to sync
	gpuErrchk(cudaDeviceSynchronize());

}

void RigidBodyForcePair::updateForces() {
	// get the force/torque between a pair of rigid bodies
	printf("  Updating rbPair forces\n");
	const int numGrids = gridKeyId1.size();

	// RBTODO: precompute certain common transformations and pass in kernel call
	for (int i = 0; i < numGrids; i++) {
		const int nb = numBlocks[i];
		const int k1 = gridKeyId1[i];
		const int k2 = gridKeyId2[i];

		printf("  Calculating grid forces\n");

		computeGridGridForce<<< nb, numThreads >>>
		(type1->rawDensityGrids_d[k1], type2->rawPotentialGrids_d[k2],
		 rb1->getBasis(), rb1->getPosition(), /* RBTODO: include offset from grid */
		 rb2->getBasis(), rb2->getPosition(),
		 forces_d[i], torques_d[i]);
		
		// RBTODO: ASYNCHRONOUSLY retrieve forces
	}
	gpuErrchk(cudaDeviceSynchronize());
	for (int i = 0; i < numGrids; i++) {
		const int nb = numBlocks[i];
		gpuErrchk(cudaMemcpy(forces[i], &(forces_d[i]), sizeof(Vector3)*nb,
												 cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(torques[i], &(torques_d[i]), sizeof(Vector3)*nb,
												 cudaMemcpyDeviceToHost));
	}

	gpuErrchk(cudaDeviceSynchronize());

	// sum forces + torques
	Vector3 f = Vector3(0.0f);
	Vector3 t = Vector3(0.0f);

	for (int i = 0; i < numGrids; i++) {
		const int nb = numBlocks[i];
		for (int j = 0; j < nb; j++) {
			f = f + forces[i][j];
			t = t + forces[i][j];
		}
	}

	// transform torque from lab-frame origin to rb centers
	Vector3 t1 = t - rb1->getPosition().cross( f );
	Vector3 t2 = -t - rb2->getPosition().cross( -f );

	// add forces to rbs
	rb1->addForce( f);
	rb1->addTorque(t1);
	rb2->addForce(-f);
	rb2->addTorque(t2);
	
	// integrate
	
}

void RigidBodyController::copyGridsToDevice() {
	RigidBodyType **rb_addr = new RigidBodyType*[conf.numRigidTypes];	/* temporary pointer to device pointer */

	gpuErrchk(cudaMalloc(&rbType_d, sizeof(RigidBodyType*) * conf.numRigidTypes));
	// TODO: The above line fails when there is not enough memory. If it fails, stop.

	printf("Copying RBs\n");
	// Copy rigidbody types 
	// http://stackoverflow.com/questions/16024087/copy-an-object-to-device
 	for (int i = 0; i < conf.numRigidTypes; i++) {
		printf("Working on RB %d\n",i);
		RigidBodyType& rb = conf.rigidBody[i]; // temporary for convenience
		rb.updateRaw();

		int ng = rb.numPotGrids;

		printf("  RigidBodyType %d: numGrids = %d\n", i, ng);		
		// copy potential grid data to device
		for (int gid = 0; gid < ng; gid++) { 
			RigidBodyGrid *g = &(rb.rawPotentialGrids[gid]); // convenience
			RigidBodyGrid *g_d = rb.rawDensityGrids_d[gid]; // convenience
			int len = g->getSize();
			float *tmpData;
			// tmpData = new float*[len];

			size_t sz = sizeof(RigidBodyGrid);
			gpuErrchk(cudaMalloc((void **) &(rawPotentialGrids_d[gid]), sz));
			gpuErrchk(cudaMemcpy(rawPotentialGrids_d[gid], &(rawPotentialGrids[gid]),
													 sz, cudaMemcpyHostToDevice));

			// allocate grid data on device
			// copy temporary host pointer to device pointer
			// copy data to device through temporary host pointer
			sz = sizeof(float*) * len;
			gpuErrchk(cudaMalloc((void **) &tmpData, sz)); 
			gpuErrchk(cudaMemcpy( &(rawPotentialGrids_d[gid].val), &tmpData,
														sizeof(float*), cudaMemcpyHostToDevice));
			sz = sizeof(float) * len;
			gpuErrchk(cudaMemcpy( tmpData, g->val, sz, cudaMemcpyHostToDevice));

				// RBTODO: why can't tmpData be deleted? 
			// delete[] tmpData;
		}
	}

	// density grids
 	for (int i = 0; i < conf.numRigidTypes; i++) {
		printf("Copying density grids of RB type %d\n",i);
		RigidBodyType& rb = conf.rigidBody[i];

		int ng = rb.numDenGrids;
		RigidBodyGrid * gtmp;
		size_t sz = sizeof(RigidBodyGrid)*ng;

		printf("  copying %d grids\n",ng);
		for (int gid = 0; gid < ng; gid++) {
			int gs = rb.rawDensityGrids[gid].getSize();
			printf("    grid %d contains %d values\n",gid, gs);
			for (int idx = 0; idx < gs; idx++) {
				printf("      val[%d] = %g\n",idx, rb.rawDensityGrids[gid].val[idx] );
			}
		}
		
		printf("  RigidBodyType %d: numGrids = %d\n", i, ng);		
		// copy grid data to device
		for (int gid = 0; gid < ng; gid++) { 
			RigidBodyGrid& g = rb.rawDensityGrids[gid]; // convenience
			RigidBodyGrid& g_d = rb.rawDensityGrids_d[gid]; // convenience
			int len = g.getSize();
			float *tmpData;
			// tmpData = new float*[len];

			size_t sz = sizeof(RigidBodyGrid);
			gpuErrchk(cudaMalloc((void **) &g_d, sz));
			gpuErrchk(cudaMemcpy(&g_d, &g,
													 sz, cudaMemcpyHostToDevice));

			// allocate grid data on device
			// copy temporary host pointer to device pointer
			// copy data to device through temporary host pointer
			sz = sizeof(float*) * len;
			gpuErrchk(cudaMalloc((void **) &tmpData, sz)); 
			gpuErrchk(cudaMemcpy( &(rawPotentialGrids_d[gid].val), &tmpData,
														sizeof(float*), cudaMemcpyHostToDevice));
			sz = sizeof(float) * len;
			gpuErrchk(cudaMemcpy( tmpData, g->val, sz, cudaMemcpyHostToDevice));

			// RBTODO: why can't this be deleted? 
			// delete[] tmpData;
		}
  }
	printf("Done copying RBs\n");
}


/* RigidBodyForcePair::RigidBodyForcePair(RigidBodyType* t1, RigidBodyType* t2, */
/* 																			 RigidBody* rb1, RigidBody* rb2, */
/* 																			 std::vector<int> gridKeyId1, std::vector<int> gridKeyId2) : */
/* 	type1(t1), type2(t2), rb1(rb1), rb2(rb2), gridKeyId1(gridKeyId1), gridKeyId2(gridKeyId2) { */

/* 	printf("    Constructing RB force pair...\n"); */
/* 	allocateMem(); */
/* 	printf("    Done constructing RB force pair\n"); */

/* } */
void RigidBodyForcePair::initialize() {
	printf("    Initializing (memory for) RB force pair...\n");

	const int numGrids = gridKeyId1.size();
	// RBTODO assert gridKeysIds are same size 

	// allocate memory for forces/torques
	for (int i = 0; i < numGrids; i++) {
		const int k1 = gridKeyId1[i];
		const int sz = type1->rawDensityGrids[k1].getSize();
		const int nb = sz / numThreads + ((sz % numThreads == 0) ? 0:1 );

		numBlocks.push_back(nb);
		forces.push_back( new Vector3[nb] );
		torques.push_back( new Vector3[nb] );

		forces_d.push_back( new Vector3[nb] ); // RBTODO: correct?
		torques_d.push_back( new Vector3[nb] );

		// allocate device memory for numBlocks of torque, etc.
    // printf("      Allocating device memory for forces/torques\n");
		gpuErrchk(cudaMalloc(&(forces_d[i]), sizeof(Vector3) * nb));
		gpuErrchk(cudaMalloc(&(torques_d[i]), sizeof(Vector3) * nb));
	}
	gpuErrchk(cudaDeviceSynchronize());
	// printf("    Done initializing RB force pair\n");

}

/* RigidBodyForcePair::RigidBodyForcePair(const RigidBodyForcePair& orig ) : */
	
/* } */

void RigidBodyForcePair::swap(RigidBodyForcePair& a, RigidBodyForcePair& b) {
	using std::swap;
	swap(a.type1, b.type1);
	swap(a.type2, b.type2);
	swap(a.rb1, b.rb1);
	swap(a.rb2, b.rb2);

	swap(a.gridKeyId1, b.gridKeyId1);
	swap(a.gridKeyId2, b.gridKeyId2);

	swap(a.numBlocks, b.numBlocks);

	swap(a.forces,    b.forces);
	swap(a.forces_d,  b.forces_d);
	swap(a.torques,   b.torques);
	swap(a.torques_d, b.torques_d);
}


RigidBodyForcePair::~RigidBodyForcePair() {
	printf("    Destructing RB force pair\n");
	const int numGrids = gridKeyId1.size();

	// printf("      numGrids = %d\n",numGrids);

	// RBTODO assert gridKeysIds are same size 

	// allocate memory for forces/torques
	for (int i = 0; i < numGrids; i++) {
		const int k1 = gridKeyId1[i];
		const int nb = numBlocks[i];

		// free device memory for numBlocks of torque, etc.
		// printf("      Freeing device memory for forces/torques\n");
		gpuErrchk(cudaFree( forces_d[i] ));	
		gpuErrchk(cudaFree( torques_d[i] ));
	}
	gpuErrchk(cudaDeviceSynchronize());
	
	numBlocks.clear();
	forces.clear();
	forces_d.clear();
	torques.clear();
	torques_d.clear();
}



