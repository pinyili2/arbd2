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
#include "Debug.h"

#include "RandomCPU.h"							/* RBTODO: fix this? */

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

RigidBodyController::RigidBodyController(const Configuration& c, const char* outArg) :
	conf(c), outArg(outArg) {

	if (conf.numRigidTypes > 0) {
		copyGridsToDevice();
	}

	int numRB = 0;
	// grow list of rbs
	for (int i = 0; i < conf.numRigidTypes; i++) {			
		numRB += conf.rigidBody[i].num;
		std::vector<RigidBody> tmp;
		// RBTODO: change conf.rigidBody to conf.rigidBodyType
		const int jmax = conf.rigidBody[i].num;
		for (int j = 0; j < jmax; j++) {
			String name = conf.rigidBody[i].name;
			if (jmax > 1) {
				char tmp[128];
				snprintf(tmp, 128, "#%d", j);
				name.add( tmp );
			}
			RigidBody r(name, conf, conf.rigidBody[i]);
			tmp.push_back( r );
	}
		rigidBodyByType.push_back(tmp);
}

	random = new RandomCPU(conf.seed + 1); /* +1 to avoid using same seed as RandomCUDA */
	
initializeForcePairs();
}
RigidBodyController::~RigidBodyController() {
	for (int i = 0; i < rigidBodyByType.size(); i++)
		rigidBodyByType[i].clear();
	rigidBodyByType.clear();
	delete random;
}

void RigidBodyController::initializeForcePairs() {
	// Loop over all pairs of rigid body types
	//   the references here make the code more readable, but they may incur a performance loss
	RigidBodyForcePair::createStreams();
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
						RigidBodyForcePair fp = RigidBodyForcePair(&(t1),&(t2),rb1,rb2,gridKeyId1,gridKeyId2, false);
						gpuErrchk(cudaDeviceSynchronize()); /* RBTODO: this should be extraneous */
						forcePairs.push_back( fp ); 
						printf("    done pushing RB force pair for %d:%d\n",i,j);
					}
				}
			}
		}
	}

	// add Pmfs (not a true pairwise RB interaction; hacky implementation)
	for (int ti = 0; ti < conf.numRigidTypes; ti++) {
		RigidBodyType& t1 = conf.rigidBody[ti];

		const std::vector<String>& keys1 = t1.densityGridKeys; 
		const std::vector<String>& keys2 = t1.pmfKeys;
		std::vector<int> gridKeyId1;
		std::vector<int> gridKeyId2;
		
		// Loop over all pairs of grid keys (e.g. "Elec")
		bool paired = false;
		for(int k1 = 0; k1 < keys1.size(); k1++) {
			for(int k2 = 0; k2 < keys2.size(); k2++) {
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
			
			// Loop over rigid bodies of these types
			for (int i = 0; i < rbs1.size(); i++) {
					RigidBody* rb1 = &(rbs1[i]);
					RigidBodyForcePair fp = RigidBodyForcePair(&(t1),&(t1),rb1,rb1,gridKeyId1,gridKeyId2, true);
					gpuErrchk(cudaDeviceSynchronize()); /* RBTODO: this should be extraneous */
					forcePairs.push_back( fp ); 
			}
		}
	}
}
	
void RigidBodyController::updateForces(int s) {
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

	// clear old forces
	for (int i = 0; i < rigidBodyByType.size(); i++) {
		for (int j = 0; j < rigidBodyByType[i].size(); j++) {
			RigidBody& rb = rigidBodyByType[i][j];
			rb.clearForce();
			rb.clearTorque();
		}
	}
			
	for (int i=0; i < forcePairs.size(); i++)
		forcePairs[i].callGridForceKernel(i,s);

	for (int i=0; i < forcePairs.size(); i++)
		forcePairs[i].retrieveForces();

	// RBTODO: see if there is a better way to sync
	gpuErrchk(cudaDeviceSynchronize());

	/*/ debug
	if (s %10 == 0) {
		int tmp = 0;
		for (int i = 0; i < rigidBodyByType.size(); i++) {
			for (int j = 0; j < rigidBodyByType[i].size(); j++) {
				RigidBody& rb = rigidBodyByType[i][j];
				tmp++;
				Vector3 p = rb.getPosition();
				Vector3 t = rb.torque;
				printf("RBTORQUE: %d %f %f %f %f %f %f\n", tmp, p.x, p.y, p.z, t.x,t.y,t.z);
			}
		}
	}
	*/
}
void RigidBodyController::integrate(int step) {
	// tell RBs to integrate
	for (int i = 0; i < rigidBodyByType.size(); i++) {
		for (int j = 0; j < rigidBodyByType[i].size(); j++) {
			RigidBody& rb = rigidBodyByType[i][j];
			
			// thermostat
			// rb.addLangevin( random->gaussian_vector(), random->gaussian_vector() );
			rb.addLangevin( random->gaussian_vector(), random->gaussian_vector() );
		}
	}

	if ( step % conf.outputPeriod == 0 ) { /* PRINT & INTEGRATE */
		if (step == 0) {						// first step so only start this cycle
			print(step);
			for (int i = 0; i < rigidBodyByType.size(); i++) {
				for (int j = 0; j < rigidBodyByType[i].size(); j++) {
					RigidBody& rb = rigidBodyByType[i][j];
					rb.integrate(0);	
				}
			}
		} else {										// finish last cycle
			for (int i = 0; i < rigidBodyByType.size(); i++) {
				for (int j = 0; j < rigidBodyByType[i].size(); j++) {
					RigidBody& rb = rigidBodyByType[i][j];
					rb.integrate(1);	
				}
			}
			print(step);

			// start this cycle
			for (int i = 0; i < rigidBodyByType.size(); i++) {
				for (int j = 0; j < rigidBodyByType[i].size(); j++) {
					RigidBody& rb = rigidBodyByType[i][j];
					rb.integrate(0);	
				}
			}
		}
	} else {											/* INTEGRATE ONLY */
		if (step == 0) {						// first step so only start this cycle
			print(step);
			for (int i = 0; i < rigidBodyByType.size(); i++) {
				for (int j = 0; j < rigidBodyByType[i].size(); j++) {
					RigidBody& rb = rigidBodyByType[i][j];
					rb.integrate(0);	
				}
			}
		} else {										// integrate end of last step and start of this one
			for (int i = 0; i < rigidBodyByType.size(); i++) {
				for (int j = 0; j < rigidBodyByType[i].size(); j++) {
					RigidBody& rb = rigidBodyByType[i][j];
					rb.integrate(2);	
				}
			}
		}
	}
}

// allocate and initialize an array of stream handles
cudaStream_t *RigidBodyForcePair::stream = (cudaStream_t *) malloc(NUMSTREAMS * sizeof(cudaStream_t));
// new cudaStream_t[NUMSTREAMS];
int RigidBodyForcePair::nextStreamID = 0;
void RigidBodyForcePair::createStreams() {
	for (int i = 0; i < NUMSTREAMS; i++)
		gpuErrchk( cudaStreamCreate( &(stream[i]) ) );
		// gpuErrchk( cudaStreamCreateWithFlags( &(stream[i]) , cudaStreamNonBlocking ) );
}

// RBTODO: bundle several rigidbodypair evaluations in single kernel call
void RigidBodyForcePair::callGridForceKernel(int pairId, int s) {
	// get the force/torque between a pair of rigid bodies
	/* printf("  Updating rbPair forces\n"); */
	const int numGrids = gridKeyId1.size();

	/* if (s%10 != 0) */
	/* 	pairId = -1000; */

	// RBTODO: precompute certain common transformations and pass in kernel call
	for (int i = 0; i < numGrids; i++) {
		const int nb = numBlocks[i];
		const int k1 = gridKeyId1[i];
		const int k2 = gridKeyId2[i];
		const cudaStream_t &s = stream[streamID[i]];

		/*
			ijk: index of grid value
			r: postion of point ijk in real space
			B: grid Basis
			o: grid origin
			R: rigid body orientation
			c: rigid body center

			B': R.B 
			c': R.o + c

  		/.––––––––––––––––––.
	  	| r = R.(B.ijk+o)+c |
	  	| r = B'.ijk + c'    |
	  	`––––––––––––––––––./
		*/
		Matrix3 B1 = rb1->getOrientation()*type1->densityGrids[k1].getBasis();
		Vector3 c1 = rb1->getOrientation()*type1->densityGrids[k1].getOrigin() + rb1->getPosition();
		
		Matrix3 B2;
		Vector3 c2;
		
		/* printf("  Calculating grid forces\n"); */
		if (!isPmf) {								/* pair of RBs */
			B2 = rb2->getOrientation()*type2->potentialGrids[k2].getBasis();
			c2 = rb2->getOrientation()*type2->potentialGrids[k2].getOrigin() + rb2->getPosition();
			B2 = B2.inverse();

			// RBTODO: get energy
			computeGridGridForce<<< nb, numThreads, 0, s >>>
				(type1->rawDensityGrids_d[k1], type2->rawPotentialGrids_d[k2],
				 B1, c1, B2, c2,
				 forces_d[i], torques_d[i], pairId+i);
		} else {										/* RB with a PMF */
			// B2 = type2->rawPmfs[i].getBasis(); // not 100% certain k2 should be used rather than i
			/// c2 = type2->rawPmfs[i].getOrigin();
			B2 = type2->rawPmfs[k2].getBasis();
			c2 = type2->rawPmfs[k2].getOrigin();
			B2 = B2.inverse();

			computeGridGridForce<<< nb, numThreads, 0, s >>>
				(type1->rawDensityGrids_d[k1], type2->rawPmfs_d[k2],
				 B1, c1, B2, c2,
				 forces_d[i], torques_d[i], pairId+i);
		}

	}
}

void RigidBodyForcePair::retrieveForces() {
	// sum forces + torques
	const int numGrids = gridKeyId1.size();
	Vector3 f = Vector3(0.0f);
	Vector3 t = Vector3(0.0f);

	// RBTODO better way to sync?
	for (int i = 0; i < numGrids; i++) {
		const cudaStream_t &s = stream[streamID[i]];
		const int nb = numBlocks[i];

		gpuErrchk(cudaMemcpyAsync(forces[i], forces_d[i], sizeof(Vector3)*nb,
															cudaMemcpyDeviceToHost, s));
		gpuErrchk(cudaMemcpyAsync(torques[i], torques_d[i], sizeof(Vector3)*nb,
															cudaMemcpyDeviceToHost, s));

		gpuErrchk(cudaStreamSynchronize( s ));
		
		for (int j = 0; j < nb; j++) {
			f = f + forces[i][j];
			t = t + torques[i][j];
		}
	}
	
	// transform torque from lab-frame origin to rb centers
	// add forces to rbs
	/* Vector3 tmp; */
	/* /\* tmp = rb1->position; *\/ */
	/* /\* printf("rb1->position: (%f,%f,%f)\n", tmp.x, tmp.y, tmp.z); *\/ */
	/* tmp = rb1->getPosition(); */
	/* printf("rb1->getPosition(): (%f,%f,%f)\n", tmp.x, tmp.y, tmp.z); */
	Vector3 t1 = t - rb1->getPosition().cross( f );
	rb1->addForce( f );
	rb1->addTorque(t1);

	if (!isPmf) {
		Vector3 t2 = -t - rb2->getPosition().cross( -f );
		rb2->addForce(-f);
		rb2->addTorque(t2);
	}
		
	/* printf("force: (%f,%f,%f)\n",f.x,f.y,f.z); */
	/* printf("torque: (%f,%f,%f)\n",t1.x,t1.y,t1.z); */
}

void RigidBodyController::copyGridsToDevice() {
	// RBTODO: clean this function up
	RigidBodyType **rb_addr = new RigidBodyType*[conf.numRigidTypes];	/* temporary pointer to device pointer */

	gpuErrchk(cudaMalloc(&rbType_d, sizeof(RigidBodyType*) * conf.numRigidTypes));
	// TODO: The above line fails when there is not enough memory. If it fails, stop.

	printf("Copying RBs\n");
	// Copy rigidbody types 
	// http://stackoverflow.com/questions/16024087/copy-an-object-to-device
 	for (int i = 0; i < conf.numRigidTypes; i++)
		conf.rigidBody[i].updateRaw();


	// density grids
 	for (int i = 0; i < conf.numRigidTypes; i++) {
		printf("Copying density grids of RB type %d\n",i);
		RigidBodyType& rb = conf.rigidBody[i];

		int ng = rb.numDenGrids;
		rb.rawDensityGrids_d = new RigidBodyGrid*[ng]; /* not sure this is needed */
		
		printf("  RigidBodyType %d: numGrids = %d\n", i, ng);		
		// copy grid data to device
		for (int gid = 0; gid < ng; gid++) { 
			RigidBodyGrid* g = &(rb.rawDensityGrids[gid]); // convenience
			// RigidBodyGrid* g_d = rb.rawDensityGrids_d[gid]; // convenience
			int len = g->getSize();
			float* tmpData;
			// tmpData = new float[len];

			size_t sz = sizeof(RigidBodyGrid);
			gpuErrchk(cudaMalloc((void **) &(rb.rawDensityGrids_d[gid]), sz));
			/* gpuErrchk(cudaMemcpy(rb.rawDensityGrids_d[gid], g, */
			/* 										 sz, cudaMemcpyHostToDevice)); */
			gpuErrchk(cudaMemcpy(rb.rawDensityGrids_d[gid], &(rb.rawDensityGrids[gid]),
													 sz, cudaMemcpyHostToDevice));

			// allocate grid data on device
			// copy temporary host pointer to device pointer
			// copy data to device through temporary host pointer
			sz = sizeof(float) * len;
			gpuErrchk(cudaMalloc((void **) &tmpData, sz)); 
			// gpuErrchk(cudaMemcpy( tmpData, g->val, sz, cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy( tmpData, rb.rawDensityGrids[gid].val, sz, cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy( &(rb.rawDensityGrids_d[gid]->val), &tmpData,
														sizeof(float*), cudaMemcpyHostToDevice));

			// RBTODO: why can't this be deleted? 
			// delete[] tmpData;
		}
  }

	for (int i = 0; i < conf.numRigidTypes; i++) {
		printf("Working on RB %d\n",i);
		RigidBodyType& rb = conf.rigidBody[i];

		int ng = rb.numPotGrids;
		rb.rawPotentialGrids_d = new RigidBodyGrid*[ng]; /* not 100% sure this is needed, possible memory leak */

		printf("  RigidBodyType %d: numGrids = %d\n", i, ng);		
		// copy potential grid data to device
		for (int gid = 0; gid < ng; gid++) { 
			RigidBodyGrid* g = &(rb.rawPotentialGrids[gid]); // convenience
			// RigidBodyGrid* g_d = rb.rawDensityGrids_d[gid]; // convenience
			int len = g->getSize();
			float* tmpData;
			// tmpData = new float*[len];

			size_t sz = sizeof(RigidBodyGrid);
			gpuErrchk(cudaMalloc((void **) &(rb.rawPotentialGrids_d[gid]), sz));
			gpuErrchk(cudaMemcpy( rb.rawPotentialGrids_d[gid], &(rb.rawPotentialGrids[gid]),
													 sz, cudaMemcpyHostToDevice ));

			// allocate grid data on device
			// copy temporary host pointer to device pointer
			// copy data to device through temporary host pointer
			sz = sizeof(float) * len;
			gpuErrchk(cudaMalloc((void **) &tmpData, sz)); 
			// sz = sizeof(float) * len;
			gpuErrchk(cudaMemcpy( tmpData, rb.rawPotentialGrids[gid].val, sz, cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy( &(rb.rawPotentialGrids_d[gid]->val), &tmpData,
														sizeof(float*), cudaMemcpyHostToDevice));
			
				// RBTODO: why can't tmpData be deleted? 
			// delete[] tmpData;
		}
	}

	for (int i = 0; i < conf.numRigidTypes; i++) {
		printf("Copying PMFs for RB %d\n",i);
		RigidBodyType& rb = conf.rigidBody[i];

		int ng = rb.numPmfs;
		rb.rawPmfs_d = new RigidBodyGrid*[ng]; /* not 100% sure this is needed, possible memory leak */

		printf("  RigidBodyType %d: numPmfs = %d\n", i, ng);		

		// copy pmf grid data to device
		for (int gid = 0; gid < ng; gid++) { 
			RigidBodyGrid g = rb.rawPmfs[gid];
			int len = g.getSize();
			float* tmpData;
			// tmpData = new float*[len];

			size_t sz = sizeof(RigidBodyGrid);
			gpuErrchk(cudaMalloc((void **) &(rb.rawPmfs_d[gid]), sz));
			gpuErrchk(cudaMemcpy( rb.rawPmfs_d[gid], &g,
													 sz, cudaMemcpyHostToDevice ));

			// allocate grid data on device
			// copy temporary host pointer to device pointer
			// copy data to device through temporary host pointer
			sz = sizeof(float) * len;
			gpuErrchk(cudaMalloc((void **) &tmpData, sz)); 
			// sz = sizeof(float) * len;
			gpuErrchk(cudaMemcpy( tmpData, rb.rawPmfs[gid].val, sz, cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy( &(rb.rawPmfs_d[gid]->val), &tmpData,
														sizeof(float*), cudaMemcpyHostToDevice));
			
		}
	}
	
	gpuErrchk(cudaDeviceSynchronize());
	printf("Done copying RBs\n");

	/* // DEBUG */
	/* RigidBodyType& rb = conf.rigidBody[0]; */
	/* printRigidBodyGrid<<<1,1>>>( rb.rawPotentialGrids_d[0] ); */
	/* gpuErrchk(cudaDeviceSynchronize()); */
	/* printRigidBodyGrid<<<1,1>>>( rb.rawDensityGrids_d[0] ); */
	/* gpuErrchk(cudaDeviceSynchronize()); */
}

void RigidBodyController::print(int step) {
	// modeled after outputExtendedData() in Controller.C
	if ( step >= 0 ) {
		// Write RIGID BODY trajectory file
		if ( step % conf.outputPeriod == 0 ) {
			if ( ! trajFile.rdbuf()->is_open() ) {
	      // open file
	      printf("OPENING RIGID BODY TRAJECTORY FILE\n");
				// RBTODO: backup_file(simParams->rigidBodyTrajectoryFile);

				char fname[140];
				strcpy(fname,outArg);
				strcat(fname, ".rb-traj");
	      trajFile.open(fname);
				
	      while (!trajFile) {
					/* if ( errno == EINTR ) {
						printf("Warning: Interrupted system call opening RIGIDBODY trajectory file, retrying.\n");
						trajFile.clear();
						trajFile.open(simParams->rigidBodyTrajectoryFile);
						continue;
					}
					*/ 
					//char err_msg[257];
					printf("Error opening RigidBody trajectory file %s",fname);
					exit(1);
	      }
	      trajFile << "# RigidBody trajectory file" << std::endl;
	      printLegend(trajFile);
			}
			printf("WRITING RIGID BODY COORDINATES AT STEP %d\n",step);
			printData(step,trajFile);
			trajFile.flush();    
		}
    
		// Write restart File
		/* if ( simParams->restartFrequency && */
		/* 		 ((step % simParams->restartFrequency) == 0) && */
		/* 		 (step != simParams->firstTimestep) )	{ */
		if ( step % conf.outputPeriod == 0 && step != 0 ){
			printf("RIGID BODY: WRITING RESTART FILE AT STEP %d\n", step);
			char fname[140];
			strcpy(fname,outArg);
			strcat(fname, ".rigid");
			// RBTODO: NAMD_backup_file(fname,".old"); /*  */
			std::ofstream restartFile(fname);
			while (!restartFile) {
				/* RBTODO 
	      if ( errno == EINTR ) {
					printf("Warning: Interrupted system call opening rigid body restart file, retrying.\n");
					restartFile.clear();
					restartFile.open(fname);
					continue;
	      }
				*/
	      printf("Error opening rigid body restart file %s",fname);
	      exit(1); // NAMD_err(err_msg);
			}
			restartFile << "# RigidBody restart file" << std::endl;
			printLegend(restartFile);
			printData(step,restartFile);
			if (!restartFile) {
	      printf("Error writing rigid body restart file %s",fname);
	      exit(-1); // NAMD_err(err_msg);
			} 
		}
	}
	/*
	//  Output final coordinates
	if (step == FILE_OUTPUT || step == END_OF_RUN) {
		int realstep = ( step == FILE_OUTPUT ?
										 simParams->firstTimestep : simParams->N );
		iout << "WRITING RIGID BODY OUTPUT FILE AT STEP " << realstep << "\n" << endi;
		static char fname[140];
		strcpy(fname, simParams->outputFilename);
		strcat(fname, ".rigid");
		NAMD_backup_file(fname);
		std::ofstream outputFile(fname);
		while (!outputFile) {
	    if ( errno == EINTR ) {
				CkPrintf("Warning: Interrupted system call opening rigid body output file, retrying.\n");
				outputFile.clear();
				outputFile.open(fname);
				continue;
	    }
	    char err_msg[257];
	    sprintf(err_msg, "Error opening rigid body output file %s",fname);
	    NAMD_err(err_msg);
		} 
		outputFile << "# NAMD rigid body output file" << std::endl;
		printLegend(outputFile);
		printData(realstep,outputFile);
		if (!outputFile) {
	    char err_msg[257];
	    sprintf(err_msg, "Error writing rigid body output file %s",fname);
	    NAMD_err(err_msg);
		} 
	}

	//  Close trajectory file
	if (step == END_OF_RUN) {
		if ( trajFile.rdbuf()->is_open() ) {
	    trajFile.close();
	    iout << "CLOSING RIGID BODY TRAJECTORY FILE\n" << endi;
		}
	}
	*/
}
void RigidBodyController::printLegend(std::ofstream &file) {
        file << "#$LABELS step RigidBodyKey"
		 << " posX  posY  posZ"
		 << " rotXX rotXY rotXZ"
		 << " rotYX rotYY rotYZ"
		 << " rotZX rotZY rotZZ"
		 << " velX  velY  velZ"
		 << " angVelX angVelY angVelZ" << std::endl;
}
void RigidBodyController::printData(int step,std::ofstream &file) {
	// tell RBs to integrate
	for (int i = 0; i < rigidBodyByType.size(); i++) {
		for (int j = 0; j < rigidBodyByType[i].size(); j++) {
			const RigidBody& rb = rigidBodyByType[i][j];
			
			Vector3 v =  rb.getPosition();
			Matrix3 t =  rb.getOrientation();
			file << step <<" "<< rb.getKey()
					 <<" "<< v.x <<" "<< v.y <<" "<< v.z;
			file <<" "<< t.exx <<" "<< t.exy <<" "<< t.exz
					 <<" "<< t.eyx <<" "<< t.eyy <<" "<< t.eyz
					 <<" "<< t.ezx <<" "<< t.ezy <<" "<< t.ezz;
			v = rb.getVelocity();
			file <<" "<< v.x <<" "<< v.y <<" "<< v.z;
			v = rb.getAngularVelocity();
			file <<" "<< v.x <<" "<< v.y <<" "<< v.z
					 << std::endl;
		}
	}
}
/*
void RigidBodyController::integrate(int step) {
    DebugM(3, "RBC::integrate: step  "<< step << "\n" << endi);
    
    DebugM(1, "RBC::integrate: Waiting for grid reduction\n" << endi);
    gridReduction->require();
  
    const Molecule * mol = Node::Object()->molecule;

    // pass reduction force and torque to each grid
    // DebugM(3, "Summing forces on rigid bodies" << "\n" << endi);
    for (int i=0; i < mol->rbReductionIdToRigidBody.size(); i++) {
	Force f;
	Force t;
	for (int k = 0; k < 3; k++) {
	    f[k] = gridReduction->item(6*i + k);
	    t[k] = gridReduction->item(6*i + k + 3);
	}

	if (step % 100 == 1)
	    DebugM(4, "RBC::integrate: reduction/rb " << i <<":"
		   << "\n\tposition: "
		   << rigidBodyList[mol->rbReductionIdToRigidBody[i]]->getPosition()
		   <<"\n\torientation: "
		   << rigidBodyList[mol->rbReductionIdToRigidBody[i]]->getOrientation()
		   << "\n" << endi);

	DebugM(4, "RBC::integrate: reduction/rb " << i <<": "
	       << "force " << f <<": "<< "torque: " << t << "\n" << endi);
	rigidBodyList[mol->rbReductionIdToRigidBody[i]]->addForce(f);
	rigidBodyList[mol->rbReductionIdToRigidBody[i]]->addTorque(t);
    }
    
    // Langevin 
    for (int i=0; i<rigidBodyList.size(); i++) {
	// continue;  // debug
	if (rigidBodyList[i]->langevin) {
	    DebugM(1, "RBC::integrate: reduction/rb " << i
		   <<": calling langevin" << "\n" << endi);
	    rigidBodyList[i]->addLangevin(
		random->gaussian_vector(), random->gaussian_vector() );
	    // DebugM(4, "RBC::integrate: reduction/rb " << i
	    // 	   << " after langevin: force " << rigidBodyList[i]f <<": "<< "torque: " << t << "\n" << endi);
	    // <<": calling langevin" << "\n" << endi);
	}
    }
    
    if ( step >= 0 && simParams->rigidBodyOutputFrequency &&
	 (step % simParams->rigidBodyOutputFrequency) == 0 ) {
	DebugM(1, "RBC::integrate:integrating for before printing output" << "\n" << endi);
	// PRINT
	if ( step == simParams->firstTimestep ) {
	    print(step);
	    // first step so only start this cycle
	    for (int i=0; i<rigidBodyList.size(); i++)  {
		DebugM(2, "RBC::integrate: reduction/rb " << i
		       <<": starting integration cycle of step "
		       << step << "\n" << endi);
		rigidBodyList[i]->integrate(&trans[i],&rot[i],0);
	    }
	} else {
	    // finish last cycle
	    // DebugM(1, "RBC::integrate: reduction/rb " << i
	    // 	   <<": firststep: calling rb->integrate" << "\n" << endi);
	    for (int i=0; i<rigidBodyList.size(); i++) {
		DebugM(2, "RBC::integrate: reduction/rb " << i
		       <<": finishing integration cycle of step "
		       << step-1 << "\n" << endi);
		rigidBodyList[i]->integrate(&trans[i],&rot[i],1);
	    }
	    print(step);
	    // start this cycle
	    for (int i=0; i<rigidBodyList.size(); i++) {
		DebugM(2, "RBC::integrate: reduction/rb " << i
		       <<": starting integration cycle of step "
		       << step << "\n" << endi);
		rigidBodyList[i]->integrate(&trans[i],&rot[i],0);
	    }
	}
    } else {
	DebugM(1, "RBC::integrate: trans[0] before: " << trans[0] << "\n" << endi);
	if ( step == simParams->firstTimestep ) {
	    // integrate the start of this cycle
	    for (int i=0; i<rigidBodyList.size(); i++) {
		DebugM(2, "RBC::integrate: reduction/rb " << i
		       <<": starting integration cycle of (first) step "
		       << step << "\n" << endi);
		rigidBodyList[i]->integrate(&trans[i],&rot[i],0);
	    }
	} else {
	    // integrate end of last ts and start of this one 
	    for (int i=0; i<rigidBodyList.size(); i++) {
		DebugM(2, "RBC::integrate: reduction/rb " << i
		   <<": ending / starting integration cycle of step "
		   << step-1 << "-" << step << "\n" << endi);
		rigidBodyList[i]->integrate(&trans[i],&rot[i],2);
	    }
	}
	DebugM(1, "RBC::integrate: trans[0] after: " << trans[0] << "\n" << endi);
    }
    
    DebugM(3, "sendRigidBodyUpdate on step: " << step << "\n" << endi);
    if (trans.size() != rot.size())
	NAMD_die("failed sanity check\n");    
    RigidBodyMsg *msg = new RigidBodyMsg;
    msg->trans.copy(trans);	// perhaps .swap() would cause problems
    msg->rot.copy(rot);
    computeMgr->sendRigidBodyUpdate(msg);
}


RigidBodyParams* RigidBodyParamsList::find_key(const char* key) {
    RBElem* cur = head;
    RBElem* found = NULL;
    RigidBodyParams* result = NULL;
    
    while (found == NULL && cur != NULL) {
       if (!strcasecmp((cur->elem).rigidBodyKey,key)) {
        found = cur;
      } else {
        cur = cur->nxt;
      }
    }
    if (found != NULL) {
      result = &(found->elem);
    }
    return result;
}
*/

/* RigidBodyForcePair::RigidBodyForcePair(RigidBodyType* t1, RigidBodyType* t2, */
/* 																			 RigidBody* rb1, RigidBody* rb2, */
/* 																			 std::vector<int> gridKeyId1, std::vector<int> gridKeyId2) : */
/* 	type1(t1), type2(t2), rb1(rb1), rb2(rb2), gridKeyId1(gridKeyId1), gridKeyId2(gridKeyId2) { */

/* 	printf("    Constructing RB force pair...\n"); */
/* 	allocateMem(); */
/* 	printf("    Done constructing RB force pair\n"); */

/* } */
int RigidBodyForcePair::initialize() {
	printf("    Initializing (memory for) RB force pair...\n");

	const int numGrids = gridKeyId1.size();
	// RBTODO assert gridKeysIds are same size 

	// allocate memory for forces/torques
	for (int i = 0; i < numGrids; i++) {
		const int k1 = gridKeyId1[i];
		const int sz = type1->rawDensityGrids[k1].getSize();
		const int nb = sz / numThreads + ((sz % numThreads == 0) ? 0:1 );
		streamID.push_back( nextStreamID % NUMSTREAMS );
		nextStreamID++;

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
	return nextStreamID;
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



