/* #include "RigidBody.h" */
#include "RigidBodyController.h"
#include "Configuration.h"
#include "RigidBodyType.h"
#include "RigidBodyGrid.h"
#include "ComputeGridGrid.cuh"

#include <cuda_profiler_api.h>

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
	gpuErrchk(cudaDeviceSynchronize()); /* RBTODO: this should be extraneous */
	for (int i = 0; i < conf.numRigidTypes; i++)
		conf.rigidBody[i].initializeParticleLists();

	int numRB = 0;
	// grow list of rbs
	for (int i = 0; i < conf.numRigidTypes; i++) {			
		conf.rigidBody[i].copyGridsToDevice();
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
			RigidBody r(name, conf, conf.rigidBody[i], this);
			tmp.push_back( r );
		}
		rigidBodyByType.push_back(tmp);
	}

	if (conf.inputRBCoordinates.length() > 0)
		loadRBCoordinates(conf.inputRBCoordinates.val());
	
	random = new RandomCPU(conf.seed + 1); /* +1 to avoid using same seed as RandomCUDA */
	
	gpuErrchk(cudaDeviceSynchronize()); /* RBTODO: this should be extraneous */
	initializeForcePairs();
	gpuErrchk(cudaDeviceSynchronize()); /* RBTODO: this should be extraneous */
        //Boltzmann distribution
        for (int i = 0; i < rigidBodyByType.size(); i++)
        {
            for (int j = 0; j < rigidBodyByType[i].size(); j++)
            {
                RigidBody& rb = rigidBodyByType[i][j];
                // thermostat
                rb.Boltzmann();
            }
        }
}

RigidBodyController::~RigidBodyController() {
	for (int i = 0; i < rigidBodyByType.size(); i++)
		rigidBodyByType[i].clear();
	rigidBodyByType.clear();
	delete random;
}

bool RigidBodyController::loadRBCoordinates(const char* fileName) {
	char line[STRLEN];
	FILE* inp = fopen(fileName, "r");

	if (inp == NULL) {
		printf("GrandBrownTown: load RB coordinates: File '%s' does not exist\n", fileName);
		exit(-1);	   
	}

	int imax = rigidBodyByType.size();
	int i = 0;
	int jmax = rigidBodyByType[i].size();
	int j = 0;

	while (fgets(line, STRLEN, inp) != NULL) {
		// Ignore comments.
		int len = strlen(line);
		if (line[0] == '#') continue;
		if (len < 2) continue;

		String s(line);
		int numTokens = s.tokenCount();
		if (numTokens != 3+9) {
			printf("GrandBrownTown: load RB coordinates: Invalid coordinate file line: %s\n", line);
			fclose(inp);	
			exit(-1);
		}

		String* tokenList = new String[numTokens];
		s.tokenize(tokenList);
		if (tokenList == NULL) {
			printf("GrandBrownTown: load RB coordinates: Invalid coordinate file line: %s\n", line);
			fclose(inp);
			exit(-1);
		}

		RigidBody& rb = rigidBodyByType[i][j];
		rb.position = Vector3(
			(float) strtod(tokenList[0],NULL), (float) strtod(tokenList[1],NULL), (float) strtod(tokenList[2],NULL));
		rb.orientation = Matrix3(
			(float) strtod(tokenList[3],NULL), (float) strtod(tokenList[4],NULL), (float) strtod(tokenList[5],NULL),
			(float) strtod(tokenList[6],NULL), (float) strtod(tokenList[7],NULL), (float) strtod(tokenList[8],NULL),
			(float) strtod(tokenList[9],NULL), (float) strtod(tokenList[10],NULL), (float) strtod(tokenList[11],NULL));

		
		delete[] tokenList;

		j++;
		if (j == jmax) {
			i++;
			if (i == imax)
				break;
			j=0;
			jmax = rigidBodyByType[i].size();
		}
	}
	fclose(inp);
	return true;
}

		

void RigidBodyController::initializeForcePairs() {
	// Loop over all pairs of rigid body types
	//   the references here make the code more readable, but they may incur a performance loss
	RigidBodyForcePair::createStreams();
	// printf("Initializing force pairs\n");
	for (int ti = 0; ti < conf.numRigidTypes; ti++) {
		RigidBodyType& t1 = conf.rigidBody[ti];
		for (int tj = ti; tj < conf.numRigidTypes; tj++) {
			RigidBodyType& t2 = conf.rigidBody[tj];


			const std::vector<String>& keys1 = t1.densityGridKeys; 
			const std::vector<String>& keys2 = t2.potentialGridKeys;

			// printf("  Working on type pair ");
			// t1.name.printInline(); printf(":"); t2.name.print();
			
			// Loop over all pairs of grid keys (e.g. "Elec")
			std::vector<int> gridKeyId1;
			std::vector<int> gridKeyId2;
			
			// printf("  Grid keys %d:%d\n",keys1.size(),keys2.size());

			bool paired = false;
			for(int k1 = 0; k1 < keys1.size(); k1++) {
				for(int k2 = 0; k2 < keys2.size(); k2++) {
				    // printf("    checking grid keys ");
				    //	keys1[k1].printInline(); printf(":"); keys2[k2].print();
					
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

						// printf("    pushing RB force pair for %d:%d\n",i,j);
						RigidBodyForcePair fp = RigidBodyForcePair(&(t1),&(t2),rb1,rb2,gridKeyId1,gridKeyId2, false, conf.rigidBodyGridGridPeriod );
						gpuErrchk(cudaDeviceSynchronize()); /* RBTODO: this should be extraneous */
						forcePairs.push_back( fp ); 
						// printf("    done pushing RB force pair for %d:%d\n",i,j);
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
					RigidBodyForcePair fp = RigidBodyForcePair(&(t1),&(t1),rb1,rb1,gridKeyId1,gridKeyId2, true, conf.rigidBodyGridGridPeriod);
					gpuErrchk(cudaDeviceSynchronize()); /* RBTODO: this should be extraneous */
					forcePairs.push_back( fp ); 
			}
		}
	}

	// Initialize device data for RB force pairs after std::vector is done growing
	for (int i = 0; i < forcePairs.size(); i++)
		forcePairs[i].initialize();
			
}

void RigidBodyController::updateParticleLists(Vector3* pos_d) {
	for (int i = 0; i < rigidBodyByType.size(); i++) {
		for (int j = 0; j < rigidBodyByType[i].size(); j++) {
			rigidBodyByType[i][j].updateParticleList(pos_d);
		}
	}
}

void RigidBodyController::clearForceAndTorque()
{
    // clear old forces
    for (int i = 0; i < rigidBodyByType.size(); i++) 
    {
        for (int j = 0; j < rigidBodyByType[i].size(); j++) 
        {
            RigidBody& rb = rigidBodyByType[i][j];
            rb.clearForce();
            rb.clearTorque();
        }
    }
}

void RigidBodyController::updateForces(Vector3* pos_d, Vector3* force_d, int s) {
	if (s <= 1)
		gpuErrchk( cudaProfilerStart() );
	
	// Grid–particle forces
	for (int i = 0; i < rigidBodyByType.size(); i++) {
		for (int j = 0; j < rigidBodyByType[i].size(); j++) {
			RigidBody& rb = rigidBodyByType[i][j];
			rb.callGridParticleForceKernel( pos_d, force_d, s );
		}
	}

	for (int i = 0; i < rigidBodyByType.size(); i++) {
		for (int j = 0; j < rigidBodyByType[i].size(); j++) {
			RigidBody& rb = rigidBodyByType[i][j];
			rb.retrieveGridParticleForces();
		}
	}

	// Grid–Grid forces
	if ( (s % conf.rigidBodyGridGridPeriod) == 0 && forcePairs.size() > 0) {
		for (int i=0; i < forcePairs.size(); i++) {
			// TODO: performance: make this check occur less frequently
			if (forcePairs[i].isWithinPairlistDist())
				forcePairs[i].callGridForceKernel(i,s);
		}
		
		// each kernel call is followed by async memcpy for previous; now get last
		RigidBodyForcePair* fp = RigidBodyForcePair::lastRbForcePair;
		fp->retrieveForcesForGrid( fp->lastRbGridID );
		fp->lastRbGridID = -1;

		// stream sync was slower than device sync
		/* for (int i = 0; i < NUMSTREAMS; i++) { */
		/* 	const cudaStream_t &s = RigidBodyForcePair::stream[i]; */
		/* 	gpuErrchk(cudaStreamSynchronize( s ));  */
		/* } */
		gpuErrchk(cudaDeviceSynchronize());
	
		for (int i=0; i < forcePairs.size(); i++)
			if (forcePairs[i].isWithinPairlistDist())
				forcePairs[i].processGPUForces();
	}
}

void RigidBodyController::SetRandomTorques()
{
    for (int i = 0; i < rigidBodyByType.size(); i++)
    {
        for (int j = 0; j < rigidBodyByType[i].size(); j++)
        {
            RigidBody& rb = rigidBodyByType[i][j];
            rb.W1 = random->gaussian_vector();
            rb.W2 = random->gaussian_vector();
        }
    }           
}

void RigidBodyController::AddLangevin()
{
    for (int i = 0; i < rigidBodyByType.size(); i++)
    {
        for (int j = 0; j < rigidBodyByType[i].size(); j++)
        {
            RigidBody& rb = rigidBodyByType[i][j];

            //printf("%f %f %f\n",rb.W1.x,rb.W1.y,rb.W1.z);
            //printf("%f %f %f\n",rb.W2.x,rb.W2.y,rb.W2.z);

            rb.addLangevin(rb.W1,rb.W2);
        }
    }
}

void RigidBodyController::integrateDLM(int step) 
{
    // tell RBs to integrate
    for (int i = 0; i < rigidBodyByType.size(); i++) 
    {
        for (int j = 0; j < rigidBodyByType[i].size(); j++) 
        {
            RigidBody& rb = rigidBodyByType[i][j];
            rb.integrateDLM(step);
        }
    }
}


//Chris original part for Brownian motion
void RigidBodyController::integrate(int step) 
{
 	// tell RBs to integrate
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

float RigidBodyController::KineticEnergy()
{
    float e = 0.;
    int num = 0;
    for (int i = 0; i < rigidBodyByType.size(); i++) 
    {
        for (int j = 0; j < rigidBodyByType[i].size(); j++) 
        {
            RigidBody& rb = rigidBodyByType[i][j];
            e += rb.Temperature();
            num += 1;
        }
    }
    if(num > 0)
        return e / num;
    else
        return 0.;
}

// allocate and initialize an array of stream handles
cudaStream_t *RigidBodyForcePair::stream = (cudaStream_t *) malloc(NUMSTREAMS * sizeof(cudaStream_t));
int RigidBodyForcePair::nextStreamID = 0;	 /* used during stream init */
int RigidBodyForcePair::lastRbGridID = -1; /* used to schedule kernel interaction */
RigidBodyForcePair* RigidBodyForcePair::lastRbForcePair = NULL;

void RigidBodyForcePair::createStreams() {
	for (int i = 0; i < NUMSTREAMS; i++)
		gpuErrchk( cudaStreamCreate( &(stream[i]) ) );
		// gpuErrchk( cudaStreamCreateWithFlags( &(stream[i]) , cudaStreamNonBlocking ) );
}
bool RigidBodyForcePair::isWithinPairlistDist() const {
	if (isPmf) return true;
	float pairlistDist = 2.0f; /* TODO: get from conf */
	float rbDist = (rb1->getPosition() - rb2->getPosition()).length();
		
	for (int i = 0; i < gridKeyId1.size(); ++i) {
		const int k1 = gridKeyId1[i];
		const int k2 = gridKeyId2[i];
		float d1 = type1->densityGrids[k1].getRadius() + type1->densityGrids[k1].getCenter().length();
		float d2 = type2->potentialGrids[k2].getRadius() + type2->potentialGrids[k2].getCenter().length();
		if (rbDist < d1+d2+pairlistDist) 
			return true;
	}
	return false;
}
Vector3 RigidBodyForcePair::getOrigin1(const int i) {
	const int k1 = gridKeyId1[i];
	return rb1->transformBodyToLab( type1->densityGrids[k1].getOrigin() );
}
Vector3 RigidBodyForcePair::getOrigin2(const int i) {
	const int k2 = gridKeyId2[i];
	if (!isPmf)
		return rb2->transformBodyToLab( type2->potentialGrids[k2].getOrigin() );
	else
		return type2->rawPmfs[k2].getOrigin();
}		
Matrix3 RigidBodyForcePair::getBasis1(const int i) {
	const int k1 = gridKeyId1[i];
	return rb1->getOrientation()*type1->densityGrids[k1].getBasis();
}
Matrix3 RigidBodyForcePair::getBasis2(const int i) {
	const int k2 = gridKeyId2[i];
	if (!isPmf)
		return rb2->getOrientation()*type2->potentialGrids[k2].getBasis();
	else
		return type2->rawPmfs[k2].getBasis();
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
	  	| r = B'.ijk + c'   |
	  	`––––––––––––––––––./
		*/
		Matrix3 B1 = getBasis1(i);
		Vector3 c = getOrigin1(i) - getOrigin2(i);
		
		Matrix3 B2 = getBasis2(i).inverse();

		
		// RBTODO: get energy
		if (!isPmf) {								/* pair of RBs */
			computeGridGridForce<<< nb, NUMTHREADS, NUMTHREADS*2*sizeof(Vector3), s >>>
				(type1->rawDensityGrids_d[k1], type2->rawPotentialGrids_d[k2],
				 B1, B2, c,
				 forces_d[i], torques_d[i]);
		} else {										/* RB with a PMF */
			computeGridGridForce<<< nb, NUMTHREADS, NUMTHREADS*2*sizeof(Vector3), s >>>
				(type1->rawDensityGrids_d[k1], type2->rawPmfs_d[k2],
				 B1, B2, c,
				 forces_d[i], torques_d[i]);
		}
		// retrieveForcesForGrid(i); // this is slower than approach below, unsure why
		
		if (lastRbGridID >= 0)
			lastRbForcePair->retrieveForcesForGrid(lastRbGridID);
		lastRbForcePair = this;
		lastRbGridID = i;
	}
}

void RigidBodyForcePair::retrieveForcesForGrid(const int i) {
	// i: grid ID (less than numGrids)
	const cudaStream_t &s = stream[streamID[i]];
	const int nb = numBlocks[i];

	gpuErrchk(cudaMemcpyAsync(forces[i], forces_d[i], sizeof(Vector3)*nb,
														cudaMemcpyDeviceToHost, s));
	gpuErrchk(cudaMemcpyAsync(torques[i], torques_d[i], sizeof(Vector3)*nb,
														cudaMemcpyDeviceToHost, s));
	
}
void RigidBodyForcePair::processGPUForces() {
	
	const int numGrids = gridKeyId1.size();
	Vector3 f = Vector3(0.0f);
	Vector3 t = Vector3(0.0f);

	for (int i = 0; i < numGrids; i++) {
		const int nb = numBlocks[i];

		Vector3 tmpF = Vector3(0.0f);
		Vector3 tmpT = Vector3(0.0f);
			
		for (int j = 0; j < nb; j++) {
			tmpF = tmpF + forces[i][j];
			tmpT = tmpT + torques[i][j];
		}
		
		// tmpT is the torque calculated about the origin of grid k2 (e.g. c2)
		//   so here we transform torque to be about rb1
		Vector3 o2 = getOrigin2(i);
		tmpT = tmpT - (rb1->getPosition() - o2).cross( tmpF ); 

		// sum forces and torques
		f = f + tmpF;
		t = t + tmpT;
	}

	f *= updatePeriod;
	t *= updatePeriod;
	
	rb1->addForce( f );
	rb1->addTorque( t );

	if (!isPmf) {
		const Vector3 t2 = -t + (rb2->getPosition()-rb1->getPosition()).cross( f );
		rb2->addForce( -f );
		rb2->addTorque( t2 );
	}

	// printf("force: %s\n", f.toString().val());
	// printf("torque: %s\n", t.toString().val());
}

void RigidBodyController::print(int step) {
	// modeled after outputExtendedData() in Controller.C
	if ( step >= 0 ) {
		// Write RIGID BODY trajectory file
		if ( step % conf.outputPeriod == 0 ) {
			if ( ! trajFile.rdbuf()->is_open() ) {
	      // open file
			    // printf("OPENING RIGID BODY TRAJECTORY FILE\n");
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
			// printf("WRITING RIGID BODY COORDINATES AT STEP %d\n",step);
			printData(step,trajFile);
			trajFile.flush();    
		}
    
		// Write restart File
		/* if ( simParams->restartFrequency && */
		/* 		 ((step % simParams->restartFrequency) == 0) && */
		/* 		 (step != simParams->firstTimestep) )	{ */
		if ( step % conf.outputPeriod == 0 && step != 0 ){
		    // printf("RIGID BODY: WRITING RESTART FILE AT STEP %d\n", step);
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

int RigidBodyForcePair::initialize() {
    // printf("    Initializing (streams for) RB force pair...\n");

	const int numGrids = gridKeyId1.size();
	// RBTODO assert gridKeysIds are same size 

	// allocate memory for forces/torques
	for (int i = 0; i < numGrids; i++) {
		const int k1 = gridKeyId1[i];
		const int sz = type1->rawDensityGrids[k1].getSize();
		const int nb = sz / NUMTHREADS + ((sz % NUMTHREADS == 0) ? 0:1 );
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
    //printf("    Destructing RB force pair\n");
	const int numGrids = gridKeyId1.size();

	// printf("      numGrids = %d\n",numGrids);

	// RBTODO assert gridKeysIds are same size 

	// allocate memory for forces/torques
	if (streamID.size() > 0) {
		for (int i = 0; i < numGrids; i++) {
			const int k1 = gridKeyId1[i];
			const int nb = numBlocks[i];

			// free device memory for numBlocks of torque, etc.
			// printf("      Freeing device memory for forces/torques\n");
			gpuErrchk(cudaFree( forces_d[i] ));	
			gpuErrchk(cudaFree( torques_d[i] ));
		}
		gpuErrchk(cudaDeviceSynchronize());
	}
	streamID.clear();
	numBlocks.clear();
	forces.clear();
	forces_d.clear();
	torques.clear();
	torques_d.clear();
}



