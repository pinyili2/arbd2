#include <cassert>
#include "Configuration.h"
#include "RigidBodyType.h"
#include "Reservoir.h"
#include "BaseGrid.h"
#include "RigidBodyGrid.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), __FILE__, line);
      if (abort) exit(code);
   }
}

void RigidBodyType::clear() {
	num = 0;											// RBTODO: not 100% sure about this
	if (reservoir != NULL) delete reservoir;
	reservoir = NULL;
	// pmf = NULL;
	mass = 1.0;

	// TODO: make sure that this actually removes grid data
	potentialGrids.clear();
	densityGrids.clear();
	pmfs.clear();
	
	potentialGridKeys.clear();
	densityGridKeys.clear();
	pmfKeys.clear();

	
	if (numParticles != NULL) {
		for (int i=0; i < numPotGrids; ++i) {
			printf("CLEARING\n");
			if (numParticles[i] > 0) {
				delete[] particles[i];
				gpuErrchk(cudaFree( particles_d[i] ));
			}
		}
		delete[] numParticles;
		delete[] particles;
		delete[] particles_d;
		numParticles = NULL;
	}
		
	if (numPotGrids > 0) delete[] rawPotentialGrids;
	if (numDenGrids > 0) delete[] rawDensityGrids;
	rawPotentialGrids = NULL;
	rawDensityGrids = NULL;
}


// void RigidBodyType::setDampingCoeffs(float timestep, float tmp_mass, Vector3 tmp_inertia, float tmp_transDamping, float tmp_rotDamping) {
void RigidBodyType::setDampingCoeffs(float timestep) { /* MUST ONLY BE CALLED ONCE!!! */
	/*–––––––––––––––––––––––––––––––––––––––––––––––––––––––.
	| DiffCoeff = kT / dampingCoeff mass                     |
	|                                                        |
	| type->DampingCoeff has units of (1/ps)                 |
	|                                                        |
	| f[kcal/mol AA] = - dampingCoeff * momentum[amu AA/ns]  |
	|                                                        |
	| units "(1/ns) * (amu AA/ns)" "kcal_mol/AA" * 2.390e-09 |
	`–––––––––––––––––––––––––––––––––––––––––––––––––––––––*/

	/*––––––––––––––––––––––––––––––––––––––––––––––––––––.
	| < f(t) f(t') > = 2 kT dampingCoeff mass delta(t-t') |
	|                                                     |
	|  units "sqrt( k K (1/ns) amu / ns )" "kcal_mol/AA"  |
	|    * 2.1793421e-06                                  |
	`––––––––––––––––––––––––––––––––––––––––––––––––––––*/
	// RBTODO: make units consistent with rest of RB code 
	float Temp = 295; /* RBTODO: temperature should be read from grid? Or set in uniformly in config file */
	transForceCoeff = 2.1793421e-06 * Vector3::element_sqrt( 2*Temp*mass*transDamping/timestep );

	// setup for langevin
	// langevin = rbParams->langevin;
	// if (langevin) {
	// T = - dampingCoeff * angularMomentum

	// < f(t) f(t') > = 2 kT dampingCoeff inertia delta(t-t')
	rotTorqueCoeff = 2.1793421e-06 *
		Vector3::element_sqrt( 2*Temp* Vector3::element_mult(inertia,rotDamping) / timestep );


	transDamping = 2.3900574e-9 * transDamping;
	rotDamping = 2.3900574e-9 * rotDamping;

	// Also apply scale factors
	applyScaleFactors();
}
void RigidBodyType::applyScaleFactors() { /* currently this is called before raw is updated */
	applyScaleFactor(potentialGridScaleKeys, potentialGridScale, potentialGridKeys, potentialGrids);
	applyScaleFactor(densityGridScaleKeys, densityGridScale, densityGridKeys, densityGrids);
	applyScaleFactor(pmfScaleKeys, pmfScale, pmfKeys, pmfs);
}
void RigidBodyType::applyScaleFactor(
	const std::vector<String> &scaleKeys, const std::vector<float> &scaleFactors,
	const std::vector<String> &gridKeys, std::vector<BaseGrid> &grids) {
	for (int i = 0; i < scaleKeys.size(); i++) {
		const String &k1 = scaleKeys[i];
		for (int j = 0; j < gridKeys.size(); j++) {
			const String &k2 = gridKeys[j];
			if ( k1 == k2 )
				grids[j].scale( scaleFactors[i] );
		}
	}
}
	
void RigidBodyType::addGrid(String s, std::vector<String> &keys, std::vector<BaseGrid> &grids) {
	// tokenize and return
	int numTokens = s.tokenCount();
	if (numTokens != 2) {
		printf("ERROR: could not add Grid.\n"); // TODO improve this message
		exit(1);
	}
	String* token = new String[numTokens];
	s.tokenize(token);
	String key = token[0];
	BaseGrid g(token[1]);

	keys.push_back( key );
	grids.push_back( g );
}
void RigidBodyType::addPotentialGrid(String s) {
	addGrid(s, potentialGridKeys, potentialGrids);
}
void RigidBodyType::addDensityGrid(String s) {
	addGrid(s, densityGridKeys, densityGrids);
}
void RigidBodyType::addPMF(String s) {
	addGrid(s, pmfKeys, pmfs);
}

void RigidBodyType::addScaleFactor(String s, std::vector<String> &keys, std::vector<float> &vals) {
	// tokenize and return
	int numTokens = s.tokenCount();
	if (numTokens != 2) {
		printf("ERROR: could not add Grid.\n"); // TODO improve this message
		exit(1);
	}
	String* token = new String[numTokens];
	s.tokenize(token);
	String key = token[0];
	float v = (float) strtod(token[1], NULL);
	keys.push_back( key );
	vals.push_back( v );
}
void RigidBodyType::scalePotentialGrid(String s) {
	addScaleFactor(s, potentialGridScaleKeys, potentialGridScale);
}
void RigidBodyType::scaleDensityGrid(String s) {
	addScaleFactor(s, densityGridScaleKeys, densityGridScale);
}
void RigidBodyType::scalePMF(String s) {
	addScaleFactor(s, pmfScaleKeys, pmfScale);
}

// void RigidBodyType::copyGridToDevice(RigidBodyGrid* ptr_d, const RigidBodyGrid* g)  {
// 	// copy grid data
// 	float* tmpData;
// 	size_t sz = sizeof(float) * g->getSize();
// 	gpuErrchk(cudaMalloc( &tmpData, sz)); 
// 	gpuErrchk(cudaMemcpy( tmpData, g->val, sz, cudaMemcpyHostToDevice));

// 	// create temporary grid
// 	sz = sizeof(RigidBodyGrid);
// 	RigidBodyGrid* tmp;
// 	memcpy(tmp, g, sz);
// 	tmp->val = tmpData;
	
// 	// copy grid
// 	gpuErrchk(cudaMalloc(&ptr_d, sz));
// 	gpuErrchk(cudaMemcpy(ptr_d, tmp, sz, cudaMemcpyHostToDevice));
	
// 	// gpuErrchk(cudaMemcpy( &(ptr_d->val), &tmpData, sizeof(float*), cudaMemcpyHostToDevice));
// }
void RigidBodyType::copyGridToDevice(RigidBodyGrid*& ptr_d, RigidBodyGrid g)  {
	// copy grid data
	float* tmpData;
	size_t sz = sizeof(float) * g.getSize();
	gpuErrchk(cudaMalloc( &tmpData, sz)); 
	gpuErrchk(cudaMemcpy( tmpData, g.val, sz, cudaMemcpyHostToDevice));

	// set grid pointer to device 
	g.val = tmpData;
	
	// copy grid
	sz = sizeof(RigidBodyGrid);
	gpuErrchk(cudaMalloc(&ptr_d, sz));
	gpuErrchk(cudaMemcpy(ptr_d, &g, sz, cudaMemcpyHostToDevice));

	// prevent destructor from trying to delete device pointer
	g.val = NULL; 
}

void RigidBodyType::freeGridFromDevice(RigidBodyGrid* ptr_d) {
	gpuErrchk(cudaFree( ptr_d->val ));	// free grid data
	gpuErrchk(cudaFree( ptr_d ));		// free grid 
}

void RigidBodyType::copyGridsToDevice() {
	int ng = numDenGrids;
	if (ng > 0) {
	    rawDensityGrids_d = new RigidBodyGrid*[ng];
	    for (int gid = 0; gid < ng; gid++)
		copyGridToDevice(rawDensityGrids_d[gid], rawDensityGrids[gid]);
	}

	ng = numPotGrids;
	if (ng > 0) {
	    rawPotentialGrids_d = new RigidBodyGrid*[ng];
	    for (int gid = 0; gid < ng; gid++)
		copyGridToDevice(rawPotentialGrids_d[gid], rawPotentialGrids[gid]);
	}

	ng = numPmfs;
	if (ng > 0) {
	    rawPmfs_d = new RigidBodyGrid*[ng];
	    for (int gid = 0; gid < ng; gid++) {
		//RigidBodyGrid tmp = RigidBodyGrid(rawPmfs[gid]);
		//copyGridToDevice(rawPmfs_d[gid], &tmp);
		copyGridToDevice(rawPmfs_d[gid], RigidBodyGrid(rawPmfs[gid]));
	    }
	}
}

void RigidBodyType::updateRaw() {
	if (numPotGrids > 0 && rawPotentialGrids != NULL) delete[] rawPotentialGrids;
	if (numDenGrids > 0 && rawDensityGrids != NULL) delete[] rawDensityGrids;
	if (numPmfs > 0 && rawPmfs != NULL) delete[] rawPmfs;
	numPotGrids = potentialGrids.size();
	numDenGrids = densityGrids.size();
	numPmfs = pmfs.size();
	
	if (numPotGrids > 0) {
		rawPotentialGrids		= new RigidBodyGrid[numPotGrids];
		rawPotentialBases		= new Matrix3[numPotGrids];
		rawPotentialOrigins = new Vector3[numPotGrids];
	}
	if (numDenGrids > 0) {
		rawDensityGrids			= new RigidBodyGrid[numDenGrids];
		rawDensityBases			= new Matrix3[numDenGrids];
		rawDensityOrigins		= new Vector3[numDenGrids];
	}
	if (numPmfs > 0)
		rawPmfs = new BaseGrid[numPmfs];

	for (int i=0; i < numPotGrids; i++) {
		rawPotentialGrids[i]	 = potentialGrids[i];
		rawPotentialBases[i]	 = potentialGrids[i].getBasis();
		rawPotentialOrigins[i] = potentialGrids[i].getOrigin();
	}
	for (int i=0; i < numDenGrids; i++) {
		rawDensityGrids[i]		 = densityGrids[i];
		rawDensityBases[i]		 = densityGrids[i].getBasis();
		rawDensityOrigins[i]	 = densityGrids[i].getOrigin();
	}
	for (int i=0; i < numPmfs; i++)
		rawPmfs[i] = pmfs[i];	
}

void RigidBodyType::initializeParticleLists() {
	updateRaw();			   

	if (numPotGrids < 1) return;

	numParticles = new int[numPotGrids];
	particles = new int*[numPotGrids];
	particles_d = new int*[numPotGrids];

	// Loop over potential grids
	for (int i = 0; i < numPotGrids; ++i) {
		String& gridName = potentialGridKeys[i];
		numParticles[i] = 0;

		// Count the particles interacting with potential grid i
		// Loop over particle types
		for (int j = 0; j < conf->numParts; ++j) {
			// Loop over rigid body grid names associated with particle type
			const std::vector<String>& gridNames = conf->partRigidBodyGrid[j];
			for (int k = 0; k < gridNames.size(); ++k) {
				if (gridNames[k] == gridName) {
					numParticles[i] += conf->numPartsOfType[j];
				}
			}
		}

		if (numParticles[i] > 0) {

		    // allocate array of particle ids for the potential grid 
		    particles[i] = new int[numParticles[i]];
		    int pid = 0;
		
		    // Loop over particle types to count the number of particles
		    for (int j = 0; j < conf->numParts; ++j) {

			// Build temporary id array of type j particles
			int tmp[conf->numPartsOfType[j]];
			int currId = 0;
			for (int aid = 0; aid < conf->num; ++aid) {
			    if (conf->type[aid] == j)
				tmp[currId++] = aid;
			}
			if (currId == 0) continue;

			// Loop over rigid body grid names associated with particle type
			const std::vector<String>& gridNames = conf->partRigidBodyGrid[j];
			for (int k = 0; k < gridNames.size(); ++k) {
			    if (gridNames[k] == gridName) {
				// Copy type j particles to particles[i]
				memcpy( &(particles[i][pid]), tmp, sizeof(int)*currId );
				assert(currId == conf->numPartsOfType[j]);
				pid += conf->numPartsOfType[j];
			    }
			}
		    }

		    // Initialize device data
		    size_t sz = sizeof(int) * numParticles[i];
		    gpuErrchk(cudaMalloc( &(particles_d[i]), sz ));
		    gpuErrchk(cudaMemcpyAsync( particles_d[i], particles[i], sz, cudaMemcpyHostToDevice));
		}
	}
}
