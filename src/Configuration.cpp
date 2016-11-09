#include "Configuration.h"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), __FILE__, line);
      if (abort) exit(code);
   }
}

Configuration::Configuration(const char* config_file, int simNum, bool debug) :
		simNum(simNum) {
	// Read the parameters.
	//type_d = NULL;
	kTGrid_d = NULL;
	//bonds_d = NULL;
	//bondMap_d = NULL;
	//excludes_d = NULL;
	//excludeMap_d = NULL;
	//angles_d = NULL;
	//dihedrals_d = NULL;
	setDefaults();
	readParameters(config_file);

	// Get the number of particles
	printf("\nCounting particles specified in the ");
	if (restartCoordinates.length() > 0) {
    // Read them from the restart file.
		printf("restart file.\n");
		num = countRestart(restartCoordinates.val());
  } else {
    if (readPartsFromFile) readAtoms();
    if (numPartsFromFile > 0) {
      // Determine number of particles from input file (PDB-style)
      printf("input file.\n");
      num = numPartsFromFile;
    } else {
      // Sum up all particles in config file
      printf("configuration file.\n");
      //int num0 = 0;
      num = 0;
      for (int i = 0; i < numParts; i++) num += part[i].num;
      //num = num0;
    }
  } // end result: variable "num" is set


	// Set the number capacity
	printf("\nInitial particles: %d\n", num);
	if (numCap <= 0) numCap = numCapFactor*num; // max number of particles
	if (numCap <= 0) numCap = 20;

	// Allocate particle variables.
	pos = new Vector3[num * simNum];
	type = new int[num * simNum];
	serial = new int[num * simNum];
	posLast = new Vector3[num * simNum];
	name = new String[num * simNum];
	currSerial = 0;


  // Now, load the coordinates
	loadedCoordinates = false;

 // If we have a restart file - use it
	if (restartCoordinates.length() > 0) {
		loadRestart(restartCoordinates.val());
		printf("Loaded %d restart coordinates from `%s'.\n", num, restartCoordinates.val());
		printf("Particle numbers specified in the configuration file will be ignored.\n");
		loadedCoordinates = true;
	} else {
		// Load coordinates from a file?
		if (numPartsFromFile > 0) {
			loadedCoordinates = true;
			for (int i = 0; i < num; i++) {
				int numTokens = partsFromFile[i].tokenCount();

				// Break the line down into pieces (tokens) so we can process them individually
				String* tokenList = new String[numTokens];
				partsFromFile[i].tokenize(tokenList);

				int currType = 0;
				for (int j = 0; j < numParts; j++)
					if (tokenList[2] == part[j].name)
						currType = j;

				for (int s = 0; s < simNum; ++s)
					type[i + s*num] = currType;

				serial[i] = currSerial++;

				pos[i] = Vector3(atof(tokenList[3].val()),
												 atof(tokenList[4].val()),
												 atof(tokenList[5].val()));
			}
			delete[] partsFromFile;
			partsFromFile = NULL;
		} else {
			// Not loading coordinates from a file
			populate();
			if (inputCoordinates.length() > 0) {
				printf("Loading coordinates from %s ... ", inputCoordinates.val());
				loadedCoordinates = loadCoordinates(inputCoordinates.val());
				if (loadedCoordinates)
					printf("done!\n");
			}
		}
	}


	if (readBondsFromFile) readBonds();
	if (readExcludesFromFile) readExcludes();
	if (readAnglesFromFile) readAngles();
	if (readDihedralsFromFile) readDihedrals();

	kT = temperature * 0.0019872065f; // `units "k K" "kcal_mol"`
	if (temperatureGridFile.length() != 0) {
		printf("\nFound temperature grid file: %s\n", temperatureGridFile.val());
		tGrid = new BaseGrid(temperatureGridFile.val());
		printf("Loaded `%s'.\n", temperatureGridFile.val());
		printf("System size %s.\n", tGrid->getExtent().toString().val());

		// TODO: ask Max Belkin what this is about and how to remove hard-coded temps
		float ToSo = 1.0f / (295.0f * 4.634248239f); // 1 / (To * sigma(To))
		sigmaT = new BaseGrid(*tGrid);
		sigmaT->shift(-122.8305f);
		sigmaT->scale(0.0269167f);
		sigmaT->mult(*tGrid);
		sigmaT->scale(ToSo);

		kTGrid = new BaseGrid(*tGrid);
		float factor = 0.0019872065f; // `units "k K" "kcal_mol"`
		kTGrid->scale(factor);
		// char outFile[256];
		// char comment[256]; sprintf(comment,"KTGrid");
		// sprintf(outFile,"kTGrid.dx");
		// kTGrid->write(outFile, comment);
	}

	printf("\nFound %d particle types.\n", numParts);
	// Load the potential grids.
	printf("Loading the potential grids...\n");
	for (int i = 0; i < numParts; i++) {
		// Decide which type of grid is given.
		String map = partGridFile[i];
		int len = map.length();
		if (len >= 3 && map[len-3]=='.' && map[len-2]=='d' && map[len-1]=='x') {
			// A dx file. Load the old-fashioned way.
			part[i].pmf = new BaseGrid(map.val());
			part[i].meanPmf = part[i].pmf->mean();
			printf("Loaded dx grid `%s'.\n", map.val());
			printf("System size %s.\n", part[i].pmf->getExtent().toString().val());
		} else if  (len >= 4 && map[len-4]=='.' && map[len-3]=='d' && map[len-2]=='e' && map[len-1]=='f') {
			// A system definition file.
			String rootGrid = OverlordGrid::readDefFirst(map);
			OverlordGrid* over = new OverlordGrid(rootGrid.val());
			int count = over->readDef(map);
			printf("Loaded system def file `%s'.\n", map.val());
			printf("Found %d unique grids.\n", over->getUniqueGridNum());
			printf("Linked %d subgrids.\n", count);

			part[i].pmf = static_cast<BaseGrid*>(over);
			part[i].meanPmf = part[i].pmf->mean();
		} else {
			printf("WARNING: Unrecognized gridFile extension. Must be *.def or *.dx.\n");
			exit(-1);
		}

		if (partForceXGridFile[i].length() != 0) {
			part[i].forceXGrid = new BaseGrid(partForceXGridFile[i].val());
			printf("Loaded `%s'.\n", partForceXGridFile[i].val());
			printf("System size %s.\n", part[i].forceXGrid->getExtent().toString().val());
		}

		if (partForceYGridFile[i].length() != 0) {
			part[i].forceYGrid = new BaseGrid(partForceYGridFile[i].val());
			printf("Loaded `%s'.\n", partForceYGridFile[i].val());
			printf("System size %s.\n", part[i].forceYGrid->getExtent().toString().val());
		}

		if (partForceZGridFile[i].length() != 0) {
			part[i].forceZGrid = new BaseGrid(partForceZGridFile[i].val());
			printf("Loaded `%s'.\n", partForceZGridFile[i].val());
			printf("System size %s.\n", part[i].forceZGrid->getExtent().toString().val());
		}

		if (partDiffusionGridFile[i].length() != 0) {
			part[i].diffusionGrid = new BaseGrid(partDiffusionGridFile[i].val());
			printf("Loaded `%s'.\n", partDiffusionGridFile[i].val());
			printf("System size %s.\n", part[i].diffusionGrid->getExtent().toString().val());
		}

		if (temperatureGridFile.length() != 0) {
			if (partDiffusionGridFile[i].length() != 0) {
				part[i].diffusionGrid->mult(*sigmaT);
			} else {
				part[i].diffusionGrid = new BaseGrid(*sigmaT);
				part[i].diffusionGrid->scale(part[i].diffusion);
				// char outFile[256];
				// char comment[256]; sprintf(comment,"Diffusion for particle type %d", i);
				// sprintf(outFile,"diffusion%d.dx",i);
				// part[i].diffusionGrid->write(outFile, comment);
			}
		}

	}

    // Load reservoir files if any
    for (int i = 0; i < numParts; i++) {
        if (partReservoirFile[i].length() != 0) {
            printf("\nLoading the reservoirs for %s... \n", part[i].name.val());
            part[i].reservoir = new Reservoir(partReservoirFile[i].val());
            int nRes = part[i].reservoir->length();
            printf("\t -> %d reservoir(s) found in `%s'.\n", nRes, partReservoirFile[i].val());
        }
    }

    // Get the system dimensions
    // from the dimensions of supplied 3D potential maps
	sys = part[0].pmf;
	sysDim = part[0].pmf->getExtent();

// RBTODO: clean this mess up
	/* // RigidBodies... */
	/* if (numRigidTypes > 0) { */
	/* 	printf("\nCounting rigid bodies specified in the configuration file.\n"); */
	/* 	numRB = 0; */

	/* 	// grow list of rbs */
	/* 	for (int i = 0; i < numRigidTypes; i++) {			 */
	/* 		numRB += rigidBody[i].num; */

	/* 		std::vector<RigidBody> tmp; */
	/* 		for (int j = 0; j < rigidBody[i].num; j++) { */
	/* 			tmp.push_back( new RigidBody( this, rigidBody[i] ) ); */
	/* 		} */

	/* 		rbs.push_back(tmp); */
	/* 	} */
		// // state data
		// rbPos = new Vector3[numRB * simNum];
		// type = new int[numRB * simNum];

	/* } */
	/* printf("Initial RigidBodies: %d\n", numRB); */


	// Create exclusions from the exclude rule, if it was specified in the config file
	if (excludeRule != String("")) {
		int oldNumExcludes = numExcludes;
		Exclude* newExcludes = makeExcludes(bonds, bondMap, num, numBonds, excludeRule, numExcludes);
		if (excludes == NULL) {
			excludes = new Exclude[numExcludes];
		} else if (numExcludes >= excludeCapacity) {
			Exclude* tempExcludes = excludes;
			excludes = new Exclude[numExcludes];
			for (int i = 0; i < oldNumExcludes; i++)
				excludes[i] = tempExcludes[i];
			delete tempExcludes;
		}
		for (int i = oldNumExcludes; i < numExcludes; i++)
			excludes[i] = newExcludes[i - oldNumExcludes];
		printf("Built %d exclusions.\n",numExcludes);

		// Call compareExcludeIndex with qsort to sort the excludes by BOTH ind1 AND ind2
		std::sort(excludes, excludes + numExcludes, compare());

		/* Each particle may have a varying number of excludes
		 * excludeMap is an array with one element for each particle
		 * which keeps track of where a particle's excludes are stored
		 * in the excludes array.
		 * excludeMap[i].x is the index in the excludes array where the ith particle's excludes begin
		 * excludeMap[i].y is the index in the excludes array where the ith particle's excludes end
		 */
		excludeMap = new int2[numPartsFromFile];
		for (int i = 0; i < numPartsFromFile; i++) {
			excludeMap[i].x = -1;
			excludeMap[i].y = -1;
		}
		int currPart = -1;
		int lastPart = -1;
		for (int i = 0; i < numExcludes; i++) {
			if (excludes[i].ind1 != currPart) {
				currPart = excludes[i].ind1;
				excludeMap[currPart].x = i;
				if (lastPart >= 0)
					excludeMap[lastPart].y = i;
				lastPart = currPart;
			}
		}
	}

	// Count number of particles of each type
	numPartsOfType = new int[numParts];
	for (int i = 0; i < numParts; ++i) {
		numPartsOfType[i] = 0;
	}
	for (int i = 0; i < num; ++i) {
		++numPartsOfType[type[i]];
	}

	// Some geometric stuff that should be gotten rid of.
	Vector3 buffer = (sys->getCenter() + 2.0f*sys->getOrigin())/3.0f;
	initialZ = buffer.z;

	// Set the initial conditions.
	// Do the initial conditions come from restart coordinates?
	// inputCoordinates are ignored if restartCoordinates exist.
	/*
	if (restartCoordinates.length() > 0) {
		loadRestart(restartCoordinates.val());
		printf("Loaded %d restart coordinates from `%s'.\n", num, restartCoordinates.val());
		printf("Particle numbers specified in the configuration file will be ignored.\n");
	} else {
		// Set the particle types.

		// Load coordinates from a file?
		if (numPartsFromFile > 0) {
			for (int i = 0; i < num; i++) {
				int numTokens = partsFromFile[i].tokenCount();

				// Break the line down into pieces (tokens) so we can process them individually
				String* tokenList = new String[numTokens];
				partsFromFile[i].tokenize(tokenList);
				int currType = 0;
				for (int j = 0; j < numParts; j++)
					if (tokenList[2] == part[j].name)
						currType = j;
				type[i] = currType;
				serial[i] = currSerial;
				currSerial++;

				pos[i] = Vector3(atof(tokenList[3].val()), atof(tokenList[4].val()), atof(tokenList[5].val()));
			}
			if (partsFromFile != NULL) {
				delete[] partsFromFile;
				partsFromFile = NULL;
			}
		} else if (inputCoordinates.length() > 0) {
			populate();
			printf("Loading coordinates from %s.\n", inputCoordinates.val());
			bool loaded = loadCoordinates(inputCoordinates.val());
			if (loaded) 
				printf("Loaded initial coordinates from %s.\n", inputCoordinates.val());
		}
	}
	*/
	

	// Get the maximum particle radius.
	minimumSep = 0.0f;
	for (int i = 0; i < numParts; ++i)
		minimumSep = std::max(minimumSep, part[i].radius);
	minimumSep *= 2.5f; // Make it a little bigger.

	// Default outputEnergyPeriod
	if (outputEnergyPeriod < 0)
		outputEnergyPeriod = 10 * outputPeriod;
	
	// If we are running with debug ON, ask the user which force computation to use
	if (debug)
		getDebugForce();

	printf("\n");
	switchStart = cutoff - switchLen;
	if (fullLongRange == 0)
		printf("Cutting off the potential from %.10g to %.10g.\n", switchStart, switchStart+switchLen);
	
	if (fullLongRange != 0)
		printf("No cell decomposition created.\n");
}

Configuration::~Configuration() {
	// System state
	delete[] pos;
	delete[] posLast;
	delete[] type;
	delete[] name;
	
	// Particle parameters
	delete[] part;
	delete[] partGridFile;
	delete[] partForceXGridFile;
	delete[] partForceYGridFile;
	delete[] partForceZGridFile;
	delete[] partDiffusionGridFile;
	delete[] partReservoirFile;
	partRigidBodyGrid.clear();
	
	// TODO: plug memory leaks
	if (partsFromFile != NULL) delete[] partsFromFile;
	if (bonds != NULL) delete[] bonds;
	if (bondMap != NULL) delete[] bondMap;
	if (excludes != NULL) delete[] excludes;
	if (excludeMap != NULL) delete[] excludeMap;
	if (angles != NULL) delete[] angles;
	if (dihedrals != NULL) delete[] dihedrals;

	delete[] numPartsOfType;
	  
	// Table parameters
	delete[] partTableFile;
	delete[] partTableIndex0;
	delete[] partTableIndex1;

	delete[] bondTableFile;

	delete[] angleTableFile;

	delete[] dihedralTableFile;

	//if (type_d != NULL) {
		//gpuErrchk(cudaFree(type_d));
		gpuErrchk(cudaFree(sys_d));
		gpuErrchk(cudaFree(kTGrid_d));
		gpuErrchk(cudaFree(part_d));
		//gpuErrchk(cudaFree(bonds_d));
		//gpuErrchk(cudaFree(bondMap_d));
		//gpuErrchk(cudaFree(excludes_d));
		//gpuErrchk(cudaFree(excludeMap_d));
		//gpuErrchk(cudaFree(angles_d));
		//gpuErrchk(cudaFree(dihedrals_d));
	//}
}

void Configuration::copyToCUDA() {
	printf("Copying to GPU %d\n", GPUManager::current());

	BrownianParticleType **part_addr = new BrownianParticleType*[numParts];

	// Copy the BaseGrid objects and their member variables/objects
	gpuErrchk(cudaMalloc(&part_d, sizeof(BrownianParticleType*) * numParts));
	// TODO: The above line fails when there is not enough memory. If it fails, stop.
	
	for (int i = 0; i < numParts; i++) {
		BaseGrid *pmf = NULL, *diffusionGrid = NULL;
		BrownianParticleType *b = new BrownianParticleType(part[i]);
		// Copy PMF
		if (part[i].pmf != NULL) {
			float *val = NULL;
			size_t sz = sizeof(float) * part[i].pmf->getSize();
		  gpuErrchk(cudaMalloc(&pmf, sizeof(BaseGrid)));
		  gpuErrchk(cudaMalloc(&val, sz));
		  gpuErrchk(cudaMemcpyAsync(val, part[i].pmf->val, sz, cudaMemcpyHostToDevice));
		  BaseGrid *pmf_h = new BaseGrid(*part[i].pmf);
			pmf_h->val = val;
			gpuErrchk(cudaMemcpy(pmf, pmf_h, sizeof(BaseGrid), cudaMemcpyHostToDevice));
			pmf_h->val = NULL;
		}
		
		// Copy the diffusion grid
		if (part[i].diffusionGrid != NULL) {
			float *val = NULL;
			size_t sz = sizeof(float) * part[i].diffusionGrid->getSize();
		  BaseGrid *diffusionGrid_h = new BaseGrid(*part[i].diffusionGrid);
		  gpuErrchk(cudaMalloc(&diffusionGrid, sizeof(BaseGrid)));
		  gpuErrchk(cudaMalloc(&val, sz));
			diffusionGrid_h->val = val;
			gpuErrchk(cudaMemcpyAsync(diffusionGrid, diffusionGrid_h, sizeof(BaseGrid),
					cudaMemcpyHostToDevice));
		  gpuErrchk(cudaMemcpy(val, part[i].diffusionGrid->val, sz, cudaMemcpyHostToDevice));
			diffusionGrid_h->val = NULL;
		}
		
		b->pmf = pmf;
		b->diffusionGrid = diffusionGrid;
		gpuErrchk(cudaMalloc(&part_addr[i], sizeof(BrownianParticleType)));
		gpuErrchk(cudaMemcpyAsync(part_addr[i], b, sizeof(BrownianParticleType),
				cudaMemcpyHostToDevice));
	}
	// RBTODO: moved this out of preceding loop; was that correct?
	gpuErrchk(cudaMemcpyAsync(part_d, part_addr, sizeof(BrownianParticleType*) * numParts,
				cudaMemcpyHostToDevice));

	// kTGrid_d
	kTGrid_d = NULL;
	if (temperatureGridFile.length() > 0) {
		gpuErrchk(cudaMalloc(&kTGrid_d, sizeof(BaseGrid)));
		gpuErrchk(cudaMemcpyAsync(kTGrid_d, kTGrid, sizeof(BaseGrid), cudaMemcpyHostToDevice));
	}

	// type_d and sys_d
	gpuErrchk(cudaMalloc(&sys_d, sizeof(BaseGrid)));
	gpuErrchk(cudaMemcpyAsync(sys_d, sys, sizeof(BaseGrid), cudaMemcpyHostToDevice));
	/*gpuErrchk(cudaMalloc(&type_d, sizeof(int) * num * simNum));
	gpuErrchk(cudaMemcpyAsync(type_d, type, sizeof(int) * num * simNum, cudaMemcpyHostToDevice));
	
	if (numBonds > 0) {
		// bonds_d
		gpuErrchk(cudaMalloc(&bonds_d, sizeof(Bond) * numBonds));
		gpuErrchk(cudaMemcpyAsync(bonds_d, bonds, sizeof(Bond) * numBonds, cudaMemcpyHostToDevice));
		
		// bondMap_d
		gpuErrchk(cudaMalloc(&bondMap_d, sizeof(int2) * num));
		gpuErrchk(cudaMemcpyAsync(bondMap_d, bondMap, sizeof(int2) * num, cudaMemcpyHostToDevice));
	}

	if (numExcludes > 0) {
		// excludes_d
		gpuErrchk(cudaMalloc(&excludes_d, sizeof(Exclude) * numExcludes));
		gpuErrchk(cudaMemcpyAsync(excludes_d, excludes, sizeof(Exclude) * numExcludes,
				cudaMemcpyHostToDevice));
		
		// excludeMap_d
		gpuErrchk(cudaMalloc(&excludeMap_d, sizeof(int2) * num));
		gpuErrchk(cudaMemcpyAsync(excludeMap_d, excludeMap, sizeof(int2) * num,
				cudaMemcpyHostToDevice));
	}

	if (numAngles > 0) {
		// angles_d
		gpuErrchk(cudaMalloc(&angles_d, sizeof(Angle) * numAngles));
		gpuErrchk(cudaMemcpyAsync(angles_d, angles, sizeof(Angle) * numAngles,
				cudaMemcpyHostToDevice));
	}

	if (numDihedrals > 0) {
		// dihedrals_d
		gpuErrchk(cudaMalloc(&dihedrals_d, sizeof(Dihedral) * numDihedrals));
		gpuErrchk(cudaMemcpyAsync(dihedrals_d, dihedrals,
												 		  sizeof(Dihedral) * numDihedrals,
														 	cudaMemcpyHostToDevice));
	}*/
	gpuErrchk(cudaDeviceSynchronize());
}

void Configuration::setDefaults() {
    // System parameters
	outputName = "out";
	timestep = 1e-5f;
	rigidBodyGridGridPeriod = 1;
	steps = 100;
	seed = 0;
	inputCoordinates = "";
	restartCoordinates = "";
	numberFluct = 0;
	numberFluctPeriod = 200;
	interparticleForce = 1;
	tabulatedPotential = 0;
	fullLongRange = 1;
	//	kTGridFile = ""; // Commented out for an unknown reason
	temperature = 295.0f;
	temperatureGridFile = "";
	coulombConst = 566.440698f/92.0f;
	electricField = 0.0f;
	cutoff = 10.0f;
	switchLen = 2.0f;
	pairlistDistance = 2.0f;
	outputPeriod = 200;
	outputEnergyPeriod = -1;
	outputFormat = TrajectoryWriter::formatDcd;
	currentSegmentZ = -1.0f;
	numCap = 0;
	decompPeriod = 10;
	readPartsFromFile = 0;
	numPartsFromFile = 0;
	partsFromFile = NULL;
	readBondsFromFile = false;
	numBonds = 0;
	bonds = NULL;
	bondMap = NULL;
	numTabBondFiles = 0;
	readExcludesFromFile = false;
	numExcludes = 0;
	excludeCapacity = 256;
	excludes = NULL;
	excludeMap = NULL;
	excludeRule = "";
	readAnglesFromFile = false;
	numAngles = 0;
	angles = NULL;
	numTabAngleFiles = 0;
	readDihedralsFromFile = false;
	numDihedrals = 0;
	dihedrals = NULL;
	numTabDihedralFiles = 0;

	// Hidden parameters
	// Might be parameters later
	numCapFactor = 5;
}

int Configuration::readParameters(const char * config_file) {
	Reader config(config_file);
	printf("Read config file %s\n", config_file);

	// Get the number of particles.
	const int numParams = config.length();
	numParts = config.countParameter("particle");
	numRigidTypes = config.countParameter("rigidBody");

	// Allocate the particle variables.
	part = new BrownianParticleType[numParts];
	partGridFile = new String[numParts];
	partForceXGridFile = new String[numParts];
	partForceYGridFile = new String[numParts];
	partForceZGridFile = new String[numParts];
	partDiffusionGridFile = new String[numParts];
	partReservoirFile = new String[numParts];
	partRigidBodyGrid.resize(numParts);
	
	// Allocate the table variables.
	partTableFile = new String[numParts*numParts];
	partTableIndex0 = new int[numParts*numParts];
	partTableIndex1 = new int[numParts*numParts];

	// Allocate rigid body types
	rigidBody = new RigidBodyType[numRigidTypes];
	
	int btfcap = 10;
	bondTableFile = new String[btfcap];

	int atfcap = 10;
	angleTableFile = new String[atfcap];

	int dtfcap = 10;
	dihedralTableFile = new String[dtfcap];

	int currPart = -1;
	int currTab = -1;
	int currBond = -1;
	int currAngle = -1;
	int currDihedral = -1;
	int currRB = -1;

	int partClassPart =  0;
	int partClassRB   =  1;
	int currPartClass = -1;				// 0 => particle, 1 => rigidBody
	
	for (int i = 0; i < numParams; i++) {
		String param = config.getParameter(i);
		String value = config.getValue(i);
		// GLOBAL
		if (param == String("outputName"))
			outputName = value;
		else if (param == String("timestep"))
			timestep = (float) strtod(value.val(), NULL);
		else if (param == String("rigidBodyGridGridPeriod"))
			rigidBodyGridGridPeriod = atoi(value.val());
		else if (param == String("steps"))
			steps = atol(value.val());
		else if (param == String("seed"))
			seed = atoi(value.val());
		else if (param == String("inputCoordinates"))
			inputCoordinates = value;
		else if (param == String("restartCoordinates"))
			restartCoordinates = value;
		else if (param == String("temperature"))
			temperature =  (float) strtod(value.val(),NULL);
		else if (param == String("temperatureGrid"))
			temperatureGridFile = value;
		else if (param == String("numberFluct"))
			numberFluct = atoi(value.val());
		else if (param == String("numberFluctPeriod"))
			numberFluctPeriod = atoi(value.val());
		else if (param == String("interparticleForce"))
			interparticleForce = atoi(value.val());
		else if (param == String("fullLongRange") || param == String("fullElect") )
			fullLongRange = atoi(value.val());
		else if (param == String("coulombConst"))
			coulombConst = (float) strtod(value.val(), NULL);
		else if (param == String("electricField"))
			electricField = (float) strtod(value.val(), NULL);
		else if (param == String("cutoff"))
			cutoff = (float) strtod(value.val(), NULL);
		else if (param == String("switchLen"))
			switchLen = (float) strtod(value.val(), NULL);
		else if (param == String("pairlistDistance"))
			pairlistDistance = (float) strtod(value.val(), NULL);
		else if (param == String("outputPeriod"))
			outputPeriod = atoi(value.val());
		else if (param == String("outputEnergyPeriod"))
			outputEnergyPeriod = atoi(value.val());
		else if (param == String("outputFormat"))
			outputFormat = TrajectoryWriter::getFormatCode(value);
		else if (param == String("currentSegmentZ"))
			currentSegmentZ = (float) strtod(value.val(), NULL);
		else if (param == String("numCap"))
			numCap = atoi(value.val());
		else if (param == String("decompPeriod"))
			decompPeriod = atoi(value.val());
		// PARTICLES
		else if (param == String("particle")) {
			part[++currPart] = BrownianParticleType(value);
			currPartClass = partClassPart;
		}
		else if (param == String("forceXGridFile"))
			partForceXGridFile[currPart] = value;
		else if (param == String("forceYGridFile"))
			partForceYGridFile[currPart] = value;
		else if (param == String("forceZGridFile"))
			partForceZGridFile[currPart] = value;
		else if (param == String("diffusionGridFile"))
			partDiffusionGridFile[currPart] = value;
		else if (param == String("diffusion"))
			part[currPart].diffusion = (float) strtod(value.val(), NULL);
		else if (param == String("charge"))
			part[currPart].charge = (float) strtod(value.val(), NULL);
		else if (param == String("radius"))
			part[currPart].radius = (float) strtod(value.val(), NULL);
		else if (param == String("eps"))
			part[currPart].eps = (float) strtod(value.val(), NULL);
		else if (param == String("reservoirFile"))
			partReservoirFile[currPart] = value;
		else if (param == String("tabulatedPotential"))
			tabulatedPotential = atoi(value.val());
		else if (param == String("tabulatedFile"))
			readTableFile(value, ++currTab);
		else if (param == String("tabulatedBondFile")) {
			if (numTabBondFiles >= btfcap) {
				String* temp = bondTableFile;
				btfcap *= 2;	
				bondTableFile = new String[btfcap];
				for (int j = 0; j < numTabBondFiles; j++)
					bondTableFile[i] = temp[i];
				delete[] temp;
			}
			if (readBondFile(value, ++currBond))
				numTabBondFiles++;
		} else if (param == String("inputParticles")) {
			if (readPartsFromFile) {
				printf("WARNING: More than one particle file specified. Discarding new file.\n");
			} else {
				partFile = value;
				readPartsFromFile = true;
				loadedCoordinates = true;
			}
		} else if (param == String("inputBonds")) {
			if (readBondsFromFile) {
				printf("WARNING: More than one bond file specified. Discarding new bond file.\n");
			} else {
				bondFile = value;				
				readBondsFromFile = true;
			}
		} else if (param == String("inputExcludes")) {
			if (readExcludesFromFile) {
				printf("WARNING: More than one exclude file specified. Discarding new exclude file.\n");
			} else {
				excludeFile = value;				
				readExcludesFromFile = true;
			}
		} else if (param == String("exclude") or param == String("exclusion")) {
			excludeRule = value; 
		} else if (param == String("inputAngles")) {
			if (readAnglesFromFile) {
				printf("WARNING: More than one angle file specified. Discarding new angle file.\n");
			} else {
				angleFile = value;
				readAnglesFromFile = true;
			}
		} else if (param == String("tabulatedAngleFile")) {
			if (numTabAngleFiles >= atfcap) {
				String* temp = angleTableFile;
				atfcap *= 2;	
				angleTableFile = new String[atfcap];
				for (int j = 0; j < numTabAngleFiles; j++)
					angleTableFile[i] = temp[i];
				delete[] temp;
			}
			if (readAngleFile(value, ++currAngle))
				numTabAngleFiles++;
		} else if (param == String("inputDihedrals")) {
			if (readDihedralsFromFile) {
				printf("WARNING: More than one dihedral file specified. Discarding new dihedral file.\n");
			} else {
				dihedralFile = value;
				readDihedralsFromFile = true;
			}
		} else if (param == String("tabulatedDihedralFile")) {
			if (numTabDihedralFiles >= dtfcap) {
				String * temp = dihedralTableFile;
				dtfcap *= 2;
				dihedralTableFile = new String[dtfcap];
				for (int j = 0; j < numTabDihedralFiles; j++)
					dihedralTableFile[i] = temp[i];
				delete[] temp;
			}
			if (readDihedralFile(value, ++currDihedral))
				numTabDihedralFiles++;
		} else if (param == String("rigidBodyPotential")) {
			partRigidBodyGrid[currPart].push_back(value);
		}
		// RIGID BODY
		else if (param == String("rigidBody")) {
			// part[++currPart] = BrownianParticleType(value);
			rigidBody[++currRB] = RigidBodyType(value, this);
			currPartClass = partClassRB;
		}
		else if (param == String("mass"))
			rigidBody[currRB].mass = (float) strtod(value.val(), NULL);
		else if (param == String("inertia"))
			rigidBody[currRB].inertia = stringToVector3( value );
		else if (param == String("transDamping"))
			rigidBody[currRB].transDamping = stringToVector3( value );
		else if (param == String("rotDamping"))
			rigidBody[currRB].rotDamping = stringToVector3( value );

		else if (param == String("densityGrid"))
			rigidBody[currRB].addDensityGrid(value);
		else if (param == String("potentialGrid"))
			rigidBody[currRB].addPotentialGrid(value);
		else if (param == String("densityGridScale"))
			rigidBody[currRB].scaleDensityGrid(value);
		else if (param == String("potentialGridScale"))
			rigidBody[currRB].scalePotentialGrid(value);
		else if (param == String("pmfScale"))
			rigidBody[currRB].scalePMF(value);
		else if (param == String("position"))
			rigidBody[currRB].initPos = stringToVector3( value );
		else if (param == String("orientation"))
			rigidBody[currRB].initRot = stringToMatrix3( value );
		else if (param == String("inputRBCoordinates"))
			inputRBCoordinates = value;
		
		// COMMON
		else if (param == String("num")) {
			if (currPartClass == partClassPart)
				part[currPart].num = atoi(value.val());
			else if (currPartClass == partClassRB) 
				rigidBody[currRB].num = atoi(value.val());
		}
		else if (param == String("gridFile")) {
			if (currPartClass == partClassPart)
				partGridFile[currPart] = value;
			else if (currPartClass == partClassRB)
				rigidBody[currRB].addPMF(value);
		}
		// UNKNOWN
		else {
			printf("ERROR: Unrecognized keyword `%s'.\n", param.val());
			exit(1);
		}
	}

	// extra configuration for RB types
	for (int i = 0; i < numRigidTypes; i++)
		rigidBody[i].setDampingCoeffs(timestep);
	
	return numParams;
}
Vector3 Configuration::stringToVector3(String s) {
	// tokenize and return
	int numTokens = s.tokenCount();
	if (numTokens != 3) {
		printf("ERROR: could not convert input to Vector3.\n"); // TODO improve this message
		exit(1);
	}
	String* token = new String[numTokens];
	s.tokenize(token);
	Vector3 v( (float) strtod(token[0], NULL),
						 (float) strtod(token[1], NULL),
						 (float) strtod(token[2], NULL) );
	return v;
}
Matrix3 Configuration::stringToMatrix3(String s) {
	// tokenize and return
	int numTokens = s.tokenCount();
	if (numTokens != 9) {
		printf("ERROR: could not convert input to Matrix3.\n"); // TODO improve this message
		exit(1);
	}
	String* token = new String[numTokens];
	s.tokenize(token);
	Matrix3 m( (float) strtod(token[0], NULL),
						 (float) strtod(token[1], NULL),
						 (float) strtod(token[2], NULL),
						 (float) strtod(token[3], NULL),
						 (float) strtod(token[4], NULL),
						 (float) strtod(token[5], NULL),
						 (float) strtod(token[6], NULL),
						 (float) strtod(token[7], NULL),
						 (float) strtod(token[8], NULL) );
	return m;
}

void Configuration::readAtoms() {
	// Open the file
	FILE* inp = fopen(partFile.val(), "r");
	char line[256];

	// If the particle file cannot be found, exit the program
	if (inp == NULL) {
		printf("ERROR: Could not open `%s'.\n", partFile.val());
		bool found = true;
		for (int i = 0; i < numParts; i++)
			if (part[i].num == 0)
				found = false;
		if (!found) {
			printf("ERROR: Number of particles not specified in config file.\n");
			exit(1);
		}
		printf("Using default coordinates file\n");
		return;
	}

	// Our particle array has a starting capacity of 256
	// We will expand this later if we need to.
	int capacity = 256;
	numPartsFromFile = 0;
	partsFromFile = new String[capacity];
	indices = new int[capacity];
	indices[0] = 0;

	// Get and process all lines of input
	while (fgets(line, 256, inp) != NULL) {
		// Lines in the particle file that begin with # are comments
		if (line[0] == '#') continue;
		      
		String s(line);
		int numTokens = s.tokenCount();
		      
		// Break the line down into pieces (tokens) so we can process them individually
		String* tokenList = new String[numTokens];
		s.tokenize(tokenList);

		// Legitimate ATOM input lines have 6 tokens: 
		// ATOM | Index | Name | X-coord | Y-coord | Z-coord
		// A line without exactly six tokens should be discarded.
		if (numTokens != 6) {
			printf("Warning: Invalid particle file line: %s\n", line);
			return;
		}

		// Ensure that this particle's type was defined in the config file.
		// If not, discard this line.
		bool found;
		for (int j = 0; j < numParts; j++) {
			// If this particle type exists, add a new one to the list
			if (part[j].name == tokenList[2]) {
				found = true;
				part[j].num++;
			}
		}

		// If the particle's type does not exist according to the config file, discard it.
		if (!found) {
			printf("WARNING Unknown particle type %s found and discarded.\n", tokenList[2].val());
			continue;
		}

		// If we don't have enough room in our particle array, we need to expand it.
		if (numPartsFromFile >= capacity) {
			// Temporary pointers to the old arrays
			String* temp = partsFromFile;	
			int* temp2 = indices;

			// Double the capacity
			capacity *= 2;

			// Create pointers to new arrays which are twice the size of the old ones
			partsFromFile = new String[capacity];
			indices = new int[capacity];
		
			// Copy the old values into the new arrays
			for (int j = 0; j < numPartsFromFile; j++) {
				partsFromFile[j] = temp[j];
				indices[j] = temp2[j];
			}

			// delete the old arrays
			delete[] temp;
			delete[] temp2;
		}
		// Make sure the index of this particle is unique.
		// NOTE: The particle list is sorted by index. 
		bool uniqueID = true;		
		int key = atoi(tokenList[1].val());
		int mid = 0;

		// If the index is greater than the last index in the list, 
		// this particle belongs at the end of the list. Since the 
		// list is kept sorted, we know this is okay.
		if (numPartsFromFile == 0 || key > indices[numPartsFromFile - 1]) {
			indices[numPartsFromFile] = key;
			partsFromFile[numPartsFromFile++] = line;
		}
		// We need to do a binary search to figure out if
		// the index already exists in the list. 
		// The assumption is that input files SHOULD have their indices sorted in 
		// ascending order, so we shouldn't actually use the binary search 
		// or the sort (which is pretty time consuming) very often.
		else {
			int low = 0, high = numPartsFromFile - 1;
			
			while (low <= high) {
				mid = (int)((high - low) / 2 + low);
				int curr = indices[mid];
				if (curr < key) {
					low = mid + 1;
				} else if (curr > key) {
					high = mid - 1;
				} else {
					// For now, particles with non-unique IDs are simply not added to the array
					// Other possible approaches which are not yet implemented:
					// 1: Keep track of these particles and assign them new IDs after you have
					//    already added all of the other particles. 	
					// 2: Get rid of ALL particles with that ID, even the ones that have already 
					//    been added.
					printf("WARNING: Non-unique ID found: %s\n", line);
					uniqueID = false;
					break;
				}
			}
			if (uniqueID) {
				// Add the particle to the end of the array, then sort it. 
				indices[numPartsFromFile] = key;
				partsFromFile[numPartsFromFile++] = line;
				std::sort(indices, indices + numPartsFromFile);
				std::sort(partsFromFile, partsFromFile + numPartsFromFile, compare());		
			}
		}
	}
}

void Configuration::readBonds() {
	// Open the file
	FILE* inp = fopen(bondFile.val(), "r");
	char line[256];

	// If the particle file cannot be found, exit the program
	if (inp == NULL) {
		printf("WARNING: Could not open `%s'.\n", bondFile.val());
		printf("         This simulation will not use particle bonds.\n");
		return;
	}

	// Our particle array has a starting capacity of 256
	// We will expand this later if we need to.
	int capacity = 256;
	numBonds = 0;
	bonds = new Bond[capacity];

	// Get and process all lines of input
	while (fgets(line, 256, inp) != NULL) {
		
		// Lines in the particle file that begin with # are comments
		if (line[0] == '#') continue;
		      
		String s(line);
		int numTokens = s.tokenCount();
		      
		// Break the line down into pieces (tokens) so we can process them individually
		String* tokenList = new String[numTokens];
		s.tokenize(tokenList);

		// Legitimate BOND input lines have 4 tokens: 
		// BOND | OPERATION_FLAG | INDEX1 | INDEX2 | FILENAME 
		// A line without exactly five tokens should be discarded.
		if (numTokens != 5) {
			printf("WARNING: Invalid bond file line: %s\n", line);
			continue;
		}

		String op = tokenList[1];
		int ind1 = atoi(tokenList[2].val());
		int ind2 = atoi(tokenList[3].val());
		String file_name = tokenList[4];

		if (ind1 == ind2) {
			printf("WARNING: Invalid bond file line: %s\n", line);
			continue;
		}
		
		// If we don't have enough room in our bond array, we need to expand it.
		if (numBonds+1 >= capacity) { // "numBonds+1" because we are adding two bonds to array
			// Temporary pointer to the old array
			Bond* temp = bonds;	

			// Double the capacity
			capacity *= 2;

			// Create pointer to new array which is twice the size of the old one
			bonds = new Bond[capacity];
		
			// Copy the old values into the new array
			for (int j = 0; j < numBonds; j++)
				bonds[j] = temp[j];

			// delete the old array
			delete[] temp;
		}
		// Add the bond to the bond array
		// We must add it twice: Once for (ind1, ind2) and once for (ind2, ind1)
		
		// RBTODO: add ind1/2 to exclusion list here iff op == REPLACE
		/*if (op == Bond::REPLACE)
		if( (int)(op) == 1)
		{
			printf("WARNING: Bond exclusions not implemented\n");
			continue;
		}*/

		if (ind1 < 0 || ind1 >= num || ind2 < 0 || ind2 >=num) {
			printf("ERROR: Bond file line '%s' includes invalid index\n", line);
			exit(1);
		}

		Bond* b = new Bond(op, ind1, ind2, file_name);
		bonds[numBonds++] = *b;
		b = new Bond(op, ind2, ind1, file_name);
		bonds[numBonds++] = *b;
		delete[] tokenList;
	}	
	// Call compareBondIndex with qsort to sort the bonds by BOTH ind1 AND ind2
	std::sort(bonds, bonds + numBonds, compare());

	/* Each particle may have a varying number of bonds
	 * bondMap is an array with one element for each particle
	 * which keeps track of where a particle's bonds are stored
	 * in the bonds array.
	 * bondMap[i].x is the index in the bonds array where the ith particle's bonds begin
	 * bondMap[i].y is the index in the bonds array where the ith particle's bonds end
	 */
	bondMap = new int2[num];
	for (int i = 0; i < num; i++) {	
		bondMap[i].x = -1;
		bondMap[i].y = -1;
	}
	int currPart = -1;
	int lastPart = -1;
	for (int i = 0; i < numBonds; i++) {
		if (bonds[i].ind1 != currPart) {
			currPart = bonds[i].ind1;
			bondMap[currPart].x = i;
			if (lastPart >= 0) bondMap[lastPart].y = i;
			lastPart = currPart;
		}
	}
	if (bondMap[lastPart].x > 0)
		bondMap[lastPart].y = numBonds;
}

void Configuration::readExcludes()
{
	// Open the file
	FILE* inp = fopen(excludeFile.val(), "r");
	char line[256];

	// If the exclusion file cannot be found, exit the program
	if (inp == NULL) {
		printf("WARNING: Could not open `%s'.\n", excludeFile.val());
		printf("This simulation will not use exclusions.\n");
		return;
	}

	// Our particle array has a starting capacity of 256
	// We will expand this later if we need to.
	excludeCapacity = 256;
	numExcludes = 0;
	excludes = new Exclude[excludeCapacity];

	// Get and process all lines of input
	while (fgets(line, 256, inp) != NULL) {
		// Lines in the particle file that begin with # are comments
		if (line[0] == '#') continue;
		      
		String s(line);
		int numTokens = s.tokenCount();
		      
		// Break the line down into pieces (tokens) so we can process them individually
		String* tokenList = new String[numTokens];
		s.tokenize(tokenList);

		// Legitimate EXCLUDE input lines have 3 tokens: 
		// BOND | INDEX1 | INDEX2
		// A line without exactly three tokens should be discarded.
		if (numTokens != 3) {
			printf("WARNING: Invalid exclude file line: %s\n", line);
			continue;
		}
		int ind1 = atoi(tokenList[1].val());
		int ind2 = atoi(tokenList[2].val());
		if (ind1 >= num || ind2 >= num) 
			continue;
		
		// If we don't have enough room in our bond array, we need to expand it.
		if (numExcludes >= excludeCapacity) {
			// Temporary pointer to the old array
			Exclude* temp = excludes;	

			// Double the capacity
			excludeCapacity *= 2;

			// Create pointer to new array which is twice the size of the old one
			excludes = new Exclude[excludeCapacity];
		
			// Copy the old values into the new array
			for (int j = 0; j < numExcludes; j++)
				excludes[j] = temp[j];

			// delete the old array
			delete[] temp;
		}
		// Add the bond to the exclude array
		// We must add it twice: Once for (ind1, ind2) and once for (ind2, ind1)
		Exclude ex(ind1, ind2);
		excludes[numExcludes++] = ex;
		Exclude ex2(ind2, ind1);
		excludes[numExcludes++] = ex2;
		delete[] tokenList;
	}	
	// Call compareExcludeIndex with qsort to sort the excludes by BOTH ind1 AND ind2
	std::sort(excludes, excludes + numExcludes, compare());

	/* Each particle may have a varying number of excludes
	 * excludeMap is an array with one element for each particle
	 * which keeps track of where a particle's excludes are stored
	 * in the excludes array.
	 * excludeMap[i].x is the index in the excludes array where the ith particle's excludes begin
	 * excludeMap[i].y is the index in the excludes array where the ith particle's excludes end
	 */
	excludeMap = new int2[num];
	for (int i = 0; i < num; i++) {	
		excludeMap[i].x = -1;
		excludeMap[i].y = -1;
	}
	int currPart = -1;
	int lastPart = -1;
	for (int i = 0; i < numExcludes; i++) {
		if (excludes[i].ind1 != currPart) {
			currPart = excludes[i].ind1;
			if (currPart < num) {
				excludeMap[currPart].x = i;
				if (lastPart >= 0)
					excludeMap[lastPart].y = i;
				lastPart = currPart;
			}
		}
	}
}

void Configuration::readAngles() {
	FILE* inp = fopen(angleFile.val(), "r");
	char line[256];
	int capacity = 256;
	numAngles = 0;
	angles = new Angle[capacity];

	// If the angle file cannot be found, exit the program
	if (inp == NULL) {
		printf("WARNING: Could not open `%s'.\n", angleFile.val());
		printf("This simulation will not use angles.\n");
		return;
	}

	while(fgets(line, 256, inp)) {
		if (line[0] == '#') continue;
		String s(line);
		int numTokens = s.tokenCount();
		String* tokenList = new String[numTokens];
		s.tokenize(tokenList);
		
		// Legitimate ANGLE inputs have 5 tokens
		// ANGLE | INDEX1 | INDEX2 | INDEX3 | FILENAME
		// Any angle input line without exactly 5 tokens should be discarded
		if (numTokens != 5) {
			printf("WARNING: Invalid angle input line: %s\n", line);
			continue;
		}		
		
		// Discard any empty line
		if (tokenList == NULL) 
			continue;
		
		int ind1 = atoi(tokenList[1].val());
		int ind2 = atoi(tokenList[2].val());
		int ind3 = atoi(tokenList[3].val());
		String file_name = tokenList[4];
		//printf("file_name %s\n", file_name.val());
		if (ind1 >= num or ind2 >= num or ind3 >= num)
			continue;

		if (numAngles >= capacity) {
			Angle* temp = angles;
			capacity *= 2;
			angles = new Angle[capacity];
			for (int i = 0; i < numAngles; i++)
				angles[i] = temp[i];
			delete[] temp;
		}

		Angle a(ind1, ind2, ind3, file_name);
		angles[numAngles++] = a;
		delete[] tokenList;
	}
	std::sort(angles, angles + numAngles, compare());	

	// for(int i = 0; i < numAngles; i++)
	// 	angles[i].print();
}

void Configuration::readDihedrals() {
	FILE* inp = fopen(dihedralFile.val(), "r");
	char line[256];
	int capacity = 256;
	numDihedrals = 0;
	dihedrals = new Dihedral[capacity];

	// If the dihedral file cannot be found, exit the program
	if (inp == NULL) {
		printf("WARNING: Could not open `%s'.\n", dihedralFile.val());
		printf("This simulation will not use dihedrals.\n");
		return;
	}

	while(fgets(line, 256, inp)) {
		if (line[0] == '#') continue;
		String s(line);
		int numTokens = s.tokenCount();
		String* tokenList = new String[numTokens];
		s.tokenize(tokenList);
		
		// Legitimate DIHEDRAL inputs have 6 tokens
		// DIHEDRAL | INDEX1 | INDEX2 | INDEX3 | INDEX4 | FILENAME
		// Any angle input line without exactly 6 tokens should be discarded
		if (numTokens != 6) {
			printf("WARNING: Invalid dihedral input line: %s\n", line);
			continue;
		}		
		
		// Discard any empty line
		if (tokenList == NULL) 
			continue;
		
		int ind1 = atoi(tokenList[1].val());
		int ind2 = atoi(tokenList[2].val());
		int ind3 = atoi(tokenList[3].val());
		int ind4 = atoi(tokenList[4].val());
		String file_name = tokenList[5];
		//printf("file_name %s\n", file_name.val());
		if (ind1 >= num or ind2 >= num
				or ind3 >= num or ind4 >= num)
			continue;

		if (numDihedrals >= capacity) {
			Dihedral* temp = dihedrals;
			capacity *= 2;
			dihedrals = new Dihedral[capacity];
			for (int i = 0; i < numDihedrals; ++i)
				dihedrals[i] = temp[i];
			delete[] temp;
		}

		Dihedral d(ind1, ind2, ind3, ind4, file_name);
		dihedrals[numDihedrals++] = d;
		delete[] tokenList;
	}
	std::sort(dihedrals, dihedrals + numDihedrals, compare());	

	// for(int i = 0; i < numDihedrals; i++)
	// 	dihedrals[i].print();
}

void Configuration::populate() {
	int pn = 0;
	int p = 0;

	for (int i = 0; i < num; i++) {
		for (int s = 0; s < simNum; ++s)
			type[i + s*num] = p;
		serial[i] = currSerial++;
		if (++pn >= part[p].num) {
			p++;
			pn = 0;
		}
	}
}

bool Configuration::readBondFile(const String& value, int currBond) {
	int numTokens = value.tokenCount();
	if (numTokens != 1) {
		printf("ERROR: Invalid tabulatedBondFile: %s, numTokens = %d\n", value.val(), numTokens);
		return false;
	}

	String* tokenList = new String[numTokens];
	value.tokenize(tokenList);
	if (tokenList == NULL) {
		printf("ERROR: Invalid tabulatedBondFile: %s; tokenList is NULL\n", value.val());
		return false;
	}

	bondTableFile[currBond] = tokenList[0];

	printf("Tabulated Bond Potential: %s\n", bondTableFile[currBond].val() );

	return true;
}

bool Configuration::readAngleFile(const String& value, int currAngle) {
	int numTokens = value.tokenCount();
	if (numTokens != 1) {
		printf("ERROR: Invalid tabulatedAngleFile: %s, numTokens = %d\n", value.val(), numTokens);
		return false;
	}

	String* tokenList = new String[numTokens];
	value.tokenize(tokenList);
	if (tokenList == NULL) {
		printf("ERROR: Invalid tabulatedAngleFile: %s; tokenList is NULL\n", value.val());
		return false;
	}

	angleTableFile[currAngle] = tokenList[0];

	printf("Tabulated Angle Potential: %s\n", angleTableFile[currAngle].val() );

	return true;
}

bool Configuration::readDihedralFile(const String& value, int currDihedral) {
	int numTokens = value.tokenCount();
	if (numTokens != 1) {
		printf("ERROR: Invalid tabulatedDihedralFile: %s, numTokens = %d\n", value.val(), numTokens);
		return false;
	}

	String* tokenList = new String[numTokens];
	value.tokenize(tokenList);
	if (tokenList == NULL) {
		printf("ERROR: Invalid tabulatedDihedralFile: %s; tokenList is NULL\n", value.val());
		return false;
	}

	dihedralTableFile[currDihedral] = tokenList[0];

	printf("Tabulated Dihedral Potential: %s\n", dihedralTableFile[currDihedral].val() );

	return true;
}

void Configuration::loadRestart(const char* file_name) {
	char line[STRLEN];
	FILE* inp = fopen(file_name, "r");

	if (inp == NULL) {
		printf("GrandBrownTown:loadRestart File `%s' does not exist\n", file_name);
		exit(-1);
	}

	int count = 0;
	while (fgets(line, STRLEN, inp) != NULL) {
		// Ignore comments.
		int len = strlen(line);
		if (line[0] == '#') continue;
		if (len < 2) continue;

		String s(line);
		int numTokens = s.tokenCount();
		if (numTokens != 4) {
			printf("GrandBrownTown:loadRestart Invalid coordinate file line: %s\n", line);
			fclose(inp);	
			exit(-1);
		}

		String* tokenList = new String[numTokens];
		s.tokenize(tokenList);
		if (tokenList == NULL) {
			printf("GrandBrownTown:loadRestart Invalid coordinate file line: %s\n", line);
			fclose(inp);
			exit(-1);
		}

		int typ = atoi(tokenList[0]);
		float x = (float) strtod(tokenList[1],NULL);
		float y = (float) strtod(tokenList[2],NULL);
		float z = (float) strtod(tokenList[3],NULL);

		pos[count] = Vector3(x,y,z);
		type[count] = typ;
		serial[count] = currSerial;
		currSerial++;
		if (typ < 0 || typ >= numParts) {
			printf("GrandBrownTown:countRestart Invalid particle type: %d\n", typ);
			fclose(inp);
			exit(-1);
		}

		count++;
		delete[] tokenList;
	}

	fclose(inp);    
}

bool Configuration::loadCoordinates(const char* file_name) {
	char line[STRLEN];
	FILE* inp = fopen(file_name, "r");

	if (inp == NULL) return false;

	int count = 0;
	while (fgets(line, STRLEN, inp) != NULL) {
		// Ignore comments.
		int len = strlen(line);
		if (line[0] == '#') continue;
		if (len < 2) continue;

		String s(line);
		int numTokens = s.tokenCount();
		if (numTokens != 3) {
			printf("ERROR: Invalid coordinate file line: %s\n", line);
			fclose(inp);	
			return false;
		}

		String* tokenList = new String[numTokens];
		s.tokenize(tokenList);
		if (tokenList == NULL) {
			printf("ERROR: Invalid coordinate file line: %s\n", line);
			fclose(inp);
			return false;
		}

		if (count >= num) {
			printf("WARNING: Too many coordinates in coordinate file %s.\n", file_name);
			fclose(inp);
			return true;
		}

		float x = (float) strtod(tokenList[0],NULL);
		float y = (float) strtod(tokenList[1],NULL);
		float z = (float) strtod(tokenList[2],NULL);
		pos[count] = Vector3(x,y,z);
		count++;

		delete[] tokenList;
	}
	fclose(inp);

	if (count < num) {
		printf("ERROR: Too few coordinates in coordinate file.\n");
		return false;
	}
	return true;
}

// Count the number of atoms in the restart file.
int Configuration::countRestart(const char* file_name) {
	char line[STRLEN];
	FILE* inp = fopen(file_name, "r");

	if (inp == NULL) {
		printf("ERROR: countRestart File `%s' does not exist\n", file_name);
		exit(-1);
	}

	int count = 0;
	while (fgets(line, STRLEN, inp) != NULL) {
		int len = strlen(line);
		// Ignore comments.
		if (line[0] == '#') continue;
		if (len < 2) continue;

		String s(line);
		int numTokens = s.tokenCount();
		if (numTokens != 4) {
			printf("ERROR: countRestart Invalid coordinate file line: %s\n", line);
			fclose(inp);	
			exit(-1);
		}

		String* tokenList = new String[numTokens];
		s.tokenize(tokenList);
		if (tokenList == NULL) {
			printf("ERROR: countRestart Invalid coordinate file line: %s\n", line);
			fclose(inp);
			exit(-1);
		}

		int typ = atoi(tokenList[0]);
		// float x = strtod(tokenList[1],NULL);
		// float y = strtod(tokenList[2],NULL);
		// float z = strtod(tokenList[3],NULL);
		if (typ < 0 || typ >= numParts) {
			printf("ERROR: countRestart Invalid particle type: %d\n", typ);
			fclose(inp);
			exit(-1);
		}

		count++;
		delete[] tokenList;
	}

	fclose(inp);    
	return count;
}

bool Configuration::readTableFile(const String& value, int currTab) {
	int numTokens = value.tokenCount('@');
	if (numTokens != 3) {
		printf("ERROR: Invalid tabulatedFile: %s\n", value.val());
		return false;
	}

	String* tokenList = new String[numTokens];
	value.tokenize(tokenList, '@');
	if (tokenList == NULL) {
		printf("ERROR: Invalid tabulatedFile: %s\n", value.val());
		return false;
	}

	partTableIndex0[currTab] = atoi(tokenList[0]);
	partTableIndex1[currTab] = atoi(tokenList[1]);
	partTableFile[currTab] = tokenList[2];

	printf("tabulatedPotential: %d %d %s\n", partTableIndex0[currTab],
			partTableIndex1[currTab], partTableFile[currTab].val() );
	delete[] tokenList;
	return true;
}

void Configuration::getDebugForce() {
	// Allow the user to choose which force computation to use
	printf("\n");
	printf("(1) ComputeFull [Default]          (2) ComputeSoftcoreFull\n");
	printf("(3) ComputeElecFull                (4) Compute (Decomposed)\n");
	printf("(5) ComputeTabulated (Decomposed)  (6) ComputeTabulatedFull\n");

	printf("WARNING: ");
	if (tabulatedPotential) {
		if (fullLongRange) printf("(6) was specified by config file\n");
		else printf("(5) was specified by config file\n");
	} else {
		if (fullLongRange != 0) printf("(%d) was specified by config file\n", fullLongRange);
		else printf("(4) was specified by config file\n");
	}

	char buffer[256];
	int choice;
	while (true) {
		printf("Choose a force computation (1 - 6): ");
		fgets(buffer, 256, stdin);
		bool good = sscanf(buffer, "%d", &choice) && (choice >= 1 && choice <= 6);
		if (good)
			break;
	}
	switch(choice) {
		case 1:
			tabulatedPotential = 0;
			fullLongRange = 1;
			break;
		case 2:
			tabulatedPotential = 0;
			fullLongRange = 2;
			break;
		case 3:
			tabulatedPotential = 0;
			fullLongRange = 3;
			break;
		case 4:
			tabulatedPotential = 0;
			fullLongRange = 0;
			break;
		case 5:
			tabulatedPotential = 1;
			fullLongRange = 0;
			break;
		case 6:
			tabulatedPotential = 1;
			fullLongRange = 1;
			break;
		default:
			tabulatedPotential = 0;
			fullLongRange = 1;
			break;
	}
	printf("\n");
}

//////////////////////////
// Comparison operators //
//////////////////////////
bool Configuration::compare::operator()(const String& lhs, const String& rhs) {
	String* list_lhs = new String[lhs.tokenCount()];
	String* list_rhs = new String[rhs.tokenCount()];
	lhs.tokenize(list_lhs);
	rhs.tokenize(list_rhs);
	int key_lhs = atoi(list_lhs[1].val());
	int key_rhs = atoi(list_rhs[1].val());
	delete[] list_lhs;
	delete[] list_rhs;
	return key_lhs < key_rhs;
}

bool Configuration::compare::operator()(const Bond& lhs, const Bond& rhs) {
	int diff = lhs.ind1 - rhs.ind1;
	if (diff != 0)
		return lhs.ind1 < rhs.ind1;
	return lhs.ind2 < rhs.ind2;
}

bool Configuration::compare::operator()(const Exclude& lhs, const Exclude& rhs) {
	int diff = lhs.ind1 - rhs.ind1;
	if (diff != 0)
		return lhs.ind1 < rhs.ind1;
	return lhs.ind2 < rhs.ind2;
}

bool Configuration::compare::operator()(const Angle& lhs, const Angle& rhs) {
	int diff = lhs.ind1 - rhs.ind1;
	if (diff != 0)
		return lhs.ind1 < rhs.ind1;
	diff = lhs.ind2 - rhs.ind2;
	if (diff != 0)
		return lhs.ind2 < rhs.ind2;
	return lhs.ind3 < rhs.ind3;
}

bool Configuration::compare::operator()(const Dihedral& lhs, const Dihedral& rhs) {
	int diff = lhs.ind1 - rhs.ind1;
	if (diff != 0) 
		return lhs.ind1 < rhs.ind1;
	diff = lhs.ind2 - rhs.ind2;
	if (diff != 0) 
		return lhs.ind2 < rhs.ind2;
	diff = lhs.ind3 - rhs.ind3;
	if (diff != 0) 
		return lhs.ind3 < rhs.ind3;
	return lhs.ind4 < rhs.ind4;
}
