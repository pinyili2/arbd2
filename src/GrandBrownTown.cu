#include "GrandBrownTown.h"
#include "GrandBrownTown.cuh"
/* #include "ComputeGridGrid.cuh" */
#include "WKFUtils.h"
#include "BrownParticlesKernel.h"
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <thrust/device_ptr.h>
#include <fstream>

#ifdef _OPENMP
#include <omp.h>
//#else
//typedef int omp_int_t;
//inline omp_int_t omp_get_thread_num() { return 0; }
//inline omp_int_t omp_get_max_threads() { return 1; }
#endif

#ifndef gpuErrchk
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
	if (code != cudaSuccess) {
		fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
	if (abort) exit(code);
	}
}
#endif

bool GrandBrownTown::DEBUG;

cudaEvent_t START, STOP;

GrandBrownTown::GrandBrownTown(const Configuration& c, const char* outArg,
		bool debug, bool imd_on, unsigned int imd_port, int numReplicas) :
	imd_on(imd_on), imd_port(imd_port), numReplicas(numReplicas),
	//conf(c), RBC(RigidBodyController(c,outArg)) {
	conf(c){

        RBC.resize(numReplicas);      
        for(int i = 0; i < numReplicas; ++i)
        {
            RigidBodyController* rb = new RigidBodyController(c, outArg, seed, i);
            RBC[i] = rb;
        }

        //printf("%d\n",__LINE__);
        //Determine which dynamic. Han-Yi Chou
        particle_dynamic  = c.ParticleDynamicType;
        rigidbody_dynamic = c.RigidBodyDynamicType;
        ParticleInterpolationType = c.ParticleInterpolationType;
        RigidBodyInterpolationType = c.RigidBodyInterpolationType;
        //particle_langevin_integrator = c.ParticleLangevinIntegrator;
        printf("%d\n",__LINE__);
	for (int i = 0; i < numReplicas; ++i) 
        {
		std::stringstream curr_file, restart_file, out_prefix;

		if (numReplicas > 1) {
		    curr_file << outArg << '.' << i << ".curr";
		    restart_file   << outArg << '.' << i << ".restart";
		    out_prefix << outArg << '.' << i;
		} else {
		    curr_file << outArg << ".curr";
		    restart_file   << outArg << ".restart";
		    out_prefix << outArg;
		}

                outCurrFiles.push_back(curr_file.str());
                restartFiles.push_back(restart_file.str());
                outFilePrefixes.push_back(out_prefix.str());

                //Han-Yi Chou for flush out the momentum
                if(particle_dynamic == String("Langevin") || particle_dynamic == String("NoseHooverLangevin"))
                {
                    std::stringstream restart_file_p, out_momentum_prefix, out_force_prefix;
                    restart_file_p << outArg << '.' << i << ".momentum.restart";
                    out_momentum_prefix << outArg << '.' << i << ".momentum";
                    //out_force_prefix << outArg << ".force." << i;

                    restartMomentumFiles.push_back(restart_file_p.str());//Han-Yi Chou
                    outMomentumFilePrefixes.push_back(out_momentum_prefix.str());
                    //outForceFilePrefixes.push_back(out_force_prefix.str());
                }           
	}

	GrandBrownTown::DEBUG = debug;
	sysDim = c.sysDim;
	sys = c.sys;

	// Particle variables
	partsFromFile = c.partsFromFile;
	indices = c.indices;
	numPartsFromFile = c.numPartsFromFile;  // number of particle types
	bonds = c.bonds;
	numCap = c.numCap;                      // max number of particles
	num = c.num;                            // current number of particles

	// Allocate arrays of positions, types and serial numbers
	pos    = new Vector3[num * numReplicas];  // [HOST] array of particles' positions.
        // Allocate arrays of momentum Han-Yi Chou
        if(particle_dynamic == String("Langevin") || particle_dynamic == String("NoseHooverLangevin"))
        {
            momentum = new Vector3[num * numReplicas]; //[HOST] array of particles' momentum
            if(particle_dynamic == String("NoseHooverLangevin"))
                random = new float[num * numReplicas];
        }
        //printf("%d\n",__LINE__);
        //for debug
        //force = new Vector3[num * numReplicas];

	type   = new     int[num * numReplicas];  // [HOST] array of particles' types.
	serial = new     int[num * numReplicas];  // [HOST] array of particles' serial numbers.

	// Allocate things for rigid body
	// RBC = RigidBodyController(c);
  // printf("About to devicePrint\n");
	// devicePrint<<<1,1>>>(&(c.rigidBody[0]));
	// devicePrint<<<1,1>>>(RBC.rbType_d);
	cudaDeviceSynchronize();
	// printf("Done with devicePrint\n");


	
	// Replicate identical initial conditions across all replicas
	for (int r = 0; r < numReplicas; ++r) {
	  std::copy(c.type, c.type + num, type + r*num);
	  std::copy(c.serial, c.serial + num, serial + r*num);
	  if (c.copyReplicaCoordinates > 0)
	    std::copy(c.pos, c.pos + num, pos + r*num);
	}
        if (c.copyReplicaCoordinates <= 0)
          std::copy(c.pos, c.pos + numReplicas*num, pos);

        //printf("%d\n",__LINE__); 
        //Han-Yi Chou
        if(particle_dynamic == String("Langevin") || particle_dynamic == String("NoseHooverLangevin"))
            std::copy(c.momentum,c.momentum+num*numReplicas,momentum);

        //printf("%d\n",__LINE__);

	currSerial = c.currSerial;  // serial number of the next new particle
	name = c.name;              // list of particle types! useful when 'numFluct == 1'
	posLast = c.posLast;        // previous positions of particles  (used for computing ionic current)
        momLast = c.momLast;
	timeLast = c.timeLast;      // previous time (used with posLast)
	minimumSep = c.minimumSep;  // minimum separation allowed when placing new particles

	// System parameters
	outputName = c.outputName;
	timestep = c.timestep;
	steps = c.steps;
	seed = c.seed;
	temperatureGridFile = c.temperatureGridFile;
	inputCoordinates = c.inputCoordinates;
	restartCoordinates = c.restartCoordinates;
	numberFluct = c.numberFluct;
	interparticleForce = c.interparticleForce;
	tabulatedPotential = c.tabulatedPotential;
	readBondsFromFile = c.readBondsFromFile;
	fullLongRange = c.fullLongRange;
	kT = c.kT;
	temperature = c.temperature;
	coulombConst = c.coulombConst;
	electricField = c.electricField;
	cutoff = c.cutoff;
	switchLen = c.switchLen;
	outputPeriod = c.outputPeriod;
	outputEnergyPeriod = c.outputEnergyPeriod;
	outputFormat = c.outputFormat;
	currentSegmentZ = c.currentSegmentZ;
	numberFluctPeriod = c.numberFluctPeriod;
	decompPeriod = c.decompPeriod;
	numCapFactor = c.numCapFactor;
	kTGrid = c.kTGrid;
	tGrid = c.tGrid;
	sigmaT = c.sigmaT;

	// Parameter files
	partTableFile = c.partTableFile;
	bondTableFile = c.bondTableFile;
	angleTableFile = c.angleTableFile;
	dihedralTableFile = c.dihedralTableFile;

	// Other parameters.
	switchStart = c.switchStart;
	maxInitialPot = c.maxInitialPot;
	initialZ = c.initialZ;

	// Particle parameters.
	part = c.part;
	numParts = c.numParts;
	numBonds = c.numBonds;
	numExcludes = c.numExcludes;
	numAngles = c.numAngles;
	numDihedrals = c.numDihedrals;
	partTableIndex0 = c.partTableIndex0;
	partTableIndex1 = c.partTableIndex1;

	numBondAngles = c.numBondAngles;

	numTabBondFiles = c.numTabBondFiles;
	bondMap = c.bondMap;
	// TODO: bondList = c.bondList;

	excludes = c.excludes;
        part = c.part;
	excludeMap = c.excludeMap;
	excludeRule = c.excludeRule;
	excludeCapacity = c.excludeCapacity;

	angles = c.angles;
	numTabAngleFiles = c.numTabAngleFiles;

	dihedrals = c.dihedrals;
	numTabDihedralFiles = c.numTabDihedralFiles;

	bondAngles = c.bondAngles;

	// Device parameters
	//type_d = c.type_d;
	part_d = c.part_d;
	sys_d = c.sys_d;
	kTGrid_d = c.kTGrid_d;
	//bonds_d = c.bonds_d;
	//bondMap_d = c.bondMap_d;
	//excludes_d = c.excludes_d;
	//excludeMap_d = c.excludeMap_d;
	//angles_d = c.angles_d;
	//dihedrals_d = c.dihedrals_d;

	printf("Setting up random number generator with seed %lu\n", seed);
	randoGen = new Random(num * numReplicas, seed);
	copyRandToCUDA();

        if(particle_dynamic == String("NoseHooverLangevin"))
            InitNoseHooverBath(num * numReplicas);

	// "Some geometric stuff that should be gotten rid of." -- Jeff Comer
	Vector3 buffer = (sys->getCenter() + 2.0f * sys->getOrigin())/3.0f;
	initialZ = buffer.z;

	// Load random coordinates if necessary Han-Yi Chou
	if (!c.loadedCoordinates) {
		//printf("Populating\n");
		//populate(); Han-Yi Chou, Actually the list is already populated 
	    initialCond();
		printf("Setting random initial conditions.\n");
	}

	// Prepare internal force computation
	 //internal = new ComputeForce(num, part, numParts, sys, switchStart, switchLen, coulombConst,
	 //			    fullLongRange, numBonds, numTabBondFiles, numExcludes, numAngles, numTabAngleFiles,
	 //			    numDihedrals, numTabDihedralFiles, c.pairlistDistance, numReplicas);
	internal = new ComputeForce(c, numReplicas);
	
	//MLog: I did the other halve of the copyToCUDA function from the Configuration class here, keep an eye on any mistakes that may occur due to the location.
	internal -> copyToCUDA(c.simNum, c.type, c.bonds, c.bondMap, c.excludes, c.excludeMap, c.angles, c.dihedrals, c.restraints, c.bondAngles );

	// TODO: check for duplicate potentials 
	if (c.tabulatedPotential) {
		printf("Loading %d tabulated non-bonded potentials...\n", numParts*numParts);
		for (int p = 0; p < numParts*numParts; p++) {
			if (partTableFile[p].length() > 0) {
				int type0 = partTableIndex0[p];
				int type1 = partTableIndex1[p];

				internal->addTabulatedPotential(partTableFile[p].val(), type0, type1);
				// printf("  Loaded %s for types %s and %s.\n", partTableFile[p].val(),
				// 		part[type0].name.val(), part[type1].name.val());
			}
		}
	}
	printf("Using %d non-bonded exclusions\n",c.numExcludes/2);

	if (c.readBondsFromFile) {
		printf("Loading %d tabulated bond potentials...\n", numTabBondFiles);
		for (int p = 0; p < numTabBondFiles; p++)
			if (bondTableFile[p].length() > 0) {
				//MLog: make sure to add to all GPUs
			    // printf("...loading %s\n",bondTableFile[p].val());
			    internal->addBondPotential(bondTableFile[p].val(), p, bonds, bondAngles);
				// printf("%s\n",bondTableFile[p].val());
			} else {
			    printf("...skipping %s (\n",bondTableFile[p].val());
			    internal->addBondPotential(bondTableFile[p].val(), p, bonds, bondAngles);
			}
			    
	}

	if (c.readAnglesFromFile) {
		printf("Loading %d tabulated angle potentials...\n", numTabAngleFiles);
		for (int p = 0; p < numTabAngleFiles; p++)
			if (angleTableFile[p].length() > 0)
			{
				//MLog: make sure to do this for every GPU
			    internal->addAnglePotential(angleTableFile[p].val(), p, angles, bondAngles);
			}
	}

	if (c.readDihedralsFromFile) {
		printf("Loading %d tabulated dihedral potentials...\n", numTabDihedralFiles);
		for (int p = 0; p < numTabDihedralFiles; p++)
			if (dihedralTableFile[p].length() > 0)
				internal->addDihedralPotential(dihedralTableFile[p].val(), p, dihedrals);
	}

	//Mlog: this is where we create the bondList.
	if (numBonds > 0) {
		bondList = new int3[ (numBonds / 2) * numReplicas ];
		int j = 0;

		for(int k = 0 ; k < numReplicas; k++)
		{
			for(int i = 0; i < numBonds; ++i)
			{
				if(bonds[i].ind1 < bonds[i].ind2)
				{
					if (bonds[i].tabFileIndex == -1) {
						fprintf(stderr,"Error: bondfile '%s' was not read with tabulatedBondFile command.\n", bonds[i].fileName.val());
						exit(1);
					}
						
					bondList[j] = make_int3( (bonds[i].ind1 + k * num), (bonds[i].ind2 + k * num), bonds[i].tabFileIndex );
					// cout << "Displaying: bondList["<< j <<"].x = " << bondList[j].x << ".\n"
					// << "Displaying: bondList["<< j <<"].y = " << bondList[j].y << ".\n"
					// << "Displaying: bondList["<< j <<"].z = " << bondList[j].z << ".\n";
					++j;
				}
			}
		}
	}
	// internal->createBondList(bondList);

	if (numAngles > 0) {
	angleList = new int4[ (numAngles) * numReplicas ];
	for(int k = 0 ; k < numReplicas; k++) {
	    for(int i = 0; i < numAngles; ++i) {
			if (angles[i].tabFileIndex == -1) {
				fprintf(stderr,"Error: anglefile '%s' was not read with tabulatedAngleFile command.\n", angles[i].fileName.val());
				exit(1);
			}
		angleList[i+k*numAngles] = make_int4( angles[i].ind1+k*num, angles[i].ind2+k*num, angles[i].ind3+k*num, angles[i].tabFileIndex );
	    }
	}
	}
	
	if (numDihedrals > 0) {
	dihedralList = new int4[ (numDihedrals) * numReplicas ];
	dihedralPotList = new  int[ (numDihedrals) * numReplicas ];
	for(int k = 0 ; k < numReplicas; k++) {
	    for(int i = 0; i < numDihedrals; ++i) {
			if (dihedrals[i].tabFileIndex == -1) {
				fprintf(stderr,"Error: dihedralfile '%s' was not read with tabulatedDihedralFile command.\n", dihedrals[i].fileName.val());
				exit(1);
			}
		dihedralList[i+k*numDihedrals] = make_int4( dihedrals[i].ind1+k*num, dihedrals[i].ind2+k*num, dihedrals[i].ind3+k*num, dihedrals[i].ind4+k*num);
		dihedralPotList[i+k*numDihedrals] = dihedrals[i].tabFileIndex;
	    }
	}
	}

	if (numBondAngles > 0) {
	bondAngleList = new int2[ (numBondAngles*3) * numReplicas ];
	for(int k = 0 ; k < numReplicas; k++) {
	    for(int i = 0; i < numBondAngles; ++i) {
			if (bondAngles[i].tabFileIndex1 == -1) {
				fprintf(stderr,"Error: bondanglefile '%s' was not read with tabulatedAngleFile command.\n", bondAngles[i].angleFileName.val());
				exit(1);
			}
			if (bondAngles[i].tabFileIndex2 == -1) {
				fprintf(stderr,"Error: bondanglefile1 '%s' was not read with tabulatedBondFile command.\n", bondAngles[i].bondFileName1.val());
				exit(1);
			}
			if (bondAngles[i].tabFileIndex3 == -1) {
				fprintf(stderr,"Error: bondanglefile2 '%s' was not read with tabulatedBondFile command.\n", bondAngles[i].bondFileName2.val());
				exit(1);
			}
			int idx = i+k*numBondAngles;
			bondAngleList[idx*3]   = make_int2( bondAngles[i].ind1+k*num, bondAngles[i].ind2+k*num );
			bondAngleList[idx*3+1] = make_int2( bondAngles[i].ind3+k*num, bondAngles[i].tabFileIndex1 );
			bondAngleList[idx*3+2] = make_int2( bondAngles[i].tabFileIndex2, bondAngles[i].tabFileIndex3 );
	    }
	}
	}

	internal->copyBondedListsToGPU(bondList,angleList,dihedralList,dihedralPotList,bondAngleList);
	
	forceInternal = new Vector3[num * numReplicas];
	if (fullLongRange != 0)
	    printf("No cell decomposition created.\n");

	// Prepare the trajectory output writer.
	for (int repID = 0; repID < numReplicas; ++repID) {
		TrajectoryWriter *w = new TrajectoryWriter(outFilePrefixes[repID].c_str(), TrajectoryWriter::getFormatName(outputFormat),
							   sys->getBox(), num, timestep, outputPeriod);
                
		writers.push_back(w);
	}

        //Preparing the writers for momentum if necessary Han-Yi Chou
        if(particle_dynamic == String("Langevin") || particle_dynamic == String("NoseHooverLangevin"))
        {
            for (int repID = 0; repID < numReplicas; ++repID) 
            {

                TrajectoryWriter *w = new TrajectoryWriter(outMomentumFilePrefixes[repID].c_str(), TrajectoryWriter::getFormatName(outputFormat),
                                                           sys->getBox(), num, timestep, outputPeriod);
                momentum_writers.push_back(w);
            }
        }
	updateNameList();
}

GrandBrownTown::~GrandBrownTown() {
	delete[] forceInternal;
        forceInternal = NULL;
	delete[] pos;
        pos = NULL;
        //Han-Yi Chou
        if(particle_dynamic == String("Langevin") || particle_dynamic == String("NoseHooverLangevin"))
        {
            delete[] momentum;
            momentum = NULL;
            if(particle_dynamic == String("NoseHooverLangevin"))
            {
                delete[] random;
                random = NULL;
            }
        }
        //for debug
        //delete[] force;

	delete[] type;
	delete[] serial;
	//delete randoGen;

	if (numBonds > 0)
		delete[] bondList;
	if (numAngles > 0)
		delete[] angleList;
	if (numDihedrals > 0) {
		delete[] dihedralList;
		delete[] dihedralPotList;
	}
        if(randoGen->states != NULL)
            {
                gpuErrchk(cudaFree(randoGen->states));
                randoGen->states = NULL;
            }
            if(randoGen->integer_h != NULL)
            {
                delete[] randoGen->integer_h;
                randoGen->integer_h = NULL;
            }
            if(randoGen->integer_d != NULL)
            {
                gpuErrchk(cudaFree(randoGen->integer_d));
                randoGen->integer_d = NULL;
            }
            if(randoGen->uniform_h != NULL)
            {
                delete[] randoGen->uniform_h;
                randoGen->uniform_h = NULL;
            }
            if(randoGen->uniform_d != NULL)
            {
                gpuErrchk(cudaFree(randoGen->uniform_d));
                randoGen->uniform_d = NULL;
            }
            //curandDestroyGenerator(randoGen->generator);
            delete randoGen;
            gpuErrchk(cudaFree(randoGen_d));
            for(std::vector<RigidBodyController*>::iterator iter = RBC.begin(); iter != RBC.end(); ++iter)
            {
                //(*iter)->~RigidBodyController();
                delete *iter;
            }
            RBC.clear();
	// Auxillary objects
	delete internal;
        internal = NULL;
	for (int i = 0; i < numReplicas; ++i)
        {
		delete writers[i];
                writers[i] = NULL;
        }

        if(particle_dynamic == String("Langevin") || particle_dynamic == String("NoseHooverLangevin"))
        {
            for (int i = 0; i < numReplicas; ++i)
            {
                delete momentum_writers[i];
                momentum_writers[i]=NULL;
             }
            //for (int i = 0; i < numReplicas; ++i)
                //delete force_writers[i];

        }
	//gpuErrchk(cudaFree(pos_d));
	//gpuErrchk(cudaFree(forceInternal_d));
	//gpuErrchk(cudaFree(randoGen_d));
	//gpuErrchk( cudaFree(bondList_d) );

	if (imd_on)
		delete[] imdForces;
	
		
}
//temporary test for Nose-Hoover Langevin dynamics
//Nose Hoover is now implement for particles.
void GrandBrownTown::RunNoseHooverLangevin()
{
    //comment this out because this is the origin points Han-Yi Chou
    // Open the files for recording ionic currents
    for (int repID = 0; repID < numReplicas; ++repID) 
    {

        writers[repID]->newFile(pos + (repID * num), name, 0.0f, num); // 'pos + (repID*num)' == array-to-pointer decay
        if(particle_dynamic == String("Langevin") || particle_dynamic == String("NoseHooverLangevin"))
            momentum_writers[repID]->newFile((momentum + repID * num), name, 0.0f, num); // 'pos + (repID*num)' == array-to-pointer decay
        //random_writers[repID]->newFile(random + (repID * num), name, 0.0f, num);
    }
    // Initialize timers (util.*)
    wkf_timerhandle timer0, timerS;
    timer0 = wkf_timer_create();
    timerS = wkf_timer_create();

    copyToCUDA();

    if(particle_dynamic == String("Langevin"))
        internal -> copyToCUDA(forceInternal, pos,momentum);
    else if(particle_dynamic == String("NoseHooverLangevin"))
        internal -> copyToCUDA(forceInternal, pos, momentum, random);
    else
        internal -> copyToCUDA(forceInternal, pos);

    // IMD Stuff
    void* sock = NULL;
    void* clientsock = NULL;
    int length;
    if (imd_on) 
    {
        printf("Setting up incoming socket\n");
        vmdsock_init();
        sock = vmdsock_create();
        clientsock = NULL;
        vmdsock_bind(sock, imd_port);

        printf("Waiting for IMD connection on port %d...\n", imd_port);
        vmdsock_listen(sock);
        while (!clientsock) 
        {
            if (vmdsock_selread(sock, 0) > 0) 
            {
                clientsock = vmdsock_accept(sock);
                if (imd_handshake(clientsock))
                    clientsock = NULL;
            }
        }
        sleep(1);
        if (vmdsock_selread(clientsock, 0) != 1 || imd_recv_header(clientsock, &length) != IMD_GO) 
        {
            clientsock = NULL;
        }
        imdForces = new Vector3[num*numReplicas];
        for (size_t i = 0; i < num; ++i) // clear old forces
            imdForces[i] = Vector3(0.0f);

    } // endif (imd_on)

    // Start timers
    wkf_timer_start(timer0);
    wkf_timer_start(timerS);

    if (fullLongRange == 0)
    {
        // cudaSetDevice(0);
        internal->decompose();
        gpuErrchk(cudaDeviceSynchronize());
        #ifdef _OPENMP
        omp_set_num_threads(4);
        #endif
        #pragma omp parallel for
        for(int i = 0; i < numReplicas; ++i)
            RBC[i]->updateParticleLists( (internal->getPos_d())+i*num, sys_d);
        gpuErrchk(cudaDeviceSynchronize());
    }

    float t; // simulation time

    int numBlocks = (num * numReplicas) / NUM_THREADS + ((num * numReplicas) % NUM_THREADS == 0 ? 0 : 1);
    int tl = temperatureGridFile.length();
    Vector3 *force_d;
    gpuErrchk(cudaMalloc((void**)&force_d, sizeof(Vector3)*num * numReplicas));

    printf("Configuration: %d particles | %d replicas\n", num, numReplicas);
    //float total_energy = 0.f;
    // Main loop over Brownian dynamics steps
    for (long int s = 1; s < steps; s++)
    {
        bool get_energy = ((s % outputEnergyPeriod) == 0);
        //At the very first time step, the force is computed
        if(s == 1)
        {
            // 'interparticleForce' - determines whether particles interact with each other
            gpuErrchk(cudaMemset((void*)(internal->getForceInternal_d()),0,num*numReplicas*sizeof(Vector3)));
            gpuErrchk(cudaMemset((void*)(internal->getEnergy()), 0, sizeof(float)*num*numReplicas));
            #ifdef _OPENMP
            omp_set_num_threads(4);
            #endif
            #pragma omp parallel for
            for(int i = 0; i < numReplicas; ++i)
                RBC[i]->clearForceAndTorque(); //Han-Yi Chou
            
            if (interparticleForce)
            {
                if (tabulatedPotential)
                {
                    switch (fullLongRange)
                    {
                        case 0: // [ N*log(N) ] interactions, + cutoff | decomposition
                            if (s % decompPeriod == 0)
                            {
                                // cudaSetDevice(0);
                                 internal -> decompose();
                                #ifdef _OPENMP
                                omp_set_num_threads(4);
                                #endif
                                #pragma omp parallel for
                                for(int i = 0; i < numReplicas; ++i)
                                    RBC[i]->updateParticleLists( (internal->getPos_d())+i*num, sys_d);
                            }
                            internal -> computeTabulated(get_energy);
                            break;
                        default:
                            internal->computeTabulatedFull(get_energy);
                            break;
                    }
                }
                else
                {
                    // Not using tabulated potentials.
                    switch (fullLongRange)
                    {
                        case 0: // Use cutoff | cell decomposition.
                            if (s % decompPeriod == 0)
                            {
                                // cudaSetDevice(0);
                                internal->decompose();
                                #ifdef _OPENMP
                                omp_set_num_threads(4);
                                #endif
                                #pragma omp parallel for
                                for(int i = 0; i < numReplicas; ++i)
                                    RBC[i]->updateParticleLists( (internal->getPos_d())+i*num, sys_d);
                            }
                            internal->compute(get_energy);
                            break;

                        case 1: // Do not use cutoff
                            internal->computeFull(get_energy);
                            break;

                        case 2: // Compute only softcore forces.
                            internal->computeSoftcoreFull(get_energy);
                            break;

                        case 3: // Compute only electrostatic forces.
                            internal->computeElecFull(get_energy);
                            break;
                    }
                }
            }//if inter-particle force
            #ifdef _OPENMP
            omp_set_num_threads(4);
            #endif
            #pragma omp parallel for
            for(int i = 0; i < numReplicas; ++i)
                RBC[i]->updateForces((internal->getPos_d())+i*num, (internal->getForceInternal_d())+i*num, s, (internal->getEnergy())+i*num, get_energy, 
                                       RigidBodyInterpolationType, sys, sys_d);
            if(rigidbody_dynamic == String("Langevin"))
            {
                #ifdef _OPENMP
                omp_set_num_threads(4);
                #endif
                #pragma omp parallel for
                for(int i = 0; i < numReplicas; ++i)
                {
                    RBC[i]->SetRandomTorques();
                    RBC[i]->AddLangevin();
                }
            }
        }//if step == 1

        gpuErrchk(cudaMemset((void*)(internal->getEnergy()), 0, sizeof(float)*num*numReplicas)); // TODO: make async
        if(particle_dynamic == String("Langevin"))
            updateKernelBAOAB<<< numBlocks, NUM_THREADS >>>(internal -> getPos_d(), internal -> getMom_d(), internal -> getForceInternal_d(), internal -> getType_d(), part_d, kT, kTGrid_d, electricField, tl, timestep, num, sys_d, randoGen_d, numReplicas, ParticleInterpolationType);
        else if(particle_dynamic == String("NoseHooverLangevin"))
            //kernel for Nose-Hoover Langevin dynamic
            updateKernelNoseHooverLangevin<<< numBlocks, NUM_THREADS >>>(internal -> getPos_d(), internal -> getMom_d(), 
            internal -> getRan_d(), internal -> getForceInternal_d(), internal -> getType_d(), part_d, kT, kTGrid_d, electricField, tl, timestep, num, sys_d, 
            randoGen_d, numReplicas, ParticleInterpolationType);
        ////For Brownian motion
        else
            updateKernel<<< numBlocks, NUM_THREADS >>>(internal -> getPos_d(), internal -> getForceInternal_d(), internal -> getType_d(),
                                                       part_d, kT, kTGrid_d, electricField, tl, timestep, num, sys_d, randoGen_d, numReplicas, 
                                                       internal->getEnergy(), get_energy, ParticleInterpolationType);

        if(rigidbody_dynamic == String("Langevin"))
        {
            #ifdef _OPENMP
            omp_set_num_threads(4);
            #endif
            #pragma omp parallel for
            for(int i = 0; i < numReplicas; ++i)
            {
                RBC[i]->integrateDLM(sys, 0);
                RBC[i]->integrateDLM(sys, 1);
            }
        }
        else
	{
            #ifdef _OPENMP
            omp_set_num_threads(4);
            #endif
            #pragma omp parallel for ordered
            for(int i = 0; i < numReplicas; ++i)
            {
                RBC[i]->integrate(sys, s);
                #pragma omp ordered
                RBC[i]->print(s);
            }
        }

        if (s % outputPeriod == 0) {
            // Copy particle positions back to CPU
	    gpuErrchk(cudaDeviceSynchronize());
            gpuErrchk(cudaMemcpy(pos, internal ->  getPos_d(), sizeof(Vector3) * num * numReplicas, cudaMemcpyDeviceToHost));
	}
        if (imd_on && clientsock && s % outputPeriod == 0)
        {
	    gpuErrchk(cudaDeviceSynchronize());
            float* coords = new float[num*3]; // TODO: move allocation out of run loop
            int* atomIds = new int[num]; // TODO: move allocation out of run loop
            int length;

            bool paused = false;
            while (vmdsock_selread(clientsock, 0) > 0 || paused)
            {
                switch (imd_recv_header(clientsock, &length))
                {
                        case IMD_DISCONNECT:
                            printf("[IMD] Disconnecting...\n");
                            imd_disconnect(clientsock);
                            clientsock = NULL;
                            sleep(5);
                            break;
                        case IMD_KILL:
                            printf("[IMD] Killing...\n");
                            imd_disconnect(clientsock);
                            clientsock = NULL;
                            steps = s; // Stop the simulation at this step
                            sleep(5);
                            break;
                        case IMD_PAUSE:
                            paused = !paused;
                            break;
                        case IMD_GO:
                            printf("[IMD] Caught IMD_GO\n");
                            break;
                        case IMD_MDCOMM:
                            for (size_t i = 0; i < num; ++i) // clear old forces
                                imdForces[i] = Vector3(0.0f);

                            if (imd_recv_mdcomm(clientsock, length, atomIds, coords))
                            {
                                printf("[IMD] Error receiving forces\n");
                            }
                            else
                            {
                                for (size_t j = 0; j < length; ++j)
                                {
                                    int i = atomIds[j];
                                    imdForces[i] = Vector3(coords[j*3], coords[j*3+1], coords[j*3+2]) * conf.imdForceScale;
                                }
                            }
                            break;
                        default:
                            printf("[IMD] Something weird happened. Disconnecting..\n");
                            break;
                }
            }
            if (clientsock)
            {
                    // float* coords = new float[num*3]; // TODO: move allocation out of run loop
                    for (size_t i = 0; i < num; i++)
                    {
                        const Vector3& p = pos[i];
                        coords[3*i] = p.x;
                        coords[3*i+1] = p.y;
                        coords[3*i+2] = p.z;
                    }
                    imd_send_fcoords(clientsock, num, coords);
            }
                delete[] coords;
                delete[] atomIds;
        }

        #ifdef _OPENMP
        omp_set_num_threads(4);
        #endif
        #pragma omp parallel for
        for(int i = 0; i < numReplicas; ++i) 
            RBC[i]->clearForceAndTorque();
        if (imd_on && clientsock)
            internal->setForceInternalOnDevice(imdForces); // TODO ensure replicas are mutually exclusive with IMD
	else {
            gpuErrchk(cudaMemsetAsync((void*)(internal->getForceInternal_d()),0,num*numReplicas*sizeof(Vector3)));
    	}

        if (interparticleForce)
        {
            // 'tabulatedPotential' - determines whether interaction is described with tabulated potentials or formulas
            if (tabulatedPotential)
            {
                switch (fullLongRange)
                {
                    case 0: // [ N*log(N) ] interactions, + cutoff | decomposition
                        if (s % decompPeriod == 0)
                        {
                            internal -> decompose();
                            #ifdef _OPENMP
                            omp_set_num_threads(4);
                            #endif
                            #pragma omp parallel for
                            for(int i = 0; i < numReplicas; ++i)
                                RBC[i]->updateParticleLists( (internal->getPos_d())+i*num, sys_d);
                        }
                        internal -> computeTabulated(get_energy);
                        break;
                    default: // [ N^2 ] interactions, no cutoff | decompositions
                        internal->computeTabulatedFull(get_energy);
                        break;
                }
            }
            else
            {
                // Not using tabulated potentials.
                switch (fullLongRange)
                {
                        case 0: // Use cutoff | cell decomposition.
                            if (s % decompPeriod == 0)
                            {
                               internal->decompose();
                               #ifdef _OPENMP
                               omp_set_num_threads(4);
                               #endif
                               #pragma omp parallel for
                               for(int i = 0; i < numReplicas; ++i)
                                   RBC[i]->updateParticleLists( (internal->getPos_d())+i*num, sys_d);
                            }
                            internal->compute(get_energy);
                            break;
                        case 1: // Do not use cutoff
                            internal->computeFull(get_energy);
                            break;

                        case 2: // Compute only softcore forces.
                            internal->computeSoftcoreFull(get_energy);
                            break;

                        case 3: // Compute only electrostatic forces.
                            internal->computeElecFull(get_energy);
                            break;
                }
            }
        }
        //compute the force for rigid bodies
        #ifdef _OPENMP
        omp_set_num_threads(4);
        #endif
        #pragma omp parallel for
        for(int i = 0; i < numReplicas; ++i)
            RBC[i]->updateForces((internal->getPos_d())+i*num, (internal->getForceInternal_d())+i*num, s, (internal->getEnergy())+i*num, get_energy, 
                                 RigidBodyInterpolationType, sys, sys_d);

        if(particle_dynamic == String("Langevin") || particle_dynamic == String("NoseHooverLangevin"))
            LastUpdateKernelBAOAB<<< numBlocks, NUM_THREADS >>>(internal -> getPos_d(), internal -> getMom_d(), internal -> getForceInternal_d(), 
            internal -> getType_d(), part_d, kT, kTGrid_d, electricField, tl, timestep, num, sys_d, randoGen_d, numReplicas, internal->getEnergy(), get_energy, 
            ParticleInterpolationType);
            //gpuErrchk(cudaDeviceSynchronize());

        if(rigidbody_dynamic == String("Langevin"))
        {
            #ifdef _OPENMP
            omp_set_num_threads(4);
            #endif
            #pragma omp parallel for ordered
            for(int i = 0; i < numReplicas; ++i)
            {
                RBC[i]->SetRandomTorques();
                RBC[i]->AddLangevin();
                RBC[i]->integrateDLM(sys, 2);
                #pragma omp ordered
                RBC[i]->print(s);
            }
        }

        if (s % outputPeriod == 0)
        {
            if(particle_dynamic == String("Langevin") || particle_dynamic == String("NoseHooverLangevin"))
            {
                gpuErrchk(cudaMemcpy(momentum, internal ->  getMom_d(), sizeof(Vector3) * num * numReplicas, cudaMemcpyDeviceToHost));
            }
            t = s*timestep;
            // Loop over all replicas
            for (int repID = 0; repID < numReplicas; ++repID)
            {

                if (numberFluct == 1)
                    updateNameList(); // no need for it here if particles stay the same

                // Write the trajectory.
                writers[repID]->append(pos + (repID*num), name, serial, t, num);

                if(particle_dynamic == String("Langevin") || particle_dynamic == String("NoseHooverLangevin"))
                {
                    momentum_writers[repID]->append(momentum + (repID * num), name, serial, t, num);
                    //force_writers[repID]->append(force + (repID * num), name, serial, t, num);
                }
            }
            // TODO: Currently, not compatible with replicas. Needs a fix.
            if (numberFluct == 1)
                updateReservoirs();

           remember(t);
        }
        if (get_energy)
        {
                wkf_timer_stop(timerS);
                t = s * timestep;
                // Simulation progress and statistics.
                float percent = (100.0f * s) / steps;
                float msPerStep = wkf_timer_time(timerS) * 1000.0f / outputEnergyPeriod;
                float nsPerDay = numReplicas * timestep / msPerStep * 864E5f;

                // Nice thousand separator
                setlocale(LC_NUMERIC, "");

                // Do the output
                printf("\rStep %ld [%.2f%% complete | %.3f ms/step | %.3f ns/day]",s, percent, msPerStep, nsPerDay);
        //}
        //if (get_energy)
        //{

                // Copy positions from GPU to CPU.
                //gpuErrchk(cudaMemcpy(pos, internal->getPos_d(), sizeof(Vector3)*num*numReplicas,cudaMemcpyDeviceToHost));
                float e = 0.f;
                float V = 0.f;
                thrust::device_ptr<float> en_d(internal->getEnergy());
                V = (thrust::reduce(en_d, en_d+num*numReplicas)) / numReplicas;
                if(particle_dynamic == String("Langevin") || particle_dynamic == String("NoseHooverLangevin"))
                {
                    gpuErrchk(cudaMemcpy(momentum, internal->getMom_d(), sizeof(Vector3)*num*numReplicas,cudaMemcpyDeviceToHost));
                    e = KineticEnergy();
                }   
                std::fstream energy_file;
                energy_file.open( (outFilePrefixes[0]+".energy.dat").c_str(), std::fstream::out | std::fstream::app);
                if(energy_file.is_open())
                {
                    energy_file << "Kinetic Energy: " << e*num*0.5f*(2.388458509e-1) << " (kT) "<< std::endl;
                    energy_file << "Potential Energy: " << V << " (kcal/mol) " << std::endl;
                    energy_file.close();
                }
                else
                {
                    std::cout << "Error in opening energy files\n";
                }
                
                if(rigidbody_dynamic == String("Langevin"))
                {
                    #ifdef _OPENMP
                    omp_set_num_threads(4);
                    #endif
                    #pragma omp parallel for
                    for(int i = 0; i < numReplicas; ++i)
                        RBC[i]->KineticEnergy();
                }
                std::fstream rb_energy_file;
                rb_energy_file.open( (outFilePrefixes[0]+".rb_energy.dat").c_str(), std::fstream::out | std::fstream::app);
                if(rb_energy_file.is_open())
                {
                    float k_tol = 0.f;
                    float v_tol = 0.f;
                    float (RigidBody::*func_ptr)();
                    #ifdef _OPENMP
                    omp_set_num_threads(4);
                    #endif
                    #pragma omp parallel for private(func_ptr) reduction(+:k_tol,v_tol)
                    for(int i = 0; i < numReplicas; ++i)
                    {
                        func_ptr = &RigidBody::getKinetic;
                        k_tol += RBC[i]->getEnergy(func_ptr);
                        func_ptr = &RigidBody::getEnergy;
                        v_tol += RBC[i]->getEnergy(func_ptr);
                    }
                    rb_energy_file << "Kinetic Energy "   << k_tol/numReplicas << " (kT)" << std::endl;
                    rb_energy_file << "Potential Energy " << v_tol/numReplicas << " (kcal/mol)" << std::endl;
                    rb_energy_file.close();
                }
                else
                {
                    std::cout << "Error in opening rb energy files\n"; 
                }

                // Write restart files for each replica.
                for (int repID = 0; repID < numReplicas; ++repID)
                    writeRestart(repID);

                wkf_timer_start(timerS);
         } // s % outputEnergyPeriod
     } // done with all Brownian dynamics steps

     if (imd_on and clientsock)
     {
            if (vmdsock_selread(clientsock, 0) == 1)
            {
                int length;
                switch (imd_recv_header(clientsock, &length))
                {
                    case IMD_DISCONNECT:
                        printf("\n[IMD] Disconnecting...\n");
                        imd_disconnect(clientsock);
                        clientsock = NULL;
                        sleep(5);
                        break;
                    case IMD_KILL:
                        printf("\n[IMD] Killing...\n");
                        imd_disconnect(clientsock);
                        clientsock = NULL;
                        sleep(5);
                        break;
                    default:
                        printf("\n[IMD] Something weird happened. Disconnecting..\n");
                        break;
                }
            }
     }
     // Stop the main timer.
     wkf_timer_stop(timer0);

     // Compute performance data.
     const float elapsed = wkf_timer_time(timer0); // seconds
     int tot_hrs = (int) std::fmod(elapsed / 3600.0f, 60.0f);
     int tot_min = (int) std::fmod(elapsed / 60.0f, 60.0f);
     float tot_sec   = std::fmod(elapsed, 60.0f);

     printf("\nFinal Step: %d\n", (int) steps);

     printf("Total Run Time: ");
     if (tot_hrs > 0) printf("%dh%dm%.1fs\n", tot_hrs, tot_min, tot_sec);
     else if (tot_min > 0) printf("%dm%.1fs\n", tot_min, tot_sec);
     else printf("%.2fs\n", tot_sec);

     gpuErrchk(cudaFree(force_d));
} // GrandBrownTown::run()

// Run the Brownian Dynamics steps.
void GrandBrownTown::run() {

        RunNoseHooverLangevin();
#if 0
	printf("\n\n");
	Vector3 runningNetForce(0.0f);
	
	// Open the files for recording ionic currents
	for (int repID = 0; repID < numReplicas; ++repID) {
		writers[repID]->newFile(pos + (repID * num), name, 0.0f, num); // 'pos + (repID*num)' == array-to-pointer decay
	}
        //Han-Yi Chou
        if(particle_dynamic == String("Langevin"))
        {
            for (int repID = 0; repID < numReplicas; ++repID)
            {
                momentum_writers[repID]->newFile(momentum + (repID * num), name, 0.0f, num); // 'pos + (repID*num)' == array-to-pointer decay
                //force_writers[repID]->newFile(force + (repID * num), name, 0.0f, num);
            }
        }

	// Save (remember) particle positions for later (analysis)
	remember(0.0f);

	// Initialize timers (util.*)
	wkf_timerhandle timer0, timerS;
	timer0 = wkf_timer_create();
	timerS = wkf_timer_create();

	copyToCUDA();
        if(particle_dynamic == String("Langevin"))
	    internal -> copyToCUDA(forceInternal, pos,momentum);
        else
            internal -> copyToCUDA(forceInternal, pos);

	// IMD Stuff
	void* sock = NULL;
	void* clientsock = NULL;
	int length;
	if (imd_on) {
		printf("Setting up incoming socket\n");
		vmdsock_init();
		sock = vmdsock_create();
		clientsock = NULL;
		vmdsock_bind(sock, imd_port);

		printf("Waiting for IMD connection on port %d...\n", imd_port);
		vmdsock_listen(sock);
		while (!clientsock) {
			if (vmdsock_selread(sock, 0) > 0) {
				clientsock = vmdsock_accept(sock);
				if (imd_handshake(clientsock))
					clientsock = NULL;
			}
		}
		sleep(1);
		if (vmdsock_selread(clientsock, 0) != 1 ||
				imd_recv_header(clientsock, &length) != IMD_GO) {
			clientsock = NULL;
		}
		imdForces = new Vector3[num*numReplicas];
		for (size_t i = 0; i < num; ++i) // clear old forces
			imdForces[i] = Vector3(0.0f);

	} // endif (imd_on)

	// Start timers
	wkf_timer_start(timer0);
	wkf_timer_start(timerS);

	// We haven't done any steps yet.
	// Do decomposition if we have to
	if (fullLongRange == 0)
	{
		// cudaSetDevice(0);
		internal->decompose();
		gpuErrchk(cudaDeviceSynchronize());
		RBC.updateParticleLists( internal->getPos_d() );
	}

	float t; // simulation time

        int numBlocks = (num * numReplicas) / NUM_THREADS + ((num * numReplicas) % NUM_THREADS == 0 ? 0 : 1);
        int tl = temperatureGridFile.length();
        Vector3 *force_d;
        gpuErrchk(cudaMalloc((void**)&force_d, sizeof(Vector3)*num * numReplicas));

	printf("Configuration: %d particles | %d replicas\n", num, numReplicas);

	// Main loop over Brownian dynamics steps
	for (long int s = 1; s < steps; s++) 
        {
            // Compute the internal forces. Only calculate the energy when we are about to output.
	    bool get_energy = ((s % outputEnergyPeriod) == 0);
            //At the very first time step, the force is computed
            if(s == 1) 
            {
                // 'interparticleForce' - determines whether particles interact with each other
                gpuErrchk(cudaMemset((void*)(internal->getForceInternal_d()),0,num*numReplicas*sizeof(Vector3)));
                RBC.clearForceAndTorque(); //Han-Yi Chou

		if (interparticleForce) 
                {
                    //gpuErrchk(cudaMemset((void*)(internal->getForceInternal_d()),0,num*numReplicas*sizeof(Vector3)));
                    //RBC.clearForceAndTorque(); //Han-Yi Chou

	            // 'tabulatedPotential' - determines whether interaction is described with tabulated potentials or formulas
		    if (tabulatedPotential) 
                    {
		        // Using tabulated potentials
		        // 'fullLongRange' - determines whether 'cutoff' is used
		        // 0 - use cutoff (cell decomposition) [ N*log(N) ]
		        // 1 - do not use cutoff [ N^2 ]
		        switch (fullLongRange) 
                        {
		            case 0: // [ N*log(N) ] interactions, + cutoff | decomposition
                            //I want to replace this
                            // by checking how far particles move to decide whether the pair list should be updated
                            //if(NeedUpdatePairList()) //ToDo, Han-Yi Chou
                            //{
                            //internal-> decompose();
                            //  RBC.updateParticleLists( internal->getPos_d() );
                            //}
			        if (s % decompPeriod == 0)
		                {
		                    // cudaSetDevice(0);
				    internal -> decompose();
				    RBC.updateParticleLists( internal->getPos_d() );
			        }
						
			        //MLog: added Bond* bondList to the list of passed in variables.
			        /*energy = internal->computeTabulated(forceInternal_d, pos_d, type_d, bonds_d, bondMap_d, excludes_d, excludeMap_d,	angles_d, dihedrals_d, get_energy);*/
		               // energy = internal -> computeTabulated(get_energy);
			        internal -> computeTabulated(get_energy);
			        break;
			    default: 
                               // [ N^2 ] interactions, no cutoff | decompositions
			        internal->computeTabulatedFull(get_energy);
			        break;
		        }
                    } 
                    else 
                    {
		        // Not using tabulated potentials.
		        switch (fullLongRange) 
                        {
                            case 0: // Use cutoff | cell decomposition.
		               if (s % decompPeriod == 0)
		               {
		                   // cudaSetDevice(0);
				   internal->decompose();
				   RBC.updateParticleLists( internal->getPos_d() );
			       }
			       internal->compute(get_energy);
			       break;

			    case 1: // Do not use cutoff
		                internal->computeFull(get_energy);
				break;

			    case 2: // Compute only softcore forces.
		                internal->computeSoftcoreFull(get_energy);
				break;

		            case 3: // Compute only electrostatic forces.
				internal->computeElecFull(get_energy);
				break;
		        }
	            }
		}//if inter-particle force

	        gpuErrchk(cudaDeviceSynchronize());
		RBC.updateForces(internal->getPos_d(), internal->getForceInternal_d(), s, RigidBodyInterpolationType);
                if(rigidbody_dynamic == String("Langevin"))
                {
                    RBC.SetRandomTorques();
                    RBC.AddLangevin();
                }
		gpuErrchk(cudaDeviceSynchronize());
            }//if step == 1

          
	    //Han-Yi Chou
            //update the rigid body positions and orientation
            //So far only brownian dynamics is used
            //RBC.integrate(sys, s);

	    if(particle_dynamic == String("Langevin"))
                updateKernelBAOAB<<< numBlocks, NUM_THREADS >>>(internal -> getPos_d(), internal -> getMom_d(), internal -> getForceInternal_d(), internal -> getType_d(), part_d, kT, kTGrid_d, electricField, tl, timestep, num, sys_d, randoGen_d, numReplicas);
            //For Brownian motion
            else
                updateKernel<<< numBlocks, NUM_THREADS >>>(internal -> getPos_d(), internal -> getForceInternal_d(), internal -> getType_d(), 
                                                           part_d, kT, kTGrid_d, electricField, tl, timestep, num, sys_d, randoGen_d, numReplicas);


            if(rigidbody_dynamic == String("Langevin"))
            {
                RBC.integrateDLM(sys,0);
                RBC.integrateDLM(sys,1);
            }
            else
                RBC.integrate(sys,s);


            if (s % outputPeriod == 0) {
                // Copy particle positions back to CPU
		gpuErrchk(cudaDeviceSynchronize());
                gpuErrchk(cudaMemcpy(pos, internal ->  getPos_d(), sizeof(Vector3) * num * numReplicas, cudaMemcpyDeviceToHost));
	    }

            //compute again the new force with new positions.
            //reset the internal force, I hope. Han-Yi Chou
            //First clear the old force
            if (imd_on && clientsock && s % outputPeriod == 0) 
            {
                float* coords = new float[num*3]; // TODO: move allocation out of run loop
                int* atomIds = new int[num]; // TODO: move allocation out of run loop
                int length;

                bool paused = false;
                while (vmdsock_selread(clientsock, 0) > 0 || paused) 
                {
                    switch (imd_recv_header(clientsock, &length)) 
                    {
                        case IMD_DISCONNECT:
                            printf("[IMD] Disconnecting...\n");
                            imd_disconnect(clientsock);
                            clientsock = NULL;
                            sleep(5);
                            break;
                        case IMD_KILL:
                            printf("[IMD] Killing...\n");
                            imd_disconnect(clientsock);
                            clientsock = NULL;
                            steps = s; // Stop the simulation at this step
                            sleep(5);
                            break;
                        case IMD_PAUSE:
                            paused = !paused;
                            break;
                        case IMD_GO:
                            printf("[IMD] Caught IMD_GO\n");
                            break;
                        case IMD_MDCOMM:
                            for (size_t i = 0; i < num; ++i) // clear old forces
                                imdForces[i] = Vector3(0.0f);

                            if (imd_recv_mdcomm(clientsock, length, atomIds, coords)) 
                            {
                                printf("[IMD] Error receiving forces\n");
                            } 
                            else 
                            {
                                for (size_t j = 0; j < length; ++j) 
                                {
                                    int i = atomIds[j];
                                    imdForces[i] = Vector3(coords[j*3], coords[j*3+1], coords[j*3+2]) * conf.imdForceScale;
                                }
                            }
                            break;
                        default:
                            printf("[IMD] Something weird happened. Disconnecting..\n");
                            break;
                    }
                }
                if (clientsock) 
                {
                    // float* coords = new float[num*3]; // TODO: move allocation out of run loop
                    for (size_t i = 0; i < num; i++) 
                    {
                        const Vector3& p = pos[i];
                        coords[3*i] = p.x;
                        coords[3*i+1] = p.y;
                        coords[3*i+2] = p.z;
                    }
                    imd_send_fcoords(clientsock, num, coords);
                }
                delete[] coords;
                delete[] atomIds;
            }
            if (imd_on && clientsock) 
                internal->setForceInternalOnDevice(imdForces); // TODO ensure replicas are mutually exclusive with IMD
            //else
            //{ 
                //int numBlocks = (num * numReplicas) / NUM_THREADS + (num * numReplicas % NUM_THREADS == 0 ? 0 : 1);
                //MLog: along with calls to internal (ComputeForce class) this function should execute once per GPU.
                //clearInternalForces<<< numBlocks, NUM_THREADS >>>(internal->getForceInternal_d(), num*numReplicas);
                //use cudaMemset instead
                //gpuErrchk(cudaMemset((void*)(internal->getForceInternal_d()),0,num*numReplicas*sizeof(Vector3)));
                //RBC.clearForceAndTorque();
            //}
            //compute the new force for particles
            RBC.clearForceAndTorque();
            gpuErrchk(cudaMemset((void*)(internal->getForceInternal_d()),0,num*numReplicas*sizeof(Vector3)));
            //RBC.clearForceAndTorque();

            if (interparticleForce) 
            {
                // 'tabulatedPotential' - determines whether interaction is described with tabulated potentials or formulas
                if (tabulatedPotential)
                {
                    switch (fullLongRange) 
                    {
                        case 0: // [ N*log(N) ] interactions, + cutoff | decomposition
                            if (s % decompPeriod == 0)
                            {
                                internal -> decompose();
                                RBC.updateParticleLists( internal->getPos_d() );
                            }
                            internal -> computeTabulated(get_energy);
                            break;
                        default: // [ N^2 ] interactions, no cutoff | decompositions
                            internal->computeTabulatedFull(get_energy);
                            break;
                    }
                } 
                else 
                {
                    // Not using tabulated potentials.
                    switch (fullLongRange) 
                    {
                        case 0: // Use cutoff | cell decomposition.
                            if (s % decompPeriod == 0)
                            {
                               internal->decompose();
                               RBC.updateParticleLists( internal->getPos_d() );
                            }
                            internal->compute(get_energy);
                            break;
                        case 1: // Do not use cutoff
                            internal->computeFull(get_energy);
                            break;

                        case 2: // Compute only softcore forces.
                            internal->computeSoftcoreFull(get_energy);
                            break;

                        case 3: // Compute only electrostatic forces.
                            internal->computeElecFull(get_energy);
                            break;
                    }
                }
            }

            //compute the force for rigid bodies
            RBC.updateForces(internal->getPos_d(), internal->getForceInternal_d(), s, RigidBodyInterpolationType);
            //Han-Yi Chou
            //For BAOAB, the last update is only to update the momentum
            // gpuErrchk(cudaDeviceSynchronize());

            if(particle_dynamic == String("Langevin"))
                LastUpdateKernelBAOAB<<< numBlocks, NUM_THREADS >>>(internal -> getPos_d(), internal -> getMom_d(), internal -> getForceInternal_d(), internal -> getType_d(), part_d, kT, kTGrid_d, electricField, tl, timestep, num, sys_d, randoGen_d, numReplicas);

           //gpuErrchk(cudaDeviceSynchronize());

           if(rigidbody_dynamic == String("Langevin"))
           {
               RBC.SetRandomTorques();
               RBC.AddLangevin();
               RBC.integrateDLM(sys,2);
               RBC.print(s);
           }
 
           if (s % outputPeriod == 0) 
           {
                if(particle_dynamic == String("Langevin"))
                {
		    // TODO: make async
                    gpuErrchk(cudaMemcpy(momentum, internal ->  getMom_d(), sizeof(Vector3) * num * numReplicas, cudaMemcpyDeviceToHost));
                    /*
                    gpuErrchk(cudaMemcpy(force, force_d, sizeof(Vector3) * num * numReplicas, cudaMemcpyDeviceToHost));
                    Vector3 f(0.f), p(0.f), r0(0.f);
                    double total_mass = 0.;
                    for(int i = 0; i < num * numReplicas; ++i)
                    {
                        f = f + force[i];
                        p = p + momentum[i];
                        total_mass += part[type[i]].mass;
                        r0 = r0 + part[type[i]].mass * pos[i];
                    }
                    printf("The COM is %f %f %f\n",r0.x / total_mass, r0.y / total_mass, r0.z / total_mass);
                    printf("The total momentum is %f %f %f\n",p.x, p.y, p.z);
                    printf("The total force %f %f %f\n",f.x, f.y, f.z);
                    */
		    // Output trajectories (to files)
		}
                //RBC.print(s);
                t = s*timestep;

		// Loop over all replicas
		for (int repID = 0; repID < numReplicas; ++repID) 
                {

                    if (numberFluct == 1) 
                        updateNameList(); // no need for it here if particles stay the same

                    // Write the trajectory.
		    writers[repID]->append(pos + (repID*num), name, serial, t, num);

                    if(particle_dynamic == String("Langevin"))
                    {
                        momentum_writers[repID]->append(momentum + (repID * num), name, serial, t, num);
                        //force_writers[repID]->append(force + (repID * num), name, serial, t, num);
                    }

	        }

                // TODO: Currently, not compatible with replicas. Needs a fix.
		if (numberFluct == 1) 
                    updateReservoirs();

		remember(t);
            }
	    // Output energy.
            if (get_energy) 
            {
	        wkf_timer_stop(timerS);
	        t = s * timestep;
	        // Simulation progress and statistics.
	        float percent = (100.0f * s) / steps;
	        float msPerStep = wkf_timer_time(timerS) * 1000.0f / outputEnergyPeriod;
	        float nsPerDay = numReplicas * timestep / msPerStep * 864E5f;

	        // Nice thousand separator
	        setlocale(LC_NUMERIC, "");

	        // Do the output
	        printf("\rStep %ld [%.2f%% complete | %.3f ms/step | %.3f ns/day]",s, percent, msPerStep, nsPerDay);

	        // Copy positions from GPU to CPU.
	        gpuErrchk(cudaMemcpy(pos, internal->getPos_d(), sizeof(Vector3)*num*numReplicas,cudaMemcpyDeviceToHost));
                if(particle_dynamic == String("Langevin"))
                {
                    //gpuErrchk(cudaMemcpy(momentum, internal->getMom_d(), sizeof(Vector3)*num*numReplicas,cudaMemcpyDeviceToHost));
                    gpuErrchk(cudaMemcpy(momentum, internal->getMom_d(), sizeof(Vector3)*num*numReplicas,cudaMemcpyDeviceToHost));
                    float e = KineticEnergy();
                    printf(" The kinetic energy is %f \n",e*(2.388458509e-1));
                }
                if(rigidbody_dynamic == String("Langevin"))
                {
                    //gpuErrchk(cudaMemcpy(momentum, internal->getMom_d(), sizeof(Vector3)*num*numReplicas,cudaMemcpyDeviceToHost));
                    float e = RotKineticEnergy();
                    printf(" The Rotational kinetic energy is %f \n",e*(2.388458509e-1));
                }
       
                // Write restart files for each replica.
                for (int repID = 0; repID < numReplicas; ++repID)
                    writeRestart(repID);
                
	        wkf_timer_start(timerS);
            } // s % outputEnergyPeriod
	} // done with all Brownian dynamics steps

	// If IMD is on & our socket is still open.
	if (imd_on and clientsock) 
        {
            if (vmdsock_selread(clientsock, 0) == 1) 
            {
	        int length;
		switch (imd_recv_header(clientsock, &length)) 
                {
		    case IMD_DISCONNECT:
		        printf("\n[IMD] Disconnecting...\n");
			imd_disconnect(clientsock);
			clientsock = NULL;
			sleep(5);
			break;
		    case IMD_KILL:
		        printf("\n[IMD] Killing...\n");
			imd_disconnect(clientsock);
			clientsock = NULL;
			sleep(5);
			break;
		    default:
		        printf("\n[IMD] Something weird happened. Disconnecting..\n");
			break;
	        }
            }
	}
	// Stop the main timer.
	wkf_timer_stop(timer0);

	// Compute performance data.
	const float elapsed = wkf_timer_time(timer0); // seconds
	int tot_hrs = (int) std::fmod(elapsed / 3600.0f, 60.0f);
	int tot_min = (int) std::fmod(elapsed / 60.0f, 60.0f);
	float tot_sec	= std::fmod(elapsed, 60.0f);

	printf("\nFinal Step: %d\n", (int) steps);

	printf("Total Run Time: ");
	if (tot_hrs > 0) printf("%dh%dm%.1fs\n", tot_hrs, tot_min, tot_sec);
	else if (tot_min > 0) printf("%dm%.1fs\n", tot_min, tot_sec);
	else printf("%.2fs\n", tot_sec);

        gpuErrchk(cudaFree(force_d));
#endif
} // GrandBrownTown::run()
// --------------------------------------------
// Populate lists of types and serial numbers.
//
void GrandBrownTown::populate() {
	for (int repID = 0; repID < numReplicas; repID++) {
		const int offset = repID * num;
		int pn = 0;
		int p = 0;
		for (int i = 0; i < num; i++) {
			type[i + offset] = p;
			serial[i + offset] = currSerial++;

			if (++pn >= part[p].num) {
				p++;
				pn = 0;
			}
		}
	}
}



void GrandBrownTown::writeRestart(int repID) const 
{
    FILE* out   = fopen(restartFiles[repID].c_str(), "w");
    const int offset = repID * num;

    for (int i = 0; i < num; ++i) 
    {
        const int ind = i + offset;
        const Vector3& p = pos[ind];
	fprintf(out, "%d %.10g %.10g %.10g\n", type[ind], p.x, p.y, p.z); 
    }
    fclose(out);

    if(particle_dynamic == String("Langevin") || particle_dynamic == String("NoseHooverLangevin"))
    {
        out = fopen(restartMomentumFiles[repID].c_str(), "w");
        
        for (int i = 0; i < num; ++i) 
        {
            const int ind = i + offset;
            const Vector3& p = momentum[ind];
            fprintf(out, "%d %.10g %.10g %.10g\n", type[ind], p.x, p.y, p.z); 
        }
        fclose(out);
    }
   
}

/*the center is defined by the first pmf*/
void GrandBrownTown::initialCondCen() {
	for (int i = 0; i < num; i++)
		pos[i] = part[ type[i] ].pmf->getCenter();
}


// Set random initial positions for all particles and replicas
void GrandBrownTown::initialCond() {
	for (int repID = 0; repID < numReplicas; repID++) {
		const int offset = repID * num;
		for (int i = 0; i < num; i++) {
			pos[i + offset] = findPos(type[i + offset]);
		}
	}
}

// A couple old routines for getting particle positions.
Vector3 GrandBrownTown::findPos(int typ) {
    // TODO: sum over grids
	Vector3 r;
	const BrownianParticleType& pt = part[typ];
	do {
		const float rx = sysDim.x * randoGen->uniform(); 
		const float ry = sysDim.y * randoGen->uniform();
		const float rz = sysDim.z * randoGen->uniform();
		r = sys->wrap( Vector3(rx, ry, rz) );
	} while (pt.pmf->interpolatePotential(r) > *pt.meanPmf);
	return r;
}


Vector3 GrandBrownTown::findPos(int typ, float minZ) {
	Vector3 r;
	const BrownianParticleType& pt = part[typ];
	do {
		const float rx = sysDim.x * randoGen->uniform();
		const float ry = sysDim.y * randoGen->uniform();
		const float rz = sysDim.z * randoGen->uniform();
		r = sys->wrap( Vector3(rx, ry, rz) );
	} while (pt.pmf->interpolatePotential(r) > *pt.meanPmf and fabs(r.z) > minZ);
	return r;
}

//Compute the kinetic energy of particle and rigid body Han-Yi Chou
float GrandBrownTown::KineticEnergy()
{
    float *vec_red, *energy;
    float particle_energy;

    gpuErrchk(cudaMalloc((void**)&vec_red, 512*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&energy, sizeof(float)));
    gpuErrchk(cudaMemset((void*)energy,0,sizeof(float)));

    BrownParticlesKineticEnergy<64><<<dim3(512),dim3(64)>>>(internal->getMom_d(), internal -> getType_d(), part_d, vec_red, numReplicas*num);
    gpuErrchk(cudaDeviceSynchronize());

    Reduction<64><<<dim3(1),dim3(64)>>>(vec_red, energy, 512);

    gpuErrchk(cudaMemcpy((void*)&particle_energy, energy, sizeof(float), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(vec_red));
    gpuErrchk(cudaFree(energy));

    return 2. * particle_energy / kT / num / numReplicas; //In the unit of 0.5kT
}
/*
void GrandBrownTown::RotKineticEnergy()
{
    RBC.KineticEnergy();

    return 2. * e / numReplicas / kT; //In the unit of 0.5kT
}
*/
void GrandBrownTown::InitNoseHooverBath(int N)
{
    printf("Entering Nose-Hoover Langevin\n");
    int count = 0;

    for(int i = 0; i < N; ++i)
    {
        int typ = type[i];
        double mu = part[typ].mu;
        
        double sigma = sqrt(kT / mu);

        float tmp = sigma * randoGen->gaussian();
        random[(size_t)count] = tmp;
        ++count;
    }
    printf("Done in nose-hoover bath\n");
}
// -----------------------------------------------------------------------------
// Initialize file for recording ionic current
void GrandBrownTown::newCurrent(int repID) const {
    /*
	FILE* out = fopen(outCurrFiles[repID].c_str(), "w");
	fclose(out);
    */
}


// -----------------------------------------------------------------------------
// Record the ionic current flowing through the entire system
void GrandBrownTown::writeCurrent(int repID, float t) const {
    return;
    /*
	FILE* out = fopen(outCurrFiles[repID].c_str(), "a");
	fprintf(out, "%.10g %.10g %d\n", 0.5f*(t+timeLast), current(t), num);
	fclose(out);
    */
}


// -----------------------------------------------------------------------------
// Record the ionic current in a segment -segZ < z < segZ
void GrandBrownTown::writeCurrentSegment(int repID, float t, float segZ) const {
    return;
    /*
	FILE* out = fopen(outCurrFiles[repID].c_str(), "a");
	int i;
	fprintf(out, "%.10g ", 0.5f * (t + timeLast));
	for (i = -1; i < numParts; i++)
		fprintf(out, "%.10g ", currentSegment(t,segZ,i));
	fprintf(out, "%d\n", num);
	fclose(out);
    */
}


// ----------------------------------------------------
// Compute the current in nanoamperes for entire system
//
float GrandBrownTown::current(float t) const {
	float curr = 0.0f;
	float dt = timeLast - t;

	for (int i = 0; i < num; i++) {
		Vector3 d = sys->wrapDiff(pos[i]-posLast[i]);
		curr += part[type[i]].charge*d.z/(sysDim.z*dt)*1.60217733e-1f;
	}
	return curr;
}


// -----------------------------------------------------
// Compute the current in nanoamperes for a restricted segment (-segZ < z < segZ).
//
float GrandBrownTown::currentSegment(float t, float segZ, int carrier) const {
	float curr = 0.0f;
	float dt = t - timeLast;

	for (int i = 0; i < num; i++) {
		float z0 = posLast[i].z;
		float z1 = pos[i].z;

		// Ignore carriers outside the range for both times.
		if (fabs(z0) > segZ && fabs(z1) > segZ) continue;

		// Cut the pieces outside the range.
		if (z0 < -segZ) z0 = -segZ;
		if (z1 < -segZ) z1 = -segZ;
		if (z0 > segZ) z0 = segZ;
		if (z1 > segZ) z1 = segZ;

		float dz = sys->wrapDiff(z1 - z0, sysDim.z);
		if ( carrier == type[i] || carrier == -1) {
			curr += part[type[i]].charge*dz/(2.0f*segZ*dt)*1.60217733e-1f;
		}
	}
	return curr;
}


int GrandBrownTown::getReservoirCount(int partInd, int resInd) const {
	int count = 0;
	const Reservoir* res = part[partInd].reservoir;
	for (int i = 0; i < num; ++i)
		if (type[i] == partInd and res->inside(i, pos[i]))
			count++;
	return count;
}

IndexList GrandBrownTown::getReservoirList(int partInd, int resInd) const {
	IndexList ret;
	const Reservoir* res = part[partInd].reservoir;
	for (int i = 0; i < num; ++i)
		if (type[i] == partInd and res->inside(resInd, pos[i]))
			ret.add(i);
	return ret;
}

Vector3 GrandBrownTown::freePosition(Vector3 r0, Vector3 r1, float minDist) {
	const int maxTries = 1000;
	bool tooClose = true;
	Vector3 r;
	Vector3 d = r1 - r0;
	float minDist2 = minDist*minDist;

	const CellDecomposition& decomp = internal->getDecomp();
	const CellDecomposition::cell_t *cells = decomp.getCells();

	int tries = 0;
	while (tooClose) {
		r.x = r0.x + d.x*randoGen->uniform();
		r.y = r0.y + d.y*randoGen->uniform();
		r.z = r0.z + d.z*randoGen->uniform();

		tooClose = false;
		// Check to make sure we are not too near another particle.
		const CellDecomposition::cell_t cell = decomp.getCell(decomp.getCellID(r));
		for (int i = -1; i <= 1; ++i) {
			for (int j = -1; j <= 1; ++j) {
				for (int k = -1; k <= 1; ++k) {
					int nID = decomp.getNeighborID(cell, i, j, k);
					// TODO: Determine which replica to use to look for free position.
					const CellDecomposition::range_t range = decomp.getRange(nID, 0);
					for (int n = range.first; n < range.last; ++n) {
						Vector3 dj = pos[ cells[n].particle ];
						if (dj.length2() < minDist2) {
							tooClose = true;
							break;
						}
					}
				}
			}
		}

		// Don't try too many times.
		if (++tries > maxTries) {
			printf("WARNING: freePosition too many tries to find free position.\n");
			break;
		}
	}

	return r;
}

// -----------------------------------------------------------------------------
// Update the list of particle names[] for simulations with varying number of
// particles
void GrandBrownTown::updateNameList() {
	if (outputFormat == TrajectoryWriter::formatTraj) {
		char typeNum[64];
		for (int i = 0; i < num; ++i) {
			sprintf(typeNum, "%d", type[i]);
			name[i] = typeNum;
		}
	} else {
		for (int i = 0; i < num; ++i)
			name[i] = part[ type[i] ].name;
	}
}

// -----------------------------------------------------------------------------
// Save particle positions for analysis purposes.
// TODO: Fix for multiple replicas.
void GrandBrownTown::remember(float t) {
	timeLast = t;
	std::copy(pos, pos + num * numReplicas, posLast);
        if(particle_dynamic == String("Langevin") || particle_dynamic == String("NoseHooverLangevin"))
            std::copy(momentum, momentum + num * numReplicas, momLast);
}

// -----------------------------------------------------------------------------
// Delete particles listed in the list 'p'
void GrandBrownTown::deleteParticles(IndexList& p) {
	int n = 0;
	for (int i = 0; i < num; ++i) {
		pos[n] = pos[i];
		type[n] = type[i];
		serial[n] = serial[i];
		if (p.find(i) == -1) n++;
	}
	num = n;
}


// -----------------------------------------------------------------------------
// Add particles, obey numCap limit
void GrandBrownTown::addParticles(int n, int typ) {
	if (num + n > numCap) n = numCap - num;

	for (int i = num; i < num + n; i++) {
		pos[i] = findPos(typ, initialZ);
		type[i] = typ;
		serial[i] = currSerial;
		currSerial++;
	}
	num += n;
}

// -----------------------------------------------------------------------------
// Add particles randomly within a region between r0 and r1.
// TODO: Fix for CUDA.
void GrandBrownTown::addParticles(int n, int typ, Vector3 r0, Vector3 r1) {
	if (num + n > numCap) n = numCap - num;

	Vector3 d = r1 - r0;
	for (int i = num; i < num + n; ++i) {
		Vector3 r;
		r.x = r0.x + d.x * randoGen->uniform();
		r.y = r0.y + d.y * randoGen->uniform();
		r.z = r0.z + d.z * randoGen->uniform();

		pos[i] = r;
		type[i] = typ;
		serial[i] = currSerial++;
	}
	num += n;
}

// -----------------------------------------------------------------------------
// Add particles randomly within the region defined by r0 and r1. Maintain a
// minimum distance of minDist between particles.
// TODO: Fix for CUDA.
void GrandBrownTown::addParticles(int n, int typ, Vector3 r0, Vector3 r1, float minDist) {
	if (num + n > numCap) n = numCap - num;
	const int n0 = num;
	for (int i = n0; i < n0 + n; i++) {
		// Generate a position for the new particle.
		pos[i] = freePosition(r0, r1, minDist);
		type[i] = typ;
		num++;
		// Update the cell decomposition
		internal->updateNumber(num); /* RBTODO: unsure if type arg is ok */
	}
}

// -----------------------------------------------------------------------------
// Add or delete particles in the reservoirs. Reservoirs are not wrapped.
void GrandBrownTown::updateReservoirs() {
	bool numberChange = false;
	for (int p = 0; p < numParts; ++p) {
		if (part[p].reservoir == NULL) continue;

		const int n = part[p].reservoir->length();

		for (int res = 0; res < n; res++) {
			// Get the current number of particles in the reservoir.
			IndexList resPart = getReservoirList(p, res);
			int numberCurr = resPart.length();

			// Determine the new number for this particle from a Poisson distribution.
			float number0 = part[p].reservoir->getMeanNumber(res);
			int number = randoGen->poisson(number0);

			// If the number is the same nothing needs to be done.
			if (number == numberCurr) continue;

			if (number < numberCurr) {
				int dn = numberCurr - number;

				// We need to delete particles.  Choose them at random.
				IndexList delPart;
				int pick = static_cast<int>(randoGen->uniform() *numberCurr) % numberCurr;
				if (pick + dn >= numberCurr) {
					int dn0 = dn - (numberCurr - pick);
					delPart = resPart.range(pick, numberCurr-1);
					delPart.add(resPart.range(0, dn0-1));
				} else {
					delPart = resPart.range(pick, pick + dn-1);
				}

				deleteParticles(delPart);
				numberChange = true;
			} else {
				// We need to add particles.
				Vector3 r0 = part[p].reservoir->getOrigin(res);
				Vector3 r1 = part[p].reservoir->getDestination(res);
				addParticles(number - numberCurr, p, r0, r1, minimumSep);
				numberChange = true;
			}
		} // end reservoir loop
	} // end particle loop

	if (numberChange)
		internal->updateNumber(num);
}

void GrandBrownTown::copyRandToCUDA() {
	gpuErrchk(cudaMalloc((void**)&randoGen_d, sizeof(Random)));
        gpuErrchk(cudaMemcpy(&(randoGen_d->states), &(randoGen->states), sizeof(curandState_t*),cudaMemcpyHostToDevice));
}


// -----------------------------------------------------------------------------
// Allocate memory on GPU(s) and copy to device
void GrandBrownTown::copyToCUDA() {
	/* const size_t tot_num = num * numReplicas;
	gpuErrchk(cudaMalloc(&pos_d, sizeof(Vector3) * tot_num));
	gpuErrchk(cudaMemcpyAsync(pos_d, pos, sizeof(Vector3) * tot_num,
														cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc(&forceInternal_d, sizeof(Vector3) * num * numReplicas));
	gpuErrchk(cudaMemcpyAsync(forceInternal_d, forceInternal, sizeof(Vector3) * tot_num,
														cudaMemcpyHostToDevice));*/

	// gpuErrchk(cudaDeviceSynchronize());
}

/*void GrandBrownTown::createBondList()
{
	size_t size = (numBonds / 2) * numReplicas * sizeof(int3);
	gpuErrchk( cudaMalloc( &bondList_d, size ) );
	gpuErrchk( cudaMemcpyAsync( bondList_d, bondList, size, cudaMemcpyHostToDevice) );

	for(int i = 0 ; i < (numBonds / 2) * numReplicas ; i++)
	{
		cout << "Displaying: bondList_d["<< i <<"].x = " << bondList[i].x << ".\n"
			<< "Displaying: bondList_d["<< i <<"].y = " << bondList[i].y << ".\n"
			<< "Displaying: bondList_d["<< i <<"].z = " << bondList[i].z << ".\n";

	}
}*/
