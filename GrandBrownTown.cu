#include "GrandBrownTown.h"
#include "GrandBrownTown.cuh"
/* #include "ComputeGridGrid.cuh" */

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true) {
	if (code != cudaSuccess) {
		fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
	if (abort) exit(code);
	}
}

bool GrandBrownTown::DEBUG;

cudaEvent_t START, STOP;

GrandBrownTown::GrandBrownTown(const Configuration& c, const char* outArg,
		const long int randomSeed, bool debug, bool imd_on, unsigned int imd_port, int numReplicas) :
	imd_on(imd_on), imd_port(imd_port), numReplicas(numReplicas),
	conf(c), RBC(RigidBodyController(c)) {

	for (int i = 0; i < numReplicas; i++) {
		std::stringstream curr_file, restart_file, out_prefix;

		curr_file << outArg << '.' << i << ".curr";
		restart_file << outArg << '.' << i << ".restart";
		out_prefix << outArg << '.' << i;

		outCurrFiles.push_back(curr_file.str());
		restartFiles.push_back(restart_file.str());
		outFilePrefixes.push_back(out_prefix.str());
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
	// TODO: add an option to generate random initial conditions for all replicas
	for (int r = 0; r < numReplicas; ++r) {
		std::copy(c.pos, c.pos + num, pos + r*num);
		std::copy(c.type, c.type + num, type + r*num);
		std::copy(c.serial, c.serial + num, serial + r*num);
	}

	currSerial = c.currSerial;  // serial number of the next new particle
	name = c.name;              // list of particle types! useful when 'numFluct == 1'
	posLast = c.posLast;        // previous positions of particles  (used for computing ionic current)
	timeLast = c.timeLast;      // previous time (used with posLast)
	minimumSep = c.minimumSep;  // minimum separation allowed when placing new particles

	// System parameters
	outputName = c.outputName;
	timestep = c.timestep;
	steps = c.steps;
	seed = c.seed;
	temperatureGrid = c.temperatureGrid;
	inputCoordinates = c.inputCoordinates;
	restartCoordinates = c.restartCoordinates;
	numberFluct = c.numberFluct;
	interparticleForce = c.interparticleForce;
	tabulatedPotential = c.tabulatedPotential;
	readBondsFromFile = c.readBondsFromFile;
	fullLongRange = c.fullLongRange;
	kT = c.kT;
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

	numTabBondFiles = c.numTabBondFiles;
	bondMap = c.bondMap;

	excludes = c.excludes;
	excludeMap = c.excludeMap;
	excludeRule = c.excludeRule;
	excludeCapacity = c.excludeCapacity;

	angles = c.angles;
	numTabAngleFiles = c.numTabAngleFiles;

	dihedrals = c.dihedrals;
	numTabDihedralFiles = c.numTabDihedralFiles;

	// Device parameters
	type_d = c.type_d;
	part_d = c.part_d;
	sys_d = c.sys_d;
	kTGrid_d = c.kTGrid_d;
	bonds_d = c.bonds_d;
	bondMap_d = c.bondMap_d;
	excludes_d = c.excludes_d;
	excludeMap_d = c.excludeMap_d;
	angles_d = c.angles_d;
	dihedrals_d = c.dihedrals_d;

	// Seed random number generator
	long unsigned int r_seed;
	if (seed == 0) {
		long int r0 = randomSeed;
		for (int i = 0; i < 4; i++)
			r0 *= r0 + 1;
		r_seed = time(NULL) + r0;
	} else {
		r_seed = seed + randomSeed;
	}
	printf("Setting up Random Generator\n");
	randoGen = new Random(num * numReplicas, r_seed);
	printf("Random Generator Seed: %lu -> %lu\n", randomSeed, r_seed);

	// "Some geometric stuff that should be gotten rid of." -- Jeff Comer
	Vector3 buffer = (sys->getCenter() + 2.0f * sys->getOrigin())/3.0f;
	initialZ = buffer.z;

	// Load random coordinates if necessary
	if (!c.loadedCoordinates) {
		printf("Populating\n");
		populate();
		initialCond();
		printf("Set random initial conditions.\n");
	}

	// Prepare internal force computation
	internal = new ComputeForce(num, part, numParts, sys, switchStart, switchLen, coulombConst,
			fullLongRange, numBonds, numTabBondFiles, numExcludes, numAngles, numTabAngleFiles,
			numDihedrals, numTabDihedralFiles, numReplicas);

	if (c.tabulatedPotential) {
		printf("Loading the tabulated potentials...\n");
		for (int p = 0; p < numParts*numParts; p++) {
			if (partTableFile[p].length() > 0) {
				int type0 = partTableIndex0[p];
				int type1 = partTableIndex1[p];

				internal->addTabulatedPotential(partTableFile[p].val(), type0, type1);
				printf("Loaded %s for types %s and %s.\n", partTableFile[p].val(),
						part[type0].name.val(), part[type1].name.val());
			}
		}
	}

	if (c.readBondsFromFile) {
		printf("Loading the tabulated bond potentials...\n");
		for (int p = 0; p < numTabBondFiles; p++)
			if (bondTableFile[p].length() > 0) {
				internal->addBondPotential(bondTableFile[p].val(), p, bonds, bonds_d);
				printf("%s\n",bondTableFile[p].val());
			}
	}

	if (c.readAnglesFromFile) {
		printf("Loading the tabulated angle potentials...\n");
		for (int p = 0; p < numTabAngleFiles; p++)
			if (angleTableFile[p].length() > 0)
				internal->addAnglePotential(angleTableFile[p].val(), p, angles, angles_d);
	}

	if (c.readDihedralsFromFile) {
		printf("Loading the tabulated dihedral potentials...\n");
		for (int p = 0; p < numTabDihedralFiles; p++)
			if (dihedralTableFile[p].length() > 0)
				internal->addDihedralPotential(dihedralTableFile[p].val(), p, dihedrals, dihedrals_d);
	}

	forceInternal = new Vector3[num * numReplicas];

	if (fullLongRange != 0)
		printf("No cell decomposition created.\n");

	// Prepare the trajectory output writer.
	for (int repID = 0; repID < numReplicas; ++repID) {
		TrajectoryWriter *w =
				new TrajectoryWriter(outFilePrefixes[repID].c_str(),
														 TrajectoryWriter::getFormatName(outputFormat),
														 sys->getBox(), num, timestep, outputPeriod);
		writers.push_back(w);
	}
	updateNameList();
}

GrandBrownTown::~GrandBrownTown() {
	delete[] forceInternal;
	delete[] pos;
	delete[] type;
	delete[] serial;

	delete randoGen;

	// Auxillary objects
	delete internal;
	for (int i = 0; i < numReplicas; ++i)
		delete writers[i];

	gpuErrchk(cudaFree(pos_d));
	gpuErrchk(cudaFree(forceInternal_d));
	gpuErrchk(cudaFree(randoGen_d));
}

// Run the Brownian Dynamics steps.
void GrandBrownTown::run() {
	printf("\n");

	// Open the files for recording ionic currents
	for (int repID = 0; repID < numReplicas; ++repID) {
		newCurrent(repID);
		writers[repID]->newFile(pos + (repID * num), name, 0.0f, num); // 'pos + (repID*num)' == array-to-pointer decay
	}

	// Save (remember) particle positions for later (analysis)
	remember(0.0f);

	// Initialize timers (util.*)
	rt_timerhandle cputimer;
	cputimer = rt_timer_create();
	rt_timerhandle timer0, timerS;
	timer0 = rt_timer_create();
	timerS = rt_timer_create();

	copyToCUDA();

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
	} // endif (imd_on)

	// Start timers
	rt_timer_start(timer0);
	rt_timer_start(timerS);


	// We haven't done any steps yet.
	// Do decomposition if we have to
	if (fullLongRange == 0)
		internal->decompose(pos_d);

	float t; // simulation time

	printf("Configuration: %d particles | %d replicas\n", num, numReplicas);

	// Main loop over Brownian dynamics steps
	for (long int s = 1; s < steps; s++) {
		// Compute the internal forces. Only calculate the energy when we are about to output.
		bool get_energy = ((s % outputEnergyPeriod) == 0);
		float energy = 0.0f;

		// Set the timer
		rt_timer_start(cputimer);

		// 'interparticleForce' - determines whether particles interact with each other
		if (interparticleForce) {

			// 'tabulatedPotential' - determines whether interaction is described with tabulated potentials or formulas
			if (tabulatedPotential) {
				// Using tabulated potentials

				// 'fullLongRange' - determines whether 'cutoff' is used
				// 0 - use cutoff (cell decomposition) [ N*log(N) ]
				// 1 - do not use cutoff [ N^2 ]
				switch (fullLongRange) {
					case 0: // [ N*log(N) ] interactions, + cutoff | decomposition
						if (s % decompPeriod == 0)
							internal->decompose(pos_d);
						energy = internal->computeTabulated(forceInternal_d, pos_d, type_d,
																								bonds_d, bondMap_d,
																								excludes_d, excludeMap_d,
																								angles_d, dihedrals_d,
																								get_energy);
						break;
					default: // [ N^2 ] interactions, no cutoff | decompositions
						energy =
								internal->computeTabulatedFull(forceInternal_d, pos_d, type_d,
																							 bonds_d, bondMap_d,
																							 excludes_d, excludeMap_d,
																							 angles_d, dihedrals_d,
																							 get_energy);
						break;
				}
			
			} else {
				// Not using tabulated potentials.
				
				switch (fullLongRange) {

					case 0: // Use cutoff | cell decomposition.
						if (s % decompPeriod == 0)
							internal->decompose(pos_d);
						energy =
								internal->compute(forceInternal_d, pos_d, type_d, get_energy);
						break;

					case 1: // Do not use cutoff
						energy = internal->computeFull(forceInternal_d,
																					 pos_d, type_d, get_energy);
						break;

					case 2: // Compute only softcore forces.
						energy = internal->computeSoftcoreFull(forceInternal_d,
																									 pos_d, type_d, get_energy);
						break;

					case 3: // Compute only electrostatic forces.
						energy = internal->computeElecFull(forceInternal_d,
																							 pos_d, type_d, get_energy);
						break;
				}
			}
		}

		/* Time force computations.
		rt_timer_stop(cputimer);
		float dt1 = rt_timer_time(cputimer);
		printf("Force Computation Time: %f ms\n", dt1 * 1000);
		rt_timer_start(cputimer);
		// */

		// Make sure the force computation has completed without errors before continuing.
		//gpuErrchk(cudaPeekAtLastError()); // Does not work on old GPUs (like mine). TODO: write a better wrapper around Peek
		gpuErrchk(cudaDeviceSynchronize());

		// printf("  Computed energies\n");

		// int numBlocks = (num * numReplicas) / NUM_THREADS + 1;
		int numBlocks = (num * numReplicas) / NUM_THREADS + (num * numReplicas % NUM_THREADS == 0 ? 0 : 1);
		int tl = temperatureGrid.length();

		// Call the kernel to update the positions of each particle
		updateKernel<<< numBlocks, NUM_THREADS >>>(pos_d, forceInternal_d, type_d,
																							 part_d, kT, kTGrid_d,
																							 electricField, tl, timestep, num,
																							 sys_d, randoGen_d, numReplicas);
		//gpuErrchk(cudaPeekAtLastError()); // Does not work on old GPUs (like mine). TODO: write a better wrapper around Peek
		
		
		/* Time position computations.
		rt_timer_stop(cputimer);
		float dt2 = rt_timer_time(cputimer);
		printf("Position Update Time: %f ms\n", dt2 * 1000);
		*/

		RBC.updateForces();
		/* 	for (int j = 0; j < t->num; j++) { */
		/* computeGridGridForce<<< numBlocks, NUM_THREADS >>>(grid1_d, grid2_d); */
		
		// int numBlocks = (numRB ) / NUM_THREADS + (num * numReplicas % NUM_THREADS == 0 ? 0 : 1);
		
		
		Vector3 force0(0.0f);

		if (imd_on && clientsock && s % outputPeriod == 0) {
			int length;
			if (vmdsock_selread(clientsock, 0) == 1) {
				switch (imd_recv_header(clientsock, &length)) {
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
					default:
						printf("[IMD] Something weird happened. Disconnecting..\n");
						break;
				}
			}
			if (clientsock) {
				gpuErrchk(cudaMemcpy(pos, pos_d, sizeof(Vector3) * num, cudaMemcpyDeviceToHost));
				float* coords = new float[num * 3];
				for (size_t i = 0; i < num; i++) {
					Vector3 p = pos[i];
					coords[3*i] = p.x;
					coords[3*i+1] = p.y;
					coords[3*i+2] = p.z;
				}
				imd_send_fcoords(clientsock, num, coords);
				delete[] coords;
			}
		}

		// Output trajectories (to files)
		if (s % outputPeriod == 0) {
			// Copy particle positions back to CPU
			gpuErrchk(cudaMemcpy(pos, pos_d, sizeof(Vector3) * num * numReplicas,
					cudaMemcpyDeviceToHost));

			// Nanoseconds computed
			t = s*timestep;

			// Loop over all replicas
			for (int repID = 0; repID < numReplicas; ++repID) {

				// *Analysis*: ionic current
				if (currentSegmentZ <= 0.0f) writeCurrent(repID, t);
				else writeCurrentSegment(repID, t, currentSegmentZ);

				// 
				if (numberFluct == 1) updateNameList(); // no need for it here if particles stay the same
														// TODO: doublecheck

				// Write the trajectory.
				writers[repID]->append(pos + (repID*num), name, serial, t, num);
			}

			// Handle particle fluctuations.
			// TODO: Currently, not compatible with replicas. Needs a fix.
			if (numberFluct == 1) updateReservoirs();

			// Store the current positions.
			// We must do this after particles have been added.
			remember(t);
		} // s % outputPeriod == 0


		// Output energy.
		if (get_energy) {
			// Stop the timer.
			rt_timer_stop(timerS);

			// Copy back forces to display (internal only)
			gpuErrchk(cudaMemcpy(&force0, forceInternal_d, sizeof(Vector3), cudaMemcpyDeviceToHost));

			// Nanoseconds computed
			t = s * timestep;

			// Simulation progress and statistics.
			float percent = (100.0f * s) / steps;
			float msPerStep = rt_timer_time(timerS) * 1000.0f / outputEnergyPeriod;
			float nsPerDay = numReplicas * timestep / msPerStep * 864E5f;

			// Nice thousand separator
			setlocale(LC_NUMERIC, "");

			// Do the output
			printf("Step %ld [%.2f%% complete | %.3f ms/step | %.3f ns/day]\n",
						 s, percent, msPerStep, nsPerDay);
		/*	printf("T: %.5g ns | E: %.5g | F: %.5g %.5g %.5g\n",
						 t, energy, force0.x, force0.y, force0.z);
		*/

			// Copy positions from GPU to CPU.
			gpuErrchk(cudaMemcpy(pos, pos_d, sizeof(Vector3) * num * numReplicas,
													 cudaMemcpyDeviceToHost));

			// Write restart files for each replica.
			for (int repID = 0; repID < numReplicas; ++repID)
				writeRestart(repID);

			// restart the timer
			rt_timer_start(timerS);
		} // s % outputEnergyPeriod
	} // done with all Brownian dynamics steps

	// If IMD is on & our socket is still open.
	if (imd_on and clientsock) {
		if (vmdsock_selread(clientsock, 0) == 1) {
			int length;
			switch (imd_recv_header(clientsock, &length)) {
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
					sleep(5);
					break;
				default:
					printf("[IMD] Something weird happened. Disconnecting..\n");
					break;
			}
		}
	}

	// Stop the main timer.
	rt_timer_stop(timer0);

	// Compute performance data.
	const float elapsed = rt_timer_time(timer0); // seconds
	int tot_hrs = (int) std::fmod(elapsed / 3600.0f, 60.0f);
	int tot_min = (int) std::fmod(elapsed / 60.0f, 60.0f);
	float tot_sec	= std::fmod(elapsed, 60.0f);

	printf("Final Step: %d\n", (int) steps);

	printf("Total Run Time: ");
	if (tot_hrs > 0) printf("%dh%dm%.1fs\n", tot_hrs, tot_min, tot_sec);
	else if (tot_min > 0) printf("%dm%.1fs\n", tot_min, tot_sec);
	else printf("%.2fs\n", tot_sec);

	gpuErrchk(cudaMemcpy(pos, pos_d, sizeof(Vector3) * num * numReplicas, cudaMemcpyDeviceToHost));

	// Write the restart file (once again)
	for (int repID = 0; repID < numReplicas; ++repID)
		writeRestart(repID);

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



void GrandBrownTown::writeRestart(int repID) const {
	FILE* out = fopen(restartFiles[repID].c_str(), "w");
	const int offset = repID * num;
	for (int i = 0; i < num; ++i) {
		const int ind = i + offset;
		const Vector3& p = pos[ind];
		fprintf(out, "%d %.10g %.10g %.10g\n", type[ind], p.x, p.y, p.z);
	}
	fclose(out);
}


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
	Vector3 r;
	const BrownianParticleType& pt = part[typ];
	do {
		const float rx = sysDim.x * randoGen->uniform();
		const float ry = sysDim.y * randoGen->uniform();
		const float rz = sysDim.z * randoGen->uniform();
		r = sys->wrap( Vector3(rx, ry, rz) );
	} while (pt.pmf->interpolatePotential(r) > pt.meanPmf);
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
	} while (pt.pmf->interpolatePotential(r) > pt.meanPmf and fabs(r.z) > minZ);
	return r;
}

// -----------------------------------------------------------------------------
// Initialize file for recording ionic current
void GrandBrownTown::newCurrent(int repID) const {
	FILE* out = fopen(outCurrFiles[repID].c_str(), "w");
	fclose(out);
}


// -----------------------------------------------------------------------------
// Record the ionic current flowing through the entire system
void GrandBrownTown::writeCurrent(int repID, float t) const {
	FILE* out = fopen(outCurrFiles[repID].c_str(), "a");
	fprintf(out, "%.10g %.10g %d\n", 0.5f*(t+timeLast), current(t), num);
	fclose(out);
}


// -----------------------------------------------------------------------------
// Record the ionic current in a segment -segZ < z < segZ
void GrandBrownTown::writeCurrentSegment(int repID, float t, float segZ) const {
	FILE* out = fopen(outCurrFiles[repID].c_str(), "a");
	int i;
	fprintf(out, "%.10g ", 0.5f * (t + timeLast));
	for (i = -1; i < numParts; i++)
		fprintf(out, "%.10g ", currentSegment(t,segZ,i));
	fprintf(out, "%d\n", num);
	fclose(out);
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
		internal->updateNumber(pos, num);
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
		internal->updateNumber(pos, num);
}

// -----------------------------------------------------------------------------
// Allocate memory on GPU(s) and copy to device
void GrandBrownTown::copyToCUDA() {
	const size_t tot_num = num * numReplicas;
	gpuErrchk(cudaMalloc(&pos_d, sizeof(Vector3) * tot_num));
	gpuErrchk(cudaMemcpyAsync(pos_d, pos, sizeof(Vector3) * tot_num,
														cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc(&forceInternal_d, sizeof(Vector3) * num * numReplicas));
	gpuErrchk(cudaMemcpyAsync(forceInternal_d, forceInternal, sizeof(Vector3),
														cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc(&randoGen_d, sizeof(Random)));
	gpuErrchk(cudaMemcpyAsync(randoGen_d, randoGen, sizeof(Random),
														cudaMemcpyHostToDevice));
	gpuErrchk(cudaDeviceSynchronize());
}
