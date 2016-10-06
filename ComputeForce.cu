///////////////////////////////////////////////////////////////////////
// Brownian dynamics base class
// Author: Jeff Comer <jcomer2@illinois.edu>

#include "ComputeForce.h"
#include "ComputeForce.cuh"
#include <cuda_profiler_api.h>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

cudaEvent_t start, stop;

void runSort(int2 *d1, int *d2, float *key,
				int2 *scratch1, int  *scratch2, float *scratchKey,
				unsigned int count);

ComputeForce::ComputeForce(int num, const BrownianParticleType part[],
													 int numParts, const BaseGrid* g, float switchStart,
													 float switchLen, float electricConst,
													 int fullLongRange, int numBonds, int numTabBondFiles,
													 int numExcludes, int numAngles, int numTabAngleFiles,
													 int numDihedrals, int numTabDihedralFiles,
													 int numReplicas) :
		num(num), numParts(numParts), sys(g), switchStart(switchStart),
		switchLen(switchLen), electricConst(electricConst),
		cutoff2((switchLen + switchStart) * (switchLen + switchStart)),
		decomp(g->getBox(), g->getOrigin(), switchStart + switchLen, numReplicas),
		numBonds(numBonds), numTabBondFiles(numTabBondFiles),
		numExcludes(numExcludes), numAngles(numAngles),
		numTabAngleFiles(numTabAngleFiles), numDihedrals(numDihedrals),
		numTabDihedralFiles(numTabDihedralFiles), numReplicas(numReplicas) {
	// Allocate the parameter tables.
	decomp_d = NULL;

	tableEps = new float[numParts * numParts];
	tableRad6 = new float[numParts * numParts];
	tableAlpha = new float[numParts * numParts];

	const size_t tableSize = sizeof(float) * numParts * numParts;
	gpuErrchk(cudaMalloc(&tableEps_d, tableSize));
	gpuErrchk(cudaMalloc(&tableRad6_d, tableSize));
	gpuErrchk(cudaMalloc(&tableAlpha_d, tableSize));
	gpuErrchk(cudaMalloc(&sys_d, sizeof(BaseGrid)));
	gpuErrchk(cudaMemcpyAsync(sys_d, sys, sizeof(BaseGrid), cudaMemcpyHostToDevice));
	// Form the parameter tables.
	makeTables(part);

	gpuErrchk(cudaMemcpyAsync(tableAlpha_d, tableAlpha, tableSize,
			cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyAsync(tableEps_d, tableEps, tableSize,
			cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyAsync(tableRad6_d, tableRad6, tableSize,
			cudaMemcpyHostToDevice));

	// Create the potential table
	tablePot = new TabulatedPotential*[numParts * numParts];
	tablePot_addr = new TabulatedPotential*[numParts * numParts];
	for (int i = 0; i < numParts*numParts; i++) {
		tablePot_addr[i] = NULL;
		tablePot[i] = NULL;
	}
	gpuErrchk(cudaMalloc(&tablePot_d, sizeof(TabulatedPotential*) * numParts * numParts));

	// Create the bond table
	tableBond = new TabulatedPotential*[numTabBondFiles];
	tableBond_addr = new TabulatedPotential*[numTabBondFiles];
	for (int i = 0; i < numTabBondFiles; i++) {
		tableBond_addr[i] = NULL;
		tableBond[i] = NULL;
	}
	gpuErrchk(cudaMalloc(&tableBond_d, sizeof(TabulatedPotential*) * numTabBondFiles));

	// Create the angle table
	tableAngle = new TabulatedAnglePotential*[numTabAngleFiles];
	tableAngle_addr = new TabulatedAnglePotential*[numTabAngleFiles];
	for (int i = 0; i < numTabAngleFiles; i++) {
		tableAngle_addr[i] = NULL;
		tableAngle[i] = NULL;
	}
	gpuErrchk(cudaMalloc(&tableAngle_d, sizeof(TabulatedAnglePotential*) * numTabAngleFiles));

	// Create the dihedral table
	tableDihedral = new TabulatedDihedralPotential*[numTabDihedralFiles];
	tableDihedral_addr = new TabulatedDihedralPotential*[numTabDihedralFiles];
	for (int i = 0; i < numTabDihedralFiles; i++) {
		tableDihedral_addr[i] = NULL;
		tableDihedral[i] = NULL;
	}
	gpuErrchk(cudaMalloc(&tableDihedral_d, sizeof(TabulatedDihedralPotential*) * numTabDihedralFiles));

	//Calculate the number of blocks the grid should contain
	gridSize =  num / NUM_THREADS + 1;

	// Create and allocate the energy arrays
	gpuErrchk(cudaMalloc(&energies_d, sizeof(float) * num));
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
}

ComputeForce::~ComputeForce() {
	delete[] tableEps;
	delete[] tableRad6;
	delete[] tableAlpha;
	cudaFree(tableEps_d);
	cudaFree(tableAlpha_d);
	cudaFree(tableRad6_d);

	for (int j = 0; j < numParts * numParts; ++j)
		delete tablePot[j];
	delete[] tablePot;
	delete[] tablePot_addr;

	for (int j = 0; j < numTabBondFiles; ++j)
		delete tableBond[j];
	delete[] tableBond;
	delete[] tableBond_addr;
	gpuErrchk(cudaFree(tableBond_d));

	for (int j = 0; j < numTabAngleFiles; ++j)
		if (tableAngle[j] != NULL)
			delete tableAngle[j];
	delete[] tableAngle;
	delete[] tableAngle_addr;

	if(type_d != NULL)
	{
		gpuErrchk(cudaFree(tableAngle_d));

		gpuErrchk(cudaFree(energies_d));

		gpuErrchk(cudaFree(sys_d));

		gpuErrchk( cudaFree(pos_d) );
		gpuErrchk( cudaFree(forceInternal_d) );
		gpuErrchk( cudaFree(type_d) );
		gpuErrchk( cudaFree(bonds_d) );
		gpuErrchk( cudaFree(bondMap_d) );
		gpuErrchk( cudaFree(excludes_d) );
		gpuErrchk( cudaFree(excludeMap_d) );
		gpuErrchk( cudaFree(angles_d) );
		gpuErrchk( cudaFree(dihedrals_d) );
		gpuErrchk( cudaFree(bondList_d) );
	}
}

void ComputeForce::updateNumber(int newNum) {
	if (newNum == num or newNum < 0) return;

	// Set the new number.
	num = newNum;

	// Reallocate the neighbor list.
	//delete[] neigh;
	//neigh = new IndexList[num];
	decompose();

	printf("updateNumber() called\n");
	// Reallocate CUDA arrays

	// Recalculate the number of blocks in the grid
	gridSize = 0;
	while ((int)sqrt(NUM_THREADS) * gridSize < num)
		++gridSize;

	gpuErrchk(cudaFree(energies_d));
	gpuErrchk(cudaMalloc(&energies_d, sizeof(float) * gridSize));
}

void ComputeForce::makeTables(const BrownianParticleType part[]) {
	for (int i = 0; i < numParts; ++i) {
		const BrownianParticleType& pi = part[i];
		for (int j = 0; j < numParts; ++j) {
			const BrownianParticleType& pj = part[j];
			int ind = i * numParts + j;
			tableEps[ind] = sqrtf(pi.eps * pj.eps);
			float r = pi.radius + pj.radius;
			tableRad6[ind] = r * r * r * r * r * r;
			tableAlpha[ind] = electricConst * pi.charge * pj.charge;
		}
	}
}

bool ComputeForce::addTabulatedPotential(String fileName, int type0, int type1) {
	if (type0 < 0 or type0 >= numParts) return false;
	if (type1 < 0 or type1 >= numParts) return false;

	int ind = type0 + type1 * numParts;
	int ind1 = type1 + type0 * numParts;

	// If an entry already exists for this particle type, delete it
	if (tablePot[ind] != NULL) {
		delete tablePot[ind];
		gpuErrchk(cudaFree(tablePot_addr[ind]));
		tablePot[ind] = NULL;
		tablePot_addr[ind] = NULL;
	}
	if (tablePot[ind1] != NULL) {
		gpuErrchk(cudaFree(tablePot_addr[ind1]));
		delete tablePot[ind1];
		tablePot[ind1] = NULL;
		tablePot_addr[ind1] = NULL;
	}

	tablePot[ind] = new TabulatedPotential(fileName);
	tablePot[ind]->truncate(switchStart, sqrtf(cutoff2), 0.0f);
	tablePot[ind1] = new TabulatedPotential(*(tablePot[ind]));

	TabulatedPotential* t = new TabulatedPotential(*tablePot[ind]);

	// Copy tablePot[ind] to the device
	float *v0, *v1, *v2, *v3;
	size_t sz_n = sizeof(float) * tablePot[ind]->n;
	gpuErrchk(cudaMalloc(&v0, sz_n));
	gpuErrchk(cudaMalloc(&v1, sz_n));
	gpuErrchk(cudaMalloc(&v2, sz_n));
	gpuErrchk(cudaMalloc(&v3, sz_n));
	gpuErrchk(cudaMemcpyAsync(v0, tablePot[ind]->v0, sz_n, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyAsync(v1, tablePot[ind]->v1, sz_n, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyAsync(v2, tablePot[ind]->v2, sz_n, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyAsync(v3, tablePot[ind]->v3, sz_n, cudaMemcpyHostToDevice));
	t->v0 = v0; t->v1 = v1;
	t->v2 = v2; t->v3 = v3;
	gpuErrchk(cudaMalloc(&tablePot_addr[ind], sizeof(TabulatedPotential)));
	gpuErrchk(cudaMemcpy(tablePot_addr[ind], t, sizeof(TabulatedPotential), cudaMemcpyHostToDevice));
	t->v0 = NULL; t->v1 = NULL;
	t->v2 = NULL; t->v3 = NULL;
	delete t;
	/** Same thing for ind1 **/
	t = new TabulatedPotential(*tablePot[ind1]);
	sz_n = sizeof(float) * tablePot[ind1]->n;
	gpuErrchk(cudaMalloc(&v0, sz_n));
	gpuErrchk(cudaMalloc(&v1, sz_n));
	gpuErrchk(cudaMalloc(&v2, sz_n));
	gpuErrchk(cudaMalloc(&v3, sz_n));
	gpuErrchk(cudaMemcpyAsync(v0, tablePot[ind1]->v0, sz_n, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyAsync(v1, tablePot[ind1]->v1, sz_n, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyAsync(v2, tablePot[ind1]->v2, sz_n, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyAsync(v3, tablePot[ind1]->v3, sz_n, cudaMemcpyHostToDevice));
	t->v0 = v0; t->v1 = v1;
	t->v2 = v2; t->v3 = v3;
	gpuErrchk(cudaMalloc(&tablePot_addr[ind1], sizeof(TabulatedPotential)));
	gpuErrchk(cudaMemcpy(tablePot_addr[ind1], t, sizeof(TabulatedPotential), cudaMemcpyHostToDevice));
	t->v0 = NULL; t->v1 = NULL;
	t->v2 = NULL; t->v3 = NULL;
	delete t;
	gpuErrchk(cudaMemcpy(tablePot_d, tablePot_addr,
			sizeof(TabulatedPotential*) * numParts * numParts, cudaMemcpyHostToDevice));

	return true;
}

bool ComputeForce::addBondPotential(String fileName, int ind, Bond bonds[])
{
	if (tableBond[ind] != NULL) {
		delete tableBond[ind];
		gpuErrchk(cudaFree(tableBond_addr[ind]));
		tableBond[ind] = NULL;
		tableBond_addr[ind] = NULL;
	}
	tableBond[ind] = new TabulatedPotential(fileName);
	tableBond[ind]->truncate(switchStart, sqrtf(cutoff2), 0.0f);

	for (int i = 0; i < numBonds; ++i)
		if (bonds[i].fileName == fileName)
			bonds[i].tabFileIndex = ind;

	gpuErrchk(cudaMemcpyAsync(bonds_d, bonds, sizeof(Bond) * numBonds, cudaMemcpyHostToDevice));

	// Copy tableBond[ind] to the device
	float *v0, *v1, *v2, *v3;
	size_t sz_n = sizeof(float) * tableBond[ind]->n;
	gpuErrchk(cudaMalloc(&v0, sz_n));
	gpuErrchk(cudaMalloc(&v1, sz_n));
	gpuErrchk(cudaMalloc(&v2, sz_n));
	gpuErrchk(cudaMalloc(&v3, sz_n));
	gpuErrchk(cudaMemcpyAsync(v0, tableBond[ind]->v0, sz_n, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyAsync(v1, tableBond[ind]->v1, sz_n, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyAsync(v2, tableBond[ind]->v2, sz_n, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyAsync(v3, tableBond[ind]->v3, sz_n, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc(&tableBond_addr[ind], sizeof(TabulatedPotential)));
	TabulatedPotential t = TabulatedPotential(*tableBond[ind]);
	t.v0 = v0; t.v1 = v1;
	t.v2 = v2; t.v3 = v3;
	gpuErrchk(cudaMemcpyAsync(tableBond_addr[ind], &t,
			sizeof(TabulatedPotential), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(tableBond_d, tableBond_addr,
			sizeof(TabulatedPotential*) * numTabBondFiles, cudaMemcpyHostToDevice));
	t.v0 = NULL; t.v1 = NULL;
	t.v2 = NULL; t.v3 = NULL;
	return true;
}

bool ComputeForce::addAnglePotential(String fileName, int ind, Angle* angles) {
	if (tableAngle[ind] != NULL) {
		delete tableAngle[ind];
		gpuErrchk(cudaFree(tableAngle_addr[ind]));
		tableAngle[ind] = NULL;
		tableAngle_addr[ind] = NULL;
	}

	tableAngle[ind] = new TabulatedAnglePotential(fileName);
	TabulatedAnglePotential *t = new TabulatedAnglePotential(*tableAngle[ind]);

	// Copy tableAngle[ind] to the device
	float *pot;
	int size = tableAngle[ind]->size;
	gpuErrchk(cudaMalloc(&pot, sizeof(float) * size));
	gpuErrchk(cudaMemcpyAsync(pot, tableAngle[ind]->pot, sizeof(float) * size, cudaMemcpyHostToDevice));
	t->pot = pot;
	gpuErrchk(cudaMalloc(&tableAngle_addr[ind], sizeof(TabulatedAnglePotential)));
	gpuErrchk(cudaMemcpy(tableAngle_addr[ind], t, sizeof(TabulatedAnglePotential), cudaMemcpyHostToDevice));
	t->pot = NULL;
	delete t;

	gpuErrchk(cudaMemcpyAsync(tableAngle_d, tableAngle_addr,
			sizeof(TabulatedAnglePotential*) * numTabAngleFiles, cudaMemcpyHostToDevice));

	for (int i = 0; i < numAngles; i++)
		if (angles[i].fileName == fileName)
			angles[i].tabFileIndex = ind;

	gpuErrchk(cudaMemcpy(angles_d, angles, sizeof(Angle) * numAngles,
			cudaMemcpyHostToDevice));
	return true;
}

bool ComputeForce::addDihedralPotential(String fileName, int ind, Dihedral dihedrals[])
{
	for (int i = 0; i < numDihedrals; i++)
		if (dihedrals[i].fileName == fileName)
			dihedrals[i].tabFileIndex = ind;

	gpuErrchk(cudaMemcpyAsync(dihedrals_d, dihedrals, sizeof(Dihedral) * numDihedrals,
			cudaMemcpyHostToDevice));

	if (tableDihedral[ind] != NULL) {
		delete tableDihedral[ind];
		gpuErrchk(cudaFree(tableDihedral_addr[ind]));
		tableDihedral[ind] = NULL;
		tableDihedral_addr[ind] = NULL;
	}

	tableDihedral[ind] = new TabulatedDihedralPotential(fileName);
	TabulatedDihedralPotential t = TabulatedDihedralPotential(*tableDihedral[ind]);

	// Copy tableAngle[ind] to the device
	float *pot;
	int size = tableDihedral[ind]->size;
	gpuErrchk(cudaMalloc(&pot, sizeof(float) * size));
	gpuErrchk(cudaMemcpyAsync(pot, tableDihedral[ind]->pot,
			sizeof(float) * size, cudaMemcpyHostToDevice));
	t.pot = pot;

	gpuErrchk(cudaMalloc(&tableDihedral_addr[ind], sizeof(TabulatedDihedralPotential)));
	gpuErrchk(cudaMemcpyAsync(tableDihedral_addr[ind], &t,
			sizeof(TabulatedDihedralPotential), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(tableDihedral_d, tableDihedral_addr,
			sizeof(TabulatedDihedralPotential*) * numTabDihedralFiles, cudaMemcpyHostToDevice));
	t.pot = NULL;
	return true;
}

void ComputeForce::decompose() {
	gpuErrchk( cudaProfilerStart() );
	// Reset the cell decomposition.
	bool newDecomp = false;
	if (decomp_d)
		cudaFree(decomp_d);
	else
		newDecomp = true;
		
	decomp.decompose_d(pos_d, num);
	decomp_d = decomp.copyToCUDA();

	// Update pairlists using cell decomposition (not sure this is really needed or good) 
	//RBTODO updatePairlists<<< nBlocks, NUM_THREADS >>>(pos_d, num, numReplicas, sys_d, decomp_d);	


	/* size_t free, total; */
	/* { */
	/* 	cuMemGetInfo(&free,&total); */
	/* 	printf("Free memory: %zu / %zu\n", free, total); */
	/* } */
	
	// initializePairlistArrays
	int nCells = decomp.nCells.x * decomp.nCells.y * decomp.nCells.z;
	int blocksPerCell = 10;
	if (newDecomp) {
		// RBTODO: free memory elsewhere
		// allocate device data
		// initializePairlistArrays<<< 1, 32 >>>(10*nCells*blocksPerCell);
		const int maxPairs = 1<<25;
		gpuErrchk(cudaMalloc(&numPairs_d,       sizeof(int)));

		gpuErrchk(cudaMalloc(&pairLists_d,      sizeof(int2)*maxPairs));
		gpuErrchk(cudaMalloc(&pairTabPotType_d, sizeof(int)*maxPairs));

		gpuErrchk(cudaDeviceSynchronize());
	}

	
	/* cuMemGetInfo(&free,&total); */
	/* printf("Free memory: %zu / %zu\n", free, total); */
	
	const int NUMTHREADS = 128;
	//const size_t nBlocks = (num * numReplicas) / NUM_THREADS + 1;
	const size_t nBlocks = nCells*blocksPerCell;

	/* clearPairlists<<< 1, 32 >>>(pos, num, numReplicas, sys_d, decomp_d); */
	/* gpuErrchk(cudaDeviceSynchronize()); */
	/* pairlistTest<<< nBlocks, NUMTHREADS >>>(pos, num, numReplicas, */
	/* 																					 sys_d, decomp_d, nCells, blocksPerCell, */
	/* 																					 numPairs_d, pairListListI_d, pairListListJ_d); */
	/* gpuErrchk(cudaDeviceSynchronize());	 */

	int tmp = 0;
	gpuErrchk(cudaMemcpyAsync(numPairs_d, &tmp,	sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaDeviceSynchronize());
	
	float pairlistdist2 = (sqrt(cutoff2) + 2.0f);
	pairlistdist2 = pairlistdist2*pairlistdist2;
	
	createPairlists<<< 2048, 64 >>>(pos_d, num, numReplicas, sys_d, decomp_d, nCells, numPairs_d, pairLists_d, numParts, type_d, pairTabPotType_d, pairlistdist2);
	/* createPairlistsOld<<< nBlocks, NUMTHREADS >>>(pos, num, numReplicas, */
	/* 																					 sys_d, decomp_d, nCells, blocksPerCell, */
	/* 																					 numPairs_d, pairLists_d, */
	/* 																					 numParts, type, pairTabPotType_d, pairlistdist2); */

	gpuErrchk(cudaDeviceSynchronize()); /* RBTODO: sync needed here? */
	// if (false)
	{ // sort pairlist
		int numPairs;
		gpuErrchk(cudaMemcpyAsync( &numPairs, numPairs_d, sizeof(int), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaDeviceSynchronize()); /* RBTODO: sync needed here? */
		printf("here, %d pairs\n", numPairs);
		/* runSort(pairLists_d, pairTabPotType_d, pairDists_d, */
		/* 				pairLists_s, pairTabPotType_s, pairDists_s, */
		/* 				numPairs); */
		/* printf("done\n"); */
		
		/* // RBTODO: sort pairListInd as well!!! (i.e. roll your own sort!) */
		/* // thrust::sort_by_key( pairDists_d, pairDists_d+numPairs_d, pairLists_d ); */
		/* // thrust::sort_by_key( pairDists_d, pairDists_d+numPairs_d, pairLists_d ); */
		/* gpuErrchk(cudaDeviceSynchronize()); /\* RBTODO: sync needed here? *\/ */
	}
}

IndexList ComputeForce::decompDim() const {
	IndexList ret;
	ret.add(decomp.getNx());
	ret.add(decomp.getNy());
	ret.add(decomp.getNz());
	return ret;
}

CellDecomposition ComputeForce::getDecomp() { return decomp; }

float ComputeForce::decompCutoff() { return decomp.getCutoff(); }

// TODO: Fix this
int* ComputeForce::neighborhood(Vector3 r) {
	// return decomp.getCell(r)->getNeighbors();
	return NULL;
}

float ComputeForce::computeFull(bool get_energy) {
	float energy = 0.0f;
	gridSize = (num * numReplicas) / NUM_THREADS + 1;
	dim3 numBlocks(gridSize, 1, 1);
	dim3 numThreads(NUM_THREADS, 1, 1);

	// Call the kernel to calculate forces
	computeFullKernel<<< numBlocks, numThreads >>>(forceInternal_d, pos_d, type_d, tableAlpha_d,
		tableEps_d, tableRad6_d, num, numParts, sys_d, energies_d, gridSize,
		numReplicas, get_energy);

	// Calculate energy based on the array created by the kernel
	if (get_energy) {
		gpuErrchk(cudaDeviceSynchronize());
		thrust::device_ptr<float> en_d(energies_d);
		energy = thrust::reduce(en_d, en_d + num);
	}

	return energy;
}

float ComputeForce::computeSoftcoreFull(bool get_energy) {
	float energy = 0.0f;
	gridSize = (num * numReplicas) / NUM_THREADS + 1;
	dim3 numBlocks(gridSize, 1, 1);
	dim3 numThreads(NUM_THREADS, 1, 1);

	// Call the kernel to calculate forces
	computeSoftcoreFullKernel<<<numBlocks, numThreads>>>(forceInternal_d, pos_d, type_d,
			tableEps_d, tableRad6_d, num, numParts, sys_d, energies_d, gridSize,
			numReplicas, get_energy);

	// Calculate energy based on the array created by the kernel
	if (get_energy) {
		cudaDeviceSynchronize();
		thrust::device_ptr<float> en_d(energies_d);
		energy = thrust::reduce(en_d, en_d + num);
	}

	return energy;
}

float ComputeForce::computeElecFull(bool get_energy) {
	float energy = 0.0f;

	gridSize = num/NUM_THREADS + 1;
	dim3 numBlocks(gridSize, 1, 1);
	dim3 numThreads(NUM_THREADS, 1, 1);

	// Call the kernel to calculate forces
	computeElecFullKernel<<<numBlocks, numThreads>>>(forceInternal_d, pos_d, type_d,
			tableAlpha_d, num, numParts, sys_d, energies_d, gridSize, numReplicas,
			get_energy);

	// Calculate energy based on the array created by the kernel
	if (get_energy) {
		gpuErrchk(cudaDeviceSynchronize());
		thrust::device_ptr<float> en_d(energies_d);
		energy = thrust::reduce(en_d, en_d + num);
	}

	return energy;
}


float ComputeForce::compute(bool get_energy) {
	float energy = 0.0f;

	gridSize = (num * numReplicas) / NUM_THREADS + 1;
	dim3 numBlocks(gridSize, 1, 1);
	dim3 numThreads(NUM_THREADS, 1, 1);

	// Call the kernel to calculate forces
	computeKernel<<<numBlocks, numThreads>>>(forceInternal_d, pos_d, type_d,
			tableAlpha_d, tableEps_d, tableRad6_d, num, numParts, sys_d,
			decomp_d, energies_d, switchStart, switchLen, gridSize, numReplicas,
			get_energy);

	gpuErrchk(cudaDeviceSynchronize());
	// Calculate the energy based on the array created by the kernel
	if (get_energy) {
		gpuErrchk(cudaDeviceSynchronize());
		thrust::device_ptr<float> en_d(energies_d);
		energy = thrust::reduce(en_d, en_d + num);
	}

	return energy;
}

//MLog: added Bond* bondList to the list of passed in variables.
/*float ComputeForce::computeTabulated(Vector3* force, Vector3* pos, int* type,
		Bond* bonds, int2* bondMap, Exclude* excludes, int2* excludeMap,
		Angle* angles, Dihedral* dihedrals, bool get_energy, Bond* bondList) {*/
float ComputeForce::computeTabulated(bool get_energy) {
	float energy = 0.0f;

	gridSize = (num * numReplicas) / NUM_THREADS + 1;
	dim3 numBlocks(gridSize, 1, 1);
	dim3 numThreads(NUM_THREADS, 1, 1);
	
	// Call the kernel to calculate the forces
	// int nb = (decomp.nCells.x * decomp.nCells.y * decomp.nCells.z);
	// int nb = (1+(decomp.nCells.x * decomp.nCells.y * decomp.nCells.z)) * 75; /* RBTODO: number of pairLists */
	const int nb = 800;
	// printf("ComputeTabulated\n");
	gpuErrchk(cudaDeviceSynchronize());

	// RBTODO: get_energy
	//if (get_energy)
	if (false) 
	{
		clearEnergies<<< nb, numThreads >>>(energies_d,num);
		gpuErrchk(cudaDeviceSynchronize());
		computeTabulatedEnergyKernel<<< nb, numThreads >>>(forceInternal_d, pos_d, sys_d,
						cutoff2, numPairs_d, pairLists_d, pairTabPotType_d, tablePot_d, energies_d);
	}
	
	else
	{
		computeTabulatedKernel<<< nb, numThreads >>>(forceInternal_d, pos_d, sys_d,
						cutoff2, numPairs_d, pairLists_d, pairTabPotType_d, tablePot_d);
	}
	/* printPairForceCounter<<<1,32>>>(); */

	//Mlog: the commented function doesn't use bondList, uncomment for testing.
	//if(bondMap_d != NULL && tableBond_d != NULL)
	if(bondList_d != NULL && tableBond_d != NULL)

	{
	    //computeTabulatedBonds <<<numBlocks, numThreads>>> ( force, pos, num, numParts, sys_d, bonds, bondMap_d, numBonds, numReplicas, energies_d, get_energy, tableBond_d);
	computeTabulatedBonds <<<nb, numThreads>>> ( forceInternal_d, pos_d, sys_d, numReplicas*numBonds/2, bondList_d, tableBond_d);
	}

	if (angleList_d != NULL && tableAngle_d != NULL)
		computeTabulatedAngles<<<nb, numThreads>>>(forceInternal_d, pos_d, sys_d, numAngles*numReplicas, angleList_d, tableAngle_d);

	if (dihedralList_d != NULL && tableDihedral_d != NULL)
		computeTabulatedDihedrals<<<nb, numThreads>>>(forceInternal_d, pos_d, sys_d, numDihedrals*numReplicas, dihedralList_d, dihedralPotList_d, tableDihedral_d);


	// Calculate the energy based on the array created by the kernel
	// TODO: return energy
	// if (get_energy) {
	// 	gpuErrchk(cudaDeviceSynchronize());
	// 	thrust::device_ptr<float> en_d(energies_d);
	// 	energy = thrust::reduce(en_d, en_d + num);
	// }

	return energy;
}

float ComputeForce::computeTabulatedFull(bool get_energy) {
	energy = 0.0f;

	gridSize = (num * numReplicas) / NUM_THREADS + 1;
	dim3 numBlocks(gridSize, 1, 1);
	dim3 numThreads(NUM_THREADS, 1, 1);

	// Call the kernel to calculate forces
	computeTabulatedFullKernel<<< numBlocks, numThreads >>>(forceInternal_d, pos_d, type_d,	tablePot_d, tableBond_d, num, numParts, sys_d, bonds_d, bondMap_d, numBonds, excludes_d, excludeMap_d, numExcludes, energies_d, gridSize, numReplicas, get_energy, angles_d);
	gpuErrchk(cudaDeviceSynchronize());

	computeAngles<<< numBlocks, numThreads >>>(forceInternal_d, pos_d, angles_d, tableAngle_d,
																						 numAngles, num, sys_d, energies_d,
																						 get_energy);
	gpuErrchk(cudaDeviceSynchronize());
	computeDihedrals<<< numBlocks, numThreads >>>(forceInternal_d, pos_d, dihedrals_d,
																							  tableDihedral_d, numDihedrals,
																								num, sys_d, energies_d,
																								get_energy);
	// Calculate the energy based on the array created by the kernel
	if (get_energy) {
		gpuErrchk(cudaDeviceSynchronize());
		thrust::device_ptr<float> en_d(energies_d);
		energy = thrust::reduce(en_d, en_d + num);
	}

	return energy;
}

void ComputeForce::copyToCUDA(Vector3* forceInternal, Vector3* pos)
{
	const size_t tot_num = num * numReplicas;

	gpuErrchk(cudaMalloc(&pos_d, sizeof(Vector3) * tot_num));
	gpuErrchk(cudaMemcpyAsync(pos_d, pos, sizeof(Vector3) * tot_num, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc(&forceInternal_d, sizeof(Vector3) * num * numReplicas));
	gpuErrchk(cudaMemcpyAsync(forceInternal_d, forceInternal, sizeof(Vector3) * tot_num, cudaMemcpyHostToDevice));

	gpuErrchk(cudaDeviceSynchronize());
}

void ComputeForce::copyToCUDA(int simNum, int *type, Bond* bonds, int2* bondMap, Exclude* excludes, int2* excludeMap, Angle* angles, Dihedral* dihedrals)
{
	// type_d
	gpuErrchk(cudaMalloc(&type_d, sizeof(int) * num * simNum));
	gpuErrchk(cudaMemcpyAsync(type_d, type, sizeof(int) * num * simNum, cudaMemcpyHostToDevice));
	
	if (numBonds > 0)
	{
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
	}

	gpuErrchk(cudaDeviceSynchronize());
}

// void ComputeForce::createBondList(int3 *bondList)
// {
// 	size_t size = (numBonds / 2) * numReplicas * sizeof(int3);
// 	gpuErrchk( cudaMalloc( &bondList_d, size ) );
// 	gpuErrchk( cudaMemcpyAsync( bondList_d, bondList, size, cudaMemcpyHostToDevice) );

// 	for(int i = 0 ; i < (numBonds / 2) * numReplicas ; i++)
// 	{
// 		cout << "Displaying: bondList_d["<< i <<"].x = " << bondList[i].x << ".\n"
// 			<< "Displaying: bondList_d["<< i <<"].y = " << bondList[i].y << ".\n"
// 			<< "Displaying: bondList_d["<< i <<"].z = " << bondList[i].z << ".\n";

// 	}
// }

void ComputeForce::copyBondedListsToGPU(int3 *bondList, int4 *angleList, int4 *dihedralList, int *dihedralPotList) {

	
	size_t size;

	if (numBonds > 0) {
	size = (numBonds / 2) * numReplicas * sizeof(int3);
	gpuErrchk( cudaMalloc( &bondList_d, size ) );
	gpuErrchk( cudaMemcpyAsync( bondList_d, bondList, size, cudaMemcpyHostToDevice) );
	}
	
	if (numAngles > 0) {
    size = numAngles * numReplicas * sizeof(int4);
    gpuErrchk( cudaMalloc( &angleList_d, size ) );
    gpuErrchk( cudaMemcpyAsync( angleList_d, angleList, size, cudaMemcpyHostToDevice) );
	}
	
	if (numDihedrals > 0) {
    size = numDihedrals * numReplicas * sizeof(int4);
    gpuErrchk( cudaMalloc( &dihedralList_d, size ) );
    gpuErrchk( cudaMemcpyAsync( dihedralList_d, dihedralList, size, cudaMemcpyHostToDevice) );

    size = numDihedrals * numReplicas * sizeof(int);
    gpuErrchk( cudaMalloc( &dihedralPotList_d, size ) );
    gpuErrchk( cudaMemcpyAsync( dihedralPotList_d, dihedralPotList, size, cudaMemcpyHostToDevice) );
	}
}
