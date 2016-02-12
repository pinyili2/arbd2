///////////////////////////////////////////////////////////////////////
// Brownian dynamics base class
// Author: Jeff Comer <jcomer2@illinois.edu>
#ifndef COMPUTEFORCE_H
#define COMPUTEFORCE_H

#ifdef __CUDACC__
    #define HOST __host__
    #define DEVICE __device__
#else
    #define HOST 
    #define DEVICE 
#endif

#include "BaseGrid.h"
#include "BrownianParticleType.h"
#include "CellDecomposition.h"
#include "JamesBond.h"
#include "TabulatedPotential.h"
#include "useful.h"
#include "Exclude.h"
#include "Angle.h"
#include "TabulatedAngle.h"
#include "TabulatedDihedral.h"

#include <cuda.h>
#include <thrust/transform_reduce.h>	// thrust::reduce
#include <thrust/functional.h>				// thrust::plus

const unsigned int NUM_THREADS = 256;

class ComputeForce {
public:
	ComputeForce(int num, const BrownianParticleType* part, int numParts,
			const BaseGrid* g, float switchStart, float switchLen,
			float electricConst, int fullLongRange, int numBonds,
			int numTabBondFiles, int numExcludes, int numAngles, int numTabAngleFiles,
			int numDihedrals, int numTabDihedralFiles, int numReplicas = 1);
	~ComputeForce();

	void updateNumber(Vector3* pos, int newNum);
	void makeTables(const BrownianParticleType* part);

	bool addTabulatedPotential(String fileName, int type0, int type1);
	bool addBondPotential(String fileName, int ind, Bond* bonds, Bond* bonds_d);
	bool addAnglePotential(String fileName, int ind, Angle* angles, Angle* angles_d);
	bool addDihedralPotential(String fileName, int ind, Dihedral* dihedrals, Dihedral* dihedrals_d);

	void decompose(Vector3* pos);
	
	CellDecomposition getDecomp();
	IndexList decompDim() const;

	float decompCutoff();

	// Does nothing
	int* neighborhood(Vector3 r);

	float computeSoftcoreFull(Vector3* force, Vector3* pos, int* type, bool get_energy);
	float computeElecFull(Vector3* force, Vector3* pos, int* type, bool get_energy);
	
	float compute(Vector3* force, Vector3* pos, int* type, bool get_energy);
	float computeFull(Vector3* force, Vector3* pos, int* type, bool get_energy);
	
	float computeTabulated(Vector3* force, Vector3* pos, int* type,
			Bond* bonds, int2* bondMap, Exclude* excludes, int2* excludeMap,
			Angle* angles, Dihedral* dihedrals, bool get_energy);
	float computeTabulatedFull(Vector3* force, Vector3* pos, int* type,
			Bond* bonds, int2* bondMap, Exclude* excludes, int2* excludeMap,
			Angle* angles, Dihedral* dihedrals, bool get_energy);

	HOST DEVICE
	static EnergyForce coulombForce(Vector3 r, float alpha,float start, float len);

	HOST DEVICE
	static EnergyForce coulombForceFull(Vector3 r, float alpha);

	HOST DEVICE
	static EnergyForce softcoreForce(Vector3 r, float eps, float rad6);

private:
	int numReplicas;
	int num;
	int numParts;
	int numBonds;
	int numExcludes;
	int numTabBondFiles;
	int numAngles;
	int numTabAngleFiles;
	int numDihedrals;
	int numTabDihedralFiles;
	float *tableEps, *tableRad6, *tableAlpha;
	TabulatedPotential **tablePot;
	TabulatedPotential **tableBond;
	TabulatedAnglePotential **tableAngle;
	TabulatedDihedralPotential **tableDihedral;
	const BaseGrid* sys;
	float switchStart, switchLen, electricConst, cutoff2;
	CellDecomposition decomp;
	int numTablePots;
	float energy;

	// Device Variables
	BaseGrid* sys_d;
	CellDecomposition* decomp_d;
	float *energies_d;
	float *tableEps_d, *tableRad6_d, *tableAlpha_d;
	int gridSize;
	TabulatedPotential **tablePot_d, **tablePot_addr;
	TabulatedPotential **tableBond_d, **tableBond_addr;
	TabulatedAnglePotential **tableAngle_d, **tableAngle_addr;
	TabulatedDihedralPotential **tableDihedral_d, **tableDihedral_addr;

	// Pairlists
	int *pairIds_d;
	int numPairs;
	int numPairs_d;	
};

#endif
