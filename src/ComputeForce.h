///////////////////////////////////////////////////////////////////////
// Brownian dynamics base class
// Author: Jeff Comer <jcomer2@illinois.edu>
#pragma once

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

// Simple classes
#include "Restraint.h"
#include "useful.h"
#include "Exclude.h"
#include "Angle.h"
#include "JamesBond.h"
#include "TabulatedPotential.h"
#include "TabulatedAngle.h"
#include "TabulatedDihedral.h"
#include "ProductPotential.h"
#include "GPUManager.h"

// #include <map>

#include <cstdio>
// #include <cuda_runtime.h>
#include <thrust/transform_reduce.h>	// thrust::reduce
#include <thrust/functional.h>				// thrust::plus

#ifdef USE_BOOST
#include <boost/unordered_map.hpp>
typedef boost::unordered_map<String,unsigned int> XpotMap;
inline std::size_t hash_value(String const& s) {
    if (s.length() == 0) return 0;
    return boost::hash_range(s.val(), s.val()+s.length());
}
#else
#include <map>
typedef std::map<String,unsigned int> XpotMap;
inline std::size_t hash_value(String const& s) {
    if (s.length() == 0) return 0;
    return hash_value(s.val());
}
#endif



const unsigned int NUM_THREADS = 256;

// Configuration
class Configuration;

class ComputeForce {
public:
    ComputeForce(const Configuration &c, const int numReplicas);
    ~ComputeForce();
    
	void updateNumber(int newNum);
	void makeTables(const BrownianParticleType* part);

	bool addTabulatedPotential(String fileName, int type0, int type1);
	bool addBondPotential(String fileName, int ind, Bond* bonds, BondAngle* bondAngles);
	bool addAnglePotential(String fileName, int ind, Angle* angles, BondAngle* bondAngles);
	bool addDihedralPotential(String fileName, int ind, Dihedral* dihedrals);

	void decompose();
	
	CellDecomposition getDecomp();
	IndexList decompDim() const;

	float decompCutoff();

	// Does nothing
	int* neighborhood(Vector3 r);

	float computeSoftcoreFull(bool get_energy);
	float computeElecFull(bool get_energy);
	
	float compute(bool get_energy);
	float computeFull(bool get_energy);
	
	//MLog: the commented function doesn't use bondList, uncomment for testing.
	/*float computeTabulated(Vector3* force, Vector3* pos, int* type,
			Bond* bonds, int2* bondMap, Exclude* excludes, int2* excludeMap,
			Angle* angles, Dihedral* dihedrals, bool get_energy);*/
	float computeTabulated(bool get_energy);
	float computeTabulatedFull(bool get_energy);
	
	//MLog: new copy function to allocate memory required by ComputeForce class.
	void copyToCUDA(Vector3* forceInternal, Vector3* pos);
	void copyToCUDA(int simNum, int *type, Bond* bonds, int2* bondMap, Exclude* excludes, int2* excludeMap, Angle* angles, Dihedral* dihedrals, const Restraint* const restraints, const BondAngle* const bondAngles,
			const XpotMap simple_potential_map,
			const std::vector<SimplePotential> simple_potentials,
			const ProductPotentialConf* const product_potential_confs);
        void copyToCUDA(Vector3* forceInternal, Vector3* pos, Vector3* mom);
        void copyToCUDA(Vector3* forceInternal, Vector3* pos, Vector3* mom, float* random);
	
	// void createBondList(int3 *bondList);
	void copyBondedListsToGPU(int3 *bondList, int4 *angleList, int4 *dihedralList, int *dihedralPotList, int4 *bondAngleList);
	    
	//MLog: because of the move of a lot of private variables, some functions get starved necessary memory access to these variables, below is a list of functions that return the specified private variable.
	Vector3* getPos_d()
	{
		return pos_d;
	}
        Vector3* getMom_d() const
        {
            return mom_d;
        }
        float* getRan_d()
        {
            return ran_d;
        }

	Vector3* getForceInternal_d()
	{
		return forceInternal_d;
	}
	void setForceInternalOnDevice(Vector3* f);

	int* getType_d()
	{
		return type_d;
	}

	Bond* getBonds_d()
	{
		return bonds_d;
	}

	int2* getBondMap_d()
	{
		return bondMap_d;
	}

	Exclude* getExcludes_d()
	{
		return excludes_d;
	}

	int2* getExcludeMap_d()
	{
		return excludeMap_d;
	}

	Angle* getAngles_d()
	{
		return angles_d;
	}

	Dihedral* getDihedrals_d()
	{
		return dihedrals_d;
	}

	int3* getBondList_d()
	{
		return bondList_d;
	}
	
        float* getEnergy()
        {
            return energies_d;
        }
	HOST DEVICE
	static EnergyForce coulombForce(Vector3 r, float alpha,float start, float len);

	HOST DEVICE
	static EnergyForce coulombForceFull(Vector3 r, float alpha);

	HOST DEVICE
	static EnergyForce softcoreForce(Vector3 r, float eps, float rad6);

private:
	static GPUManager gpuman;

	// Configuration* c;
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
	float pairlistdist2;
	int2 *pairLists_d;
	cudaTextureObject_t pairLists_tex;

	int *pairTabPotType_d;
	cudaTextureObject_t pairTabPotType_tex;

	int *numPairs_d;

        //Han-Yi Chou
        int *CellNeighborsList;	
	//MLog: List of variables that need to be moved over to ComputeForce class. Members of this list will be set to static to avoid large alterations in working code, thereby allowing us to access these variables easily.
	cudaTextureObject_t neighbors_tex;

	//BrownianParticleType* part;
	//float electricConst;
	//int fullLongRange;
	Vector3* pos_d;
	cudaTextureObject_t pos_tex;
        Vector3* mom_d;
        float*   ran_d;
	Vector3* forceInternal_d;
	int* type_d; 

	Bond* bonds_d; 
	int2* bondMap_d; 

	Exclude* excludes_d; 
	int2* excludeMap_d; 

	Angle* angles_d;
	Dihedral* dihedrals_d;

	int numBondAngles;
	BondAngle* bondAngles_d;
	int4* bondAngleList_d;

    int numProductPotentials;
    float** simple_potential_pots_d;
    SimplePotential* simple_potentials_d;
    int* product_potential_particles_d;
    SimplePotential* product_potentials_d;
    uint2* product_potential_list_d;
    unsigned short* productCount_d;

	int3* bondList_d;
	int4* angleList_d;
	int4* dihedralList_d;
	int* dihedralPotList_d;

	int numRestraints;
	int* restraintIds_d;
	Vector3* restraintLocs_d;
	float* restraintSprings_d;

};
