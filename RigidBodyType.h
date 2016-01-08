// RigidBodyType.h (2015)
// Author: Chris Maffeo <cmaffeo2@illinois.edu>

#pragma once
#include <vector>
/* #include <thrust/host_vector.h> */
/* #include <thrust/device_vector.h> */
#include "Reservoir.h"
#include "BaseGrid.h"
#include "RigidBodyGrid.h"
#include "useful.h"

#include <cstdio>

class RigidBodyType {
private:
	// Deletes all members
	void clear();
	// void copy(const RigidBodyType& src);

	void addGrid(String s, std::vector<String> &keys, std::vector<BaseGrid> &grids);
	void addScaleFactor(String s, std::vector<String> &keys, std::vector<float> &vals);
	void applyScaleFactors();
	void applyScaleFactor(
		const std::vector<String> &scaleKeys, const std::vector<float> &scaleFactors,
		const std::vector<String> &gridKeys, std::vector<BaseGrid> &grids);
	
public:
/* RigidBodyType(const String& name = "") : */
/* 	name(name), num(0), */
/* 		reservoir(NULL), mass(1.0f), inertia(), transDamping(), */
/* 		rotDamping(), potentialGrids(NULL), densityGrids(NULL), */
/* 		potentialGrids_D(NULL), densityGrids_D(NULL) { } */

RigidBodyType(const String& name = "") :
	name(name), num(0),
	reservoir(NULL), mass(1.0f), inertia(), transDamping(),
	rotDamping(), numPotGrids(0), numDenGrids(0), numPmfs(0),
	initPos(), initRot(Matrix3(1.0f))  { }
	
	/* RigidBodyType(const RigidBodyType& src) { copy(src); } */
	~RigidBodyType() { clear(); }

	/* RigidBodyType& operator=(const RigidBodyType& src); */

  void addPotentialGrid(String s);
	void addDensityGrid(String s);
	void addPMF(String s);
  void scalePotentialGrid(String s);
	void scaleDensityGrid(String s);
	void scalePMF(String s);

	void updateRaw();
	void setDampingCoeffs(float timestep);
	
public:

	
	String name;
	int num; // number of particles of this type

	Reservoir* reservoir;

	float mass;
	Vector3 inertia;
	Vector3 transDamping;
	Vector3 rotDamping;
	Vector3 transForceCoeff;
	Vector3 rotTorqueCoeff;

	Vector3 initPos;	
	Matrix3 initRot;
	
	std::vector<String> potentialGridKeys;
	std::vector<String> densityGridKeys;
	std::vector<String> pmfKeys;

	std::vector<BaseGrid> potentialGrids;
	std::vector<BaseGrid> densityGrids;
	std::vector<BaseGrid> pmfs;

	std::vector<String> potentialGridScaleKeys;
	std::vector<String> densityGridScaleKeys;
	std::vector<String> pmfScaleKeys;

	std::vector<float> potentialGridScale;
	std::vector<float> densityGridScale;
	std::vector<float> pmfScale;

	
	// RBTODO: clear std::vectors after initialization, (but keep offsets)
	// duplicates of std::vector grids for device
	int numPotGrids;
	int numDenGrids;
	int numPmfs;
	RigidBodyGrid* rawPotentialGrids;
	RigidBodyGrid* rawDensityGrids;
	BaseGrid* rawPmfs;
	Matrix3* rawPotentialBases;
	Matrix3* rawDensityBases;
	Vector3* rawPotentialOrigins;
	Vector3* rawDensityOrigins;		

	// device pointers
	RigidBodyGrid** rawPotentialGrids_d;
	RigidBodyGrid** rawDensityGrids_d;
	RigidBodyGrid** rawPmfs_d;
	
};

