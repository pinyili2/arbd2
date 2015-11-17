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

public:
/* RigidBodyType(const String& name = "") : */
/* 	name(name), num(0), */
/* 		reservoir(NULL), mass(1.0f), inertia(), transDamping(), */
/* 		rotDamping(), potentialGrids(NULL), densityGrids(NULL), */
/* 		potentialGrids_D(NULL), densityGrids_D(NULL) { } */

RigidBodyType(const String& name = "") :
	name(name), num(0),
		reservoir(NULL), mass(1.0f), inertia(), transDamping(),
		rotDamping(), numPotGrids(0), numDenGrids(0) { }
	
	/* RigidBodyType(const RigidBodyType& src) { copy(src); } */
	~RigidBodyType() { clear(); }

	/* RigidBodyType& operator=(const RigidBodyType& src); */

  void addPotentialGrid(String s);
	void addDensityGrid(String s);
	void updateRaw();
	void setDampingCoeffs(float timestep, float tmp_mass, Vector3 tmp_inertia,
												 float tmp_transDamping, float tmp_rotDamping);
	
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

	std::vector<String> potentialGridKeys;
	std::vector<String> densityGridKeys;

	std::vector<BaseGrid> potentialGrids;
	std::vector<BaseGrid> densityGrids;
	
	// for device
	int numPotGrids;
	int numDenGrids;
	RigidBodyGrid* rawPotentialGrids;
	RigidBodyGrid* rawDensityGrids;
	Matrix3* rawPotentialBases;
	Matrix3* rawDensityBases;
	Vector3* rawPotentialOrigins;
	Vector3* rawDensityOrigins;		
};

