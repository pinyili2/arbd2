// RigidBodyType.h (2015)
// Author: Chris Maffeo <cmaffeo2@illinois.edu>

#pragma once
// #include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "Reservoir.h"
#include "useful.h"
#include "BaseGrid.h"

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
		rotDamping() {
		/* potentialGrids = 	*(new thrust::host_vector<BaseGrid>()); */
		/* densityGrids = 	*(new thrust::host_vector<BaseGrid>()); */
		/* potentialGrids = 	*(new thrust::host_vector<BaseGrid>()); */
		/* densityGrids = 	*(new thrust::host_vector<BaseGrid>()); */

		/* thrust::host_vector<BaseGrid> potentialGrids; */
		/* thrust::host_vector<BaseGrid> densityGrids; */
		/* thrust::device_vector<BaseGrid> potentialGrids_D; */
		/* thrust::device_vector<BaseGrid> densityGrids_D; */

		potentialGrids = thrust::host_vector<BaseGrid>();
		densityGrids	 = thrust::host_vector<BaseGrid>();
		potentialGrids = thrust::host_vector<BaseGrid>();
		densityGrids	 = thrust::host_vector<BaseGrid>();

	}


	
	/* RigidBodyType(const RigidBodyType& src) { copy(src); } */
	~RigidBodyType() { clear(); }

	/* RigidBodyType& operator=(const RigidBodyType& src); */

  void addPotentialGrid(String s);
	void addDensityGrid(String s);
	
public:
	String name;
	int num; // number of particles of this type

	Reservoir* reservoir;

	float mass;
	Vector3 inertia;
	Vector3 transDamping;
	Vector3 rotDamping;

	thrust::host_vector<BaseGrid> potentialGrids;
	thrust::host_vector<BaseGrid> densityGrids;
	thrust::device_vector<BaseGrid> potentialGrids_D;
	thrust::device_vector<BaseGrid> densityGrids_D;
};
