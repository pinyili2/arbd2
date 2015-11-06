// RigidBodyType.h (2015)
// Author: Chris Maffeo <cmaffeo2@illinois.edu>

#pragma once
// #include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "Reservoir.h"
#include "useful.h"
#include "BaseGrid.h"

// Stores particle type's potential grid and other information
struct KeyGrid {
	String key;
	BaseGrid grid;
KeyGrid() :
	key(NULL), grid() { }	
};

class RigidBodyType {
private:
	// Deletes all members
	void clear();
	// void copy(const RigidBodyType& src);

	KeyGrid createKeyGrid(String s);

public:
RigidBodyType(const String& name = "") :
	name(name), num(0),
		reservoir(NULL), mass(1.0f), inertia(), transDamping(),
		rotDamping(), potentialGrids(NULL), densityGrids(NULL),
		potentialGrids_D(NULL), densityGrids_D(NULL) { }
	
	/* RigidBodyType(const RigidBodyType& src) { copy(src); } */
	~RigidBodyType() { clear(); }

	/* RigidBodyType& operator=(const RigidBodyType& src); */

	// crop
	// Crops all BaseGrid members
	// @param  boundries to crop to (x0, y0, z0) -> (x1, y1, z1);
	//         whether to change the origin
	// @return success of function (if false nothing was done)
	/* bool crop(int x0, int y0, int z0, int x1, int y1, int z1, bool keep_origin); */

  void addPotentialGrid(String s);
	void addDensityGrid(String s);
	
public:
	String name;
	int num; // number of particles of this type

	Reservoir* reservoir;
	/* BaseGrid* pmf; */

	float mass;
	Vector3 inertia;
	Vector3 transDamping;
	Vector3 rotDamping;

	/* std::vector<KeyGrid> potentialGrids; */
	/* std::vector<KeyGrid> densityGrids; */
	thrust::host_vector<KeyGrid> potentialGrids;
	thrust::device_vector<KeyGrid> potentialGrids_D;
	thrust::host_vector<KeyGrid> densityGrids;
	thrust::device_vector<KeyGrid> densityGrids_D;
};
