//////////////////////////////////////////////////////////////////////
// Copy of BaseGrid with some modificaitons
// 
#ifndef RBBASEGRID_H
#define RBBASEGRID_H
// #pragma once

#ifdef __CUDACC__
    #define HOST __host__
    #define DEVICE __device__
#else
    #define HOST 
    #define DEVICE 
#endif

#include "BaseGrid.h"
#include "useful.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda.h>

using namespace std;

#define STRLEN 512

// RBTODO: integrate with existing grid code?

class RigidBodyGrid { 
	friend class SparseGrid;
private:
  // Initialize the variables that get used a lot.
  // Also, allocate the main value array.
  void init();

public:

	/*                               \
	| CONSTRUCTORS, DESTRUCTORS, I/O |
	\===============================*/
	
	// RBTODO Fix?
	RigidBodyGrid(); // cmaffeo2 (2015) moved this out of protected, cause I wanted RigidBodyGrid in a struct
  // The most obvious of constructors.
	RigidBodyGrid(int nx0, int ny0, int nz0);

  // Make an orthogonal grid given the box dimensions and resolution.
  RigidBodyGrid(Vector3 box, float dx);

  // The box gives the system geometry.
  // The grid point numbers define the resolution.
  RigidBodyGrid(Matrix3 box, int nx0, int ny0, int nz0);

  // The box gives the system geometry.
  // dx is the approx. resolution.
  // The grid spacing is always a bit larger than dx.
  RigidBodyGrid(Matrix3 box, Vector3 origin0, float dx);

  // The box gives the system geometry.
  // dx is the approx. resolution.
  // The grid spacing is always a bit smaller than dx.
  RigidBodyGrid(Matrix3 box, float dx);

  // Make a copy of a BaseGrid grid.
  RigidBodyGrid(const BaseGrid& g);

  // Make an exact copy of a grid.
  RigidBodyGrid(const RigidBodyGrid& g);

  RigidBodyGrid mult(const RigidBodyGrid& g);

  RigidBodyGrid& operator=(const RigidBodyGrid& g);

  // Make a copy of a grid, but at a different resolution.
  RigidBodyGrid(const RigidBodyGrid& g, int nx0, int ny0, int nz0);
  
	virtual ~RigidBodyGrid();

	/*             \
	| DATA METHODS |
	\=============*/
		
	void zero();
  
  bool setValue(int j, float v);

  bool setValue(int ix, int iy, int iz, float v);

  virtual float getValue(int j) const;

  virtual float getValue(int ix, int iy, int iz) const;

  // Vector3 getPosition(int ix, int iy, int iz) const;
  HOST DEVICE Vector3 getPosition(int j) const;
	HOST DEVICE Vector3 getPosition(int j, Matrix3 basis, Vector3 origin) const;
		
  /* // Does the point r fall in the grid? */
  /* // Obviously this is without periodic boundary conditions. */
  /* bool inGrid(Vector3 r) const; */

  /* bool inGridInterp(Vector3 r) const; */

  /* Vector3 transformTo(Vector3 r) const; */

  /* Vector3 transformFrom(Vector3 l) const; */

  IndexList index(int j) const;
  int indexX(int j) const;
  int indexY(int j) const;
  int indexZ(int j) const;
  int index(int ix, int iy, int iz) const;
  
  /* int index(Vector3 r) const; */
  /* int nearestIndex(Vector3 r) const; */

  HOST DEVICE inline int length() const { return size; }

  HOST DEVICE inline int getNx() const {return nx;}
  HOST DEVICE inline int getNy() const {return ny;}
  HOST DEVICE inline int getNz() const {return nz;}
  HOST DEVICE inline int getSize() const {return size;}

  
  // Add a fixed value to the grid.
  void shift(float s);

  // Multiply the grid by a fixed value.
  void scale(float s);

	/*         \
	| COMPUTED |
	\=========*/
	
  // Get the mean of the entire grid.
  float mean() const;
	
  // Get the potential at the closest node.
  /* virtual float getPotential(Vector3 pos) const; */

	DEVICE float interpolatePotentialLinearly(const Vector3& l) const;
	DEVICE Vector3 interpolateForceDLinearly(const Vector3& l) const;

	HOST DEVICE float interpolateDiffX(const float wx, const float wy, const float wz, float g1[4][4][4]) const;
  HOST DEVICE float interpolateDiffY(const float wx, const float wy, const float wz, float g1[4][4][4]) const;
	HOST DEVICE float interpolateDiffZ(const float wx, const float wy, const float wz, float g1[4][4][4]) const;

  HOST DEVICE float interpolatePotential(const Vector3& l) const;

  HOST DEVICE inline static int wrap(int i, int n) {
		if (i < 0) {
			i %= n;
			i += n;
		}
		// The portion above allows i == n, so no else keyword.
		if (i >= n) {
			i %= n;
		} 
		return i;
	}

	/** interpolateForce() to be used on CUDA Device **/
	DEVICE Vector3 interpolateForceD(Vector3 l) const;

  inline virtual Vector3 interpolateForce(Vector3 pos) const {
		Vector3 f;
 		Vector3 l = pos;
		int homeX = int(floor(l.x));
		int homeY = int(floor(l.y));
		int homeZ = int(floor(l.z));
		// Get the array jumps with shifted indices.
		int jump[3];
		jump[0] = nz*ny;
		jump[1] = nz;
		jump[2] = 1;
		// Shift the indices in the home array.
		int home[3];
		home[0] = homeX;
		home[1] = homeY;
		home[2] = homeZ;

		// Get the interpolation coordinates.
		const float wx = l.x - homeX;
		const float wy = l.y - homeY;
		const float wz = l.z - homeZ;

		// Find the values at the neighbors.
		float g1[4][4][4];
		//RBTODO parallelize?
		for (int ix = 0; ix < 4; ix++) {
			for (int iy = 0; iy < 4; iy++) {
				for (int iz = 0; iz < 4; iz++) {
	  			// Wrap around the periodic boundaries. 
					int jx = ix-1 + home[0];
					jx = wrap(jx, nx);
					int jy = iy-1 + home[1];
					jy = wrap(jy, ny);
					int jz = iz-1 + home[2];
					jz = wrap(jz, nz);
					int ind = jz*jump[2] + jy*jump[1] + jx*jump[0];
					g1[ix][iy][iz] = val[ind];
				}
			}
		}  
		f.x = interpolateDiffX( wx, wy, wz, g1 );
		f.y = interpolateDiffY( wx, wy, wz, g1 );
		f.z = interpolateDiffZ( wx, wy, wz, g1 );
		// Vector3 f1 = basisInv.transpose().transform(f);
		// return f1;
		return f;
	}

  // Wrap coordinate: 0 <= x < l
  HOST DEVICE inline float wrapFloat(float x, float l) const {
		int image = int(floor(x/l));
		x -= image*l;
		return x;
  }
  
  // Wrap distance: -0.5*l <= x < 0.5*l
  HOST DEVICE static inline float wrapDiff(float x, float l) {
		int image = int(floor(x/l));
		x -= image*l;
		if (x >= 0.5f * l)
			x -= l;
		return x;
  }

  // Wrap vector, 0 <= x < lx  &&  0 <= y < ly  &&  0 <= z < lz
  HOST DEVICE inline Vector3 wrap(Vector3 l) const {
    l.x = wrapFloat(l.x, nx);
    l.y = wrapFloat(l.y, ny);
    l.z = wrapFloat(l.z, nz);
    return l;
  }

  // Wrap vector distance, -0.5*l <= x < 0.5*l  && ...
  HOST DEVICE inline Vector3 wrapDiff(Vector3 l) const {
    l.x = wrapDiff(l.x, nx);
    l.y = wrapDiff(l.y, ny);
    l.z = wrapDiff(l.z, nz);
		return l;
  }

  /* Vector3 wrapDiffNearest(Vector3 r) const; */

  // Includes the home node.
  // indexBuffer must have a size of at least 27.
  void getNeighbors(int j, int* indexBuffer) const;
  
  // Get the values at the neighbors of a node.
  // Note that homeX, homeY, and homeZ do not need to be wrapped,
  // since we do it here.
  void getNeighborValues(NeighborList* neigh, int homeX, int homeY, int homeZ) const;
  inline void setVal(float* v) { val = v; }
	
public:
  int nx, ny, nz;
  int size;
  float* val;
};

#endif
