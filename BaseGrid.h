//////////////////////////////////////////////////////////////////////
// Grid base class that does just the basics.
// Author: Jeff Comer <jcomer2@illinois.edu>
#ifndef BASEGRID_H
#define BASEGRID_H


#ifdef __CUDACC__
    #define HOST __host__
    #define DEVICE __device__
#else
    #define HOST 
    #define DEVICE 
#endif

#include "useful.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda.h>


using namespace std;

#define STRLEN 512

class NeighborList {
public:
  float v[3][3][3];
};

class BaseGrid {
  friend class SparseGrid;
 
private:
  // Initialize the variables that get used a lot.
  // Also, allocate the main value array.
	HOST DEVICE
  void init();

public:
	HOST DEVICE
		BaseGrid(); // cmaffeo2 (2015) moved this out of protected, cause I wanted BaseGrid in a struct

  // The most obvious of constructors.
	HOST DEVICE
		BaseGrid(Matrix3 basis0, Vector3 origin0, int nx0, int ny0, int nz0);

  // Make an orthogonal grid given the box dimensions and resolution.
  BaseGrid(Vector3 box, float dx);

  // The box gives the system geometry.
  // The grid point numbers define the resolution.
  BaseGrid(Matrix3 box, int nx0, int ny0, int nz0);

  // The box gives the system geometry.
  // dx is the approx. resolution.
  // The grid spacing is always a bit larger than dx.
  BaseGrid(Matrix3 box, Vector3 origin0, float dx);

  // The box gives the system geometry.
  // dx is the approx. resolution.
  // The grid spacing is always a bit smaller than dx.
  BaseGrid(Matrix3 box, float dx);

  // Make an exact copy of a grid.
  BaseGrid(const BaseGrid& g);

  BaseGrid mult(const BaseGrid& g);

  BaseGrid& operator=(const BaseGrid& g);

  // Make a copy of a grid, but at a different resolution.
  BaseGrid(const BaseGrid& g, int nx0, int ny0, int nz0);

  // Read a grid from a file.
  BaseGrid(const char* fileName);
  
  // Write without comments.
  virtual void write(const char* fileName) const;

  // Writes the grid as a file in the dx format.
  virtual void write(const char* fileName, const char* comments) const;

  // Writes the grid data as a single column in the order:
  // nx ny nz ox oy oz dxx dyx dzx dxy dyy dzy dxz dyz dzz val0 val1 val2 ...
  virtual void writeData(const char* fileName);
 
  // Write the valies in a single column.
  virtual void writePotential(const char* fileName) const;
  
	HOST DEVICE
	virtual ~BaseGrid();

  void zero();
  
  bool setValue(int j, float v);

  bool setValue(int ix, int iy, int iz, float v);

  virtual float getValue(int j) const;

  virtual float getValue(int ix, int iy, int iz) const;

  Vector3 getPosition(int ix, int iy, int iz) const;

  Vector3 getPosition(int j) const;

  // Does the point r fall in the grid?
  // Obviously this is without periodic boundary conditions.
  bool inGrid(Vector3 r) const;

  bool inGridInterp(Vector3 r) const;

  Vector3 transformTo(Vector3 r) const;

  Vector3 transformFrom(Vector3 l) const;

  IndexList index(int j) const;
  int indexX(int j) const;
  int indexY(int j) const;
  int indexZ(int j) const;
  int index(int ix, int iy, int iz) const;
  
  int index(Vector3 r) const;

  int nearestIndex(Vector3 r) const;

  HOST DEVICE inline int length() const {
		return size;
	}
  void setBasis(const Matrix3& b);
  void setOrigin(const Vector3& o);

  HOST DEVICE inline Vector3 getOrigin() const {return origin;}
  HOST DEVICE inline Matrix3 getBasis() const {return basis;}
  HOST DEVICE inline Matrix3 getInverseBasis() const {return basisInv;}
  HOST DEVICE inline int getNx() const {return nx;}
  HOST DEVICE inline int getNy() const {return ny;}
  HOST DEVICE inline int getNz() const {return nz;}
  HOST DEVICE inline int getSize() const {return size;}

  
  // A matrix defining the basis for the entire system.
  Matrix3 getBox() const;
  // The longest diagonal of the system.
  Vector3 getExtent() const;
  // The longest diagonal of the system.
  float getDiagonal() const;
  // The position farthest from the origin.
  Vector3 getDestination() const;
  // The center of the grid.
  Vector3 getCenter() const;
  // The volume of a single cell.
  float getCellVolume() const;
  // The volume of the entire system.
  float getVolume() const;
  Vector3 getCellDiagonal() const;

  // Add a fixed value to the grid.
  void shift(float s);

  // Multiply the grid by a fixed value.
  void scale(float s);

  // Get the mean of the entire grid.
  float mean() const;

  // Compute the average profile along an axis.
  // Assumes that the grid axis with index "axis" is aligned with the world axis of index "axis".
  void averageProfile(const char* fileName, int axis);

  // Get the potential at the closest node.
  virtual float getPotential(Vector3 pos) const;

	// crop
	// Cuts the grid down
	// @param		boundries to crop to (x0, y0, z0) -> (x1, y1, z1);
	//					whether to change the origin
	// @return	success of the function
	bool crop(int x0, int y0, int z0, int x1, int y1, int z1, bool keep_origin);

  // Added by Rogan for times when simpler calculations are required.
  virtual float interpolatePotentialLinearly(Vector3 pos) const;

  HOST DEVICE inline float interpolateDiffX(Vector3 pos, float w[3], float g1[4][4][4]) const {
    float a0, a1, a2, a3;
    
    // Mix along x, taking the derivative.
    float g2[4][4];
    for (int iy = 0; iy < 4; iy++) {
      for (int iz = 0; iz < 4; iz++) {
				a3 = 0.5f*(-g1[0][iy][iz] + 3.0f*g1[1][iy][iz] - 3.0f*g1[2][iy][iz] + g1[3][iy][iz]);
				a2 = 0.5f*(2.0f*g1[0][iy][iz] - 5.0f*g1[1][iy][iz] + 4.0f*g1[2][iy][iz] - g1[3][iy][iz]);
				a1 = 0.5f*(-g1[0][iy][iz] + g1[2][iy][iz]);
				a0 = g1[1][iy][iz];

				//g2[iy][iz] = a3*w[0]*w[0]*w[0] + a2*w[0]*w[0] + a1*w[0] + a0;
				g2[iy][iz] = 3.0f*a3*w[0]*w[0] + 2.0f*a2*w[0] + a1;
      }
    }


    // Mix along y.
    float g3[4];
    for (int iz = 0; iz < 4; iz++) {
      a3 = 0.5f*(-g2[0][iz] + 3.0f*g2[1][iz] - 3.0f*g2[2][iz] + g2[3][iz]);
      a2 = 0.5f*(2.0f*g2[0][iz] - 5.0f*g2[1][iz] + 4.0f*g2[2][iz] - g2[3][iz]);
      a1 = 0.5f*(-g2[0][iz] + g2[2][iz]);
      a0 = g2[1][iz];
   
      g3[iz] = a3*w[1]*w[1]*w[1] + a2*w[1]*w[1] + a1*w[1] + a0;
    }

    // Mix along z.
    a3 = 0.5f*(-g3[0] + 3.0f*g3[1] - 3.0f*g3[2] + g3[3]);
    a2 = 0.5f*(2.0f*g3[0] - 5.0f*g3[1] + 4.0f*g3[2] - g3[3]);
    a1 = 0.5f*(-g3[0] + g3[2]);
    a0 = g3[1];
 
    float retval = -(a3*w[2]*w[2]*w[2] + a2*w[2]*w[2] + a1*w[2] + a0);
    return retval;
  }

  HOST DEVICE inline float interpolateDiffY(Vector3 pos, float w[3], float g1[4][4][4]) const {
    float a0, a1, a2, a3;
  
    // Mix along x, taking the derivative.
    float g2[4][4];
    for (int iy = 0; iy < 4; iy++) {
      for (int iz = 0; iz < 4; iz++) {
				a3 = 0.5f*(-g1[0][iy][iz] + 3.0f*g1[1][iy][iz] - 3.0f*g1[2][iy][iz] + g1[3][iy][iz]);
				a2 = 0.5f*(2.0f*g1[0][iy][iz] - 5.0f*g1[1][iy][iz] + 4.0f*g1[2][iy][iz] - g1[3][iy][iz]);
				a1 = 0.5f*(-g1[0][iy][iz] + g1[2][iy][iz]);
				a0 = g1[1][iy][iz];

				g2[iy][iz] = a3*w[0]*w[0]*w[0] + a2*w[0]*w[0] + a1*w[0] + a0;
      }
    }

    // Mix along y.
    float g3[4];
    for (int iz = 0; iz < 4; iz++) {
      a3 = 0.5f*(-g2[0][iz] + 3.0f*g2[1][iz] - 3.0f*g2[2][iz] + g2[3][iz]);
      a2 = 0.5f*(2.0f*g2[0][iz] - 5.0f*g2[1][iz] + 4.0f*g2[2][iz] - g2[3][iz]);
      a1 = 0.5f*(-g2[0][iz] + g2[2][iz]);
      a0 = g2[1][iz];
   
      //g3[iz] = a3*w[1]*w[1]*w[1] + a2*w[1]*w[1] + a1*w[1] + a0;
      g3[iz] = 3.0f*a3*w[1]*w[1] + 2.0f*a2*w[1] + a1;
    }

    // Mix along z.
    a3 = 0.5f*(-g3[0] + 3.0f*g3[1] - 3.0f*g3[2] + g3[3]);
    a2 = 0.5f*(2.0f*g3[0] - 5.0f*g3[1] + 4.0f*g3[2] - g3[3]);
    a1 = 0.5f*(-g3[0] + g3[2]);
    a0 = g3[1];

    return -(a3*w[2]*w[2]*w[2] + a2*w[2]*w[2] + a1*w[2] + a0);
  }

  HOST DEVICE inline float interpolateDiffZ(Vector3 pos, float w[3], float g1[4][4][4]) const {
    float a0, a1, a2, a3;
  
    // Mix along x, taking the derivative.
    float g2[4][4];
    for (int iy = 0; iy < 4; iy++) {
      for (int iz = 0; iz < 4; iz++) {
				a3 = 0.5f*(-g1[0][iy][iz] + 3.0f*g1[1][iy][iz] - 3.0f*g1[2][iy][iz] + g1[3][iy][iz]);
				a2 = 0.5f*(2.0f*g1[0][iy][iz] - 5.0f*g1[1][iy][iz] + 4.0f*g1[2][iy][iz] - g1[3][iy][iz]);
				a1 = 0.5f*(-g1[0][iy][iz] + g1[2][iy][iz]);
				a0 = g1[1][iy][iz];

				g2[iy][iz] = a3*w[0]*w[0]*w[0] + a2*w[0]*w[0] + a1*w[0] + a0;
      }
    }

    // Mix along y.
    float g3[4];
    for (int iz = 0; iz < 4; iz++) {
      a3 = 0.5f*(-g2[0][iz] + 3.0f*g2[1][iz] - 3.0f*g2[2][iz] + g2[3][iz]);
      a2 = 0.5f*(2.0f*g2[0][iz] - 5.0f*g2[1][iz] + 4.0f*g2[2][iz] - g2[3][iz]);
      a1 = 0.5f*(-g2[0][iz] + g2[2][iz]);
      a0 = g2[1][iz];
   
      g3[iz] = a3*w[1]*w[1]*w[1] + a2*w[1]*w[1] + a1*w[1] + a0;
    }

    // Mix along z.
    a3 = 0.5f*(-g3[0] + 3.0f*g3[1] - 3.0f*g3[2] + g3[3]);
    a2 = 0.5f*(2.0f*g3[0] - 5.0f*g3[1] + 4.0f*g3[2] - g3[3]);
    a1 = 0.5f*(-g3[0] + g3[2]);
    a0 = g3[1];

    return -(3.0f*a3*w[2]*w[2] + 2.0f*a2*w[2] + a1);
  }
  
  HOST DEVICE inline float interpolatePotential(Vector3 pos) const {
    // Find the home node.
    Vector3 l = basisInv.transform(pos - origin);
    int homeX = int(floor(l.x));
    int homeY = int(floor(l.y));
    int homeZ = int(floor(l.z));
    
    // Get the array jumps.
    int jump[3];
    jump[0] = nz*ny;
    jump[1] = nz;
    jump[2] = 1;

    // Shift the indices in the home array.
    int home[3];
    home[0] = homeX;
    home[1] = homeY;
    home[2] = homeZ;

    // Get the grid dimensions.
    int g[3];
    g[0] = nx;
    g[1] = ny;
    g[2] = nz;

    // Get the interpolation coordinates.
    float w[3];
    w[0] = l.x - homeX;
    w[1] = l.y - homeY;
    w[2] = l.z - homeZ;

    // Find the values at the neighbors.
    float g1[4][4][4];
    for (int ix = 0; ix < 4; ix++) {
	  	int jx = ix-1 + home[0];
	  	jx = wrap(jx, g[0]);
      for (int iy = 0; iy < 4; iy++) {
	  		int jy = iy-1 + home[1];
	  		jy = wrap(jy, g[1]);
				for (int iz = 0; iz < 4; iz++) {
	  			// Wrap around the periodic boundaries. 
	  			int jz = iz-1 + home[2];
	  			jz = wrap(jz, g[2]);
	  
	  			int ind = jz*jump[2] + jy*jump[1] + jx*jump[0];
	  			g1[ix][iy][iz] = val[ind];
				}
      }
    }
    float a0, a1, a2, a3;
  
    // Mix along x.
    float g2[4][4];
    for (int iy = 0; iy < 4; iy++) {
      for (int iz = 0; iz < 4; iz++) {
				a3 = 0.5f*(-g1[0][iy][iz] + 3.0f*g1[1][iy][iz] - 3.0f*g1[2][iy][iz] + g1[3][iy][iz]);
				a2 = 0.5f*(2.0f*g1[0][iy][iz] - 5.0f*g1[1][iy][iz] + 4.0f*g1[2][iy][iz] - g1[3][iy][iz]);
				a1 = 0.5f*(-g1[0][iy][iz] + g1[2][iy][iz]);
				a0 = g1[1][iy][iz];

				g2[iy][iz] = a3*w[0]*w[0]*w[0] + a2*w[0]*w[0] + a1*w[0] + a0;
      }
    }

    // Mix along y.
    float g3[4];
    for (int iz = 0; iz < 4; iz++) {
      a3 = 0.5f*(-g2[0][iz] + 3.0f*g2[1][iz] - 3.0f*g2[2][iz] + g2[3][iz]);
      a2 = 0.5f*(2.0f*g2[0][iz] - 5.0f*g2[1][iz] + 4.0f*g2[2][iz] - g2[3][iz]);
      a1 = 0.5f*(-g2[0][iz] + g2[2][iz]);
      a0 = g2[1][iz];
   
      g3[iz] = a3*w[1]*w[1]*w[1] + a2*w[1]*w[1] + a1*w[1] + a0;
    }

    // Mix along z.
    a3 = 0.5f*(-g3[0] + 3.0f*g3[1] - 3.0f*g3[2] + g3[3]);
    a2 = 0.5f*(2.0f*g3[0] - 5.0f*g3[1] + 4.0f*g3[2] - g3[3]);
    a1 = 0.5f*(-g3[0] + g3[2]);
    a0 = g3[1];

    return a3*w[2]*w[2]*w[2] + a2*w[2]*w[2] + a1*w[2] + a0;
  }

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
	DEVICE inline Vector3 interpolateForceD(Vector3 pos) const {
		Vector3 f;
 		Vector3 l = basisInv.transform(pos - origin);
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

		// Shift the indices in the grid dimensions.
		int g[3];
		g[0] = nx;
		g[1] = ny;
		g[2] = nz;

		// Get the interpolation coordinates.
		float w[3];
		w[0] = l.x - homeX;
		w[1] = l.y - homeY;
		w[2] = l.z - homeZ;
		// Find the values at the neighbors.
		float g1[4][4][4];
		for (int ix = 0; ix < 4; ix++) {
			for (int iy = 0; iy < 4; iy++) {
				for (int iz = 0; iz < 4; iz++) {
	  			// Wrap around the periodic boundaries. 
					int jx = ix-1 + home[0];
					jx = wrap(jx, g[0]);
					int jy = iy-1 + home[1];
					jy = wrap(jy, g[1]);
					int jz = iz-1 + home[2];
					jz = wrap(jz, g[2]);
					int ind = jz*jump[2] + jy*jump[1] + jx*jump[0];
					g1[ix][iy][iz] = val[ind];
				}
			}
		}  
		f.x = interpolateDiffX(pos, w, g1);
		f.y = interpolateDiffY(pos, w, g1);
		f.z = interpolateDiffZ(pos, w, g1);
		Vector3 f1 = basisInv.transpose().transform(f);
		return f1;
	}

  inline virtual Vector3 interpolateForce(Vector3 pos) const {
		Vector3 f;
 		Vector3 l = basisInv.transform(pos - origin);
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

		// Shift the indices in the grid dimensions.
		int g[3];
		g[0] = nx;
		g[1] = ny;
		g[2] = nz;

		// Get the interpolation coordinates.
		float w[3];
		w[0] = l.x - homeX;
		w[1] = l.y - homeY;
		w[2] = l.z - homeZ;
		// Find the values at the neighbors.
		float g1[4][4][4];
		for (int ix = 0; ix < 4; ix++) {
			for (int iy = 0; iy < 4; iy++) {
				for (int iz = 0; iz < 4; iz++) {
	  			// Wrap around the periodic boundaries. 
					int jx = ix-1 + home[0];
					jx = wrap(jx, g[0]);
					int jy = iy-1 + home[1];
					jy = wrap(jy, g[1]);
					int jz = iz-1 + home[2];
					jz = wrap(jz, g[2]);
					int ind = jz*jump[2] + jy*jump[1] + jx*jump[0];
					g1[ix][iy][iz] = val[ind];
				}
			}
		}  
		f.x = interpolateDiffX(pos, w, g1);
		f.y = interpolateDiffY(pos, w, g1);
		f.z = interpolateDiffZ(pos, w, g1);
		Vector3 f1 = basisInv.transpose().transform(f);
		return f1;
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
  HOST DEVICE inline virtual Vector3 wrap(Vector3 r) const {
    Vector3 l = basisInv.transform(r - origin);
    l.x = wrapFloat(l.x, nx);
    l.y = wrapFloat(l.y, ny);
    l.z = wrapFloat(l.z, nz);
    return basis.transform(l) + origin;
  }

  // Wrap vector distance, -0.5*l <= x < 0.5*l  && ...
  HOST DEVICE inline Vector3 wrapDiff(Vector3 r) const {
    Vector3 l = basisInv.transform(r);
    l.x = wrapDiff(l.x, nx);
    l.y = wrapDiff(l.y, ny);
    l.z = wrapDiff(l.z, nz);
    return basis.transform(l);
  }

  Vector3 wrapDiffNearest(Vector3 r) const;

  // Includes the home node.
  // indexBuffer must have a size of at least 27.
  void getNeighbors(int j, int* indexBuffer) const;
  
  // Get the values at the neighbors of a node.
  // Note that homeX, homeY, and homeZ do not need to be wrapped,
  // since we do it here.
  void getNeighborValues(NeighborList* neigh, int homeX, int homeY, int homeZ) const;
  inline void setVal(float* v) { val = v; }

public:
  Vector3 origin;
  Matrix3 basis;
  int nx, ny, nz;
  int nynz;
  int size;
  Matrix3 basisInv;
public:
  float* val;
};
#endif
