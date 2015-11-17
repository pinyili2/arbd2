//////////////////////////////////////////////////////////////////////
// Grid base class that does just the basics.
// Author: Jeff Comer <jcomer2@illinois.edu>

#include "RigidBodyGrid.h"
#include <cuda.h>

#define STRLEN 512

// Initialize the variables that get used a lot.
// Also, allocate the main value array.
void RigidBodyGrid::init() {
	nynz = ny*nz;
	size = nx*ny*nz;
	val = new float[size];
}
RigidBodyGrid::RigidBodyGrid() {
	RigidBodyGrid tmp(1,1,1);
	val = new float[1];
	*this = tmp;									// TODO: verify that this is OK
	
	// origin = Vector3();
	// nx = 1;
	// ny = 1;
	// nz = 1;
	
	// init();
	// zero();
}

// The most obvious of constructors.
RigidBodyGrid::RigidBodyGrid(int nx0, int ny0, int nz0) {
	nx = abs(nx0);
	ny = abs(ny0);
	nz = abs(nz0);
	
	init();
	zero();
}

// Make an orthogonal grid given the box dimensions and resolution.
RigidBodyGrid::RigidBodyGrid(Vector3 box, float dx) {
	dx = fabsf(dx);
	box.x = fabsf(box.x);
	box.y = fabsf(box.y);
	box.z = fabsf(box.z);

	// Tile the grid into the system box.
	// The grid spacing is always a bit smaller than dx.
	nx = int(ceilf(box.x/dx));
	ny = int(ceilf(box.y/dx));
	nz = int(ceilf(box.z/dx));
	if (nx <= 0) nx = 1;
	if (ny <= 0) ny = 1;
	if (nz <= 0) nz = 1;
	
	init();
	zero();
}

// The box gives the system geometry.
// The grid point numbers define the resolution.
RigidBodyGrid::RigidBodyGrid(Matrix3 box, int nx0, int ny0, int nz0) {
	nx = nx0;
	ny = ny0;
	nz = nz0;

	// Tile the grid into the system box.
	if (nx <= 0) nx = 1;
	if (ny <= 0) ny = 1;
	if (nz <= 0) nz = 1;

	init();
	zero();
}

// The box gives the system geometry.
// dx is the approx. resolution.
// The grid spacing is always a bit larger than dx.
RigidBodyGrid::RigidBodyGrid(Matrix3 box, Vector3 origin0, float dx) {
	dx = fabs(dx);
	
	// Tile the grid into the system box.
	// The grid spacing is always a bit larger than dx.
	nx = int(floor(box.ex().length()/dx))-1;
	ny = int(floor(box.ey().length()/dx))-1;
	nz = int(floor(box.ez().length()/dx))-1;
	if (nx <= 0) nx = 1;
	if (ny <= 0) ny = 1;
	if (nz <= 0) nz = 1;

	init();
	zero();
}

// The box gives the system geometry.
// dx is the approx. resolution.
// The grid spacing is always a bit smaller than dx.
RigidBodyGrid::RigidBodyGrid(Matrix3 box, float dx) {
	dx = fabs(dx);
	
	// Tile the grid into the system box.
	// The grid spacing is always a bit smaller than dx.
	nx = int(ceilf(box.ex().length()/dx));
	ny = int(ceilf(box.ey().length()/dx));
	nz = int(ceilf(box.ez().length()/dx));
	if (nx <= 0) nx = 1;
	if (ny <= 0) ny = 1;
	if (nz <= 0) nz = 1;

	init();
	zero();
}

// Make an exact copy of a grid.
RigidBodyGrid::RigidBodyGrid(const RigidBodyGrid& g) {
	nx = g.nx;
	ny = g.ny;
	nz = g.nz;
	
	init();
	for (int i = 0; i < size; i++) val[i] = g.val[i];
}

RigidBodyGrid RigidBodyGrid::mult(const RigidBodyGrid& g) {
	for (int i = 0; i < size; i++) val[i] *= g.val[i];
	return *this;
}

RigidBodyGrid& RigidBodyGrid::operator=(const RigidBodyGrid& g) {
	delete[] val;
	val = NULL;
	nx = g.nx;
	ny = g.ny;
	nz = g.nz;
	
	init();
	for (int i = 0; i < size; i++) val[i] = g.val[i];

	return *this;
}


// Make a copy of a grid, but at a different resolution.
RigidBodyGrid::RigidBodyGrid(const RigidBodyGrid& g, int nx0, int ny0, int nz0) : nx(nx0),  ny(ny0), nz(nz0) {
	if (nx <= 0) nx = 1;
	if (ny <= 0) ny = 1;
	if (nz <= 0) nz = 1;

	// // Tile the grid into the box of the template grid.
	// Matrix3 box = g.getBox();

	init();

	// Do an interpolation to obtain the values.
	for (int i = 0; i < size; i++) {
		Vector3 r = getPosition(i);
		val[i] = g.interpolatePotential(r);
	}
}

RigidBodyGrid::~RigidBodyGrid() {
	if (val != NULL)
		delete[] val;
}

void RigidBodyGrid::zero() {
	for (int i = 0; i < size; i++) val[i] = 0.0f;
}

bool RigidBodyGrid::setValue(int j, float v) {
	if (j < 0 || j >= size) return false;
	val[j] = v;
	return true;
}

bool RigidBodyGrid::setValue(int ix, int iy, int iz, float v) {
	if (ix < 0 || ix >= nx) return false;
	if (iy < 0 || iy >= ny) return false;
	if (iz < 0 || iz >= nz) return false;
	int j = iz + iy*nz + ix*ny*nz;

	val[j] = v;
	return true;
}

float RigidBodyGrid::getValue(int j) const {
	if (j < 0 || j >= size) return 0.0f;
	return val[j];
}

float RigidBodyGrid::getValue(int ix, int iy, int iz) const {
	if (ix < 0 || ix >= nx) return 0.0f;
	if (iy < 0 || iy >= ny) return 0.0f;
	if (iz < 0 || iz >= nz) return 0.0f;
	
	int j = iz + iy*nz + ix*ny*nz;
	return val[j];
}

Vector3 RigidBodyGrid::getPosition(int j) const {
	int iz = j%nz;
	int iy = (j/nz)%ny;
	int ix = j/(nz*ny);

	return Vector3(ix,iy,iz);
}

Vector3 RigidBodyGrid::getPosition(int j, Matrix3 basis, Vector3 origin) const {
	int iz = j%nz;
	int iy = (j/nz)%ny;
	int ix = j/(nz*ny);

	return basis.transform(Vector3(ix, iy, iz)) + origin;
}

// // Does the point r fall in the grid?
// // Obviously this is without periodic boundary conditions.
// bool RigidBodyGrid::inGrid(Vector3 r,) const {
// 	Vector3 l = basisInv.transform(r-origin);

// 	if (l.x < 0.0f || l.x >= nx) return false;
// 	if (l.y < 0.0f || l.y >= ny) return false;
// 	if (l.z < 0.0f || l.z >= nz) return false;
// 	return true;
// }

// bool RigidBodyGrid::inGridInterp(Vector3 r) const {
// 	Vector3 l = basisInv.transform(r-origin);

// 	if (l.x < 2.0f || l.x >= nx-3.0f) return false;
// 	if (l.y < 2.0f || l.y >= ny-3.0f) return false;
// 	if (l.z < 2.0f || l.z >= nz-3.0f) return false;
// 	return true;
// }

// Vector3 RigidBodyGrid::transformTo(Vector3 r) const {
// 	return basisInv.transform(r-origin);
// }
// Vector3 RigidBodyGrid::transformFrom(Vector3 l) const {
// 	return basis.transform(l) + origin;
// }

IndexList RigidBodyGrid::index(int j) const {
	int iz = j%nz;
	int iy = (j/nz)%ny;
	int ix = j/(nz*ny);
	IndexList ret;
	ret.add(ix);
	ret.add(iy);
	ret.add(iz);
	return ret;
}
int RigidBodyGrid::indexX(int j) const { return j/(nz*ny); }
int RigidBodyGrid::indexY(int j) const { return (j/nz)%ny; }
int RigidBodyGrid::indexZ(int j) const { return j%nz; }
int RigidBodyGrid::index(int ix, int iy, int iz) const { return iz + iy*nz + ix*ny*nz; }

// int RigidBodyGrid::index(Vector3 r) const {
// 	Vector3 l = basisInv.transform(r-origin);
	
// 	int ix = int(floor(l.x));
// 	int iy = int(floor(l.y));
// 	int iz = int(floor(l.z));

// 	ix = wrap(ix, nx);
// 	iy = wrap(iy, ny);
// 	iz = wrap(iz, nz);
	
// 	return iz + iy*nz + ix*ny*nz;
// }

// int RigidBodyGrid::nearestIndex(Vector3 r) const {
// 	Vector3 l = basisInv.transform(r-origin);
	
// 	int ix = int(floorf(l.x + 0.5f));
// 	int iy = int(floorf(l.y + 0.5f));
// 	int iz = int(floorf(l.z + 0.5f));

// 	ix = wrap(ix, nx);
// 	iy = wrap(iy, ny);
// 	iz = wrap(iz, nz);
	
// 	return iz + iy*nz + ix*ny*nz;
// }


// Add a fixed value to the grid.
void RigidBodyGrid::shift(float s) {
	for (int i = 0; i < size; i++) val[i] += s;
}

// Multiply the grid by a fixed value.
void RigidBodyGrid::scale(float s) {
	for (int i = 0; i < size; i++) val[i] *= s;
}

// Get the mean of the entire grid.
float RigidBodyGrid::mean() const {
	float sum = 0.0f;
	for (int i = 0; i < size; i++) sum += val[i];
	return sum/size;
}

// Get the potential at the closest node.
// float RigidBodyGrid::getPotential(Vector3 pos) const {
// 	// Find the nearest node.
// 	int j = nearestIndex(pos);

// 	return val[j];
// }


// Added by Rogan for times when simpler calculations are required.
float RigidBodyGrid::interpolatePotentialLinearly(Vector3 pos) const {
	// Find the home node.
	Vector3 l = pos;
	int homeX = int(floorf(l.x));
	int homeY = int(floorf(l.y));
	int homeZ = int(floorf(l.z));
	
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
	float g1[2][2][2];
	for (int ix = 0; ix < 2; ix++) {
		for (int iy = 0; iy < 2; iy++) {
			for (int iz = 0; iz < 2; iz++) {
				// Wrap around the periodic boundaries. 
				int jx = ix + home[0];
				jx = wrap(jx, g[0]);
				int jy = iy + home[1];
				jy = wrap(jy, g[1]);
				int jz = iz + home[2];
				jz = wrap(jz, g[2]);
				
				int ind = jz*jump[2] + jy*jump[1] + jx*jump[0];
				g1[ix][iy][iz] = val[ind];
			}
		}
	}

	// Mix along x.
	float g2[2][2];
	for (int iy = 0; iy < 2; iy++) {
		for (int iz = 0; iz < 2; iz++) {
			// p = w[0] * g[0][iy][iz] + (1-w[0]) * g[1][iy][iz]
			g2[iy][iz] = (1.0f-w[0])*g1[0][iy][iz] + w[0]*g1[1][iy][iz];
		}
	}

	// Mix along y.
	float g3[2];
	for (int iz = 0; iz < 2; iz++) {
		g3[iz] = (1.0f-w[1])*g2[0][iz] + w[1]*g2[1][iz];
	}

	// DEBUG
	//printf("(0,0,0)=%.1f (0,0,1)=%.1f (0,1,0)=%.1f (0,1,1)=%.1f (1,0,0)=%.1f (1,0,1)=%.1f (1,1,0)=%.1f (1,1,1)=%.1f ",
	//   g1[0][0][0], g1[0][0][1], g1[0][1][0], g1[0][1][1], g1[1][0][0], g1[1][0][1], g1[1][1][0], g1[1][1][1] );
	//printf ("%.2f\n",(1.0-w[2])*g3[0] + w[2]*g3[1]);

	// Mix along z
	return (1.0f-w[2])*g3[0] + w[2]*g3[1];
}



// Vector3 RigidBodyGrid::wrapDiffNearest(Vector3 r) const {
// 	Vector3 l = r;
// 	l.x = wrapDiff(l.x, nx);
// 	l.y = wrapDiff(l.y, ny);
// 	l.z = wrapDiff(l.z, nz);

// 	float length2 = l.length2();

// 	for (int dx = -1; dx <= 1; dx++) {
// 		for (int dy = -1; dy <= 1; dy++) {
// 			for (int dz = -1; dz <= 1; dz++) {
// 				//if (dx == 0 && dy == 0 && dz == 0) continue;
// 				Vector3 tmp = Vector3(l.x+dx*nx, l.y+dy*ny, l.z+dz*nz);
// 				if (tmp.length2() < length2) {
// 					l = tmp;
// 					length2 = l.length2();
// 				}
// 			}
// 		}
// 	}

// 	return l;
// 	// return basis.transform(l);
// }


// Includes the home node.
// indexBuffer must have a size of at least 27.
void RigidBodyGrid::getNeighbors(int j, int* indexBuffer) const {
	int jx = indexX(j);
	int jy = indexY(j);
	int jz = indexZ(j);

	int k = 0;
	for (int ix = -1; ix <= 1; ix++) {
		for (int iy = -1; iy <= 1; iy++) {
			for (int iz = -1; iz <= 1; iz++) {
				int ind = wrap(jz+iz,nz) + nz*wrap(jy+iy,ny) + nynz*wrap(jx+ix,nx);
				indexBuffer[k] = ind;
				k++;
			}
		}
	}
}


// Get the values at the neighbors of a node.
// Note that homeX, homeY, and homeZ do not need to be wrapped,
// since we do it here.
void RigidBodyGrid::getNeighborValues(NeighborList* neigh, int homeX, int homeY, int homeZ) const {
	for (int ix = -1; ix <= 1; ix++) {
		for (int iy = -1; iy <= 1; iy++) {
			for (int iz = -1; iz <= 1; iz++) {
				int ind = wrap(homeZ+iz,nz) + nz*wrap(homeY+iy,ny) + nynz*wrap(homeX+ix,nx);
				neigh->v[ix+1][iy+1][iz+1] = val[ind];
			}
		}
	}
}  
