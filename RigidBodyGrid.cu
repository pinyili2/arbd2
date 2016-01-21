//////////////////////////////////////////////////////////////////////
// Grid base class that does just the basics.
// Author: Jeff Comer <jcomer2@illinois.edu>

#include "RigidBodyGrid.h"
#include <cuda.h>

#define STRLEN 512

// Initialize the variables that get used a lot.
// Also, allocate the main value array.
void RigidBodyGrid::init() {
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

RigidBodyGrid::RigidBodyGrid(const BaseGrid& g) {
	nx = g.nx;
	ny = g.ny;
	nz = g.nz;
	
	init();
	for (int i = 0; i < size; i++) val[i] = g.val[i];
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

HOST DEVICE float RigidBodyGrid::interpolatePotential(const Vector3& l) const {
	// Find the home node.
	const int homeX = int(floor(l.x));
	const int homeY = int(floor(l.y));
	const int homeZ = int(floor(l.z));
	
	float g3[4];
	for (int iz = 0; iz < 4; iz++) {
		float g2[4];
		for (int iy = 0; iy < 4; iy++) {
			// Fetch values from nearby
			float g1[4];
			for (volatile  int ix = 0; ix < 4; ix++) {
				volatile int jx = (ix-1 + homeX);
				volatile int jy = (iy-1 + homeY);
				volatile int jz = (iz-1 + homeZ);
				const int ind = jz + jy*nz + jx*nz*ny;
				g1[ix] = jz < 0 || jz >= nz || jy < 0 || jy >= ny || jx < 0 || jx >= nx ?
					0 : val[ind];
			}
			// Mix along x.
			const float a3 = 0.5f*(-g1[0] + 3.0f*g1[1] - 3.0f*g1[2] + g1[3]);
			const float a2 = 0.5f*(2.0f*g1[0] - 5.0f*g1[1] + 4.0f*g1[2] - g1[3]);
			const float a1 = 0.5f*(-g1[0] + g1[2]);
			const float a0 = g1[1];
			const float wx = l.x - homeX;
			g2[iy] = a3*wx*wx*wx + a2*wx*wx + a1*wx + a0;
		}
		// Mix along y.
		const float a3 = 0.5f*(-g2[0] + 3.0f*g2[1] - 3.0f*g2[2] + g2[3]);
		const float a2 = 0.5f*(2.0f*g2[0] - 5.0f*g2[1] + 4.0f*g2[2] - g2[3]);
		const float a1 = 0.5f*(-g2[0] + g2[2]);
		const float a0 = g2[1];
		const float wy = l.y - homeY;
		g3[iz] = a3*wy*wy*wy + a2*wy*wy + a1*wy + a0;
	}
	// Mix along z.
	const float a3 = 0.5f*(-g3[0] + 3.0f*g3[1] - 3.0f*g3[2] + g3[3]);
	const float a2 = 0.5f*(2.0f*g3[0] - 5.0f*g3[1] + 4.0f*g3[2] - g3[3]);
	const float a1 = 0.5f*(-g3[0] + g3[2]);
	const float a0 = g3[1];
	const float wz = l.z - homeZ;
	return a3*wz*wz*wz + a2*wz*wz + a1*wz + a0;
}

/** interpolateForce() to be used on CUDA Device **/
DEVICE ForceEnergy RigidBodyGrid::interpolateForceD(const Vector3 l) const {
	Vector3 f;
	// Vector3 l = basisInv.transform(pos - origin);
	const int homeX = int(floor(l.x));
	const int homeY = int(floor(l.y));
	const int homeZ = int(floor(l.z));
	const float wx = l.x - homeX;
	const float wy = l.y - homeY;
	const float wz = l.z - homeZ;
	const float wx2 = wx*wx;

	// RBTODO: test against cpu algorithm; also see if its the same algorithm used by NAMD
	// RBTODO: test NAMD alg. for speed
	
	/* f.x */
	float g3[3][4];
	for (int iz = 0; iz < 4; iz++) {
		float g2[2][4];
		const int jz = (iz + homeZ - 1);
		for (int iy = 0; iy < 4; iy++) {
			float v[4];
			const int jy = (iy + homeY - 1);
			for (int ix = 0; ix < 4; ix++) {
				const int jx = (ix + homeX - 1);
				const int ind = jz + jy*nz + jx*nz*ny;
				v[ix] = jz < 0 || jz >= nz || jy < 0 || jy >= ny || jx < 0 || jx >= nx ?
					0 : val[ind];
			}
			const float a3 = 0.5f*(-v[0] + 3.0f*v[1] - 3.0f*v[2] + v[3])*wx2;
			const float a2 = 0.5f*(2.0f*v[0] - 5.0f*v[1] + 4.0f*v[2] - v[3])*wx;
			const float a1 = 0.5f*(-v[0] + v[2]);
			g2[0][iy] = 3.0f*a3 + 2.0f*a2 + a1;				/* f.x (derivative) */
			g2[1][iy] = a3*wx + a2*wx + a1*wx + v[1]; /* f.y & f.z */
		}

		// Mix along y.
		{
			g3[0][iz] = 0.5f*(-g2[0][0] + 3.0f*g2[0][1] - 3.0f*g2[0][2] + g2[0][3])*wy*wy*wy +
				0.5f*(2.0f*g2[0][0] - 5.0f*g2[0][1] + 4.0f*g2[0][2] - g2[0][3])      *wy*wy +
				0.5f*(-g2[0][0] + g2[0][2])                                          *wy +
				g2[0][1];
		}

		{
			const float a3 = 0.5f*(-g2[1][0] + 3.0f*g2[1][1] - 3.0f*g2[1][2] + g2[1][3])*wy*wy;
			const float a2 = 0.5f*(2.0f*g2[1][0] - 5.0f*g2[1][1] + 4.0f*g2[1][2] - g2[1][3])*wy;
			const float a1 = 0.5f*(-g2[1][0] + g2[1][2]);
			g3[1][iz] = 3.0f*a3 + 2.0f*a2 + a1;						/* f.y */
			g3[2][iz] = a3*wy + a2*wy + a1*wy + g2[1][1]; /* f.z */
		}
	}

	// Mix along z.
	{
		f.x = -0.5f*(-g3[0][0] + 3.0f*g3[0][1] - 3.0f*g3[0][2] + g3[0][3])*wz*wz*wz +
			-0.5f*(2.0f*g3[0][0] - 5.0f*g3[0][1] + 4.0f*g3[0][2] - g3[0][3])*wz*wz +
			-0.5f*(-g3[0][0] + g3[0][2])                                    *wz -
			g3[0][1];
	}
	{
		f.y = -0.5f*(-g3[1][0] + 3.0f*g3[1][1] - 3.0f*g3[1][2] + g3[1][3])*wz*wz*wz +
			-0.5f*(2.0f*g3[1][0] - 5.0f*g3[1][1] + 4.0f*g3[1][2] - g3[1][3])*wz*wz +
			-0.5f*(-g3[1][0] + g3[1][2])                                    *wz -
			g3[1][1];
	}
	{
		f.z = -1.5f*(-g3[2][0] + 3.0f*g3[2][1] - 3.0f*g3[2][2] + g3[2][3])*wz*wz -
			(2.0f*g3[2][0] - 5.0f*g3[2][1] + 4.0f*g3[2][2] - g3[2][3])      *wz -
			0.5f*(-g3[2][0] + g3[2][2]);
	}
	float e = 0.5f*(-g3[2][0] + 3.0f*g3[2][1] - 3.0f*g3[2][2] + g3[2][3])*wz*wz*wz +
		0.5f*(2.0f*g3[2][0] - 5.0f*g3[2][1] + 4.0f*g3[2][2] - g3[2][3])    *wz*wz +
		0.5f*(-g3[2][0] + g3[2][2])                                        *wz +
		g3[2][1];
	
	return ForceEnergy(f,e);
}


DEVICE float RigidBodyGrid::interpolatePotentialLinearly(const Vector3& l) const {
	// Find the home node.
	const int homeX = int(floor(l.x));
	const int homeY = int(floor(l.y));
	const int homeZ = int(floor(l.z));

	float g3[2];
	for (int iz = 0; iz < 2; iz++) {
		float g2[2];
		for (int iy = 0; iy < 2; iy++) {
			// Fetch values from nearby
			float g1[2];
			for (volatile int ix = 0; ix < 2; ix++) {
				volatile int jx = (ix + homeX);
				volatile int jy = (iy + homeY);
				volatile int jz = (iz + homeZ);
				const int ind = jz + jy*nz + jx*nz*ny;
				g1[ix] = jz < 0 || jz >= nz || jy < 0 || jy >= ny || jx < 0 || jx >= nx ?
					0 : val[ind];
			}
			// Mix along x.
			const float wx = l.x - homeX;
			g2[iy] = wx * (g1[1] - g1[0]) + g1[0];
		}
		// Mix along y.
		const float wy = l.y - homeY;
		g3[iz] = wy * (g2[1] - g2[0]) + g2[0];
	}
	// Mix along z.
	const float wz = l.z - homeZ;
	return wz * (g3[1] - g3[0]) + g3[0];
}
DEVICE ForceEnergy RigidBodyGrid::interpolateForceDLinearly(const Vector3& l) const {
	// Find the home node.
	const int homeX = int(floor(l.x));
	const int homeY = int(floor(l.y));
	const int homeZ = int(floor(l.z));

	Vector3 f;

	const float wx = l.x - homeY;
	const float wy = l.y - homeY;	
	const float wz = l.z - homeZ;

	float v[2][2][2];
	for (int iz = 0; iz < 2; iz++) {
		int jz = (iz + homeZ);
		for (int iy = 0; iy < 2; iy++) {
			int jy = (iy + homeY);
			for (int ix = 0; ix < 2; ix++) {
				int jx = (ix + homeX);
				int ind = jz + jy*nz + jx*nz*ny;
				v[ix][iy][iz] = jz < 0 || jz >= nz || jy < 0 || jy >= ny || jx < 0 || jx >= nx ?
					0 : val[ind];
			}
		}
	}

	float g3[3][2];
	for (int iz = 0; iz < 2; iz++) {
		float g2[2][2];
		for (int iy = 0; iy < 2; iy++) {
			g2[0][iy] = (v[1][iy][iz] - v[0][iy][iz]); /* f.x */
			g2[1][iy] = wx * (v[1][iy][iz] - v[0][iy][iz]) + v[0][iy][iz]; /* f.y & f.z */
		}
		// Mix along y.
		g3[0][iz] = wy * (g2[0][1] - g2[0][0]) + g2[0][0];
		g3[1][iz] = (g2[1][1] - g2[1][0]) + g2[1][0];
		g3[2][iz] = wy * (g2[1][1] - g2[1][0]) + g2[1][0];
	}
	// Mix along z.
	f.x = -(wz * (g3[0][1] - g3[0][0]) + g3[0][0]);
	f.y = -(wz * (g3[1][1] - g3[1][0]) + g3[1][0]);
	f.z = -      (g3[2][1] - g3[2][0]);
	float e = wz * (g3[2][1] - g3[2][0]) + g3[2][0];
	return ForceEnergy(f,e);
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
				int ind = wrap(jz+iz,nz) + nz*wrap(jy+iy,ny) + ny*nz*wrap(jx+ix,nx);
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
				int ind = wrap(homeZ+iz,nz) + nz*wrap(homeY+iy,ny) + ny*nz*wrap(homeX+ix,nx);
				neigh->v[ix+1][iy+1][iz+1] = val[ind];
			}
		}
	}
}  
