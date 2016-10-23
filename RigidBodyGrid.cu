//////////////////////////////////////////////////////////////////////
// Grid base class that does just the basics.
// Author: Jeff Comer <jcomer2@illinois.edu>

#include "RigidBodyGrid.h"
#include <cuda.h>

#define STRLEN 512

	/*                               \
	| CONSTRUCTORS, DESTRUCTORS, I/O |
	\===============================*/

RigidBodyGrid::RigidBodyGrid() {
	RigidBodyGrid tmp(1,1,1);
	val = new float[1];
	*this = tmp;									// TODO: verify that this is OK
}

// The most obvious of constructors.
RigidBodyGrid::RigidBodyGrid(int nx0, int ny0, int nz0) {
	nx = abs(nx0);
	ny = abs(ny0);
	nz = abs(nz0);
	
	val = new float[nx*ny*nz];
	zero();
}

RigidBodyGrid::RigidBodyGrid(const BaseGrid& g) {
	nx = g.nx;
	ny = g.ny;
	nz = g.nz;
	
	val = new float[nx*ny*nz];
	for (int i = 0; i < nx*ny*nz; i++) val[i] = g.val[i];
}

// Make an exact copy of a grid.
RigidBodyGrid::RigidBodyGrid(const RigidBodyGrid& g) {
	nx = g.nx;
	ny = g.ny;
	nz = g.nz;
	
	val = new float[nx*ny*nz];
	for (int i = 0; i < nx*ny*nz; i++) val[i] = g.val[i];
}

RigidBodyGrid RigidBodyGrid::mult(const RigidBodyGrid& g) {
	for (int i = 0; i < nx*ny*nz; i++) val[i] *= g.val[i];
	return *this;
}

RigidBodyGrid& RigidBodyGrid::operator=(const RigidBodyGrid& g) {
	delete[] val;
	val = NULL;
	nx = g.nx;
	ny = g.ny;
	nz = g.nz;
	
	val = new float[nx*ny*nz];
	for (int i = 0; i < nx*ny*nz; i++) val[i] = g.val[i];

	return *this;
}

RigidBodyGrid::~RigidBodyGrid() {
	if (val != NULL)
		delete[] val;
}

void RigidBodyGrid::zero() {
	for (int i = 0; i < nx*ny*nz; i++) val[i] = 0.0f;
}

bool RigidBodyGrid::setValue(int j, float v) {
	if (j < 0 || j >= nx*ny*nz) return false;
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
	if (j < 0 || j >= nx*ny*nz) return 0.0f;
	return val[j];
}

float RigidBodyGrid::getValue(int ix, int iy, int iz) const {
	if (ix < 0 || ix >= nx) return 0.0f;
	if (iy < 0 || iy >= ny) return 0.0f;
	if (iz < 0 || iz >= nz) return 0.0f;
	
	int j = iz + iy*nz + ix*ny*nz;
	return val[j];
}

Vector3 RigidBodyGrid::getPosition(const int j) const {
	/* const int iz = j%nz; */
	/* const int iy = (j/nz)%ny; */
	/* const int ix = j/(nz*ny); */
	const int jy = j/nz;
	const int jx = jy/ny;

	const int iz = j - jy*nz;
	const int iy = jy - jx*ny;
	// const int ix = jx;

	return Vector3(jx,iy,iz);
}

Vector3 RigidBodyGrid::getPosition(int j, Matrix3 basis, Vector3 origin) const {
	int iz = j%nz;
	int iy = (j/nz)%ny;
	int ix = j/(nz*ny);

	return basis.transform(Vector3(ix, iy, iz)) + origin;
}

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

// Add a fixed value to the grid.
void RigidBodyGrid::shift(float s) {
	for (int i = 0; i < nx*ny*nz; i++) val[i] += s;
}

// Multiply the grid by a fixed value.
void RigidBodyGrid::scale(float s) {
	for (int i = 0; i < nx*ny*nz; i++) val[i] *= s;
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
	f.x = -0.5f*(-g3[0][0] + 3.0f*g3[0][1] - 3.0f*g3[0][2] + g3[0][3])*wz*wz*wz +
		-0.5f*(2.0f*g3[0][0] - 5.0f*g3[0][1] + 4.0f*g3[0][2] - g3[0][3])*wz*wz +
		-0.5f*(-g3[0][0] + g3[0][2])                                    *wz -
		g3[0][1];
	f.y = -0.5f*(-g3[1][0] + 3.0f*g3[1][1] - 3.0f*g3[1][2] + g3[1][3])*wz*wz*wz +
		-0.5f*(2.0f*g3[1][0] - 5.0f*g3[1][1] + 4.0f*g3[1][2] - g3[1][3])*wz*wz +
		-0.5f*(-g3[1][0] + g3[1][2])                                    *wz -
		g3[1][1];
	f.z = -1.5f*(-g3[2][0] + 3.0f*g3[2][1] - 3.0f*g3[2][2] + g3[2][3])*wz*wz -
		(2.0f*g3[2][0] - 5.0f*g3[2][1] + 4.0f*g3[2][2] - g3[2][3])      *wz -
		0.5f*(-g3[2][0] + g3[2][2]);
	float e = 0.5f*(-g3[2][0] + 3.0f*g3[2][1] - 3.0f*g3[2][2] + g3[2][3])*wz*wz*wz +
		0.5f*(2.0f*g3[2][0] - 5.0f*g3[2][1] + 4.0f*g3[2][2] - g3[2][3])    *wz*wz +
		0.5f*(-g3[2][0] + g3[2][2])                                        *wz +
		g3[2][1];
	
	return ForceEnergy(f,e);
}

DEVICE ForceEnergy RigidBodyGrid::interpolateForceDLinearly(const Vector3& l) const {
	// Find the home node.
	const int homeX = int(floor(l.x));
	const int homeY = int(floor(l.y));
	const int homeZ = int(floor(l.z));

	Vector3 f;

	const float wx = l.x - homeX;
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
		g3[1][iz] = (g2[1][1] - g2[1][0]);
		g3[2][iz] = wy * (g2[1][1] - g2[1][0]) + g2[1][0];
	}
	// Mix along z.
	f.x = -(wz * (g3[0][1] - g3[0][0]) + g3[0][0]);
	f.y = -(wz * (g3[1][1] - g3[1][0]) + g3[1][0]);
	f.z = -      (g3[2][1] - g3[2][0]);
	float e = wz * (g3[2][1] - g3[2][0]) + g3[2][0];
	return ForceEnergy(f,e);
}

