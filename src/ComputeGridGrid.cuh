// Included in RigidBodyController.cu
#pragma once
#include "useful.h"
#define NUMTHREADS 96
#define WARPSIZE 32

class RigidBodyGrid;

extern __global__
void computeGridGridForce(const RigidBodyGrid* rho, const RigidBodyGrid* u,
				const Matrix3 basis_rho, const Matrix3 basis_u_inv,
				const Vector3 origin_rho_minus_origin_u,
				Vector3 * retForce, Vector3 * retTorque);

extern __global__
void computePartGridForce(const Vector3* __restrict__ pos, Vector3* particleForce,
				const int num, const int* __restrict__ particleIds,
				const RigidBodyGrid* __restrict__ u,
				const Matrix3 basis_u_inv, const Vector3 origin_u,
				Vector3* __restrict__ retForce, Vector3* __restrict__ retTorque);

extern __global__
void createPartlist(const Vector3* __restrict__ pos,
				const int numTypeParticles, const int* __restrict__ typeParticles_d,
				int* numParticles_d, int* particles_d,
				const Vector3 gridCenter, const float radius2);
	
extern __global__
void printRigidBodyGrid(const RigidBodyGrid* rho);
