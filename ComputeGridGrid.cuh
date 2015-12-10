#pragma once
/* #include "RigidBodyGrid.h" */
/* #include "useful.h" */

__global__
void computeGridGridForce(const RigidBodyGrid* rho, const RigidBodyGrid* u,
													Matrix3 basis_rho, Vector3 origin_rho,
													Matrix3 basis_u,   Vector3 origin_u,
													Vector3 * retForce, Vector3 * retTorque) {

	printf("ComputeGridGridForce called\n");
	extern __shared__ Vector3 force [];
	extern __shared__ Vector3 torque [];
	
  // RBTODO http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
	const unsigned int tid = threadIdx.x;
	const unsigned int r_id = blockIdx.x * blockDim.x + threadIdx.x;
	
	// RBTODO parallelize transform
	if (r_id > rho->size) {				// skip threads with no data
		force[tid] = Vector3(0.0f);
		torque[tid] = Vector3(0.0f);
		return;
	}
	// Maybe: Tile grid data into shared memory
	//   RBTODO: think about localizing regions of grid datas
	Vector3 r_ijk = rho->getPosition(r_id); /* i,j,k value of voxel */
	Vector3 r_pos = basis_rho.transform( r_ijk ) + origin_rho;
	Vector3 u_ijk_float = basis_u.transform( r_pos - origin_u );

	
	float r_val = rho->val[r_id];
	float energy = r_val * u->interpolatePotential( u_ijk_float ); 
	
	// RBTODO combine interp methods and reduce repetition! 
	Vector3 f = r_val*u->interpolateForceD( u_ijk_float ); /* in coord frame of u */
	f = basis_u.inverse().transpose().transform( f ); /* transform to lab frame */

	// Calculate torque about lab-frame origin 
	Vector3 t = r_pos.cross(f);				// RBTODO: test if sign is correct!
	

	force[tid] = f;
	torque[tid] = t;
	__syncthreads();

	// Reduce force and torques
	// http://www.cuvilib.com/Reduction.pdf
	// RBTODO optimize
	for (unsigned int offset = blockDim.x/2; offset > 0; offset >>= 1) {
		if (tid < offset) {
			unsigned int oid = tid + offset;
			force[tid] = force[tid] + force[oid];
			torque[tid] = torque[tid] + torque[oid];
		}
		__syncthreads();
	}

	if (tid == 0) {
		retForce[blockIdx.x] = force[0];
		retTorque[blockIdx.x] = force[0];
	}
	
	// RBTODO 3rd-law forces + torques (maybe outside kernel)
	// must reference rigid bodies in some way
}
