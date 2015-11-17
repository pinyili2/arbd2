#pragma once
/* #include "RigidBodyGrid.h" */
/* #include "useful.h" */

__global__
void computeGridGridForce(const RigidBodyGrid& rho, const RigidBodyGrid& u,
													Matrix3 basis_rho, Vector3 origin_rho,
													Matrix3 basis_u,   Vector3 origin_u) {
  // RBTODO http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
	const unsigned int r_id = blockIdx.x * blockDim.x + threadIdx.x;
	
	// RBTODO parallelize transform
	if (r_id > rho.size)					// skip threads with no data 
		return;
	
	// Maybe: Tile grid data into shared memory
	//   RBTODO: think about localizing regions of grid data
	Vector3 p = rho.getPosition(r_id, basis_rho, origin_rho);
	float val = rho.val[r_id];

	// RBTODO potential basis and rho!
	
	// RBTODO combine interp methods and reduce repetition! 
	float energy = val*u.interpolatePotential(p); 
	Vector3 f = val*u.interpolateForceD(p);
	Vector3 t = p.cross(f);				// test if sign is correct!

	// RBTODO reduce forces and torques
	// http://www.cuvilib.com/Reduction.pdf

	// RBTODO 3rd-law forces + torques (maybe outside kernel)

	// must reference rigid bodies in some way
}
