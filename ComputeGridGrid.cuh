#pragma once
#include "RigidBodyGrid.h"

__global__
void computeGridGridForce(RigidBodyGrid* rho, RigidBodyGrid* u) {
	unsigned int bidx = blockIdx.x;
	unsigned int tidx = threadIdx.x;

	// GTX 980 has ~ 4 x 10x10x10 floats available for shared memory
	//    but code should work without access to this
	
	// RBTODO rewrite to use shared memory, break problem into subgrids
	// will lookups of 

// http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/

	const unsigned int r_id = bidx * blockDim.x + threadIdx.x;
	
	// RBTODO parallelize transform
	if (r_id > rho->size)					// skip threads with no data 
		return;

	// Tile grid data into shared memory
	//   RBTODO: think about localizing regions of grid data
	Vector3 p = rho->getPosition(r_id);
	float val = rho->val[r_id];

	// RBTODO reduce these
	// http://www.cuvilib.com/Reduction.pdf
	float energy = ;
	Vector3 f = u->interpolateForceD(p);
	Vector3 t = f * 
		
}
