#pragma once
/* #include "RigidBodyGrid.h" */
/* #include "useful.h" */

// #define NUMTHREADS 256

__global__
void computeGridGridForce(const RigidBodyGrid* rho, const RigidBodyGrid* u,
													const Matrix3 basis_rho, const Vector3 origin_rho,
													const Matrix3 basis_u_inv, const Vector3 origin_u,
													Vector3 * retForce, Vector3 * retTorque, int gridNum) {

	// printf("ComputeGridGridForce called\n");
	/* extern __shared__ Vector3 force []; */
	__shared__ Vector3 force [NUMTHREADS];
	__shared__ Vector3 torque [NUMTHREADS];
	
  // RBTODO http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
	const int tid = threadIdx.x;
	const int r_id = blockIdx.x * blockDim.x + threadIdx.x;

	/* return; */
	// RBTODO parallelize transform
	if (r_id >= rho->size) {				// skip threads with no data
		force[tid] = Vector3(0.0f);
		torque[tid] = Vector3(0.0f);
		return;
	}
	/* printf("tid,r_id: %d,%d\n",tid,r_id); */
	/* if (tid == 0) { */
	/* 	printf("%f",rho->val[r_id]); */
	/* } */
	/* __syncthreads; */
	/* return; */

	// RBTODO: reduce registers used; commenting out interpolatePotential / interpolateForceD still uses ~40 registers, otherwise 236!!!
	
	// RBTODO: maybe organize data into compact regions for grid data for better use of shared memory...
	Vector3 r_pos = rho->getPosition(r_id); /* i,j,k value of voxel */
	r_pos = basis_rho.transform( r_pos ) + origin_rho; /* real space */
	// Vector3 u_ijk_float = basis_u.transform( r_pos - origin_u );
	// const Matrix3 basis_u_inv = basis_u.inverse();
	const Vector3 u_ijk_float = basis_u_inv.transform( r_pos - origin_u );
	
	// const float r_val = rho->val[r_id];
	/* const float energy = r_val * u->interpolatePotential( u_ijk_float );  */
	// const float energy = 0.0f;

	// RBTODO What about non-unit delta?
	// RBTODO combine interp methods and reduce repetition! 
	Vector3 f = u->interpolateForceD( u_ijk_float ); /* in coord frame of u */
	const float r_val = rho->val[r_id];
	f = r_val*f;
	f = basis_u_inv.transpose().transform( f ); /* transform to lab frame */
	force[tid] = f;

	// Calculate torque about lab-frame origin 
	torque[tid] = r_pos.cross(f);				// RBTODO: test if sign is correct!
	

	/*/ debug forces
	float cutoff = 1e-3;
	if (gridNum >= 0 && (abs(f.x) >= cutoff || abs(f.y) >= cutoff || abs(f.z) >= cutoff))
		printf("GRIDFORCE: %d %f %f %f %f %f %f\n", gridNum, r_pos.x,r_pos.y,r_pos.z,f.x,f.y,f.z);
	*/
	
	// Reduce force and torques
	// http://www.cuvilib.com/Reduction.pdf
	// RBTODO optimize
	__syncthreads();
	for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
		if (tid < offset) {
			int oid = tid + offset;
			force[tid] = force[tid] + force[oid];
			torque[tid] = torque[tid] + torque[oid];
		}
		__syncthreads();
	}

	if (tid == 0) {
		retForce[blockIdx.x] = force[0];
		retTorque[blockIdx.x] = torque[0];
		/* printf("GPU force: (%f,%f,%f)\n", force[0].x, force[0].y, force[0].z); */
		/* printf("GPU force0: (%f,%f,%f)\n", f.x, f.y, f.z); */
	}
}

__global__
void printRigidBodyGrid(const RigidBodyGrid* rho) {
	printf("Printing an RB of size %d\n",rho->size);
	for (int i=0; i < rho->size; i++)
		printf("  val[%d] = %f\n", i, rho->val[i]);
}
