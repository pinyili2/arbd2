// Included in RigidBodyController.cu
#pragma once


__global__
void computeGridGridForce(const RigidBodyGrid* rho, const RigidBodyGrid* u,
													const Matrix3 basis_rho, const Matrix3 basis_u_inv,
													const Vector3 origin_rho_minus_origin_u,
													Vector3 * retForce, Vector3 * retTorque) {

	extern __shared__ Vector3 s[];
	Vector3 *force = s;
	Vector3 *torque = &s[NUMTHREADS];

  // RBTODO: http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops
	const int tid = threadIdx.x;
	const int r_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (r_id >= rho->getSize()) { // skip threads with nothing to do
		force[tid] = Vector3(0.0f);
		torque[tid] = Vector3(0.0f);
		return;
	}

	// RBTODO: reduce registers used;
	//   commenting out interpolateForceD still uses ~40 registers
	//   -- the innocuous-looking fn below is responsible; consumes ~17 registers!
	Vector3 r_pos= rho->getPosition(r_id); /* i,j,k value of voxel */

	r_pos = basis_rho.transform( r_pos ) + origin_rho_minus_origin_u; /* real space */
	const Vector3 u_ijk_float = basis_u_inv.transform( r_pos );

	// RBTODO What about non-unit delta?
	/* Vector3 tmpf  = Vector3(0.0f); */
	/* float tmpe = 0.0f; */
	/* const ForceEnergy fe = ForceEnergy( tmpf, tmpe); */
	const ForceEnergy fe = u->interpolateForceD( u_ijk_float ); /* in coord frame of u */
	force[tid] = fe.f;

	const float r_val = rho->val[r_id]; /* maybe move to beginning of function?  */
	force[tid] = basis_u_inv.transpose().transform( r_val*force[tid] ); /* transform to lab frame, with correct scaling factor */

	// Calculate torque about origin_u in the lab frame
	torque[tid] = r_pos.cross(force[tid]);				// RBTODO: test if sign is correct!
	

	// Reduce force and torques
	// http://www.cuvilib.com/Reduction.pdf
	// RBTODO optimize further, perhaps
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
	}
}

__global__
void printRigidBodyGrid(const RigidBodyGrid* rho) {
	printf("Printing an RB of size %d\n",rho->size);
	for (int i=0; i < rho->size; i++)
		printf("  val[%d] = %f\n", i, rho->val[i]);
}
