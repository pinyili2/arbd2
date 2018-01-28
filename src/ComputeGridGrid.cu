// Included in RigidBodyController.cu
#include "ComputeGridGrid.cuh"
#include "RigidBodyGrid.h"
#include "CudaUtil.cuh"
//RBTODO handle periodic boundaries
//RBTODO: add __restrict__, benchmark (Q: how to restrict member data?)
__global__
void computeGridGridForce(const RigidBodyGrid* rho, const RigidBodyGrid* u, const Matrix3 basis_rho, const Matrix3 basis_u_inv, const Vector3 origin_rho_minus_origin_u,
			ForceEnergy* retForce, Vector3 * retTorque, int scheme) 
{

	extern __shared__ ForceEnergy s[];
	ForceEnergy *force = s;
	//Vector3 *torque = &s[NUMTHREADS];
        ForceEnergy *torque = &s[NUMTHREADS];

  // RBTODO: http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops
	const int tid = threadIdx.x;
	const int r_id = blockIdx.x * blockDim.x + threadIdx.x;

	force[tid] = ForceEnergy(0.f,0.f);
	torque[tid] = ForceEnergy(0.f,0.f);
	if (r_id < rho->getSize()) { // skip threads with nothing to do
		// RBTODO: reduce registers used;
		//   commenting out interpolateForceD still uses ~40 registers
		//   -- the innocuous-looking fn below is responsible; consumes ~17 registers!
		Vector3 r_pos= rho->getPosition(r_id); /* i,j,k value of voxel */

		r_pos = basis_rho.transform( r_pos ) + origin_rho_minus_origin_u; /* real space */
		const Vector3 u_ijk_float = basis_u_inv.transform( r_pos );
		// RBTODO: Test for non-unit delta
		/* Vector3 tmpf  = Vector3(0.0f); */
		/* float tmpe = 0.0f; */
		/* const ForceEnergy fe = ForceEnergy( tmpf, tmpe); */

                ForceEnergy fe;
                if(!scheme)
		    fe = u->interpolateForceDLinearly( u_ijk_float ); /* in coord frame of u */
                else
                    fe = u->interpolateForceD( u_ijk_float );

		force[tid] = fe;
                //force[tid].e = fe.e;

		const float r_val = rho->val[r_id]; /* maybe move to beginning of function?  */
		force[tid].f = basis_u_inv.transpose().transform( r_val*(force[tid].f) ); /* transform to lab frame, with correct scaling factor */
		// Calculate torque about origin_u in the lab frame
		torque[tid].f = r_pos.cross(force[tid].f);
	}

	// Reduce force and torques
	// http://www.cuvilib.com/Reduction.pdf
	// RBTODO optimize further, perhaps
	// assert( NUMTHREADS==32 || NUMTHREADS==64 || NUMTHREADS==128 || NUMTHREADS==256 || NUMTHREADS==512 );
	__syncthreads();
	for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
		if (tid < offset) {
			int oid = tid + offset;
                        //if(get_energy)
                            //force[tid].e = force[tid].e + force[oid].e;
			force[tid] = force[tid] + force[oid];
			torque[tid] = torque[tid] + torque[oid];
		}
		__syncthreads();
	}

	if (tid == 0) {
                //retForce[blockIdx.x].e = force[0].e;
		retForce[blockIdx.x] = force[0];
		retTorque[blockIdx.x] = torque[0].f;
	}
}

__global__
void computePartGridForce(const Vector3* __restrict__ pos, Vector3* particleForce,
				const int num, const int* __restrict__ particleIds, 
				const RigidBodyGrid* __restrict__ u,
				const Matrix3 basis_u_inv, const Vector3 origin_u,
				ForceEnergy* __restrict__ retForce, Vector3* __restrict__ retTorque, float* __restrict__ energy, bool get_energy, int scheme) {

	extern __shared__ ForceEnergy s[];
	ForceEnergy *force  = s;
	ForceEnergy *torque = &s[NUMTHREADS];
  	
	const int tid = threadIdx.x;
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	force[tid]  = ForceEnergy(0.f, 0.f);
	torque[tid] = ForceEnergy(0.f,0.f);
	if (i < num) {
		const int id = particleIds[i];
		Vector3 p = pos[id] - origin_u;
		// TODO: wrap to center of u
		const Vector3 u_ijk_float = basis_u_inv.transform( p );

                ForceEnergy fe;
                if(!scheme)                       
		    fe = u->interpolateForceDLinearly( u_ijk_float ); /* in coord frame of u */
                else
                    fe = u->interpolateForceD( u_ijk_float );
                
		force[tid] = fe;
                //force[tid].e = fe.e;
                if(get_energy)
                    atomicAdd(&energy[id], fe.e);
		force[tid].f = basis_u_inv.transpose().transform( force[tid].f ); /* transform to lab frame */
		atomicAdd( &particleForce[id], force[tid].f ); // apply force to particle
		
		// Calculate torque about origin_u in the lab frame
		torque[tid].f = p.cross(force[tid].f);				// RBTODO: test sign
	}

	// Reduce force and torques
	// assert( NUMTHREADS==32 || NUMTHREADS==64 || NUMTHREADS==128 || NUMTHREADS==256 || NUMTHREADS==512 );
	__syncthreads();
	for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
		if (tid < offset) {
			int oid = tid + offset;
                        //if(get_energy)
                            //force[tid].e = force[tid].e + force[oid].e;
			force[tid] = force[tid] + force[oid];
			torque[tid] = torque[tid] + torque[oid];
		}
		__syncthreads();
	}
	
	if (tid == 0) {
		retForce[blockIdx.x] = force[0];
		retTorque[blockIdx.x] = torque[0].f;
	}
}

__global__
void createPartlist(const Vector3* __restrict__ pos,
				const int numTypeParticles, const int* __restrict__ typeParticles_d,
				int* numParticles_d, int* particles_d,
				const Vector3 gridCenter, const float radius2) {
	const int tid = threadIdx.x;
	const int warpLane = tid % WARPSIZE; /* RBTODO: optimize */
	
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numTypeParticles) {
		int aid = typeParticles_d[i];
		float dist = (pos[aid] - gridCenter).length2();

		if (dist <= radius2) {
			int tmp = atomicAggInc(numParticles_d, warpLane);
			particles_d[tmp] = aid;
		}
	}
}		

__global__
void printRigidBodyGrid(const RigidBodyGrid* rho) {
  printf("Printing an RB of size %d\n",rho->size);
  for (int i=0; i < rho->size; i++)
	printf("  val[%d] = %f\n", i, rho->val[i]);
}
