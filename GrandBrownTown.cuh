// GrandBrownTown.cuh
//
// Terrance Howard <heyterrance@gmail.com>

#pragma once

__device__
Vector3 step(Vector3 r0, float kTlocal, Vector3 force, float diffusion,
						 float timestep, BaseGrid *sys, Random *randoGen, int num);

__device__
Vector3 step(Vector3 r0, float kTlocal, Vector3 force, float diffusion,
						 Vector3 diffGrad, float timestep, BaseGrid *sys,
						 Random *randoGen, int num);

__global__
void updateKernel(Vector3 pos[], Vector3 forceInternal[],
									int type[], BrownianParticleType* part[],
									float kT, BaseGrid* kTGrid,
									float electricField, int tGridLength,
									float timestep, int num, BaseGrid* sys,
									Random* randoGen, int numReplicas) {
	// Calculate this thread's ID
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Loop over ALL particles in ALL replicas
	if (idx < num * numReplicas) {
		const int t = type[idx];
		Vector3& p = pos[idx];

		const BrownianParticleType& pt = *part[t];

		// Compute external force
		Vector3 forceExternal = Vector3(0.0f, 0.0f, pt.charge * electricField);

		BaseGrid* pmf = pt.pmf;

		// Find the home node.
		Vector3 l = pmf->getInverseBasis().transform(p - pmf->getOrigin());
		int homeX = int(floorf(l.x));
		int homeY = int(floorf(l.y));
		int homeZ = int(floorf(l.z));

		// Shift the indices in the grid dimensions.
		int gX = pmf->getNx();
		int gY = pmf->getNy();
		int gZ = pmf->getNz();

		// Get the interpolation coordinates.
		float w[3];
		w[0] = l.x - homeX;
		w[1] = l.y - homeY;
		w[2] = l.z - homeZ;

		// Find the values at the neighbors.
		float g1[4][4][4];
		for (int ix = 0; ix < 4; ix++) {
			int jx = ix-1 + homeX;
			jx = pmf->wrap(jx, gX);

			for (int iy = 0; iy < 4; iy++) {
				int jy = iy-1 + homeY;
				jy = pmf->wrap(jy, gY);

				for (int iz = 0; iz < 4; iz++) {
					int jz = iz-1 + homeZ;
					jz = pmf->wrap(jz, gZ);
					// Wrap around the periodic boundaries.
					int ind = jz + jy*gZ + jx*gZ*gY;
					g1[ix][iy][iz] = pmf->val[ind];
				}
			}
		}

		// Interpolate this particle's change in X, Y, and Z axes
		Vector3 f;
		f.x = pmf->interpolateDiffX(p, w, g1);
		f.y = pmf->interpolateDiffY(p, w, g1);
		f.z = pmf->interpolateDiffZ(p, w, g1);
		Vector3 forceGrid = pmf->getInverseBasis().transpose().transform(f);


#ifndef FORCEGRIDOFF
		// Add a force defined via 3D FORCE maps (not 3D potential maps)
		if (pt.forceXGrid != NULL) forceGrid.x += pt.forceXGrid->interpolatePotential(p);
		if (pt.forceYGrid != NULL) forceGrid.y += pt.forceYGrid->interpolatePotential(p);
		if (pt.forceZGrid != NULL) forceGrid.z += pt.forceZGrid->interpolatePotential(p);
#endif

		// Compute total force:
		//	  Internal:  interaction between particles
		//	  External:  electric field (now this is basically a constant vector)
		//	  forceGrid: ADD force due to PMF or other potentials defined in 3D space
		Vector3 force = forceInternal[idx] + forceExternal + forceGrid;

		if (idx == 0)
			forceInternal[idx] = force; // write it back out for force0 in run()

		// Get local kT value
		float kTlocal = (tGridLength == 0) ? kT : kTGrid->interpolatePotential(p);

		// Update the particle's position using the calculated values for time, force, etc.
		if (pt.diffusionGrid == NULL) {
			p = step(p, kTlocal, force, pt.diffusion, timestep, sys, randoGen, num);
		} else {
			float diffusion = pt.diffusionGrid->interpolatePotential(p);
			Vector3 diffGrad = pt.diffusionGrid->interpolateForceD(p);
			p = step(p, kTlocal, force, diffusion, -diffGrad, timestep, sys, randoGen, num);
		}
	}
}

__device__
Vector3 step(Vector3 r0, float kTlocal, Vector3 force, float diffusion,
						 float timestep, BaseGrid *sys, Random *randoGen, int num) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	Vector3 rando = randoGen->gaussian_vector(idx, num);
	Vector3 r = r0 + (force * diffusion * timestep / kTlocal) + (sqrtf(2.0f * diffusion * timestep) * rando);
	Vector3 l = sys->getInverseBasis().transform(r - sys->getOrigin());
	l.x = sys->wrapFloat(l.x, sys->getNx());
	l.y = sys->wrapFloat(l.y, sys->getNy());
	l.z = sys->wrapFloat(l.z, sys->getNz());
	return sys->getBasis().transform(l) + sys->getOrigin();
}

__device__
Vector3 step(Vector3 r0, float kTlocal, Vector3 force, float diffusion,
						 Vector3 diffGrad, float timestep, BaseGrid *sys,
						 Random *randoGen, int num) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	Vector3 rando = randoGen->gaussian_vector(idx, num);

	Vector3 r = r0 + (diffusion * timestep / kTlocal) * force
							+ timestep * diffGrad
							+ sqrtf(2.0f * diffusion * timestep) * rando;

	Vector3 l = sys->getInverseBasis().transform(r - sys->getOrigin());
	l.x = sys->wrapFloat(l.x, sys->getNx());
 	l.y = sys->wrapFloat(l.y, sys->getNy());
 	l.z = sys->wrapFloat(l.z, sys->getNz());

	return sys->getBasis().transform(l) + sys->getOrigin();
}

__global__ void devicePrint(RigidBodyType* rb) {
	printf("RigidBodyType: numGrids = %d\n", rb->numPotGrids);
}
// __device__ void devicePrint(BaseGrid g) {
// 	printf("RigidBodyType: numGrids = %d\n", numPotGrids);
// };
