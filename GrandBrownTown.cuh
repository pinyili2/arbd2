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
void clearInternalForces(Vector3 forceInternal[], int num, int numReplicas) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num * numReplicas)
		forceInternal[idx] = Vector3(0.0f);
}
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

	 	/* printf("atom %d: forceInternal: %f %f %f\n", idx, forceInternal[idx].x, forceInternal[idx].y, forceInternal[idx].z);  */

		// Compute external force
		Vector3 forceExternal = Vector3(0.0f, 0.0f, pt.charge * electricField);

		// Compute PMF
		ForceEnergy fe = pt.pmf->interpolateForceDLinearly(p);

#ifndef FORCEGRIDOFF
		// Add a force defined via 3D FORCE maps (not 3D potential maps)
		if (pt.forceXGrid != NULL) fe.f.x += pt.forceXGrid->interpolatePotentialLinearly(p);
		if (pt.forceYGrid != NULL) fe.f.y += pt.forceYGrid->interpolatePotentialLinearly(p);
		if (pt.forceZGrid != NULL) fe.f.z += pt.forceZGrid->interpolatePotentialLinearly(p);		
#endif

		// Compute total force:
		//	  Internal:  interaction between particles
		//	  External:  electric field (now this is basically a constant vector)
		//	  forceGrid: ADD force due to PMF or other potentials defined in 3D space
		Vector3 force = forceInternal[idx] + forceExternal + fe.f;

		if (idx == 0)
			forceInternal[idx] = force; // write it back out for force0 in run()

		// Get local kT value
		float kTlocal = (tGridLength == 0) ? kT : kTGrid->interpolatePotentialLinearly(p); /* periodic */

		// Update the particle's position using the calculated values for time, force, etc.
		float diffusion = pt.diffusion;
		Vector3 diffGrad = Vector3(0.0f);

		if (pt.diffusionGrid != NULL) {
			// printf("atom %d: pos: %f %f %f\n", idx, p.x, p.y, p.z);
			// p = pt.diffusionGrid->wrap(p); // illegal mem access; no origin/basis?

		Vector3 gridCenter = pt.diffusionGrid->origin +
			pt.diffusionGrid->basis.transform( Vector3(0.5*pt.diffusionGrid->nx,
																								 0.5*pt.diffusionGrid->ny,
																								 0.5*pt.diffusionGrid->nz)); 
		Vector3 p2 = p - gridCenter;
		p2 = sys->wrapDiff( p2 ) + gridCenter;
		/* p2 = sys->wrap( p2 ); */
		/* p2 = p2 - gridCenter; */
		/* printf("atom %d: ps2: %f %f %f\n", idx, p2.x, p2.y, p2.z); */
		
		ForceEnergy diff = pt.diffusionGrid->interpolateForceDLinearlyPeriodic(p2);
			diffusion = diff.e;
			diffGrad = diff.f;
		}
		
		/* printf("atom %d: pos: %f %f %f\n", idx, p.x, p.y, p.z); */
		/* printf("atom %d: force: %f %f %f\n", idx, force.x, force.y, force.z); */
		/* printf("atom %d: kTlocal, diffusion, timestep: %f, %f, %f\n", */
		/* 			 idx, kTlocal, diffusion, timestep); */
		
		pos[idx] = step(p, kTlocal, force, diffusion, -diffGrad, timestep, sys, randoGen, num);
	}
}

__device__
Vector3 step(Vector3 r0, float kTlocal, Vector3 force, float diffusion,
						 Vector3 diffGrad, float timestep, BaseGrid *sys,
						 Random *randoGen, int num) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	Vector3 rando = randoGen->gaussian_vector(idx, num);

	diffusion *= timestep;
	
	Vector3 r = r0 + (diffusion / kTlocal) * force
							+ timestep * diffGrad
							+ sqrtf(2.0f * diffusion) * rando;
	Vector3 l = sys->getInverseBasis().transform(r - sys->getOrigin());
	l.x = sys->wrapFloat(l.x, sys->getNx());
 	l.y = sys->wrapFloat(l.y, sys->getNy());
 	l.z = sys->wrapFloat(l.z, sys->getNz());
	return sys->getBasis().transform(l) + sys->getOrigin();
}

__global__ void devicePrint(RigidBodyType* rb[]) {
	// printf("Device printing\n");
	int i = 0;
	printf("RigidBodyType %d: numGrids = %d\n", i, rb[i]->numPotGrids);
	printf("  RigidBodyType %d: potGrid: %p\n", i, rb[i]->rawPotentialGrids);
	int j = 0;
	printf("  RigidBodyType %d: potGrid[%d]: %p\n", i, j, &(rb[i]->rawPotentialGrids[j]));
	printf("  RigidBodyType %d: potGrid[%d] size: %d\n", i, j, rb[i]->rawPotentialGrids[j].getSize());
	// BaseGrid g = rb[i]->rawPotentialGrids[j];
	// for (int k = 0; k < rb[i]->rawPotentialGrids[j].size(); k++)
	for (int k = 0; k < rb[i]->rawPotentialGrids[j].getSize(); k++)
		printf("    rbType_d[%d]->potGrid[%d].val[%d]: %g\n",
					 i, j, k, rb[i]->rawPotentialGrids[j].val[k]);
	// i, j, k, rb[i]->rawPotentialGrids[j]).val[k];
	
}

// __global__ void devicePrint(RigidBodyType* rb[]) {
// 	// printf("Device printing\n");
// 	int i = 0;
// 	printf("RigidBodyType %d: numGrids = %d\n", i, rb[i]->numPotGrids);
// 	printf("RigidBodyType %d: potGrid: %p\n", i, rb[i]->rawPotentialGrids);
// 	int j = 0;
// 	printf("RigidBodyType %d: potGrid[%d]: %p\n", i, &(rb[i]->rawPotentialGrids[j]));
// 	BaseGrid g = rb[i]->rawPotentialGrids[j];
// 	// for (int k = 0; k < rb[i]->rawPotentialGrids[j].size(); k++)
// 	for (int k = 0; k < g->getSize(); k++)
// 		printf("rbType_d[%d]->potGrid[%d].val[%d]: %g\n",
// 					 i, j, k, g.val[k]);
// 	// i, j, k, rb[i]->rawPotentialGrids[j]).val[k];
	
// }
