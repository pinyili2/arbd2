// ComputeForce.cuh
//
// Terrance Howard <heyterrance@gmail.com>

#pragma once
#include "CudaUtil.cuh"
#include <assert.h>


__host__ __device__
EnergyForce ComputeForce::coulombForce(Vector3 r, float alpha,
																			 float start, float len) {
	float d = r.length();
	
	if (d >= start + len)
		return EnergyForce();
	if (d <= start) {
		float energy = alpha/d - alpha/start + 0.5f*alpha/(start*start)*len;
		Vector3 force = -alpha/(d*d*d)*r;

		return EnergyForce(energy, force);
	}

	// Switching.
	float c = alpha / (start * start);
	float energy = 0.5f/len * c * (start + len - d) * (start + len - d);
	Vector3 force = -c * (1.0f - (d - start) / len) / d * r;
	return EnergyForce(energy, force);
}

__host__ __device__
EnergyForce ComputeForce::coulombForceFull(Vector3 r, float alpha) {
	float d = r.length();
	return EnergyForce(alpha/d, -alpha/(d*d*d)*r);
}

__host__ __device__
EnergyForce ComputeForce::softcoreForce(Vector3 r, float eps, float rad6) {
	const float d2 = r.length2();
	const float d6 = d2*d2*d2;

	Vector3 force = -12*eps*(rad6*rad6/(d6*d6*d2) - rad6/(d6*d2))*r;
		if (isnan(force.x) or isnan(force.y) or isnan(force.z))
			printf(">>>> Damn.\n");

	if (d6 < rad6) {
		const float d6_2 = d6 * d6;
		const float rad6_2 = rad6 * rad6;
		float e = eps * ((rad6_2 / (d6_2)) - (2.0f * rad6 / d6)) + eps;
		Vector3 f = -12.0f * eps * (rad6_2 / (d6_2 * d2) - rad6 / (d6 * d2)) * r;
		return EnergyForce(e, f);
	}

	return EnergyForce();
}

__global__
void computeFullKernel(Vector3 force[], Vector3 pos[], int type[],
											 float tableAlpha[], float tableEps[], float tableRad6[],
											 int num, int numParts, BaseGrid* sys, float g_energies[],
											 int gridSize, int numReplicas, bool get_energy) {
	// Calculate the ID of each thread.
	// The combination of (X, Y) is unique among all threads
	// id_in_block is unique to a single thread block
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float energy_local = 0.0f;

	// For all threads representing a valid pair of particles
	if (i < num * numReplicas) {
		const int repID = i / num;
		const int typei = type[i];
		const Vector3& posi = pos[i];
		float alpha, eps, rad6;
		Vector3 force_local(0.0f);
		int typej = -1;

		for (int j = repID * num; j < (repID + 1) * num; ++j) {
			if (i == j) continue;
			int newj = type[j];
			if (typej != newj) {
				// Get new values if type[j-1] != type[j]
				// Save time; avoid reading from memory
				typej = newj;
				alpha = tableAlpha[typei * numParts + typej];
				eps = tableEps[typei * numParts + typej];
				rad6 = tableRad6[typei * numParts + typej];
			}
			Vector3 dr = sys->wrapDiff(pos[j] - posi);
			EnergyForce fc = ComputeForce::coulombForceFull(dr, alpha);
			EnergyForce fh = ComputeForce::softcoreForce(dr, eps, rad6);
			// Only update the force in the X particle.
			// Another thread will handle the Y particle
			force_local += fc.f + fh.f;

			// Check if there is a bond between these two.
			// If there is, and the bond's flag is ADD, then add some more force to
			// the interaction.

			// Only update the energy once. The other thread handling x and y will
			// not update the energy.
			if (get_energy && j > i && i < num)
				energy_local += fc.e + fh.e;
		}
		force[i] = force_local;
		if (get_energy && i < num)
			g_energies[i] = energy_local;
	}
}

__global__
void computeSoftcoreFullKernel(Vector3 force[], Vector3 pos[], int type[],
															 float tableEps[], float tableRad6[],
															 int num, int numParts, BaseGrid* sys,
															 float g_energies[], int gridSize,
															 int numReplicas, bool get_energy) {
	// Calculate the ID of each thread.
	// The combination of (X, Y) is unique among all threads
	// id_in_block is unique to a single thread block
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float energy_local = 0.0f;
	// For all threads representing a valid pair of particles
	if (i < num * numReplicas) {
		const int repID = i / num;
		const int typei = type[i];
		const Vector3& posi = pos[i];
		float eps, rad6;
		Vector3 force_local(0.0f);

		int typej = -1;
		for (int j = repID * num; j < (repID + 1) * num; ++j) {
			if (i == j) continue;
			int newj = type[j];
			if (typej != newj) {
				typej = newj;
				eps = tableEps[typei * numParts + typej];
				rad6 = tableRad6[typei * numParts + typej];
			}

			Vector3 dr = sys->wrapDiff(pos[j] - posi);
			EnergyForce fh = ComputeForce::softcoreForce(dr, eps, rad6);
			// Only update the force in the particle i
			// Another thread will handle the particle j
			force_local += fh.f;
			// Only update the energy once. The other thread handling x and y
			// will not update the energy
			if (get_energy && j > i)
				energy_local += fh.e;
		}
		force[i] = force_local;
		if (get_energy)
			g_energies[i] = energy_local;
	}
}

__global__
void computeElecFullKernel(Vector3 force[], Vector3 pos[], int type[],
													 float tableAlpha[], int num, int numParts,
													 BaseGrid* sys, float g_energies[],
													 int gridSize, int numReplicas,
													 bool get_energy) {
	// Calculate the ID of each thread.
	// The combination of (X, Y) is unique among all threads
	// id_in_block is unique to a single thread block
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	float energy_local = 0.0f;
	// For all threads representing a valid pair of particles
	if (i < num * numReplicas) {
		const int repID = i / num;
		const int typei = type[i];
		const Vector3 posi = pos[i];
		float alpha;

		Vector3 force_local(0.0f);
		int typej = -1;
		for (int j = repID * num; j < num * (repID-1); j++) {
			if (i == j) continue;
			int newj = type[j];
			if (typej != newj) {
				typej = newj;
				alpha = tableAlpha[typei * numParts + typej];
			}
			const Vector3 dr = sys->wrapDiff(pos[j] - posi);
			EnergyForce fc = ComputeForce::coulombForceFull(dr, alpha);
			// Only update the force in the X particle.
			// Another thread will handle the Y particle
			force_local += fc.f;
			// Only update the energy once. The other thread handling x and y
			// will not update the energy
			if (get_energy and j > i)
				energy_local += fc.e;
		}
		force[i] = force_local;
		if (get_energy)
			g_energies[i] = energy_local;
	}
}


// RBTODO: remove global device variables for fast prototyping of pairlist kernel
__device__ int g_numPairs = 0;
__device__ int* g_pairI;
__device__ int* g_pairJ;

__global__
void createPairlists(Vector3 pos[], int num, int numReplicas,
										 BaseGrid* sys, CellDecomposition* decomp) {
	  // Loop over threads searching for atom pairs
  //   Each thread has designated values in shared memory as a buffer
  //   A sync operation periodically moves data from shared to global
	const int NUMTHREADS = 32;		/* RBTODO: fix */
	__shared__ int spid[NUMTHREADS];
	const int tid = threadIdx.x;
	const int ai = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid == 0) g_numPairs = 0;	/* RBTODO: be more efficient */
	__syncthreads(); 
	
	int numPairs = g_numPairs; 

	const int MAXPAIRSPERTHREAD = 256; /* RBTODO Hard limit likely too low */
	int pairI[MAXPAIRSPERTHREAD];
	int pairJ[MAXPAIRSPERTHREAD];
	int pid = 0;
	
	// ai - index of the particle in the original, unsorted array
	if (ai < (num-1) * numReplicas) {
		const int repID = ai / (num-1);
		// const int typei = type[i]; // maybe irrelevent
		const Vector3 posi = pos[ai];
		
		// RBTODO: if this approach works well, make cell decomposition finer so limit still works

		// TODO: Fix this: Find correct celli (add a new function to
		//       CellDecomposition, binary search over cells)
		CellDecomposition::cell_t celli = decomp->getCellForParticle(ai);
		const CellDecomposition::cell_t* pairs = decomp->getCells();
		for (int x = -1; x <= 1; ++x) {
			for (int y = -1; y <= 1; ++y) {
				for (int z = -1; z <= 1; ++z) {					
					if (x+y+z != -3) { // SYNC THREADS
						__syncthreads();				
						assert(pid < MAXPAIRSPERTHREAD);

						// find where each thread should put its pairs in global list
						spid[tid] = pid;
						exclIntCumSum(spid,NUMTHREADS); // returns cumsum with spid[0] = 0

						for (int d = 0; d < pid; d++) {
							g_pairI[g_numPairs + d + spid[tid]] = pairI[d];
							g_pairJ[g_numPairs + d + spid[tid]] = pairJ[d];
						}
						// update global index
						__syncthreads();
						if (tid == NUMTHREADS) g_numPairs += spid[tid] + pid;
						pid = 0;
					} // END SYNC THREADS 
	
					const int nID = decomp->getNeighborID(celli, x, y, z);				
					if (nID < 0) continue; // Skip if got wrong or duplicate cell.
					const CellDecomposition::range_t range = decomp->getRange(nID, repID);
					
					for (int n = range.first; n < range.last; n++) {
						const int aj = pairs[n].particle;

						// RBTODO: skip exclusions
						if (aj <= ai) continue;

						pairI[pid] = ai;
						pairJ[pid] = aj;
						pid++;							/* RBTODO synchronize across threads somehow */
					} 	// n
				} 		// z				
			} 			// y
		} 				// x
	}						/* replicas */
}
	
__global__
void computeKernel(Vector3 force[], Vector3 pos[], int type[],
									 float tableAlpha[], float tableEps[], float tableRad6[],
									 int num, int numParts, BaseGrid* sys,
									 CellDecomposition* decomp,
									 float g_energies[], float switchStart, float switchLen,
									 int gridSize, int numReplicas, bool get_energy) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	float energy_local = 0.0f;

	// i - index of the particle in the original, unsorted array
	if (i < num * numReplicas) {
		const int repID = i / num;
		const int typei = type[i];
		const Vector3 posi = pos[i];
		// TODO: Fix this: Find correct celli (add a new function to
		//       CellDecomposition, binary search over cells)
		CellDecomposition::cell_t celli = decomp->getCellForParticle(i);
		float alpha(0.0f), eps(0.0f), rad6(0.0f);
		Vector3 force_local(0.0f);

		const CellDecomposition::cell_t* pairs = decomp->getCells();
		for (int x = -1; x <= 1; ++x) {
			for (int y = -1; y <= 1; ++y) {
				for (int z = -1; z <= 1; ++z) {
					const int nID = decomp->getNeighborID(celli, x, y, z);
					// Skip if got wrong or duplicate cell.
					if (nID < 0) continue;

					const CellDecomposition::range_t range = decomp->getRange(nID, repID);

					int typej = -1;
					for (int n = range.first; n < range.last; ++n) {
						const int j = pairs[n].particle;
						if (j < 0)
//							printf("%d -> pairs[%d].particle %d\n", i, n, j);
						if (j == i) continue;
						const int newj = type[j];
						// Update values.
						if (typej != newj) {
							typej = newj;
							alpha = tableAlpha[typei * numParts + typej];
							eps = tableEps[typei * numParts + typej];
							rad6 = tableRad6[typei * numParts + typej];
						}

						const Vector3 dr = sys->wrapDiff(pos[j] - posi);
						if (dr.length() < 1e-4) {
//							printf("dr = %g << 1: %d -> %d on [%d, %d)\n",
//										 dr.length(), i, j, range.first, range.last);
						}

						const EnergyForce fc =
								ComputeForce::coulombForce(dr, alpha, switchStart, switchLen);
						const EnergyForce fh = ComputeForce::softcoreForce(dr, eps, rad6);

						force_local += fc.f + fh.f;
						energy_local += 0.5f * (fc.e + fh.e);
					} 	// n
				} 		// z
			} 			// y
		} 				// x
		force[i] = force_local;
		if (isnan(force_local.x) or isnan(force_local.y) or isnan(force_local.z)) {
//			printf("Nan FORCE!\n");
			force[i] = Vector3(0.0f);
		}
		if (get_energy)
			g_energies[i] = energy_local;
	}
}


// ============================================================================
// Kernel computes forces between Brownian particles (ions)
// using cell decomposition
//
__global__ void computeTabulatedKernel(Vector3* force, Vector3* pos, int* type,
		TabulatedPotential** tablePot, TabulatedPotential** tableBond,
		int num, int numParts, BaseGrid* sys,
		Bond* bonds, int2* bondMap, int numBonds,
		float* g_energies, float cutoff2, int gridSize,
		int numReplicas, bool get_energy) {
	
	const int NUMTHREADS = 32;		/* RBTODO: fix */
	__shared__ EnergyForce fe[NUMTHREADS];
	__shared__ int atomI[NUMTHREADS];
	__shared__ int atomJ[NUMTHREADS];

	int numPairs = g_numPairs; 

	const int tid = threadIdx.x;
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;

	// loop over particle pairs in pairlist
	if (gid < numPairs) {
		// BONDS: RBTODO: handle through an independent kernel call
		// RBTODO: handle exclusions in other kernel call
		const int ai = g_pairI[tid];
		const int aj = g_pairJ[tid];
		
		// Particle's type and position
		const int ind = type[ai] + type[aj] * numParts; /* RBTODO: why is numParts here? */

		// RBTODO: implement wrapDiff2, returns dr2
		const Vector3 dr = sys->wrapDiff(pos[aj] - pos[ai]);

    // Calculate the force based on the tabulatedPotential file
		fe[tid] = EnergyForce(0.0f, Vector3(0.0f));
		if (tablePot[ind] != NULL && dr.length2() <= cutoff2)
			fe[tid] = tablePot[ind]->compute(dr);

		// RBTODO: think of a better approach; e.g. shared atomicAdds that are later reduced in global memory
		atomicAdd( &force[ai], fe[tid].f );
		atomicAdd( &force[aj], -fe[tid].f );

		// RBTODO: why are energies calculated for each atom? Could be reduced
		if (get_energy && aj > ai)
			atomicAdd( &(g_energies[ai]), fe[tid].e );		
	}
	
}


// =============================================================================
// Kernel computes forces between Brownian particles (ions)
// NOT using cell decomposition
//
__global__
void computeTabulatedFullKernel(Vector3 force[], Vector3 pos[], int type[],
																TabulatedPotential* tablePot[],
																TabulatedPotential* tableBond[],
																int num, int numParts, BaseGrid *sys,
																Bond bonds[], int2 bondMap[], int numBonds,
																Exclude excludes[], int2 excludeMap[],
																int numExcludes, float g_energies[],
																int gridSize, int numReplicas, bool get_energy,
																Angle angles[]) {
	// Thread's unique ID.
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	// Initialize interaction energy (per particle)
	float energy_local = 0.0f;

	// Loop over ALL particles in ALL replicas
	if (i < num * numReplicas) {
		const int repID = i / num;

		// Each particle may have a varying number of bonds.
		// bondMap is an array with one element for each particle which keeps track
		// of where a particle's bonds are stored in the bonds array.
		// bondMap[i].x is the index in the bonds array where the ith particle's
		// bonds begin.
		// bondMap[i].y is the index in the bonds array where the ith particle's
		// bonds end.
		const int bond_start	= (bondMap != NULL) ? bondMap[i - repID * num].x : -1;
		const int bond_end 		= (bondMap != NULL) ? bondMap[i - repID * num].y : -1;

		// currBond is the index in the bonds array that we should look at next
		// currBond is initialized to bond_start because that is the first index of the
		// bonds array where this particle's bonds are stored
		int currBond = bond_start;

		// nextBond is the ID number of the next particle that this particle is bonded to
		// If this particle has at least one bond, then nextBond is initialized to be the
	 	// first particle that this particle is bonded to
		int nextBond = (bond_start >= 0) ? bonds[bond_start].ind2 : -1;

		// Same as for bonds, but for exclusions now
		const int ex_start 	= (excludeMap != NULL) ? excludeMap[i].x : -1;
		const int ex_end 		= (excludeMap != NULL) ? excludeMap[i].y : -1;
		int currEx = ex_start;
		int nextEx = (ex_start >= 0) ? excludes[ex_start].ind2 : -1;

		// Particle's type and position
		const int typei = type[i];
		const Vector3& posi = pos[i];

		// Initialize force_local - force on a particle (i)
		Vector3 force_local(0.0f);

		int typej = -1;
		int ind = -1;

		// Loop over ALL particles in a replica, where current particle belongs to
		const size_t first 	= repID * num;
		const size_t last	= first + num;
		for (int j = first; j < last; ++j) {
			if (i == j) continue;
			
			int newj = type[j];
			if (typej != newj) {
				typej = newj;
				ind = typei + typej * numParts;
			}

			Vector3 dr = sys->wrapDiff(pos[j] - posi);

			EnergyForce ft(0.0f, Vector3(0.0f));

			if (nextEx == (j - repID * num))
				nextEx = (currEx < ex_end - 1) ? excludes[++currEx].ind2 : -1;
			else if (tablePot[ind] != NULL)
				ft = tablePot[ind]->compute(dr);

			// If the next bond we want is the same as j, then there is a bond between
			// particles i and j.
			if (nextBond == (j - repID * num) and tableBond != NULL) {
				// If the user has specified the REPLACE option for this bond, then
				// overwrite the force we calculated from the regular tabulated
				// potential. If the user has specified the ADD option, then add the bond
				// force to the tabulated potential value.
				EnergyForce bond_ef = tableBond[bonds[currBond].tabFileIndex]->compute(dr);
				switch (bonds[currBond].flag) {
					case Bond::REPLACE:	ft  = bond_ef; break;
					case Bond::ADD:     ft += bond_ef; break;
				}

				// Increment currBond, so that we can find the index of the next particle
				// that this particle is bonded to
				if (currBond < bond_end - 1) nextBond = bonds[++currBond].ind2;
				else nextBond = -1;
			}

			force_local += ft.f;
			if (get_energy and j > i)
				energy_local += ft.e;

		} // Loop over all particles (i != j) in replica repID

		force[i] = force_local;
		if (get_energy and i < num)
			g_energies[i] = energy_local;
	}
}

__global__
void computeAngles(Vector3 force[], Vector3 pos[],
									 Angle angles[], TabulatedAnglePotential* tableAngle[],
									 int numAngles, int num, BaseGrid* sys,
									 float g_energies[], bool get_energy) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float energy_local = 0.0f;
	Vector3 force_local(0.0f);
	if (idx < num) {
		for (int i = 0; i < numAngles; ++i) {
			Angle& a = angles[i];
			const int ind = a.getIndex(idx);
			if (ind >= 0) {
				EnergyForce ef = tableAngle[a.tabFileIndex]->compute(&a, pos, sys, ind);
				force_local += ef.f;
				if (ind == 1 and get_energy)
					energy_local += ef.e;
			}
		}
		force[idx] += force_local;
		if (get_energy)
			g_energies[idx] += energy_local;
	}
}

__global__
void computeDihedrals(Vector3 force[], Vector3 pos[],
											Dihedral dihedrals[],
											TabulatedDihedralPotential* tableDihedral[],
											int numDihedrals, int num, BaseGrid* sys, float g_energies[],
											bool get_energy) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float energy_local = 0.0f;
	Vector3 force_local(0.0f);
	if (idx < num) {
		for (int i = 0; i < numDihedrals; ++i) {
			Dihedral& d = dihedrals[i];
			int ind = d.getIndex(idx);
			if (ind >= 0) {
				EnergyForce ef = tableDihedral[d.tabFileIndex]->compute(&d, pos, sys, ind);
				force_local += ef.f;
				if (ind == 1 and get_energy)
					energy_local += ef.e;
			}
		}
		force[idx] += force_local;
		if (get_energy)
			g_energies[idx] += energy_local;
	}
}
