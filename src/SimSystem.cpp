#include "SimSystem.h"

void CellDecomposer::decompose(SimSystem& sys, ResourceCollection& resources) {
    BoundaryConditions& bcs = sys.boundary_conditions;
    const Length& cutoff = sys.cutoff;

    // Initialize workers; TODO move this out of decompose
    std::vector<Proxy<Worker>> workers;
    for (auto& r: resources.resources) {
	Worker w(sys.cutoff, sys.patches);
	auto w_p = send(r, w);
	workers.push_back( w_p );
    }
    
    Vector3 min = sys.get_min();
    Vector3 max = sys.get_max();
    Vector3 dr = max-min;

    // For starters, distribute patches uniformly among available resources
    Vector3 n_p_v = (dr / cutoff).element_floor();
    size_t n_r = resources.resources.size();

    size_t n_p = static_cast<size_t>(round(n_p_v[0]*n_p_v[1]*n_p_v[2]));
    for (size_t i = 0; i < n_p; ++i) {
	auto w_p = workers[i / floor(n_p/n_r)];
	w_p.create_patch();
	// other stuff
    }
	
    
    
    std::vector<Proxy<Patch>> new_patches;

    
    // Count particles in each new_patch
    //   (ultimately this must be distributed, but via what mechanisms? Decomposer is only on controlling thread in current implementation, but perhaps each thread should have its own copy)
    // Then add particles from old patches to new one
    
    sys.patches = new_patches;
}
