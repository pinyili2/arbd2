#include "SimManager.h"
#include <memory>

void SimManager::run() {
    std::cout << "running" << std::endl;

    // SimSystem sys = SimSystem();
    // Patch p(10,0,0,sys);

    Patch p;

    //ProxyPatch p2(10,0,0);

    // p.add_compute( std::make_unique<LocalPairForce>() );
    // p.add_compute( std::make_unique<NeighborPairForce>() );

#ifdef USE_CUDA
    p.add_compute( std::make_unique<BDIntegrateCUDA>() );
    p.add_compute( std::make_unique<LocalBondedCUDA>() );
#else
    p.add_compute( std::make_unique<BDIntegrate>() );
    p.add_compute( std::make_unique<LocalBonded>() );
#endif
    
    for (size_t step = 0; step < 10; ++step) {
	printf("Step\n");
	p.compute();
#ifdef USE_CUDA
	cudaDeviceSynchronize();
#endif
    }    
};
