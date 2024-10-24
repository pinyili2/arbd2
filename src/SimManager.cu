#include "SimManager.h"
#include "Random.h"
#include <memory>

void SimManager::run() {
    LOGINFO("Running!");
    // TODO: process a command queue?

    // SimSystem sys = SimSystem();
    // Patch p(10,0,0,sys);

    Patch p;
    LOGINFO("here2!");
//*
    RandomCPU rng{};
    LOGINFO("there2!");
    // RandomAcc rng{};
    //ProxyPatch p2(10,0,0);

    // p.add_compute( std::make_unique<LocalPairForce>() );
    // p.add_compute( std::make_unique<NeighborPairForce>() );

// #ifdef USE_CUDA
//     p.add_compute( std::make_unique<BDIntegrateCUDA>() );
//     p.add_compute( std::make_unique<LocalBondedCUDA>() );
// #else
//     p.add_compute( std::make_unique<BDIntegrate>() );
//     p.add_compute( std::make_unique<LocalBonded>() );
// #endif

    LOGINFO("here!");
    auto tmp = Random::get_gaussian_state(&rng);
    LOGINFO("there!");
    for (size_t step = 0; step < 10; ++step) {
	LOGINFO("where!");
	LOGINFO("Step {:d}: random", step); // {:0.2f}", step, Random::gaussian(&rng,(RandomCPU::state_t*) nullptr));
	p.compute();
#ifdef USE_CUDA
	cudaDeviceSynchronize();
#endif
    }
    // */
// #ifdef USE_CUDA
//     RandomGPU<128>::launch_test_kernel<64>((size_t) 1);
//     cudaDeviceSynchronize();
// #endif

};
