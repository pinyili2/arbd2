#pragma once

#include "GPUManager.h"
#include "Types.h"
#include "Proxy.h"
#include <map>

// #include <cuda.h>
// #include <curand_kernel.h>
// #include <curand.h>

// #define cuRandchk(ans) { cuRandAssert((ans), __FILE__, __LINE__); }
// inline void cuRandAssert(curandStatus code, const char *file, int line, bool abort=true) {
// 	if (code != CURAND_STATUS_SUCCESS) {
// 		fprintf(stderr, "CURAND Error: %d (%s:%d)\n", code, file, line);
// 		if (abort) exit(code);
// 	}
// }

// class RandomState;

// Parallel (multi-threaded) RNG interface with implementations hidden in Random/*h
class Random {
public:
    struct Conf {
	enum Backend { Default, CUDA, CPU };
	Backend backend;
	size_t seed;
	size_t num_threads;
	// explicit operator int() const { return backend*3*num_threads + num_threads + seed; };
    };
    // size_t thread_idx;

    struct State;
    
    static Random* GetRandom(Conf&& conf);

    // Methods for maninpulating the state
    // virtual void init(size_t num, unsigned long seed, size_t offset) = 0;
    
    // Methods for generating random values
    virtual inline float gaussian() = 0;
    virtual inline float gaussian(size_t idx, size_t thread) = 0;
    // HOST DEVICE inline float gaussian(RandomState* state);
    virtual inline Vector3 gaussian_vector(size_t idx, size_t thread) = 0;
    
    virtual unsigned int integer() = 0;
    virtual unsigned int poisson(float lambda) = 0;
    virtual float uniform() = 0;
    
    // void reorder(int *a, int n);
    
protected:
    static std::map<Conf, Random*> _objects;
	// Random() : states(NULL), integer_h(NULL), integer_d(NULL), uniform_h(NULL), uniform_d(NULL), gaussian_h(NULL), gaussian_d(NULL) { }
	// Random(int num, unsigned long seed=0) : states(NULL), integer_h(NULL), integer_d(NULL), uniform_h(NULL), uniform_d(NULL), gaussian_h(NULL), gaussian_d(NULL) {		
	// 	init(num, seed);
	// }

};

// class RandomImpl : public Random {
// public:
//     // Methods for maninpulating the state
//     void init(size_t num, unsigned long seed, size_t offset);

//     // // Methods for generating random values
//     HOST inline float gaussian();
//     HOST inline float gaussian(size_t idx, size_t thread);
//     // HOST DEVICE inline float gaussian(RandomState* state) {};
//     HOST inline Vector3 gaussian_vector(size_t idx, size_t thread);
//     unsigned int integer() { return 0; };
//     unsigned int poisson(float lambda) { return 0; };
//     float uniform() { return 0; };
    
//     // // void reorder(int *a, int n);

// };

#ifdef USE_CUDA
// class RandomImplCUDA : public Random {
// public:
//     // Methods for maninpulating the state
//     void init(size_t num, unsigned long seed, size_t offset);

//     // // Methods for generating random values
//     HOST inline float gaussian();
//     HOST inline float gaussian(size_t idx, size_t thread);
//     // HOST DEVICE inline float gaussian(RandomState* state) {};
//     HOST inline Vector3 gaussian_vector(size_t idx, size_t thread);
//     unsigned int integer() { return 0; };
//     unsigned int poisson(float lambda) { return 0; };
//     float uniform() { return 0; };

// };
#endif
