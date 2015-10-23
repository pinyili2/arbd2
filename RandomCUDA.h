#ifndef RANDOM_CUDA_H
#define RANDOM_CUDA_H

// #include "/usr/include/linux/cuda.h"
// #include "/usr/local/encap/cuda-4.0/include/cuda_runtime.h"
// #include "/usr/local/encap/cuda-4.0/include/curand_kernel.h"
// #include "/usr/local/encap/cuda-4.0/include/curand.h"
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>

#include "common.h"
#include "useful.h"
#include "ComputeForce.h"

#ifdef __CUDACC__
    #define HOST __host__
    #define DEVICE __device__
#else
    #define HOST 
    #define DEVICE 
#endif

class Random {
private:
	static const size_t RAND_N = 512; // max random numbers stored

	curandState_t *states;
	curandGenerator_t generator;
	unsigned int *integer_h, *integer_d;
	float *uniform_h, *uniform_d;
	size_t integer_n, uniform_n;

public:

	Random() : states(NULL), integer_h(NULL), integer_d(NULL), uniform_h(NULL), uniform_d(NULL) { }
	Random(int num, unsigned long seed=0) : states(NULL), integer_h(NULL), integer_d(NULL), uniform_h(NULL), uniform_d(NULL)  {		
		init(num, seed);
	}

	void init(int num, unsigned long seed);

	DEVICE inline float gaussian(int idx, int num) {
		// TODO do stuff
		if (idx < num)
			return curand_normal(&states[idx]);
		return 0.0f;
	}

	DEVICE inline Vector3 gaussian_vector(int idx, int num) {
		// TODO do stuff
		float x = gaussian(idx, num);
		float y = gaussian(idx, num);
		float z = gaussian(idx, num);
		return Vector3(x, y, z);
	}

	unsigned int integer();
	unsigned int poisson(float lambda);
	float uniform();

	void reorder(int *a, int n);
};

#endif
