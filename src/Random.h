#pragma once

#include "GPUManager.h"
#include "Types.h"
#include "Proxy.h"
#include <map>

#include <cuda.h>
#include <cuda_runtime.h>
#ifdef USE_CUDA
#include <curand_kernel.h>
#include <curand.h>

#define cuRandchk(ans) { cuRandAssert((ans), __FILE__, __LINE__); }
inline void cuRandAssert(curandStatus code, const char *file, int line, bool abort=true) {
	if (code != CURAND_STATUS_SUCCESS) {
		fprintf(stderr, "CURAND Error: %d (%s:%d)\n", code, file, line);
		if (abort) exit(code);
	}
}
#endif

namespace Random {
    // Interface
    template<typename RNG>
    HOST DEVICE inline typename RNG::state_t* get_gaussian_state(RNG *random) {
	// return nullptr;
	return random->get_gaussian_state();
    }

    template<typename RNG>
    HOST DEVICE void set_gaussian_state(RNG *random, typename RNG::state_t* state) {
	return random->set_gaussian_state(state);
    }

    // template<typename RNG>
    // HOST DEVICE auto gaussian(RNG __restrict__ * random, typename RNG::state_t __restrict__ * state) {
    // 	return random->gaussian(state);
    // }

    template<typename RNG>
    HOST DEVICE float gaussian(RNG *random, typename RNG::state_t* state) {
	return random->gaussian(state);
    }
}

class RandomCPU {
public:
    using state_t = void;

    // Methods for maninpulating the state
    void init(size_t num, unsigned long seed, size_t offset) {
	// 
    };

    // Methods for generating random values
    HOST DEVICE inline float gaussian(state_t* state) {
	LOGWARN("RandomCPU::gaussian(): not implemented");
	return 0;
    };

    HOST DEVICE inline state_t* get_gaussian_state() { return nullptr; };

    HOST DEVICE inline void set_gaussian_state(state_t* state) {};

    // // HOST DEVICE inline float gaussian(RandomState* state) {};
    // HOST inline Vector3 gaussian_vector(size_t idx, size_t thread) {
    // 	return Vector3();
    // };
    // unsigned int integer() {
    // 	return 0;
    // };
    // unsigned int poisson(float lambda) {
    // 	return 0;
    // };
    // float uniform() {
    // 	return 0;
    // };
    // void reorder(int *a, int n);
};

#ifdef USE_CUDA

class RandomAcc {
public:
    using state_t = void;

    // Methods for maninpulating the state
    void init(size_t num, unsigned long seed, size_t offset) {
	// 
    };

    // Methods for generating random values
    template<typename S>
    HOST DEVICE inline float gaussian(S* state) {
	LOGWARN("RandomCPU::gaussian(): not implemented");
	return 0;
    };

    template<typename S>
    HOST DEVICE inline S* get_gaussian_state() { return nullptr; };

    template<typename S>
    HOST DEVICE inline void set_gaussian_state(S* state) {};

    // // HOST DEVICE inline float gaussian(RandomState* state) {};
    // HOST inline Vector3 gaussian_vector(size_t idx, size_t thread) {
    // 	return Vector3();
    // };
    // unsigned int integer() {
    // 	return 0;
    // };
    // unsigned int poisson(float lambda) {
    // 	return 0;
    // };
    // float uniform() {
    // 	return 0;
    // };
    // void reorder(int *a, int n);
};


// Forward declarations
template<size_t num_states> class RandomGPU;
template<size_t num_states>
__global__ void RandomGPU__test_kernel(RandomGPU<num_states>* rng, size_t num_vals, float* buf, unsigned long seed, size_t offset);

template<size_t num_states = 128>
class RandomGPU {
    static_assert(num_states > 0);
public:
    using state_t = curandStateXORWOW_t;

    // DEVICE RandomGPU(Random::Conf c, Resource& location) : num_states{c.num_threads}, states{nullptr} {

    HOST DEVICE RandomGPU() : states{nullptr} {
	// Not sure how this
	assert( threadIdx.x + blockIdx.x == 0 );
	
	LOGINFO("Creating RandomGPU");
	// assert( location.type == Resource::GPU );
	states = new state_t[num_states];
    }
    HOST DEVICE ~RandomGPU() {
	LOGINFO("Destroying RandomGPU");
	assert( threadIdx.x + blockIdx.x == 0 );
	if (states != nullptr) delete [] states;
	states = nullptr;
    }

    HOST static RandomGPU<num_states>* instantiate_on_GPU(RandomGPU<num_states> *dest = nullptr) {
	using RNG = RandomGPU<num_states>;
	RNG* rng = reinterpret_cast<RNG*>(new char[sizeof(RNG)]); // avoid calling constructor
	if (dest == nullptr) {
	    gpuErrchk(cudaMalloc((void**)&dest, sizeof(RNG)));
	}
	gpuErrchk(cudaMalloc((void**)&(rng->states), sizeof(state_t)*num_states));
	gpuErrchk(cudaMemcpy(dest, rng, sizeof(RNG),cudaMemcpyHostToDevice));
	rng->states = nullptr;
	return dest;
    }
	
    HOST DEVICE inline float gaussian(state_t* state) {
#ifdef __CUDA_ARCH__    
	return curand_normal( state );
#else
	return 0.0f;
#endif
    };
    
    // Useful for getting the RNG state for the thread from global
    // memory, putting it in local memory for use
    HOST DEVICE inline state_t* get_gaussian_state() {
#ifdef __CUDA_ARCH__    
	const size_t& i = threadIdx.x + blockDim.x*blockIdx.x;
	return (i < num_states && states != nullptr) ? &(states[i]) : nullptr;
#else
	return (state_t*) nullptr;
#endif
    };

    HOST DEVICE inline void set_gaussian_state(state_t* state) {
#ifdef __CUDA_ARCH__    
	const size_t& i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i < num_states && state != nullptr) states[i] = *state;
#endif
    };

    HOST DEVICE inline Vector3 gaussian_vector( state_t* state) {
	return Vector3( gaussian(state), gaussian(state), gaussian(state) );
    };

    template<size_t kernel_threads = 64> 
    static void launch_test_kernel(unsigned long seed, size_t offset = 0, size_t num_vals = 4*num_states, RandomGPU<num_states>* rng = nullptr) {
	using RNG = RandomGPU<num_states>;
	static_assert( (num_states >= kernel_threads) );
	static_assert( (num_states % kernel_threads) == 0 );
    
	bool rng_was_null = (rng == nullptr);
	if (rng_was_null) rng = RNG::instantiate_on_GPU();

	float *buf_d, *buf;

	LOGTRACE("cudaMalloc {}", num_vals);
	gpuErrchk(cudaMalloc((void**)&buf_d, num_vals*sizeof(float)));
	LOGTRACE("...launching kernel");
	RandomGPU__test_kernel<num_states><<<1+(num_states-1)/kernel_threads, kernel_threads>>>(rng, num_vals, buf_d, seed, offset);
	buf = new float[num_vals];
	gpuErrchk(cudaDeviceSynchronize());
	LOGTRACE("...copying data back");
    
	gpuErrchk(cudaMemcpy(buf, buf_d, num_vals*sizeof(float),cudaMemcpyDeviceToHost));
	float sum = 0;
	float sum2 = 0;
	for (size_t i = 0; i < num_vals; ++i) {
	    sum += buf[i];
	    sum2 += buf[i]*buf[i];
	}
	LOGINFO("RNG gaussian mean, std, count: {}, {}, {}", sum/num_vals, sqrt((sum2/num_vals) - sum*sum/(num_vals*num_vals)), num_vals);

	if (rng_was_null) cudaFree( rng );
	cudaFree( buf_d );
	delete[] buf;
    };

private:
    state_t* states;	// RNG states stored in global memory 
};

template<size_t num_states = 128>
__global__ void RandomGPU__test_kernel(RandomGPU<num_states>* rng, size_t num_vals, float* buf, unsigned long seed, size_t offset = 0) {
    using RNG = RandomGPU<num_states>;
    // printf("THREAD %d\n", threadIdx.x);
    auto local_state = Random::get_gaussian_state<RNG>(rng);
    for (size_t idx = threadIdx.x + blockIdx.x*blockDim.x; idx < num_states; idx += blockDim.x * gridDim.x) {
	assert(blockIdx.x == 0);
	// curand_init(unsigned long long seed,
	// 	    unsigned long long sequence,
	// 	    unsigned long long offset,
	// 	    curandStateXORWOW_t *state)
	curand_init(seed, idx, offset, local_state);
    }
	
    for (size_t idx = threadIdx.x + blockIdx.x*blockDim.x; idx < num_vals; idx += blockDim.x * gridDim.x) {
	buf[idx] = Random::gaussian<RNG>(rng, local_state);
    }
    Random::set_gaussian_state<RNG>(rng, local_state);
};

#endif
