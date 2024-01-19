#include "Random.h"

#ifdef USE_CUDA
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>

#endif

#define cuRandchk(ans) { cuRandAssert((ans), __FILE__, __LINE__); }
inline void cuRandAssert(curandStatus code, const char *file, int line, bool abort=true) {
	if (code != CURAND_STATUS_SUCCESS) {
		fprintf(stderr, "CURAND Error: %d (%s:%d)\n", code, file, line);
		if (abort) exit(code);
	}
}


std::map<Random::Conf, Random*> Random::_objects;

bool operator<(const Random::Conf x, const Random::Conf y) {
    return x.backend == y.backend ? (x.seed == y.seed ?
				     (x.num_threads < y.num_threads) :
				     x.seed < y.seed) : x.backend < y.backend;
}

// class RandomDEPRECATED {
// public:
// 	static const size_t RAND_N = 1024*4; // max random numbers stored

// 	curandState_t *states;
// 	curandGenerator_t generator;
// 	unsigned int *integer_h, *integer_d;
// 	float *uniform_h, *uniform_d;
// 	float *gaussian_h, *gaussian_d;
// 	size_t integer_n, uniform_n, gaussian_n;

// public:

// 	Random() : states(NULL), integer_h(NULL), integer_d(NULL), uniform_h(NULL), uniform_d(NULL), gaussian_h(NULL), gaussian_d(NULL) { }
// 	Random(int num, unsigned long seed=0) : states(NULL), integer_h(NULL), integer_d(NULL), uniform_h(NULL), uniform_d(NULL), gaussian_h(NULL), gaussian_d(NULL) {		
// 		init(num, seed);
// 	}

// 	void init(int num, unsigned long seed);

// 	DEVICE inline float gaussian(int idx, int num) {
// 		// TODO do stuff
// 		if (idx < num)
// 			return curand_normal(&states[idx]);
// 		return 0.0f;
// 	}
// 	DEVICE inline float gaussian(curandState* state) {
// 		return curand_normal(state);
// 	}

// 	DEVICE inline Vector3 gaussian_vector(int idx, int num) {
// 		// TODO do stuff
// 		if (idx < num) {
// 			curandState localState = states[idx];
// 			Vector3 v = gaussian_vector(&localState);
// 			states[idx] = localState;
// 			return v;
// 		} else return Vector3(0.0f);			
// 	}
// 	DEVICE inline Vector3 gaussian_vector(curandState* state) {
// 		float x = gaussian(state);
// 		float y = gaussian(state);
// 		float z = gaussian(state);
// 		return Vector3(x, y, z);
// 	}

// 	unsigned int integer();
// 	unsigned int poisson(float lambda);
// 	float uniform();

// 	HOST inline float gaussian() {
// 	    if (gaussian_n < 1) {
// 		cuRandchk(curandGenerateNormal(generator, gaussian_d, RAND_N, 0, 1));
// 		gpuErrchk(cudaMemcpy(gaussian_h, gaussian_d, sizeof(float) * RAND_N, cudaMemcpyDeviceToHost));
// 	    }
// 	    return gaussian_h[--gaussian_n];
// 	}

// 	void reorder(int *a, int n);
// };

// void Random::init(int num, unsigned long seed) {
// 	if (states != NULL)
// 		gpuErrchk(cudaFree(states));
// 	gpuErrchk(cudaMalloc(&states, sizeof(curandState) * num));
// 	int nBlocks = num / NUM_THREADS + 1;
// 	initKernel<<< nBlocks, NUM_THREADS >>>(seed, states, num);
// 	gpuErrchk(cudaDeviceSynchronize());

// 	// Create RNG and set seed
// 	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW);
// 	curandSetPseudoRandomGeneratorSeed(generator, seed);
	
// 	if (uniform_d != NULL) 
//         {
//             gpuErrchk(cudaFree(uniform_d));
//             uniform_d = NULL;
//         }
//         if(integer_d!=NULL)
//         {
//             gpuErrchk(cudaFree(integer_d));
//             integer_d = NULL;
//         }
//         if(gaussian_d!=NULL)
//         {
//             gpuErrchk(cudaFree(gaussian_d));
//             gaussian_d = NULL;
//         }
//         if(integer_h!=NULL)
//         {
// 	    delete[] integer_h;
//             integer_h = NULL;
//         }
//         if(uniform_h!=NULL)
//         {
// 	    delete[] uniform_h;
//             uniform_h = NULL;
// 	}
//         if(gaussian_h!=NULL)
//         {
// 	    delete[] gaussian_h;
//             gaussian_h = NULL;
//         }
// 	gpuErrchk(cudaMalloc((void**)&uniform_d, sizeof(float) * RAND_N));
// 	gpuErrchk(cudaMalloc((void**)&integer_d, sizeof(unsigned int) * RAND_N));
// 	gpuErrchk(cudaMalloc((void**)&gaussian_d, sizeof(float) * RAND_N));
// 	integer_h = new unsigned int[RAND_N];
// 	uniform_h = new float[RAND_N];
// 	gaussian_h = new float[RAND_N];
// 	uniform_n = 0;
// 	integer_n = 0;
// 	gaussian_n = 0;
// }

// float Random::uniform() {
// 	if (uniform_n < 1) {
// 		cuRandchk(curandGenerateUniform(generator, uniform_d, RAND_N));
// 		gpuErrchk(cudaMemcpy(uniform_h, uniform_d, sizeof(float) * RAND_N, cudaMemcpyDeviceToHost));
// 		uniform_n = RAND_N;
// 	}
// 	return uniform_h[--uniform_n];
// }

// unsigned int Random::poisson(float lambda) {
// 	const float l = exp(-lambda);
// 	unsigned int k = 0;
// 	float p = uniform();
// 	while (p >= l) {
// 		p *= uniform();
// 		k = k + 1;
// 	}
// 	return k;
// }

// unsigned int Random::integer() {
// 	if (integer_n < 1) {
// 		curandGenerate(generator, integer_d, RAND_N);
// 		gpuErrchk(cudaMemcpy(integer_h, integer_d, sizeof(unsigned int) * RAND_N, cudaMemcpyDeviceToHost));
// 		integer_n = RAND_N;
// 	}
// 	return integer_h[--integer_n];
// }

// void Random::reorder(int a[], int n) {
// 	for (int i = 0; i < (n-1); ++i) {
// 		unsigned int j = i + (integer() % (n-i));
// 		if ( j == i )
// 			continue;
// 		std::swap<int>(a[i], a[j]);
// 		const int tmp = a[j];
// 		a[j] = a[i];
// 		a[i] = tmp;
// 	}
// }

class RandomImpl : public Random {
public:
    // Methods for maninpulating the state
    void init(size_t num, unsigned long seed, size_t offset) {
	// 
    };

    // Methods for generating random values
    HOST DEVICE inline float gaussian() {
	return 0;
    };
    HOST DEVICE inline float gaussian(size_t idx, size_t thread) {
	return 0;
    };

    // HOST DEVICE inline float gaussian(RandomState* state) {};
    HOST inline Vector3 gaussian_vector(size_t idx, size_t thread) {
	return Vector3();
    };
    unsigned int integer() {
	return 0;
    };
    unsigned int poisson(float lambda) {
	return 0;
    };
    float uniform() {
	return 0;
    };
    // void reorder(int *a, int n);
};

#ifdef USE_CUDA

__global__ 
void RandomImplCUDA_init_kernel(unsigned long seed, curandState_t *state, int num) {
       size_t idx  = blockIdx.x * blockDim.x + threadIdx.x;
       size_t step = blockDim.x * gridDim.x;
       for(size_t i = idx; i < num; i=i+step)
       {
           curandState_t local;
           // curand_init(clock64()+seed,i,0,&local);
           //curand_init(clock64(),i,0,&state[i]);
	   curand_init(seed,i,0,&local);
           state[(size_t)i] = local;
       }
}

// template<size_t NUM_THREADS = 128, size_t buffer_size = 1024, typename T=int>
// class RandomImplGPUres : public Random {
// public:

//     RandomImplGPURes(Random::Conf c, Resource& location) : location{location}, seed{c.seed}, num_states{c.num_threads}, num{0} {
// 	// Note: NUM_THREADS refers to CUDA threads, whereas c.num_threads refers to (?) number of random numbers to be generated in parallel
	
// 	LOGINFO("Creating RandomImplGPUres with seed {}", seed);
// 	assert( location.type == Resource::GPU );
// 	LOGINFO("...Sending data");

	
// 	states_d = states_d.send_children(location);
// 	buffer_d = buffer_d.send_children(location);
	    
// 	// LOGINFO("...Cleaning temporary data");
// 	// delete [] tmp.states;
// 	// delete [] tmp.buffer;

// 	// gpuErrchk(cudaMalloc(&(data.states), sizeof(curandState_t) * c.num_threads));
// 	// int nBlocks = c.num_threads / NUM_THREADS + 1;
// 	// initKernel<<< nBlocks, NUM_THREADS >>>(seed, data.states, NUM_THREADS);
// 	// gpuErrchk(cudaDeviceSynchronize());

// 	// TODO consider location whe ncreating generator!
// 	// Create RNG and set seed
// 	LOGINFO("...Creating generator");
// 	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW);
// 	curandSetPseudoRandomGeneratorSeed( generator, seed );

// 	// Set seed 
// 	LOGINFO("...Setting seed");
// 	gpuErrchk(cudaDeviceSynchronize());
// 	LOGINFO("...synced");
// 	size_t num_blocks = 1 + (c.num_threads-1)/NUM_THREADS;
// 	LOGINFO("...launching");
// 	// LOGINFO("...hmmm {}", (void*) metadata.states);
// 	RandomImplGPUres_init_kernel<<< num_blocks, NUM_THREADS >>>(seed, states_d.values, num_states);
// 	LOGINFO("...Kernel launched");
// 	gpuErrchk(cudaDeviceSynchronize());
// 	LOGINFO("...Done");

//     // Allocate temporary storage


// 	// gpuErrchk(cudaMalloc((void**) &(data.buffer), sizeof(size_t) * buffer_size));
	    
//     };

//     // struct Data {
//     // 	static_assert( sizeof(float) == sizeof(int) );
//     // 	static_assert( sizeof(float) == sizeof(T) );

//     // 	Data send_children(Resource& location) const {
//     // 	    Data tmp{};
//     // 	    tmp.states = states.send_children(location);
//     // 	    tmp.buffer = buffer.send_children(location);
//     // 	    return tmp;
//     // 	};
//     // 	void clear() {};	    
//     // };

// private:    
//     // Proxy<Data> data;
//     size_t seed;
//     curandGenerator_t generator;
//     size_t num_states;
//     Array<curandState_t> states_d{NUM_THREADS};
//     Array<T> buffer_d{buffer_size};		// Cast when you need (float) vs int?
//     Array<T> local_buffer{buffer_size};	// What about Gaussian (float) vs int?
//     size_t num;		                // number of entries left in local_buffer

//     // Methods for maninpulating the state
//     void init(size_t num, unsigned long seed, size_t offset) {
// 	// gpuErrchk(cudaMalloc((void**)&uniform_d, sizeof(float) * RAND_N));
// 	// gpuErrchk(cudaMalloc((void**)&integer_d, sizeof(unsigned int) * RAND_N));
// 	// gpuErrchk(cudaMalloc((void**)&gaussian_d, sizeof(float) * RAND_N));
// 	// integer_h = new unsigned int[RAND_N];
// 	// uniform_h = new float[RAND_N];
// 	// gaussian_h = new float[RAND_N];
// 	// uniform_n = 0;
// 	// integer_n = 0;
// 	// gaussian_n = 0;
// 	// // 
//     };

// public:    
//     // RandomImplCUDA<NUM_THREADS,buffer_size> send_children(Resource& location) const {
//     // 	RandomImplCUDA<NUM_THREADS,buffer_size> tmp{};
//     // 	tmp.states = states.send_children(location);
//     // 	tmp.buffer = buffer.send_children(location);
//     // 	return tmp;
//     // 	};
//     // 	void clear() {};	    

//     // TODO: could introduce dual buffer so random numbers can be filled asynchronously
//     // Methods for generating random values
//     HOST DEVICE inline float gaussian() {
// 	if (num == 0) {
// 	    cuRandchk(curandGenerateUniform( generator, (float*) buffer_d.values, buffer_size ));
// 	    gpuErrchk(cudaMemcpy(local_buffer.values, buffer_d.values, sizeof(float) * buffer_size, cudaMemcpyDeviceToHost));
// 	    num = buffer_size;
// 	}
// 	return *reinterpret_cast<float*>(&local_buffer[--num]);
//     };
//     HOST DEVICE inline float gaussian(size_t idx, size_t thread) {
// 	return 0;
//     };

//     // HOST DEVICE inline float gaussian(RandomState* state) {};
//     HOST DEVICE inline Vector3 gaussian_vector(size_t idx, size_t thread) {
// 	return Vector3();
//     };
// };


using curandState = curandStateXORWOW_t;
template<size_t NUM_THREADS = 128>
class RandomGPU : public Random {
public:

    DEVICE RandomGPU(Random::Conf c, Resource& location) : num_states{c.num_threads}, states{nullptr} {
	// Note: NUM_THREADS refers to CUDA threads, whereas c.num_threads refers to (?) number of random numbers to be generated in parallel
	// Not sure how this
	assert( threadIdx.x + blockIdx.x == 0 );
	
	LOGINFO("Creating RandomGPU", seed);
	assert( location.type == Resource::GPU );
	states = new curandState[num_states];
    }
    DEVICE ~RandomGPU() {
	assert( threadIdx.x + blockIdx.x == 0 );
	if (states != nullptr) delete [] states;
	states = nullptr;
    };

private:    
    size_t num_states;
    curandState* states;	// RNG states stored in global memory 
    
public:    
    // Methods for maninpulating the state
    DEVICE void init(size_t num, unsigned long seed, size_t offset = 0) {
	// curand_init(unsigned long long seed,
	// 	    unsigned long long sequence,
	// 	    unsigned long long offset,
	// 	    curandStateXORWOW_t *state)
	assert( states != nullptr );
	for (size_t idx = threadIdx.x + blockIdx.x*blockDim.x; idx < NUM_THREADS; idx += blockDim.x * gridDim.x) {
	    curand_init(seed, tidx, offset, curandState &state[idx]);
	}
    };

    // // This _might_ be made more generic by using whatever chunk of shm is needed, then returning end of pointer
    // DEVICE inline void copy_state_to_shm(curandState *shm) {
    // 	size_t idx = threadIdx.x + blockIdx.x*blockDim.x;
    // 	states_shm = &shm;
    // 	for (; idx < NUM_THREADS; idx += blockDim.x * gridDim.x) {
    // 	    (*states_shm)[idx] = states[idx];
    // 	}
    // 	// __syncthreads();

    // }
    
    DEVICE inline float gaussian() {
	size_t idx = threadIdx.x + blockIdx.x*blockDim.x;
	assert( idx < num_states );
	return curand_normal(&states[idx]);
    };

    DEVICE inline float gaussian(curandState& state) { return curand_normal( &state ); };

    // Useful for getting the RNG state for the thread from global memory, putting it in local memory for use
    DEVICE inline curandState* get_gaussian_state() {
	size_t& i = threadIdx.x + blockSize.x*blockIdx.x;
	return (i < num_states) ? &(states[i]) : nullptr;
    };

    DEVICE inline void set_gaussian_state(curandState& state) {
	size_t& i = threadIdx.x + blockSize.x*blockIdx.x;
	if (i < num_states) states[i] = state;
    };

    DEVICE inline Vector3 gaussian_vector() {
	return Vector3( gaussian(), gaussian(), gaussian() );
    };

    DEVICE inline Vector3 gaussian_vector( curandState& state) {
	return Vector3( gaussian(state), gaussian(state), gaussian(state) );
    };
};


/*
 * Note: this implementation (and probably the interface) should be
 *   reworked. Generating RNs on the GPU doesn't require or benefit
 *   from the Data buffers, so probably those should be removed from a
 *   device-resident object. Ideally this object can implement a
 *   uniform and lightweight interface that enables efficient use
 *   through the agnostic API provided by the Random class.
 */
template<size_t NUM_THREADS = 128, size_t buffer_size = 1024>
class RandomImplCUDA : public Random {
public:

    template<typename T>
    struct Data {
	Array<float> buffer{buffer_size};
	size_t num; // number of unread entries in buffer

	Data send_children(Resource& location) const {
	    Data tmp{};
	    tmp.buffer = buffer.send_children(location);
	    LOGTRACE( "Clearing RANDOM::Data::buffer..." );
	    buffer.clear(); // Not sure if this should be done here!
	    return tmp;
	};

    };
    
    RandomImplCUDA(Random::Conf c, Resource& location) : location{location}, seed{c.seed}, num_states{c.num_threads}, num{0} {
	// Note: NUM_THREADS refers to CUDA threads, whereas c.num_threads refers to (?) number of random numbers to be generated in parallel
	
	LOGINFO("Creating RandomImplCUDA with seed {}", seed);
	assert( location.type == Resource::GPU );
	LOGINFO("...Sending data");

	// Data tmp;
	// data = send(location, tmp);	
	// metadata.buffer = new size_t[buffer_size]; // TODO handle case if RandomImplCUDA is on location
	// metadata.num = 0;

	states_d = states_d.send_children(location);
	    
	// LOGINFO("...Cleaning temporary data");
	// delete [] tmp.states;
	// delete [] tmp.buffer;

	// gpuErrchk(cudaMalloc(&(data.states), sizeof(curandState_t) * c.num_threads));
	// int nBlocks = c.num_threads / NUM_THREADS + 1;
	// initKernel<<< nBlocks, NUM_THREADS >>>(seed, data.states, NUM_THREADS);
	// gpuErrchk(cudaDeviceSynchronize());

	// TODO consider location whe ncreating generator!
	// Create RNG and set seed
	LOGINFO("...Creating generator");
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW);
	curandSetPseudoRandomGeneratorSeed( generator, seed );

	// Set seed 
	LOGINFO("...Setting seed");
	gpuErrchk(cudaDeviceSynchronize());
	LOGINFO("...synced");
	size_t num_blocks = 1 + (c.num_threads-1)/NUM_THREADS;
	LOGINFO("...launching");
	// LOGINFO("...hmmm {}", (void*) metadata.states);
	RandomImplCUDA_init_kernel<<< num_blocks, NUM_THREADS >>>(c.seed, states_d.values, num_states);
	LOGINFO("...Kernel launched");
	gpuErrchk(cudaDeviceSynchronize());
	LOGINFO("...Done");

    // Allocate temporary storage


	// gpuErrchk(cudaMalloc((void**) &(data.buffer), sizeof(size_t) * buffer_size));
	    
    };

    // struct Data {
    // 	static_assert( sizeof(float) == sizeof(int) );
    // 	static_assert( sizeof(float) == sizeof(T) );

    // 	Data send_children(Resource& location) const {
    // 	    Data tmp{};
    // 	    tmp.states = states.send_children(location);
    // 	    tmp.buffer = buffer.send_children(location);
    // 	    return tmp;
    // 	};
    // 	void clear() {};	    
    // };

private:    
    Resource location;
    size_t seed;
    curandGenerator_t generator;
    size_t num_states;
    Array<curandState_t> states_d;

    // Buffers for generating numbers from CPU-resident
    Data<float>* gauss_d;
    Data<float>* poisson_d;
    Data<float>* uniform_d;
    
    Data<float> gauss;
    Data<float> poisson;
    Data<float> uniform;
    
	
    // Methods for maninpulating the state
    void init(size_t num, unsigned long seed, size_t offset) {
	// gpuErrchk(cudaMalloc((void**)&uniform_d, sizeof(float) * RAND_N));
	// gpuErrchk(cudaMalloc((void**)&integer_d, sizeof(unsigned int) * RAND_N));
	// gpuErrchk(cudaMalloc((void**)&gaussian_d, sizeof(float) * RAND_N));
	// integer_h = new unsigned int[RAND_N];
	// uniform_h = new float[RAND_N];
	// gaussian_h = new float[RAND_N];
	// uniform_n = 0;
	// integer_n = 0;
	// gaussian_n = 0;
	// // 
    };

public:    
    // RandomImplCUDA<NUM_THREADS,buffer_size> send_children(Resource& location) const {
    // 	RandomImplCUDA<NUM_THREADS,buffer_size> tmp{};
    // 	tmp.states = states.send_children(location);
    // 	tmp.buffer = buffer.send_children(location);
    // 	return tmp;
    // 	};
    // 	void clear() {};	    

    // TODO: could introduce dual buffer so random numbers can be filled asynchronously
    // Methods for generating random values
    HOST DEVICE inline float gaussian() {
#ifdef __CUDA__ARCH__
	size_t& i = threadIdx.x + blockSize.x*blockIdx.x
	if (i < num_states) {
	    return curand_normal( &(state_d[i]) );
	}
	return 0.0f;
#else
	if (num == 0) {
	    cuRandchk(curandGenerateUniform( generator, (float*) gauss.buffer_d.values, buffer_size ));
	    gpuErrchk(cudaMemcpy(local_buffer.values, buffer_d.values, sizeof(float) * buffer_size, cudaMemcpyDeviceToHost));
	    num = buffer_size;
	}
	return (&local_buffer[--num]);
#endif
    };
    
    __device__ inline float gaussian() {
    };
    
    DEVICE inline float gaussian(curandState* state) { return curand_normal( state ); };

    // Useful for getting the RNG state for the thread from global memory, putting it in local memory for use
    DEVICE inline curandState* get_gaussian_state() {
	size_t& i = threadIdx.x + blockSize.x*blockIdx.x;
	if (i < num_states) {
	    return &(state_d[i]);
	}
	return nullptr;
    };

    DEVICE inline void set_gaussian_state(curandState* state) {
	size_t& i = threadIdx.x + blockSize.x*blockIdx.x;
	if (i < num_states) state_d[i] = *state;
    };
    
    // HOST inline float gaussian(size_t idx, size_t thread) {
    // 	return 0;
    // };

    // HOST DEVICE inline float gaussian(RandomState* state) {};
    HOST inline Vector3 gaussian_vector(size_t idx, size_t thread) {
	return Vector3();
    };
    
    // unsigned int integer() {
    // 	return 0;
    // };
    // unsigned int poisson(float lambda) {
    // 	return 0;
    // };
    // float uniform() {
    // 	return 0;
    // };
};

/* GPU-resident?
template<size_t NUM_THREADS = 128, size_t buffer_size = 1024>
class RandomImplCUDA_DEPRECATE : public Random {
public:
    struct Data {
	Data() : states(nullptr), buffer(nullptr), metadata(nullptr) {};
	// Data(Data&& orig) : num_states(orig.states), states(orig.states), buffer(orig.buffer), num(orig.num) {};

	struct Metadata {
	    size_t num_states;
	    size_t num;		// Remove?
	    
	};
	Array<curandState_t> states;
	size_t* buffer;		// What about Gaussian (float) vs int?
	// static_assert( sizeof(float) == sizeof(size_t) );
	// static_assert( sizeof(int) == sizeof(size_t) );
	Metadata* metadata;
	
	Data send_children(Resource& location) const {
	    const Resource& local = Resource::Local();
	    Data tmp{*this};
	    tmp.states = nullptr;
	    tmp.buffer = nullptr;
	    
	    if (local.type == Resource::CPU) {
		if (location.type == Resource::GPU) {
		    if (states != nullptr) {
			gpuErrchk(cudaMalloc(&(tmp.states), sizeof(curandState_t) * num_states));
			gpuErrchk(cudaMemcpy(tmp.states, states, sizeof(curandState_t) * num_states, cudaMemcpyHostToDevice));
		    }
		    if (buffer != nullptr) {
			gpuErrchk(cudaMalloc(&(tmp.buffer), sizeof(size_t) * buffer_size));
			gpuErrchk(cudaMemcpy(tmp.buffer, buffer, sizeof(float) * buffer_size, cudaMemcpyHostToDevice));
		    }
		} else {
		    Exception(NotImplementedError, "");
		}
	    } else {
		Exception(NotImplementedError, "");
	    }
	    return tmp;
	};
	void clear() {
	    Exception(NotImplementedError, "");	    
	};
    };

    // Metadata stored on host even if Data is on device
    struct Metadata {
 	curandGenerator_t generator;
	curandState_t *states_d;	
	size_t* buffer_d;    // Device buffer (possibly used)
	Proxy<Data> data;    // State data that may be found elsewhere
	size_t* buffer;	     // Local buffer
	size_t num;	     // Number of elements in local buffer
    };


    RandomImplCUDA_DEPRECATE(Random::Conf c, Resource& location) {
	// Note: NUM_THREADS refers to CUDA threads, whereas c.num_threads refers to (?) number of random numbers to be generated in parallel

	LOGINFO("Creating RandomImplCUDA with seed {}", c.seed);
	
	assert( location.type == Resource::GPU );
	
	// For now create temporary buffers locally, then copy to 'location'
	//   Can optimize at a later time by avoiding temporary allocations
	Data tmp;
	tmp.states = new curandState_t[c.num_threads];
	tmp.buffer = new size_t[buffer_size];
	tmp.num = 0;

	LOGINFO("...Sending data");

	metadata.data = send(location, tmp);
	metadata.buffer = new size_t[buffer_size]; // TODO handle case if RandomImplCUDA is on location
	metadata.num = 0;
	

	LOGINFO("...Cleaning temporary data");
	delete [] tmp.states;
	delete [] tmp.buffer;

	// gpuErrchk(cudaMalloc(&(data.states), sizeof(curandState_t) * c.num_threads));
	// int nBlocks = c.num_threads / NUM_THREADS + 1;
	// initKernel<<< nBlocks, NUM_THREADS >>>(c.seed, data.states, c.num_threads);
	// gpuErrchk(cudaDeviceSynchronize());

	// TODO consider location whe ncreating generator!
	// Create RNG and set seed
	LOGINFO("...Creating generator");
	curandCreateGenerator(&(metadata.generator), CURAND_RNG_PSEUDO_XORWOW);
	curandSetPseudoRandomGeneratorSeed( metadata.generator, c.seed );

	// Set seed 
	LOGINFO("...Setting seed");
	gpuErrchk(cudaDeviceSynchronize());
	LOGINFO("...synced");
	size_t num_blocks = 1 + (c.num_threads-1)/NUM_THREADS;
	LOGINFO("...launching");
	// LOGINFO("...hmmm {}", (void*) metadata.states);
	RandomImplCUDA_init_kernel<<< num_blocks, NUM_THREADS >>>(c.seed, metadata.states, c.num_threads);
	LOGINFO("...Kernel launched");
	gpuErrchk(cudaDeviceSynchronize());
	LOGINFO("...Done");

	// Allocate temporary storage


	// gpuErrchk(cudaMalloc((void**) &(data.buffer), sizeof(size_t) * buffer_size));
	    
    };
   
    // Methods for maninpulating the state
    void init(size_t num, unsigned long seed, size_t offset) {
	// gpuErrchk(cudaMalloc((void**)&uniform_d, sizeof(float) * RAND_N));
	// gpuErrchk(cudaMalloc((void**)&integer_d, sizeof(unsigned int) * RAND_N));
	// gpuErrchk(cudaMalloc((void**)&gaussian_d, sizeof(float) * RAND_N));
	// integer_h = new unsigned int[RAND_N];
	// uniform_h = new float[RAND_N];
	// gaussian_h = new float[RAND_N];
	// uniform_n = 0;
	// integer_n = 0;
	// gaussian_n = 0;
	// // 
    };

    // Methods for generating random values
    HOST inline float gaussian() {
	if (metadata.num == 0) {
	    cuRandchk(curandGenerateUniform( metadata.generator, (float*) metadata.data.addr->buffer, buffer_size ));
	    gpuErrchk(cudaMemcpy(metadata.buffer, metadata.data.addr->buffer, sizeof(float) * buffer_size, cudaMemcpyDeviceToHost));
	    metadata.num = buffer_size-1;
	}
	return (float)(metadata.buffer[metadata.num--]);
    };
    HOST inline float gaussian(size_t idx, size_t thread) {
	return 0;
    };

    // HOST DEVICE inline float gaussian(RandomState* state) {};
    HOST inline Vector3 gaussian_vector(size_t idx, size_t thread) {
	return Vector3();
    };
    unsigned int integer() {
	return 0;
    };
    unsigned int poisson(float lambda) {
	return 0;
    };
    float uniform() {
	return 0;
    };
private:
    // Proxy<Data> data;
    Metadata metadata;
    
    // void reorder(int *a, int n);
    };
// */
#endif

Random* Random::GetRandom(Conf&& conf) {
	// Checks _randoms for a matching configuration, returns one if found, otherwise creates
	if (conf.backend == Conf::Default) {
#ifdef USE_CUDA
	    conf.backend = Conf::CUDA;
#else
	    conf.backend = Conf::CPU;
#endif
	}

	// Insert configuration into map, if it exists 
	auto emplace_result = Random::_objects.emplace(conf, nullptr);
	auto& it = emplace_result.first;
	bool& inserted = emplace_result.second;
	if (inserted) {
	    // Conf not found, so create a new one 
	    Random* tmp;
	    Resource r{conf.backend == Conf::CUDA ? Resource::GPU : Resource::CPU ,0};
	    
	    switch (conf.backend) {
	    case Conf::CUDA:
#ifdef USE_CUDA
		// TODO: replace Resource below / overload GetRandom() so it can target a resource
		tmp = new RandomImplCUDA<128,1024>(conf, r );
#else
		LOGWARN("Random::GetRandom(): CUDA disabled, creating CPU random instead");
		tmp = new RandomImpl();
#endif
		break;
	    case Conf::CPU:
		tmp = new RandomImpl();
		break;
	    default:
		Exception(ValueError, "Random::GetRandom(): Unrecognized backend");
	    }
	    it->second = tmp;
	}
	return it->second;
}
