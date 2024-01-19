#include "Random.h"

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

	states_d = states_d.send_children(location);

	// TODO consider location when creating generator!
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
	RandomImplCUDA_init_kernel<<< num_blocks, NUM_THREADS >>>(c.seed, states_d.values, num_states);
	LOGINFO("...Kernel launched");

	gpuErrchk(cudaDeviceSynchronize());
	LOGINFO("...Done");

	// Allocate temporary storage
	// gpuErrchk(cudaMalloc((void**) &(data.buffer), sizeof(size_t) * buffer_size));
    };

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
