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
    HOST inline float gaussian() {
	return 0;
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


// GPU-resident?
template<size_t NUM_THREADS = 128, size_t buffer_size = 1024>
class RandomImplCUDA : public Random {
public:
    struct Data {
	Data() : states(nullptr), buffer(nullptr), num(0) {};
	curandState_t *states;
	size_t* buffer;		// What about Gauss vs int?
	size_t num;		// Remove?
    };

    // Metadata stored on host even if Data is on device
    struct Metadata {
 	curandGenerator_t generator;
	Proxy<Data> data;		// state data may be found elsewhere
	size_t* buffer;
	size_t num;
    };


    RandomImplCUDA(Random::Conf c, Resource&& location) {
	// Note: NUM_THREADS refers to CUDA threads, whereas c.num_threads refers to (?) number of random numbers to be generated in parallel

	INFO("Creating RandomImplCUDA with seed {}", c.seed);
	
	assert( location.type == Resource::GPU );
	
	// For now create temporary buffers locally, then copy to 'location'
	//   Can optimize at a later time by avoiding temporary allocations
	Data tmp;
	tmp.states = new curandState_t[c.num_threads];
	tmp.buffer = new size_t[buffer_size];
	tmp.num = 0;

	INFO("...Sending data");

	metadata.data = send(location, tmp);
	metadata.buffer = new size_t[buffer_size]; // TODO handle case if RandomImplCUDA is on location
	metadata.num = 0;
	

	INFO("...Cleaning temporary data");
	delete [] tmp.states;
	delete [] tmp.buffer;

	// gpuErrchk(cudaMalloc(&(data.states), sizeof(curandState_t) * c.num_threads));
	// int nBlocks = c.num_threads / NUM_THREADS + 1;
	// initKernel<<< nBlocks, NUM_THREADS >>>(c.seed, data.states, c.num_threads);
	// gpuErrchk(cudaDeviceSynchronize());

	// TODO consider location whe ncreating generator!
	// Create RNG and set seed
	INFO("...Creating generator");
	curandCreateGenerator(&(metadata.generator), CURAND_RNG_PSEUDO_XORWOW);
	curandSetPseudoRandomGeneratorSeed( metadata.generator, c.seed );

	// Set seed 
	INFO("...Setting seed");
	gpuErrchk(cudaDeviceSynchronize());
	INFO("...synced");
	size_t num_blocks = 1 + (c.num_threads-1)/NUM_THREADS;
	RandomImplCUDA_init_kernel<<< num_blocks, NUM_THREADS >>>(c.seed, metadata.data.addr->states, c.num_threads);
	INFO("...Kernel launched");
	gpuErrchk(cudaDeviceSynchronize());
	INFO("...Done");

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

	    switch (conf.backend) {
	    case Conf::CUDA:
#ifdef USE_CUDA
		// TODO: replace Resource below / overload GetRandom() so it can target a resource 
		tmp = new RandomImplCUDA<128,1024>(conf, Resource{Resource::GPU,0} );
#else
		WARN("Random::GetRandom(): CUDA disabled, creating CPU random instead");
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
