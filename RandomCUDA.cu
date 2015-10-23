#include "RandomCUDA.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true) {
	if (code != cudaSuccess) {
		fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define cuRandchk(ans) { cuRandAssert((ans), __FILE__, __LINE__); }
inline void cuRandAssert(curandStatus code, char *file, int line, bool abort=true) {
	if (code != CURAND_STATUS_SUCCESS) {
		fprintf(stderr, "CURAND Error: %d %s %d\n", code, file, line);
		if (abort) exit(code);
	}
}

__global__
void initKernel(unsigned long seed, curandState_t *state, int num);

void Random::init(int num, unsigned long seed) {
	if (states != NULL)
		gpuErrchk(cudaFree(states));
	gpuErrchk(cudaMalloc(&states, sizeof(curandState) * num));
	int nBlocks = num / NUM_THREADS + 1;
	initKernel<<< nBlocks, NUM_THREADS >>>(seed, states, num);
	gpuErrchk(cudaDeviceSynchronize());

	// Create RNG and set seed
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW);
	curandSetPseudoRandomGeneratorSeed(generator, seed);
	
	if (uniform_d != NULL) {
		gpuErrchk(cudaFree(uniform_d));
		gpuErrchk(cudaFree(integer_d));
		delete[] integer_h;
		delete[] uniform_h;
	}
	gpuErrchk(cudaMalloc(&uniform_d, sizeof(float) * RAND_N));
	gpuErrchk(cudaMalloc(&integer_d, sizeof(unsigned int) * RAND_N));
	integer_h = new unsigned int[RAND_N];
	uniform_h = new float[RAND_N];
	uniform_n = 0;
	integer_n = 0;
}

float Random::uniform() {
	if (uniform_n < 1) {
		cuRandchk(curandGenerateUniform(generator, (float*) uniform_d, RAND_N));
		gpuErrchk(cudaMemcpy(uniform_h, uniform_d, sizeof(float) * RAND_N, cudaMemcpyDeviceToHost));
		uniform_n = RAND_N;
	}
	return uniform_h[--uniform_n];
}

unsigned int Random::poisson(float lambda) {
	const float l = exp(-lambda);
	unsigned int k = 0;
	float p = uniform();
	while (p >= l) {
		p *= uniform();
		k = k + 1;
	}
	return k;
}

unsigned int Random::integer() {
	if (integer_n < 1) {
		curandGenerate(generator, integer_d, RAND_N);
		gpuErrchk(cudaMemcpy(integer_h, integer_d, sizeof(unsigned int) * RAND_N, cudaMemcpyDeviceToHost));
		integer_n = RAND_N;
	}
	return integer_h[--integer_n];
}

void Random::reorder(int a[], int n) {
	for (int i = 0; i < (n-1); ++i) {
		unsigned int j = i + (integer() % (n-i));
		if ( j == i )
			continue;
		swap<int>(a[i], a[j]);
		const int tmp = a[j];
		a[j] = a[i];
		a[i] = tmp;
	}
}

__global__ void initKernel(unsigned long seed, curandState_t *state, int num) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num)
		curand_init(seed, idx, 0, &state[idx]);
}
