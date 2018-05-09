#include "GPUManager.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA Error: %s %s:%d\n", cudaGetErrorString(code), __FILE__, line);
      if (abort) exit(code);
   }
}

int GPUManager::nGPUs = 0;
bool GPUManager::is_safe = true;
std::vector<int> GPUManager::allGpus, GPUManager::gpus, GPUManager::timeouts, GPUManager::notimeouts;
std::vector<GPUPeer> GPUManager::peers;

// Currently unused
std::vector<cudaDeviceProp> GPUManager::properties;
// std::vector<cudaStream_t> GPUManager::streams;
std::vector<cudaEvent_t> GPUManager::events;

void GPUManager::init() {
	load_info();
	is_safe = false;
	gpus = allGpus;

	if (allGpus.size() == 0) {
	    fprintf(stderr, "Error: Did not find a GPU\n");
	    exit(1);
	}
}

void GPUManager::load_info() {
    gpuErrchk(cudaGetDeviceCount(&nGPUs));
	printf("Found %d GPU(s)\n", nGPUs);

	for (int dev = 0; dev < nGPUs; dev++) {
		peers.push_back(GPUPeer(dev));
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, dev);
		properties.push_back(prop);

		// Print out properties
		printf("[%d] %s ", dev, prop.name);
		allGpus.push_back(dev);
		if (prop.kernelExecTimeoutEnabled) {
			printf("(may timeout) ");
		} else {
		    notimeouts.push_back(dev);
		}

		printf("| SM %d.%d, ", prop.major, prop.minor);
		printf("%.2fGHz, ", (float) prop.clockRate * 10E-7);
		printf("%.1fGB RAM\n", (float) prop.totalGlobalMem * 7.45058e-10);

		for (int peer = 0; peer < nGPUs; peer++) {
			if (peer == dev)
				continue;
			int can_access;
			cudaDeviceCanAccessPeer(&can_access, dev, peer);
			if (can_access)
				peers[dev].gpus.push_back(peer);
		}
	}
	
}

void GPUManager::init_devices() {
	printf("Initializing devices... ");
	for (unsigned int i = 0; i < gpus.size(); i++) {
		if (i != gpus.size() - 1)
			printf("%d, ", gpus[i]);
		else if (gpus.size() > 1)
			printf("and %d\n", gpus[i]);
		else
			printf("%d\n", gpus[i]);
	}
	cudaSetDevice(gpus[0]);	
	/*
	for (size_t i = 0; i < gpus.size(); i++) {
		cudaSetDevice(gpus[i]);
		printf("Set device %d\n", gpus[i]);
		cudaStream_t s;
		cudaStreamCreate(&s);
		streams.push_back(s);

		cudaEvent_t e;
		cudaEventCreate(&e);
		events.push_back(e);
	}
	*/
}

void GPUManager::set(int gpu_id) {
	gpu_id = gpu_id % (int) gpus.size();
	cudaSetDevice(gpus[gpu_id]);
	cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
	create_streams();
}

int GPUManager::current() {
	int c;
	cudaGetDevice(&c);
	return c;
}

void GPUManager::safe(bool make_safe) {
	if (make_safe == is_safe) return;
	if (make_safe) {
		if (notimeouts.size() == 0) {
			printf("WARNING: No safe GPUs\n");
			return;
		}
		gpus = notimeouts;
		is_safe = true;
	} else {
	    gpus = allGpus;
	    is_safe = false;
	}
}

int GPUManager::getInitialGPU() {
    // TODO: check the load on the gpus and select an unused one
    for (uint i = 0; i < gpus.size(); ++i) {
	if (!properties[gpus[i]].kernelExecTimeoutEnabled)
	    return i; 
    }
    return 0;
}

cudaStream_t *GPUManager::stream = (cudaStream_t *) malloc(NUMSTREAMS * sizeof(cudaStream_t));
int GPUManager::last_stream = -1;
void GPUManager::create_streams() {
    printf("Creating streams\n");
	last_stream = -1;
    	for (int i = 0; i < NUMSTREAMS; i++)
	    gpuErrchk( cudaStreamCreate( &stream[i] ) );
	// gpuErrchk( cudaStreamCreateWithFlags( &(stream[i]) , cudaStreamNonBlocking ) );
}
