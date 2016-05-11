#include "GPUManager.h"

int GPUManager::nGPUs = 0;
bool GPUManager::is_safe = true;
std::vector<int> GPUManager::gpus, GPUManager::timeouts, GPUManager::notimeouts;
std::vector<GPUPeer> GPUManager::peers;

// Currently unused
std::vector<cudaDeviceProp> GPUManager::properties;
std::vector<cudaStream_t> GPUManager::streams;
std::vector<cudaEvent_t> GPUManager::events;

void GPUManager::init() {
	load_info();
	is_safe = true;
	gpus = notimeouts;
	// If every GPU times out, use them
	if (gpus.size() == 0) {
		printf("WARNING: Using GPUs that may time out\n");
		is_safe = false;
		gpus = timeouts;
	}
}

void GPUManager::load_info() {
	cudaGetDeviceCount(&nGPUs);
	printf("Found %d GPU(s)\n", nGPUs);

	for (int dev = 0; dev < nGPUs; dev++) {
		peers.push_back(GPUPeer(dev));
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, dev);
		properties.push_back(prop);

		// Print out properties
		printf("[%d] %s ", dev, prop.name);
		if (prop.kernelExecTimeoutEnabled) {
			printf("(may timeout) ");
			timeouts.push_back(dev);
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
		gpus.insert(gpus.end(), timeouts.begin(), timeouts.end());
		is_safe = false;
	}
}
