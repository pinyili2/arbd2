#ifndef GPU_MANAGER_H
#define GPU_MANAGER_H

#include <cstdio>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define NUMSTREAMS 8

// GPUs capable of Peer Access 
// (Currently unused)
struct GPUPeer {
	int gpu;
	std::vector<int> gpus;
	GPUPeer() : gpu(-1) {}
	GPUPeer(int gpu) : gpu(gpu) {}
};

class GPUManager {
private:
	static std::vector<int> timeouts, notimeouts;
	static void init_devices();
	static int nGPUs;
	static bool is_safe;

    static void create_streams();

public:	
    static cudaStream_t* stream;
    static int last_stream;
	static std::vector<int> allGpus;
	static std::vector<int> gpus;
	static std::vector<cudaDeviceProp> properties;
	
	static bool safe() { return is_safe; }

	// init
	// Initializes gpus and properties vector
	// Bad things may happen if this is called more than once
	static void init();

	static void load_info();

	// set
	// Set the GPU
	static void set(int gpu_id);

	// current
	// @return the current GPU a thread is using
	static int current();
	
	// safe
	// @param whether gpus should contain GPUs that may timeout
	static void safe(bool make_safe);
	
	static int getInitialGPU();

        // 
    inline const cudaStream_t& get_next_stream() {
	if (last_stream == NUMSTREAMS-1) {
	    last_stream = 0;
	} else {
            last_stream +=1;
	}
	return stream[last_stream];
    };
		

	// Currently unused
	static std::vector<GPUPeer> peers;
	static std::vector<cudaEvent_t> events;
       
};

#endif
