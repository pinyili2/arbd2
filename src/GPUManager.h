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

class GPU {
    /* Class to represent individual GPUs on a node */
    friend class GPUManager;
private:
    bool may_timeout;
    unsigned int id;
    cudaStream_t streams[NUMSTREAMS];

    int last_stream;
    bool streams_created;
    void create_streams();
    void destroy_streams();

    cudaDeviceProp properties;

public:
    GPU(unsigned int id);
    ~GPU();

    inline const cudaStream_t& get_stream(unsigned int stream_id) {
	return streams[stream_id];
    }

    inline const cudaStream_t& get_next_stream() {
	if (last_stream == NUMSTREAMS-1) {
	    last_stream = 0;
	} else {
            last_stream +=1;
	}
	return streams[last_stream];
    };
};

class GPUManager {

private:
	static std::vector<GPU> allGpus, timeouts, notimeouts;
	static void init_devices();
	static int nGPUs;
	static bool is_safe;

public:	
	static std::vector<GPU> gpus;
	
	static bool safe() { return is_safe; }

	// init
	// Initializes gpus and properties vector
	// Bad things may happen if this is called more than once
	static void init();

	static void load_info();

	static void select_gpus(std::vector<unsigned int>& gpu_ids);
	// use
	// Use the GPU using local index 0..N (not cudaGetDevice index)
	static void use(int gpu_id);

	// current
	// @return the current GPU a thread is using
	static int current();
	
	// safe
	// @param whether gpus should contain GPUs that may timeout
	static void safe(bool make_safe);
	
	static int getInitialGPU();

        // 
    inline const cudaStream_t& get_next_stream() {
	return gpus[0].get_next_stream();
    };

};

#endif
