#pragma once

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "ARBDException.h"
#include "GPUManager.h"


HOST DEVICE size_t current_thread_idx();
    


/**
 * @brief Represents a resource that can store data and perform computations.
 */
struct Resource {
    /**
     * @brief Enum to specify the type of the resource (e.g., CPU or GPU).
     */
    enum ResourceType {CPU, MPI, GPU};
    ResourceType type; ///< Type of the resource.
    size_t id; ///< Unique identifier associated with the resource.
    Resource* parent; ///< Parent resource; nullptr unless type is GPU
        
    /**
     * @brief Checks if the resource is running on the calling thread. 
     */
    HOST DEVICE bool is_running() const {
	bool ret = true;
// #ifdef __CUDA_ACC__
// 	ret = (type == GPU);
// #else
// 	ret = (type == CPU);
// #endif
	return ret;
    }

    // Q: should this return True for GPU that is attached/assigned to current thread? Currently assuming yes.
    HOST DEVICE bool is_local() const {
	bool ret = true;
#ifdef __CUDA_ACC__
	ret = (type == GPU);
	LOGWARN("Resource::is_local() not fully implemented on GPU devices; returning {}",ret);
#else
	if (type == GPU && parent != nullptr) {
	    ret = parent->is_local();
	} else {
	    ret = (current_thread_idx() == id);
	}
#endif
	return ret;
    };
    // HOST DEVICE static bool is_local() { // check if thread/gpu idx matches some global idx };

    static Resource Local() {
	LOGWARN("Resource::Local() not properly implemented");
#ifdef __CUDA_ACC__
	return Resource{ GPU, 0 };
#else
	return Resource{ CPU, 0 };
#endif
    };
    bool operator==(const Resource& other) const { return type == other.type && id == other.id; };
	
};

