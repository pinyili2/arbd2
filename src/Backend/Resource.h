#pragma once
#include "ARBDLogger.h"

// Define HOST and DEVICE macros
#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#include "Backend/CUDA/CUDAManager.h"
#else
#define HOST
#define DEVICE
#endif

#ifdef USE_SYCL
#include "Backend/SYCL/SYCLManager.h"
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif



/**
 * @brief Get current resource index (device ID or MPI rank)
 */
inline size_t caller_id() {
#ifdef __CUDACC__
#ifdef USE_CUDA
    if (cudaGetDevice(nullptr) == cudaSuccess) { 
        int device;
        cudaGetDevice(&device);
        return static_cast<size_t>(device);
    }
#endif
#endif

#ifdef USE_SYCL
    return static_cast<size_t>(ARBD::SYCL::SYCLManager::get_current_device().id());
#endif

#ifdef USE_MPI
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return static_cast<size_t>(rank);
#else
    return 0;
#endif
}

/**
 * @brief Resource representation for heterogeneous computing
 * Supports CUDA, SYCL, and MPI resource types
 */
namespace ARBD {
/**
 * @brief Resource representation for heterogeneous computing environments
 * 
 * The Resource class provides a unified interface for representing and managing
 * computational resources across different backends including CUDA GPUs, SYCL devices,
 * and MPI processes. It enables transparent resource identification and locality
 * checking in heterogeneous computing scenarios.
 * 
 * @details This class serves as a fundamental building block for resource management
 * in distributed and parallel computing environments. It supports:
 * - CUDA GPU devices for NVIDIA GPU computing
 * - SYCL devices for cross-platform parallel computing
 * - MPI processes for distributed computing
 * 
 * The resource hierarchy is supported through parent-child relationships, allowing
 * for complex resource topologies and nested resource management.
 * 
 * @note The class is designed to work in both host and device code contexts when
 * compiled with CUDA, using appropriate HOST and DEVICE decorators.
 * 
 * @example Basic Usage:
 * ```cpp
 * // Create a CUDA resource for device 0
 * ARBD::Resource cuda_res(ARBD::Resource::CUDA, 0);
 * 
 * // Create a SYCL resource for device 1
 * ARBD::Resource sycl_res(ARBD::Resource::SYCL, 1);
 * 
 * // Check if resource is local to current execution context
 * if (cuda_res.is_local()) {
 *     // Perform local operations
 * }
 * ```
 * 
 * @example Hierarchical Resources:
 * ```cpp
 * // Create parent MPI resource
 * ARBD::Resource mpi_root(ARBD::Resource::MPI, 0);
 * 
 * // Create child CUDA resource
 * ARBD::Resource cuda_child(ARBD::Resource::CUDA, 0, &mpi_root);
 * ```
 * 
 * @see ResourceType for available resource types
 * @see is_local() for locality checking
 * @see getTypeString() for human-readable type names
 */

struct Resource {
    enum ResourceType {CUDA, SYCL, MPI};
    ResourceType type;   
    size_t id;          
    Resource* parent;   

    HOST DEVICE Resource() : type(MPI), id(0), parent(nullptr) {}
    HOST DEVICE Resource(ResourceType t, size_t i) : type(t), id(i), parent(nullptr) {}
    HOST DEVICE Resource(ResourceType t, size_t i, Resource* p) : type(t), id(i), parent(p) {}

    HOST DEVICE constexpr const char* getTypeString() const {
        switch(type) {
            case MPI: return "MPI";
            case CUDA: return "CUDA";
            case SYCL: return "SYCL";
            default: return "Unknown";
        }
    }

    HOST DEVICE bool is_local() const {
        bool ret = false;

#ifdef USE_CUDA
        if (cudaGetDevice(nullptr) == cudaSuccess) {  // Are we in a CUDA context?
            int current_device;
            cudaGetDevice(&current_device);
            
            switch(type) {
                case CUDA:
                    ret = (current_device == static_cast<int>(id));
                    break;
                case SYCL:
                case MPI:
                    ret = false;  // SYCL/MPI resources are never local to CUDA
                    break;
            }
            LOGWARN("Resource::is_local(): CUDA context - type %s, device %zu, current %d, returning %d", 
                    getTypeString(), id, current_device, ret);
        } else 
#endif
#ifdef USE_SYCL
        if (type == SYCL) {
            // Check if we're in a SYCL context
            try {
                auto& current_device = SYCL::SYCLManager::get_current_device();
                ret = (current_device.id() == id);
                LOGINFO("Resource::is_local(): SYCL context - device %zu, current %u, returning %d", 
                       id, current_device.id(), ret);
            } catch (...) {
                ret = false;
                LOGINFO("Resource::is_local(): SYCL device not available, returning false");
            }
        } else
#endif
        {   // CPU/MPI context
            switch(type) {
                case CUDA:
                    if (parent != nullptr) {
                        ret = parent->is_local();
                        LOGINFO("Resource::is_local(): CPU checking CUDA parent, returning %d", ret);
                    } else {
                        ret = false;
                        LOGINFO("Resource::is_local(): CPU with no CUDA parent, returning false");
                    }
                    break;
                case SYCL:
                    if (parent != nullptr) {
                        ret = parent->is_local();
                        LOGINFO("Resource::is_local(): CPU checking SYCL parent, returning %d", ret);
                    } else {
                        ret = false;
                        LOGINFO("Resource::is_local(): CPU with no SYCL parent, returning false");
                    }
                    break;
                case MPI:
                    ret = (caller_id() == id);
                    LOGINFO("Resource::is_local(): MPI direct check - caller %zu, id %zu, returning %d", 
                           caller_id(), id, ret);
                    break;
            }
        }
        return ret;
    }

    static Resource Local() {
#ifdef USE_CUDA
        if (cudaGetDevice(nullptr) == cudaSuccess) {  // Are we in a CUDA context?
            int device;
            cudaGetDevice(&device);
            return Resource{CUDA, static_cast<size_t>(device)};
        }
#endif
#ifdef USE_SYCL
        // Check if SYCL is available and preferred
        try {
            auto& current_device = SYCL::SYCLManager::get_current_device();
            return Resource{SYCL, static_cast<size_t>(current_device.id())};
        } catch (...) {
            // Fall through to MPI if SYCL is not available
        }
#endif
        return Resource{MPI, caller_id()};
    }

    HOST DEVICE bool operator==(const Resource& other) const {
        return type == other.type && id == other.id;
    }

    HOST std::string toString() const {
        return std::string(getTypeString()) + "[" + std::to_string(id) + "]";
    }

    /**
     * @brief Check if the resource supports asynchronous operations
     */
    HOST DEVICE bool supports_async() const {
        switch(type) {
            case CUDA:
            case SYCL:
                return true;
            case MPI:
                return false;  // MPI operations are typically synchronous in this context
            default:
                return false;
        }
    }

    /**
     * @brief Get the memory space type for this resource
     */
    HOST DEVICE constexpr const char* getMemorySpace() const {
        switch(type) {
            case CUDA: return "device";
            case SYCL: return "device";
            case MPI: return "host";
            default: return "unknown";
        }
    }

    /**
     * @brief Check if this resource represents a device (GPU) or host (CPU/MPI)
     */
    HOST DEVICE bool is_device() const {
        return type == CUDA || type == SYCL;
    }

    /**
     * @brief Check if this resource represents a host (CPU/MPI)
     */
    HOST DEVICE bool is_host() const {
        return type == MPI;
    }
};

} // namespace ARBD
