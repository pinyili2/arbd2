#pragma once

#ifdef USE_MPI
#include <mpi.h>
#endif
#include "CUDA/CUDAManager.h"
#include "ARBDException.h"


namespace ARBD {

/**
 * @brief Get current resource index (device ID or MPI rank)
 */
#ifdef __CUDACC__
HOST DEVICE inline size_t caller_id() {
#ifdef USE_CUDA
    
    if (cudaGetDevice(nullptr) == cudaSuccess) { 
        int device;
        cudaGetDevice(&device);
        return static_cast<size_t>(device);
    }
#endif
#endif
#ifdef USE_MPI
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return static_cast<size_t>(rank);
#else
    return 0;
#endif
}


class FileHandle {
  FILE* m_file = nullptr;
public:
  FileHandle(const char* filename, const char* mode) : m_file(std::fopen(filename, mode)) {
      if (!m_file) {
        throw std::runtime_error(std::string("FileHandle: Failed to open file '") +filename + "' with mode '" + mode + "'.");}
  }
  ~FileHandle() {
      if (m_file) {
            std::fclose(m_file);
            m_file = nullptr; // Good practice to nullify after closing
        }
  }
  // Delete copy constructor/assignment
  FileHandle(const FileHandle&) = delete;
  FileHandle& operator=(const FileHandle&) = delete;
  // Allow move
  FileHandle(FileHandle&& other) noexcept : m_file(other.m_file) { other.m_file = nullptr; }
  FileHandle& operator=(FileHandle&& other) noexcept {
      if (this != &other) {
          if (m_file) std::fclose(m_file);
          m_file = other.m_file;
          other.m_file = nullptr;
      }
      return *this;
  }

  FILE* get() const { return m_file; }
  // operator FILE*() const { return m_file; } // If implicit conversion is desired
};
// Usage: FileHandle my_file("data.txt", "r"); // Automatically closes


/**
 * @brief Resource representation for heterogeneous computing
 */
struct Resource {
    enum ResourceType {CPU, MPI, CUDA, SYCL};
    ResourceType type;   
    size_t id;          
    Resource* parent;   

    HOST DEVICE Resource() : type(CPU), id(0), parent(nullptr) {}
    HOST DEVICE Resource(ResourceType t, size_t i) : type(t), id(i), parent(nullptr) {}
    HOST DEVICE Resource(ResourceType t, size_t i, Resource* p) : type(t), id(i), parent(p) {}

    HOST DEVICE constexpr const char* getTypeString() const {
        switch(type) {
            case CPU: return "CPU";
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
                case GPU:
                    ret = (current_device == static_cast<int>(id));
                    break;
                case CPU:
                case MPI:
                    ret = false;  // CPU/MPI resources are never local to GPU
                    break;
            }
            LOGWARN("Resource::is_local(): GPU context - type %s, device %zu, current %d, returning %d", 
                    getTypeString(), id, current_device, ret);
        } else 
#endif
        {   // CPU context
            switch(type) {
                case GPU:
                    if (parent != nullptr) {
                        ret = parent->is_local();
                        LOGINFO("Resource::is_local(): CPU checking GPU parent, returning %d", ret);
                    } else {
                        ret = false;
                        LOGINFO("Resource::is_local(): CPU with no GPU parent, returning false");
                    }
                    break;
                case CPU:
                case MPI:
                    ret = (caller_id() == id);
                    LOGINFO("Resource::is_local(): CPU direct check - type %s, caller %zu, id %zu, returning %d", 
                           getTypeString(), caller_id(), id, ret);
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
            return Resource{GPU, static_cast<size_t>(device)};
        }
#endif
        return Resource{CPU, caller_id()};
    }

    HOST DEVICE bool operator==(const Resource& other) const {
        return type == other.type && id == other.id;
    }

    HOST std::string toString() const {
        return std::string(getTypeString()) + "[" + std::to_string(id) + "]";
    }
};

} // namespace ARBD
