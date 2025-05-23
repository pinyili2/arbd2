
#pragma once

#include "ARBDException.h"
#include <vector>
#include <string>
#include <memory>
#include <functional>

#ifdef __CUDACC__
    #define HOST __host__
    #define DEVICE __device__
#else
    #define HOST
    #define DEVICE
#endif

#if defined(__CUDACC__) // NVCC
   #define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
  #define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
  #define MY_ALIGN(n) __declspec(align(n))
#else
  #error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

#ifndef USE_CUDA
struct MY_ALIGN(16) float4 {
    float4() : x(0), y(0), z(0), w(0) {};
    float4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {};
    float4 operator+(const float4& o) const {
        return float4(x+o.x, y+o.y, z+o.z, w+o.w);
    };
    float4 operator*(float s) const {
        return float4(x*s, y*s, z*s, w*s);
    };
    
    float x, y, z, w;
};
#endif

// Type traits for SFINAE
template <typename T, typename = void>
struct has_copy_to_cuda : std::false_type {};

template <typename T>
struct has_copy_to_cuda<T, decltype(std::declval<T>().copy_to_cuda(), void())> : std::true_type {};

#ifdef USE_CUDA
#include <cstdio>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

// CUDA error checking
namespace cuda {
    inline void check_error(cudaError_t error, const char* file, int line) {
        if (error != cudaSuccess) {
            throw _ARBDException(std::string(file) + ":" + std::to_string(line), 
                               CUDARuntimeError, 
                               "CUDA error: %s", cudaGetErrorString(error));
        }
    }
}

#define CUDA_CHECK(call) cuda::check_error(call, __FILE__, __LINE__)

#ifdef USE_NCCL
#include <nccl.h>
#define NCCLCHECK(cmd) do {                     \
      ncclResult_t r = cmd;                     \
      if (r!= ncclSuccess) {                    \
          printf("Failed, NCCL error %s:%d '%s'\n", \
             __FILE__,__LINE__,ncclGetErrorString(r)); \
          exit(EXIT_FAILURE);                   \
      }                                         \
  } while(0)
#endif

#define NUMSTREAMS 8

// GPUs capable of Peer Access 
struct GPUPeer {
    int gpu;
    std::vector<int> gpus;
    
    GPUPeer() : gpu(-1) {}
    explicit GPUPeer(int gpu) : gpu(gpu) {}
};

// RAII wrapper for CUDA streams
class CudaStream {
public:
    CudaStream() : stream_(nullptr) {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }
    
    explicit CudaStream(unsigned int flags) : stream_(nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
    }
    
    ~CudaStream() {
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }
    
    // Prevent copying
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    
    // Allow moving
    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }
    
    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_) {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }
    
    void synchronize() {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    
    cudaStream_t get() const { return stream_; }
    operator cudaStream_t() const { return stream_; }
    
private:
    cudaStream_t stream_;
};

class GPU {
    /* Class to represent individual GPUs on a node */
    friend class GPUManager;
private:
    bool may_timeout;
    unsigned int id;
    std::array<CudaStream, NUMSTREAMS> streams;
    int last_stream;
    bool streams_created;
    
    void create_streams();
    void destroy_streams();
    
    cudaDeviceProp properties;

public:
    explicit GPU(unsigned int id);
    ~GPU();

    inline const cudaStream_t& get_stream(unsigned int stream_id) const {
        return streams[stream_id % NUMSTREAMS].get();
    }

    inline const cudaStream_t& get_next_stream() {
        if (last_stream >= NUMSTREAMS-1) {
            last_stream = 0;
        } else {
            last_stream += 1;
        }
        return streams[last_stream].get();
    }
};

// Implementation for GPU constructor
inline GPU::GPU(unsigned int id) : id(id), last_stream(-1), streams_created(false) {
    CUDA_CHECK(cudaSetDevice(id));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, id));
    
    may_timeout = properties.kernelExecTimeoutEnabled;
    
    // Log device info
    char timeout_str[32] = "";
    if (may_timeout) {
        sprintf(timeout_str, "(may timeout) ");
    }
    
    LOGINFO("[{}] {} {}| SM {}.{} {:.2f}GHz, {:.1f}GB RAM",
         id, properties.name, timeout_str, properties.major, properties.minor,
         static_cast<float>(properties.clockRate) * 10E-7, 
         static_cast<float>(properties.totalGlobalMem) * 7.45058e-10);
    
    create_streams();
}

// Implementation for GPU destructor
inline GPU::~GPU() {
    destroy_streams();
}

// Implementations for stream methods
inline void GPU::create_streams() {
    int curr;
    CUDA_CHECK(cudaGetDevice(&curr));
    CUDA_CHECK(cudaSetDevice(id));

    if (streams_created) destroy_streams();
    last_stream = -1;
    
    for (int i = 0; i < NUMSTREAMS; i++) {
        streams[i] = CudaStream();
    }
    
    streams_created = true;
    CUDA_CHECK(cudaSetDevice(curr));
}

inline void GPU::destroy_streams() {
    int curr;
    LOGTRACE("Destroying streams");
    
    if (cudaGetDevice(&curr) == cudaSuccess) { // Avoid errors during shutdown
        CUDA_CHECK(cudaSetDevice(id));
        
        // Streams destructors will handle cleanup
        streams.clear();
        
        streams_created = false;
        CUDA_CHECK(cudaSetDevice(curr));
    }
}

class GPUManager {
private:
    static std::vector<GPU> all_gpus, timeouts, no_timeouts;
    static void init_devices();
    static int num_gpus;
    static bool is_safe;

#ifdef USE_NCCL
    static void init_comms();
    static std::vector<ncclComm_t> comms;
#endif

public:    
    static size_t all_gpu_size() { return all_gpus.size(); }
    static std::vector<GPU> gpus;
    
    static bool safe() { return is_safe; }

    // Initialize GPUs and get their properties
    static void init();

    static void load_info();

    // Select which GPUs to use
    static void select_gpus(const std::vector<unsigned int>& gpu_ids);
    
    // Use a specific GPU
    static void use(int gpu_id);

    // Synchronize on a specific GPU
    static void sync(int gpu_id);
    
    // Synchronize on all selected GPUs
    static void sync();

    // Get current device
    static int current();
    
    // Control GPU safety (timeout)
    static void safe(bool make_safe);
    
    // Get initial GPU for a new task
    static int getInitialGPU();

    // Get next available stream
    inline static const cudaStream_t& get_next_stream() {
        return gpus[0].get_next_stream();
    }

#ifdef USE_NCCL
    // NCCL broadcast operations
    template<typename T>
    static void nccl_broadcast(int root, const std::vector<T*>& send_d, 
                             const std::vector<T*>& recv_d, unsigned int size, 
                             int stream_id = -1) {
        if (gpus.size() == 1) return;
        
        ncclGroupStart();
        for (size_t i = 0; i < gpus.size(); ++i) {
            cudaStream_t stream = (stream_id >= 0) ? 
                                  gpus[i].get_stream(stream_id) : 0;
                                  
            NCCLCHECK(ncclBroadcast(
                static_cast<const void*>(send_d[i]),
                static_cast<void*>(recv_d[i]),
                size * sizeof(T) / sizeof(float),
                ncclFloat, root, comms[i], stream));
        }
        ncclGroupEnd();
    }
    
    // NCCL reduce operations
    template<typename T>
    static void nccl_reduce(int root, const std::vector<T*>& send_d, 
                          const std::vector<T*>& recv_d, unsigned int size, 
                          int stream_id = -1) {
        if (gpus.size() == 1) return;
        
        ncclGroupStart();
        for (size_t i = 0; i < gpus.size(); ++i) {
            cudaStream_t stream = (stream_id >= 0) ? 
                                 gpus[i].get_stream(stream_id) : 0;
                                 
            NCCLCHECK(ncclReduce(
                static_cast<const void*>(send_d[i]),
                static_cast<void*>(recv_d[i]),
                size * sizeof(T) / sizeof(float),
                ncclFloat, ncclSum, root, comms[i], stream));
        }
        ncclGroupEnd();
    }
#endif // USE_NCCL
};

#endif // USE_CUDA
