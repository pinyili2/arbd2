#include "CudaManager.h"
#include <format>

#ifdef USE_CUDA

namespace ARBD {

// Static member initialization
std::vector<GPUManager::GPU> GPUManager::all_gpus_;
std::vector<GPUManager::GPU> GPUManager::gpus_;
std::vector<GPUManager::GPU> GPUManager::no_timeouts_;
bool GPUManager::is_safe_{true};

#ifdef USE_NCCL
std::vector<ncclComm_t> GPUManager::comms_;
#endif

// GPU class implementation
GPUManager::GPU::GPU(unsigned int id) : id_(id) {
    CUDA_CHECK(cudaSetDevice(id_));
    CUDA_CHECK(cudaGetDeviceProperties(&properties_, id_));
    
    may_timeout_ = properties_.kernelExecTimeoutEnabled;
    
    LOGINFO("[{}] {} {}| SM {}.{} {:.2f}GHz, {:.1f}GB RAM",
         id_, properties_.name, 
         may_timeout_ ? "(may timeout) " : "",
         properties_.major, properties_.minor,
         static_cast<float>(properties_.clockRate) * 1e-7,
         static_cast<float>(properties_.totalGlobalMem) * 7.45058e-10);
    
    create_streams();
}

GPUManager::GPU::~GPU() {
    destroy_streams();
}

void GPUManager::GPU::create_streams() {
    int curr;
    CUDA_CHECK(cudaGetDevice(&curr));
    CUDA_CHECK(cudaSetDevice(id_));

    if (streams_created_) destroy_streams();
    last_stream_ = -1;
    
    for (auto& stream : streams_) {
        stream = Stream();
    }
    
    streams_created_ = true;
    CUDA_CHECK(cudaSetDevice(curr));
}

void GPUManager::GPU::destroy_streams() {
    int curr;
    LOGTRACE("Destroying streams");
    
    if (cudaGetDevice(&curr) == cudaSuccess) { // Avoid errors during shutdown
        CUDA_CHECK(cudaSetDevice(id_));
        streams_created_ = false;
        CUDA_CHECK(cudaSetDevice(curr));
    }
}

// GPUManager static methods implementation
void GPUManager::init() {
    int num_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    LOGINFO("Found {} GPU(s)", num_gpus);
    
    all_gpus_.clear();
    no_timeouts_.clear();
    
    for (int dev = 0; dev < num_gpus; dev++) {
        GPU g(dev);
        all_gpus_.push_back(std::move(g));
        if (!all_gpus_.back().may_timeout()) {
            no_timeouts_.push_back(all_gpus_.back());
        }
    }
    
    is_safe_ = false;
    if (all_gpus_.empty()) {
        ARBD_Exception(ExceptionType::ValueError, "No GPUs found");
    }
}

void GPUManager::load_info() {
    init();
    gpus_ = all_gpus_;
    init_devices();
}

void GPUManager::init_devices() {
    LOGINFO("Initializing GPU devices... ");
    std::string msg;
    
    for (size_t i = 0; i < gpus_.size(); i++) {
        if (i > 0) {
            if (i == gpus_.size() - 1) {
                msg += " and ";
            } else {
                msg += ", ";
            }
        }
        msg += std::to_string(gpus_[i].id());
        
        use(i);
        CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
        gpus_[i].create_streams();
    }
    
    LOGINFO("Initializing GPUs: {}", msg);
    use(0);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void GPUManager::select_gpus(std::span<const unsigned int> gpu_ids) {
    gpus_.clear();
    for (unsigned int id : gpu_ids) {
        if (id >= all_gpus_.size()) {
            ARBD_Exception(ExceptionType::ValueError, 
                "Invalid GPU ID: {}", id);
        }
        gpus_.push_back(all_gpus_[id]);
    }
    init_devices();
    
#ifdef USE_NCCL
    init_comms();
#endif
}

void GPUManager::use(int gpu_id) {
    gpu_id = gpu_id % static_cast<int>(gpus_.size());
    CUDA_CHECK(cudaSetDevice(gpus_[gpu_id].id()));
}

void GPUManager::sync(int gpu_id) {
    int curr;
    CUDA_CHECK(cudaGetDevice(&curr));
    CUDA_CHECK(cudaSetDevice(gpus_[gpu_id].id()));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaSetDevice(curr));
}

void GPUManager::sync() {
    if (gpus_.size() > 1) {
        int curr;
        CUDA_CHECK(cudaGetDevice(&curr));
        for (const auto& gpu : gpus_) {
            CUDA_CHECK(cudaSetDevice(gpu.id()));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaSetDevice(curr));
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

int GPUManager::current() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

void GPUManager::safe(bool make_safe) {
    if (make_safe == is_safe_) return;
    
    if (make_safe) {
        if (no_timeouts_.empty()) {
            LOGWARN("No safe GPUs available");
            return;
        }
        all_gpus_ = no_timeouts_;
        is_safe_ = true;
    } else {
        is_safe_ = false;
    }
}

int GPUManager::get_initial_gpu() {
    for (const auto& gpu : gpus_) {
        if (!gpu.may_timeout()) {
            return gpu.id();
        }
    }
    return 0;
}

#ifdef USE_NCCL
void GPUManager::init_comms() {
    if (gpus_.size() == 1) return;
    
    std::vector<int> gpu_ids;
    gpu_ids.reserve(gpus_.size());
    for (const auto& gpu : gpus_) {
        gpu_ids.push_back(gpu.id());
    }
    
    comms_.resize(gpus_.size());
    NCCL_CHECK(ncclCommInitAll(comms_.data(), gpus_.size(), gpu_ids.data()));
}
#endif

} // namespace ARBD

#endif // USE_CUDA 