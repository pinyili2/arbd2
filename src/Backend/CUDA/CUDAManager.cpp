#include "ARBDException.h"
#include "ARBDLogger.h"
#include <array>
#include <string>
#include <vector>

#ifdef USE_CUDA
#include "CUDAManager.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
namespace ARBD {
namespace CUDA {

// Static member initialization
std::vector<GPUManager::GPU> GPUManager::all_gpus_;
std::vector<GPUManager::GPU> GPUManager::gpus_;
std::vector<GPUManager::GPU> GPUManager::safe_gpus_;
std::vector<std::vector<bool>> GPUManager::peer_access_matrix_;
bool GPUManager::prefer_safe_{false};

// GPU class implementation
GPUManager::GPU::GPU(unsigned int id) : id_(id) {
  CUDA_CHECK(cudaSetDevice(id_));

#if CUDART_VERSION >= 12000
  CUDA_CHECK(cudaGetDeviceProperties_v2(&properties_, id_));
#else
  CUDA_CHECK(cudaGetDeviceProperties(&properties_, id_));
#endif

  may_timeout_ = properties_.kernelExecTimeoutEnabled;

  // Enhanced logging to match old GPUManager format
  std::string timeout_str = may_timeout_ ? "(may timeout) " : "";
  LOGINFO("[{}] {} {}| SM {}.{} {:.2f}GHz, {:.1f}GB RAM", id_, properties_.name,
          timeout_str, properties_.major, properties_.minor,
          static_cast<float>(properties_.clockRate) * 1e-6,
          static_cast<float>(properties_.totalGlobalMem) /
              (1024.0f * 1024.0f * 1024.0f));

  create_streams();
}

GPUManager::GPU::~GPU() { destroy_streams(); }

void GPUManager::GPU::create_streams() {
  int curr;
  CUDA_CHECK(cudaGetDevice(&curr));
  CUDA_CHECK(cudaSetDevice(id_));

  if (streams_created_)
    destroy_streams();
  last_stream_ = -1;

  // Create streams using CUDA stream creation (matching old implementation)
  for (size_t i = 0; i < NUM_STREAMS; ++i) {
    // Note: Stream constructor calls cudaStreamCreate internally
    streams_[i] = Stream();
  }

  streams_created_ = true;
  CUDA_CHECK(cudaSetDevice(curr));
}

void GPUManager::GPU::destroy_streams() {
  int curr;
  LOGTRACE("Destroying streams for GPU {}", id_);

  if (cudaGetDevice(&curr) == cudaSuccess) { // Avoid errors during shutdown
    CUDA_CHECK(cudaSetDevice(id_));
    streams_created_ = false;
    CUDA_CHECK(cudaSetDevice(curr));
  }
}

void GPUManager::GPU::synchronize_all_streams() {
  int curr;
  CUDA_CHECK(cudaGetDevice(&curr));
  CUDA_CHECK(cudaSetDevice(id_));
  for (auto &stream : streams_) {
    stream.synchronize();
  }

  CUDA_CHECK(cudaSetDevice(curr));
}

// GPUManager static methods implementation
void GPUManager::init() {
  int num_gpus;
  CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
  LOGINFO("Found {} GPU(s)", num_gpus);

  all_gpus_.clear();
  safe_gpus_.clear();

  for (int dev = 0; dev < num_gpus; dev++) {
    all_gpus_.emplace_back(dev);
    if (!all_gpus_.back().may_timeout()) {
      safe_gpus_.emplace_back(dev);
    }
  }

  if (all_gpus_.empty()) {
    ARBD_Exception(ExceptionType::ValueError, "No GPUs found");
  }

  // Initialize peer access matrix
  query_peer_access();
}

void GPUManager::load_info() {
  init();
  gpus_.clear();
  if (prefer_safe_) {
    for (const auto &gpu : safe_gpus_) {
      gpus_.emplace_back(gpu.id());
    }
  } else {
    for (const auto &gpu : all_gpus_) {
      gpus_.emplace_back(gpu.id());
    }
  }
  if (!gpus_.empty()) {
    init_devices();
  } else {
    LOGWARN("No GPUs selected for initialization");
  }
}

void GPUManager::init_devices() {
  LOGINFO("Initializing GPU devices...");
  std::string msg;

  // Build message string like the old implementation
  for (size_t i = 0; i < gpus_.size(); i++) {
    if (i != gpus_.size() - 1 && gpus_.size() > 1) {
      msg += std::to_string(gpus_[i].id()) + ", ";
    } else if (gpus_.size() > 1) {
      msg += "and " + std::to_string(gpus_[i].id());
    } else {
      msg += std::to_string(gpus_[i].id());
    }

    use(static_cast<int>(i));
    CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
  }

  LOGINFO("Initializing GPUs: {}", msg);
  use(0);
  CUDA_CHECK(cudaDeviceSynchronize());
}

void GPUManager::use(int gpu_id) {
  if (gpus_.empty()) {
    ARBD_Exception(ExceptionType::ValueError, "No GPUs selected");
  }
  gpu_id = gpu_id % static_cast<int>(gpus_.size());
  CUDA_CHECK(cudaSetDevice(gpus_[gpu_id].id()));
}

void GPUManager::sync(int gpu_id) {
  if (gpu_id >= static_cast<int>(gpus_.size())) {
    ARBD_Exception(ExceptionType::ValueError, "Invalid GPU index: {}", gpu_id);
  }

  int curr;
  CUDA_CHECK(cudaGetDevice(&curr));
  CUDA_CHECK(cudaSetDevice(gpus_[gpu_id].id()));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaSetDevice(curr));
}

int GPUManager::current() {
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  return device;
}

void GPUManager::prefer_safe_gpus(bool safe) {
  prefer_safe_ = safe;

  if (safe && safe_gpus_.empty()) {
    LOGWARN("No safe GPUs available (no GPUs without timeout enabled)");
    return;
  }

  // Update current GPU selection
  gpus_.clear();
  if (prefer_safe_) {
    for (const auto &gpu : safe_gpus_) {
      gpus_.emplace_back(gpu.id());
    }
  } else {
    for (const auto &gpu : all_gpus_) {
      gpus_.emplace_back(gpu.id());
    }
  }

  if (!gpus_.empty()) {
    init_devices();
  }
}

int GPUManager::get_safest_gpu() {
  if (!safe_gpus_.empty()) {
    return safe_gpus_[0].id();
  }

  if (!all_gpus_.empty()) {
    LOGWARN("No safe GPUs available, returning first available GPU");
    return all_gpus_[0].id();
  }

  ARBD_Exception(ExceptionType::ValueError, "No GPUs available");
}

void GPUManager::enable_peer_access() {
  if (gpus_.size() < 2) {
    LOGINFO("Peer access not needed with fewer than 2 GPUs");
    return;
  }

  LOGINFO("Enabling peer access between {} GPUs", gpus_.size());

  for (size_t i = 0; i < gpus_.size(); ++i) {
    CUDA_CHECK(cudaSetDevice(gpus_[i].id()));

    for (size_t j = 0; j < gpus_.size(); ++j) {
      if (i != j && can_access_peer(gpus_[i].id(), gpus_[j].id())) {
        cudaError_t result = cudaDeviceEnablePeerAccess(gpus_[j].id(), 0);
        if (result == cudaSuccess) {
          LOGDEBUG("Enabled peer access: GPU {} -> GPU {}", gpus_[i].id(),
                   gpus_[j].id());
        } else if (result != cudaErrorPeerAccessAlreadyEnabled) {
          LOGWARN("Failed to enable peer access: GPU {} -> GPU {}: {}",
                  gpus_[i].id(), gpus_[j].id(), cudaGetErrorString(result));
        }
      }
    }
  }
}

bool GPUManager::can_access_peer(int gpu1, int gpu2) {
  if (gpu1 >= static_cast<int>(peer_access_matrix_.size()) ||
      gpu2 >= static_cast<int>(peer_access_matrix_[gpu1].size())) {
    return false;
  }
  return peer_access_matrix_[gpu1][gpu2];
}

void GPUManager::set_cache_config(cudaFuncCache config) {
  for (size_t i = 0; i < gpus_.size(); ++i) {
    use(static_cast<int>(i));
    CUDA_CHECK(cudaDeviceSetCacheConfig(config));
  }
}

cudaStream_t GPUManager::get_stream(int gpu_id, size_t stream_id) {
  if (gpu_id >= static_cast<int>(gpus_.size())) {
    ARBD_Exception(ExceptionType::ValueError, "Invalid GPU index: {}", gpu_id);
  }
  return gpus_[gpu_id].get_stream(stream_id);
}

void GPUManager::query_peer_access() {
  size_t num_devices = all_gpus_.size();
  peer_access_matrix_.resize(num_devices,
                             std::vector<bool>(num_devices, false));

  for (size_t i = 0; i < num_devices; ++i) {
    for (size_t j = 0; j < num_devices; ++j) {
      if (i != j) {
        int can_access;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, all_gpus_[i].id(),
                                           all_gpus_[j].id()));
        peer_access_matrix_[i][j] = (can_access == 1);

        if (can_access) {
          LOGDEBUG("Peer access available: GPU {} -> GPU {}", all_gpus_[i].id(),
                   all_gpus_[j].id());
        }
      }
    }
  }
}

} // namespace CUDA
} // namespace ARBD
#endif // USE_CUDA
