#pragma once

#include <future>
#include "Resource.h"
#include "ARBDLogger.h"
#include "ARBDException.h"
#include <type_traits>
#include <string>
#include <cstring>

#include <typeinfo>

// Forward declare Resource to use in concepts
using ARBD::Resource;

namespace ARBD {
// template<typename T, typename...Args1, typename... Args2>
// __global__ void proxy_sync_call_kernel_noreturn(T* addr, (T::*memberFunc(Args1...)), Args2...args);
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


#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/std/utility>
// Kernels
template<typename T, typename RetType, typename...Args>
__global__ void proxy_sync_call_kernel(RetType* result, T* addr, RetType (T::*memberFunc(Args...)), Args...args) {
    if (blockIdx.x == 0) {
	*result = (addr->*memberFunc)(args...);
    }
}

template<typename T, typename... Args>
__global__ void constructor_kernel(T* __restrict__ devptr, Args...args) {
    if (blockIdx.x == 0) {
	devptr = new T{::cuda::std::forward<Args>(args)...};
    }
}
#endif



/**
 * @brief Concept to check if a type has a 'send_children' method that accepts a Resource parameter.
 */
template<typename T>
concept has_send_children = requires(T t, Resource r) {
    t.send_children(r);
};

/**
 * @brief Concept to check if a type has a 'no_send' member.
 */
template<typename T>
concept has_no_send = requires(T t) {
    t.no_send;
};

// Helper alias for C++14 compatibility (still needed for Metadata_t)
template <typename...>
using void_t = void;

// END traits

// Used by Proxy class 
template <typename T, typename = void>
struct Metadata_t {
    Metadata_t(const T& obj) {};
    Metadata_t(const Metadata_t<T>& other) {};
}; 
template <typename T>
struct Metadata_t<T, void_t<typename T::Metadata>> : T::Metadata {
    Metadata_t(const T& obj) : T::Metadata(obj) {};
    Metadata_t(const Metadata_t<T>& other) : T::Metadata(other) {};
};

template<typename T, typename Enable = void>
struct Proxy {
    /**
     * @brief Default constructor initializes the location to a default CPU resource and the address to nullptr.
     */
    static_assert(!std::is_same<T, Proxy>::value, "Cannot make a Proxy of a Proxy object");

    Proxy() : location(Resource{Resource::SYCL,0}), addr(nullptr), metadata(nullptr) {
	LOGINFO("Constructing Proxy<{}> @{}", typeid(T).name(), static_cast<void*>(this));
    };
    Proxy(const Resource& r) : location(r),  addr(nullptr), metadata(nullptr) {
	LOGINFO("Constructing Proxy<{}> @{}", typeid(T).name(), static_cast<void*>(this));
    };
    Proxy(const Resource& r, T& obj, T* dest = nullptr) : location(r), addr(dest == nullptr ? &obj : dest) {
	if (dest == nullptr) metadata = nullptr;
	else metadata = new Metadata_t<T>(obj);
	LOGINFO("Constructing Proxy<{}> @{} wrapping @{} with metadata @{}",
		typeid(T).name(), static_cast<void*>(this), static_cast<void*>(&obj), static_cast<void*>(metadata));
    };
        // Copy constructor
    Proxy(const Proxy<T>& other) : location(other.location), addr(other.addr), metadata(nullptr) {
        LOGINFO("Copy Constructing Proxy<{}> @{}", typeid(T).name(), static_cast<void*>(this));
        if (other.metadata != nullptr) {
            const Metadata_t<T>& tmp = *(other.metadata);
            metadata = new Metadata_t<T>(tmp);
        }
    };
    Proxy<T>& operator=(const Proxy<T>& other) {
        if (this != &other) {
            // Free existing resources.
            if (metadata != nullptr) delete metadata;
            location = other.location;
            addr = other.addr;
            const Metadata_t<T>& tmp = *(other.metadata);
            metadata = new Metadata_t<T>(tmp); // copy construct!
            // std::copy(other.metadata, other.metadata + sizeof(Metadata_t<T>), metadata);
      }
      return *this;
    };
    Proxy(Proxy<T>&& other) : addr(nullptr), metadata(nullptr) {
        LOGINFO("Move Constructing Proxy<{}> @{}", typeid(T).name(), static_cast<void*>(this));
        location = other.location;
        addr = other.addr;
        // For now we avoid std::move, but we may choose to change this behavior
        // const Metadata_t<T>& tmp = *(other.metadata);
        metadata = other.metadata;
        other.metadata = nullptr;
    };
    ~Proxy() {
        LOGINFO("Deconstructing Proxy<{}> @{} with metadata @{}", typeid(T).name(), static_cast<void*>(this), static_cast<void*>(metadata));
        if (metadata != nullptr) delete metadata;
    };


    
    /**
     * @brief Overloaded operator-> returns the address of the underlying object.
     * @return The address of the underlying object.
     */
    auto operator->() { return addr; };
    auto operator->() const { return addr; };

    /**
     * @brief The resource associated with the data represented by the proxy.
     */
    Resource location;	    ///< The device (thread/gpu) holding the data represented by the proxy.
    T* addr;		    ///< The address of the underlying object.
    Metadata_t<T>* metadata; ///< T-specific metadata that resides in same memory space as Proxy<T> 

    // Use two template parameter packs as suggested here: https://stackoverflow.com/questions/26994969/inconsistent-parameter-pack-deduction-with-variadic-templates
    template <typename RetType, typename... Args1, typename... Args2>
    RetType callSync(RetType (T::*memberFunc)(Args1...), Args2&&... args) {
        switch (location.type) {
	case Resource::SYCL:
	    if (location.is_local()) {
		return (addr->*memberFunc)(std::forward<Args2>(args)...);
	    } else {
            #ifdef USE_MPI
            RetType result;
            MPI_Send(args..., location.id, MPI_COMM_WORLD);
            MPI_Recv(&result, sizeof(RetType), MPI_BYTE, location.id, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            return result;
            #else
            ARBD::throw_not_implemented("Non-local CPU calls require MPI support");
            #endif
		//Exception( NotImplementedError, "Proxy::callSync() non-local CPU calls" );
	    }
	    break;
	case Resource::CUDA:
    #ifdef __CUDACC__
	    if (location.is_local()) {
		if (sizeof(RetType) > 0) {
		    // Note: this only support basic RetType objects
		    RetType* dest;
		    RetType obj;
		    gpuErrchk(cudaMalloc(&dest, sizeof(RetType)));
		    proxy_sync_call_kernel<T, RetType, Args2...><<<1,32>>>(dest, addr, addr->*memberFunc, args...);
		    // proxy_sync_call_kernel<><<<1,32>>>(dest, addr, addr->*memberFunc, args...);
		    gpuErrchk(cudaMemcpy(dest, &obj, sizeof(RetType), cudaMemcpyHostToDevice));
		    gpuErrchk(cudaFree(dest));
		    return obj;
		} else {
		    ARBD::throw_not_implemented("Proxy::callSync() local GPU calls");
		}
	    } else {
            size_t target_device = location.id;
            int current_device;
            gpuErrchk(cudaGetDevice(&current_device));
            gpuErrchk(cudaSetDevice(target_device));
            
            RetType* dest;
            RetType result;
            gpuErrchk(cudaMalloc(&dest, sizeof(RetType)));
            proxy_sync_call_kernel<T, RetType, Args2...><<<1,32>>>(dest, addr, memberFunc, args...);
            gpuErrchk(cudaMemcpy(&result, dest, sizeof(RetType), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaFree(dest));
            
            gpuErrchk(cudaSetDevice(current_device));
            return result;
		//Exception( NotImplementedError, "Proxy::callSync() non-local GPU calls" );
	    }
        #else
        ARBD::throw_not_implemented("Proxy::callSync() for CUDA GPU only defined for files compiled with nvvc");
        #endif		    		
	    break;
	case Resource::MPI:
	    #ifdef USE_MPI
           int rank, size;
           MPI_Comm_rank(MPI_COMM_WORLD, &rank);
           MPI_Comm_size(MPI_COMM_WORLD, &size);

           if (rank == location.id) {
               // Target rank executes the function
               RetType result = (addr->*memberFunc)(args...);
               // Broadcast result to all ranks
               MPI_Bcast(&result, sizeof(RetType), MPI_BYTE, location.id, MPI_COMM_WORLD);
               return result;
           } else {
               // Other ranks receive the result
               RetType result;
               MPI_Bcast(&result, sizeof(RetType), MPI_BYTE, location.id, MPI_COMM_WORLD); 
               return result;
           }
       #else
           ARBD::throw_not_implemented("MPI calls require USE_MPI flag");
       #endif
	    break;
	default:
	    ARBD::throw_value_error("Proxy::callSync(): Unknown resource type");
        }
	return RetType{};
    }

    // TODO generalize to handle void RetType 
    template <typename RetType, typename... Args1, typename... Args2>
    std::future<RetType> callAsync(RetType (T::*memberFunc)(Args1...), Args2&&... args) {
        switch (location.type) {
	case Resource::SYCL:
	    if (location.is_local()) {
		return (addr->*memberFunc)(std::forward<Args2>(args)...);
	    } else {
		ARBD::throw_not_implemented("Proxy::callAsync() non-local CPU calls");
	    }
	    break;
	case Resource::CUDA:
	#ifdef __CUDACC__
	    if (location.is_local()) {
            return std::async(std::launch::async, [this, memberFunc, args...] {
                RetType* dest;
                RetType result;
                gpuErrchk(cudaMalloc(&dest, sizeof(RetType)));
                proxy_sync_call_kernel<T, RetType, Args2...><<<1,32>>>(dest, addr, memberFunc, args...);
                gpuErrchk(cudaMemcpy(&result, dest, sizeof(RetType), cudaMemcpyDeviceToHost));
                gpuErrchk(cudaFree(dest));
                return result;
		//Exception( NotImplementedError, "Proxy::callAsync() local GPU calls" );
	    }); 
	    } else {
           return std::async(std::launch::async, [this, memberFunc, args...] {
               int current_device;
               gpuErrchk(cudaGetDevice(&current_device));
               gpuErrchk(cudaSetDevice(location.id));
               
               RetType* dest;
               RetType result;
               gpuErrchk(cudaMalloc(&dest, sizeof(RetType)));
               proxy_sync_call_kernel<T, RetType, Args2...><<<1,32>>>(dest, addr, memberFunc, args...);
               gpuErrchk(cudaMemcpy(&result, dest, sizeof(RetType), cudaMemcpyDeviceToHost));
               gpuErrchk(cudaFree(dest));
               
               gpuErrchk(cudaSetDevice(current_device));
               return result;
           });
       }
       #else
	    ARBD::throw_not_implemented("Async for GPU only defined for files compiled with nvvc");
       #endif
	    break;
	case Resource::MPI:
	    #ifdef USE_MPI
           return std::async(std::launch::async, [this, memberFunc, args...] {
               int rank, size;
               MPI_Comm_rank(MPI_COMM_WORLD, &rank);
               MPI_Comm_size(MPI_COMM_WORLD, &size);

               if (rank == location.id) {
                   RetType result = (addr->*memberFunc)(args...);
                   MPI_Bcast(&result, 1, MPI_BYTE, location.id, MPI_COMM_WORLD);
                   return result;
               } else {
                   RetType result;
                   MPI_Bcast(&result, 1, MPI_BYTE, location.id, MPI_COMM_WORLD);
                   return result;
               }
           });
       #else
        ARBD::throw_not_implemented("Async MPI calls require USE_MPI flag");
       #endif
       break;
	default:
	    ARBD::throw_value_error("Proxy::callAsync(): Unknown resource type");
        }
	return std::async(std::launch::async, [] { return RetType{}; });
    }
};

// Specialization for bool/int/float types that do not have member functions
template<typename T>
struct Proxy<T, typename std::enable_if_t<std::is_arithmetic<T>::value>> {
    /**
     * @brief Default constructor initializes the location to a default CPU resource and the address to nullptr.
     */
    Proxy() : location{Resource{Resource::SYCL,0}}, addr{nullptr} {};
    Proxy(const Resource& r, T* obj) : location{r}, addr{obj} {};

    /**
     * @brief Overloaded operator-> returns the address of the underlying object.
     * @return The address of the underlying object.
     */
    auto operator->() { return addr; }
    auto operator->() const { return addr; }

    /**
     * @brief The resource associated with the data represented by the proxy.
     */
    Resource location;	    ///< The device (thread/gpu) holding the data represented by the proxy.
    T* addr;		    ///< The address of the underlying object.
};


// using Proxy<int> = SimpleProxy<int>;



// class Proxy<int> {

//     // Define Metadata types using SFINAE
//     // template<typename=void> struct Metadata_t { };
//     // template<> struct Metadata_t<void_t<T::Metadata>> : T::Metadata { };
//     // template<typename=void> struct Metadata_t { };
//     // template<> struct Metadata_t<void_t<T::Metadata>> : T::Metadata { };
//     // using Metadata_t = Metadata_t<T>;
    
// public:

//     /**
//      * @brief Default constructor initializes the location to a default CPU resource and the address to nullptr.
//      */
//     Proxy<int>() : location(Resource{Resource::SYCL,0}), addr(nullptr) {};
//     Proxy<int>(const Resource& r, int* obj) : location(r), addr(obj) {};

//     /**
//      * @brief Overloaded operator-> returns the address of the underlying object.
//      * @return The address of the underlying object.
//      */
//     auto operator->() { return addr; }
//     auto operator->() const { return addr; }

//     /**
//      * @brief The resource associated with the data represented by the proxy.
//      */
//     Resource location;	    ///< The device (thread/gpu) holding the data represented by the proxy.
//     int* addr;		    ///< The address of the underlying object.
// };


// // Partial specialization
// template<typename T>
// using Proxy<T> = Proxy<T, std::void_t<typename T::Metadata>>
// // template<typename T>
// // class Proxy<T, typename T::Metadata> { };


/**
 * @brief Template function to send data ignoring children to a specified location.
 * @tparam T The type of the data to be sent.
 * @param location The destination resource for the data.
 * @param obj The data to be sent.
 * @param dest Optional parameter to provide a pre-allocated destination. If not provided, memory is allocated.
 * @return A Proxy representing the data at the destination location.
 */
template <typename T>
HOST inline Proxy<T> _send_ignoring_children(const Resource& location, T& obj, T* dest = nullptr) {
    LOGTRACE("   _send_ignoring_children...");
    switch (location.type) {
    case Resource::CUDA:
	LOGINFO("   GPU...");
#ifdef USE_CUDA
	if (location.is_local()) {
	    if (dest == nullptr) { // allocate if needed
		LOGTRACE("   cudaMalloc for array");
		gpuErrchk(cudaMalloc(&dest, sizeof(T)));
	    }
	    gpuErrchk(cudaMemcpy(dest, &obj, sizeof(T), cudaMemcpyHostToDevice));
	} else {
 	    ARBD::throw_not_implemented("`_send_ignoring_children(...)` on non-local GPU" );
	}
#else
	ARBD::throw_not_implemented("USE_CUDA is not enabled" );
#endif
	break;
    case Resource::SYCL:
	LOGINFO("Using SYCL...");
	if (location.is_local()) {
	    LOGINFO("   local SYCL...");
	    // if (dest == nullptr) { // allocate if needed
	    // 	LOGINFO("   allocate memory...");
	    // 	LOGTRACE("   Allocate CPU memory for {}", type_name<T>().c_str());
	    // 	dest = new T;
	    // }
	    // LOGINFO("   memcpying...");
	    // memcpy(dest, &obj, sizeof(T));
	    // dest = *obj;
	} else {
	    LOGINFO("   nonlocal...");
	    // Exception( NotImplementedError, "`_send_ignoring_children(...)` on non-local CPU" );
	}
	break;
    default:
	// error
	ARBD::throw_value_error("`_send_ignoring_children(...)` applied with unkown resource type" );
    }

    LOGINFO("   creating Proxy...");
    // Proxy<T>* ret = new Proxy<T>(location, dest); // Proxies should be explicitly removed  
    // LOGINFO("   ...done @{}", fmt::ptr(ret));
    // Proxy<T>&& ret =
    return Proxy<T>(location, obj, dest); // Proxies should be explicitly removed
    
	//LOGINFO("   ...done @{}", fmt::ptr(&ret));
    // return ret;
    // LOGINFO("   ...done @{}", fmt::ptr(ret));
    // return *ret;
}

/**
 * @brief Template function to send simple objects to a specified location without considering child objects.
 *        This version will be selected upon send(location, obj) if obj.send_children does not exist (C++20 concepts)
 * @tparam T The type of the data to be sent.
 * @param location The destination resource for the data.
 * @param obj The data to be sent.
 * @param dest Optional parameter to provide a pre-allocated destination. If not provided, memory is allocated.
 * @return A Proxy representing the data at the destination location.
 */
template <typename T>
requires (!has_send_children<T>)
HOST inline Proxy<T>& send(const Resource& location, T& obj, T* dest = nullptr) {
    LOGINFO("...Sending object {} @{} to device at {}", typeid(T).name(), static_cast<void*>(&obj), static_cast<void*>(dest));
    // Simple objects can simply be copied without worrying about contained objects and arrays
    Proxy<T>&& ret = _send_ignoring_children(location, obj, dest);
    LOGTRACE("...done sending");
    // printf("...done\n");        
    return ret;
}

/**
 * @brief Template function to send more complex objects to a specified location.
 *        This version will be selected upon send(location, obj) if obj.send_children exists (C++20 concepts)
 * @tparam T The type of the data to be sent.
 * @param location The destination resource for the data.
 * @param obj The data to be sent.
 * @param dest Optional parameter to provide a pre-allocated destination. If not provided, memory is allocated on the GPU.
 * @return A Proxy representing the data at the destination location.
 */
template <typename T>
requires has_send_children<T>
HOST inline Proxy<T> send(const Resource& location, T& obj, T* dest = nullptr) {
    // static_assert(!has_no_send<T>());
    LOGINFO("Sending complex object {} @{} to device at {}", typeid(T).name(), static_cast<void*>(&obj), static_cast<void*>(dest));
    auto dummy = obj.send_children(location); // function is expected to return an object of type obj with all pointers appropriately assigned to valid pointers on location
    Proxy<T> ret = _send_ignoring_children<T>(location, dummy, dest);
    LOGTRACE("... clearing dummy complex object");
    dummy.clear();
    LOGTRACE("... done sending");
    return ret;
}

// Utility function for constructing objects in remote memory address
// spaces, obviating the need to construct simple objects locally
// before copying. Returns a Proxy object, but in cases where the
// remote resource location is non-CPU or non-local, metadata for
// Proxy will be blank.
template<typename T, typename... Args>
Proxy<T> construct_remote(Resource location, Args&&...args) {
    //static_assert(!has_no_send<T>());
    switch (location.type) {
    case Resource::SYCL:
	if (location.is_local()) {
	    T* ptr = new T{std::forward<Args>(args)...};
	    return Proxy<T>(location, *ptr);
	} else {
	    ARBD::throw_not_implemented("construct_remote() non-local CPU calls" );
	}
	break;
    case Resource::CUDA:
    #ifdef __CUDACC__
	if (location.is_local()) {
	    T* devptr;
	    LOGWARN("construct_remote: TODO: switch to device associated with location");
	    gpuErrchk(cudaMalloc(&devptr, sizeof(T)));
	    constructor_kernel<<<1,32>>>(devptr, std::forward<Args>(args)...);
	    gpuErrchk(cudaDeviceSynchronize());
	    LOGWARN("construct_remote: proxy.metadata not set");
	    return Proxy<T>(location);
	    // Exception( NotImplementedError, "cunstruct_remote() local GPU call" );
	    // Note: this only support basic RetType objects
	    // T* dest;
	    // T obj;
	    // gpuErrchk(cudaMalloc(&dest, sizeof(RetType)));
	    // proxy_sync_call_kernel<T, RetType, Args2...><<<1,32>>>(dest, addr, addr->*memberFunc, args...);
	    // 	gpuErrchk(cudaMemcpy(dest, &obj, sizeof(RetType), cudaMemcpyHostToDevice));
	    // 	gpuErrchk(cudaFree(dest));
	} else {
	    ARBD::throw_not_implemented("cunstruct_remote() non-local GPU call" );
	}
    #else
	    ARBD::throw_not_implemented("construct_remote() for GPU only defined for files compiled with nvvc" );
    #endif	    		
	break;
    case Resource::MPI:
	ARBD::throw_not_implemented("construct_remote() for MPI" );
	break;
    default:
	ARBD::throw_value_error("construct_remote(): unknown resource type" );
    }
    return Proxy<T>{};
}

};