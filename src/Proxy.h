#pragma once

#include <future>
#include <iostream>
#include "ARBDException.h"

/**
 * @brief Represents a resource that can store data and perform computations.
 */
struct Resource {
    /**
     * @brief Enum to specify the type of the resource (e.g., CPU or GPU).
     */
    enum ResourceType {CPU, MPI, GPU};
    ResourceType type; ///< Type of the resource.
    size_t id; ///< ID or any other identifier associated with the resource.
    // HOST DEVICE static bool is_local() { // check if thread/gpu idx matches some global idx };
};

// START traits
// These ugly bits of code help implement SFINAE in C++14 and should likely be removed if a newer standard is adopted 
// https://stackoverflow.com/questions/55191505/c-compile-time-check-if-method-exists-in-template-type
/**
 * @brief Template trait to check if a method 'send_children' exists in a type.
 */
#include <type_traits>
template <typename T, typename = void>
struct has_send_children : std::false_type {};
template <typename T>
struct has_send_children<T, decltype(std::declval<T>().send_children(Resource{Resource::CPU,0}), void())> : std::true_type {};

// template <typename T, typename = void>
// struct has_metadata : std::false_type {};
// template <typename T>
// struct has_metadata<T, decltype(std::declval<T>()::Metadata, void())> : std::true_type {};

template <typename...>
using void_t = void;
// struct Metadata_t<T, decltype(std::declval<typename T::Metadata>(), void())> : T::Metadata { }; 
// END traits
template <typename T, typename = void>
struct Metadata_t { }; 
template <typename T>
struct Metadata_t<T, void_t<typename T::Metadata>> : T::Metadata { };
// struct Metadata_t<T, decltype(std::declval<typename T::Metadata>(), void())> : T::Metadata { }; 


/**
 * @brief Template class representing a proxy for the underlying data.
 * @tparam T The type of the underlying data.
 */
// C++17 way: template<typename T, typename Metadata = std::void_t<typename T::Metadata>>
// C++14 way:
//template<typename T, typename Metadata = typename std::conditional<has_metadata<T>::value, typename T::Metadata, void>::type>
template<typename T>
class Proxy {

    // Define Metadata types using SFINAE
    // template<typename=void> struct Metadata_t { };
    // template<> struct Metadata_t<void_t<T::Metadata>> : T::Metadata { };
    // template<typename=void> struct Metadata_t { };
    // template<> struct Metadata_t<void_t<T::Metadata>> : T::Metadata { };
    // using Metadata_t = Metadata_t<T>;
    
public:

    /**
     * @brief Default constructor initializes the location to a default CPU resource and the address to nullptr.
     */
    Proxy<T>() : location(Resource{Resource::CPU,0}), addr(nullptr) {};
    Proxy<T>(const Resource& r, T* obj) : location(r), addr(obj) {};

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
    Metadata_t<T>* metadata;	// Q: rename to Metadata? Has
				// consequences for Proxy<Proxy<T>>
				// that limits specialization but uses
				// T::Metadata automatically
				//
				// A: depends on how Proxy<Proxy<T>>
				// objects are used
    
    template <typename RetType, typename... Args>
    RetType callSync(RetType (T::*memberFunc)(Args...), Args&&... args) {
        switch (location.type) {
            case Resource::CPU:
	    // 	return ([&](auto&&... capturedArgs) {
            //     return (addr->*memberFunc)(std::forward<decltype(capturedArgs)>(capturedArgs)...);
            // })(std::forward<Args>(args)...);
		return (addr->*memberFunc)(std::forward<Args>(args)...);
            case Resource::GPU:
                // Handle GPU-specific logic
                std::cerr << "Error: GPU not implemented in synchronous call." << std::endl;
                // You may want to throw an exception or handle this case accordingly
                return RetType{};
            case Resource::MPI:
                // Handle MPI-specific logic
                std::cerr << "Error: MPI not implemented in synchronous call." << std::endl;
                // You may want to throw an exception or handle this case accordingly
                return RetType{};
            default:
                // Handle other cases or throw an exception
                std::cerr << "Error: Unknown resource type." << std::endl;
                // You may want to throw an exception or handle this case accordingly
                return RetType{};
        }
    }

    template <typename RetType, typename... Args>
    std::future<RetType> callAsync(RetType (T::*memberFunc)(Args...), Args... args) {
        switch (location.type) {
            case Resource::CPU:
                // Handle CPU-specific asynchronous logic
                return std::async(std::launch::async, [this, memberFunc, args...] {
                    return (addr->*memberFunc)(args...);
                });
            case Resource::GPU:
                // Handle GPU-specific asynchronous logic
                std::cerr << "Error: GPU not implemented in asynchronous call." << std::endl;
                // You may want to throw an exception or handle this case accordingly
                return std::async(std::launch::async, [] { return RetType{}; });
            case Resource::MPI:
                // Handle MPI-specific asynchronous logic
                std::cerr << "Error: MPI not implemented in asynchronous call." << std::endl;
                // You may want to throw an exception or handle this case accordingly
                return std::async(std::launch::async, [] { return RetType{}; });
            default:
                // Handle other cases or throw an exception
                std::cerr << "Error: Unknown resource type." << std::endl;
                // You may want to throw an exception or handle this case accordingly
                return std::async(std::launch::async, [] { return RetType{}; });
        }
    }
};

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
    switch (location.type) {
    case Resource::GPU:
	if (dest == nullptr) { // allocate if needed
	    // printf("   cudaMalloc for array\n");
	    gpuErrchk(cudaMalloc(&dest, sizeof(T)));
	}
	gpuErrchk(cudaMemcpy(dest, &obj, sizeof(T), cudaMemcpyHostToDevice));
	break;
    case Resource::CPU:
	// not implemented
	Exception( NotImplementedError, "`_send_ignoring_children(...)` on CPU" );
	break;
    default:
	// error
	Exception( ValueError, "`_send_ignoring_children(...)` applied with unkown resource type" );
    }

    return Proxy<T>(location, dest);
}

/**
 * @brief Template function to send simple objects to a specified location without considering child objects.
 *        This version will be selected upon send(location, obj) if obj.send_children does not exist (C++14-compatible SFINAE)
 * @tparam T The type of the data to be sent.
 * @param location The destination resource for the data.
 * @param obj The data to be sent.
 * @param dest Optional parameter to provide a pre-allocated destination. If not provided, memory is allocated.
 * @return A Proxy representing the data at the destination location.
 */
template <typename T, typename Dummy = void, typename std::enable_if_t<!has_send_children<T>::value, Dummy>* = nullptr>
HOST inline Proxy<T> send(const Resource& location, T& obj, T* dest = nullptr) {
    TRACE("...Sending object {} @{} to device at {}", type_name<T>().c_str(), fmt::ptr(&obj), fmt::ptr(dest));
    // Simple objects can simply be copied without worrying about contained objects and arrays
    auto ret = _send_ignoring_children(location, obj, dest);
    TRACE("...done sending");
    // printf("...done\n");        
    return ret;
}

/**
 * @brief Template function to send more complex objects to a specified location.
 *        This version will be selected upon send(location, obj) if obj.send_children exists (C++14-compatible SFINAE)
 * @tparam T The type of the data to be sent.
 * @param location The destination resource for the data.
 * @param obj The data to be sent.
 * @param dest Optional parameter to provide a pre-allocated destination. If not provided, memory is allocated on the GPU.
 * @return A Proxy representing the data at the destination location.
 */
template <typename T, typename Dummy = void, typename std::enable_if_t<has_send_children<T>::value, Dummy>* = nullptr>
HOST inline Proxy<T> send(const Resource& location, T& obj, T* dest = nullptr) {
    TRACE("Sending complex object {} @{} to device at {}", type_name<T>().c_str(), fmt::ptr(&obj), fmt::ptr(dest));
    auto dummy = obj.send_children(location); // function is expected to return an object of type obj with all pointers appropriately assigned to valid pointers on location
    Proxy<T> ret = _send_ignoring_children<T>(location, dummy, dest);
    TRACE("... clearing dummy complex object");
    dummy.clear();
    TRACE("... done sending");
    return ret;
}
