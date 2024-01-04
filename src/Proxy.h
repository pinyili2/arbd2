#pragma once

#include <iostream>
#include "ARBDException.h"

/**
 * @brief Represents a resource that can store data and perform computations.
 */
struct Resource {
    /**
     * @brief Enum to specify the type of the resource (e.g., CPU or GPU).
     */
    enum ResourceType {CPU, GPU};
    ResourceType type; ///< Type of the resource.
    size_t id; ///< ID or any other identifier associated with the resource.
    // HOST DEVICE static bool is_local() { // check if thread/gpu idx matches some global idx };
};

// START traits
// https://stackoverflow.com/questions/55191505/c-compile-time-check-if-method-exists-in-template-type
/**
 * @brief Template trait to check if a method 'send_children' exists in a type.
 */
#include <type_traits>
template <typename T, typename = void>
struct has_send_children : std::false_type {};

template <typename T>
struct has_send_children<T, decltype(std::declval<T>().send_children(Resource{Resource::CPU,0}), void())> : std::true_type {};
// END traits


/**
 * @brief Template class representing a proxy for the underlying data.
 * @tparam T The type of the underlying data.
 */
template<typename T>
class Proxy {
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
};

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
    printf("Sending object %s @%x to device at %x\n", type_name<T>().c_str(), &obj, dest);

    // Simple objects can simply be copied without worrying about contained objects and arrays
    auto ret = _send_ignoring_children<T>(location, obj, dest);
    printf("...done\n");        
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
    printf("Sending object %s @%x to device at %x\n", type_name<T>().c_str(), &obj, dest);
    auto dummy = obj.send_children(location); // function is expected to return an object of type obj with all pointers appropriately assigned to valid pointers on location
    Proxy<T> ret = _send_ignoring_children(location, dummy, dest);
    printf("clearing...\n");
    dummy.clear();
    printf("...done\n");    
    return ret;
}
