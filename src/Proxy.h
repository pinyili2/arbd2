#pragma once

#include <iostream>
#include "ARBDException.h"

struct Resource {
    // Represents something that can store data and compute things
    enum ResourceType {CPU, GPU};
    ResourceType type;
    size_t id; // maybe something else
    // HOST DEVICE static bool is_local() { // check if thread/gpu idx matches some global idx };
};

// START traits
// https://stackoverflow.com/questions/55191505/c-compile-time-check-if-method-exists-in-template-type
#include <type_traits>
template <typename T, typename = void>
struct has_send_children : std::false_type {};

template <typename T>
struct has_send_children<T, decltype(std::declval<T>().send_children(Resource{Resource::CPU,0}), void())> : std::true_type {};
// END traits


// Template argument is concrete class with data to be proxied
//   'Proxy' is probably a bad name because this is not quite the GoF Proxy Pattern
template<typename T>
class Proxy {
public:
    Proxy<T>() : location(Resource{Resource::CPU,0}), addr(nullptr) {};
    Proxy<T>(const Resource& r, T* obj) : location(r), addr(obj) {};
    auto operator->() { return addr; }
    auto operator->() const { return addr; }
    Resource location;
    T* addr;
};


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

// This ugly template allows overloading copy_to_cuda, depending on whether T.copy_to_cuda exists using C++14-compatible SFINAE
template <typename T, typename Dummy = void, typename std::enable_if_t<!has_send_children<T>::value, Dummy>* = nullptr>
HOST inline Proxy<T> send(const Resource& location, T& obj, T* dest = nullptr) {
    printf("Sending object %s @%x to device at %x\n", type_name<T>().c_str(), &obj, dest);

    // Simple objects can simply be copied without worrying about contained objects and arrays
    auto ret = _send_ignoring_children<T>(location, obj, dest);
    printf("...done\n");        
    return ret;
}

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
