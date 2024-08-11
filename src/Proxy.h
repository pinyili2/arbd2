#pragma once

#include <future>
#include <iostream>
#include "Resource.h"

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

// Used by Proxy class 
template <typename T, typename = void>
struct Metadata_t { }; 
template <typename T>
struct Metadata_t<T, void_t<typename T::Metadata>> : T::Metadata { };
// struct Metadata_t<T, decltype(std::declval<typename T::Metadata>(), void())> : T::Metadata { }; 


// template<typename T, typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
// struct Proxy {
//     /**
//      * @brief Default constructor initializes the location to a default CPU resource and the address to nullptr.
//      */
//     Proxy() : location{Resource{Resource::CPU,0}}, addr{nullptr} {};
//     Proxy(const Resource& r, T* obj) : location{r}, addr{obj} {};

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
//     T* addr;		    ///< The address of the underlying object.
// };

/**
 * @brief Template class representing a proxy for the underlying data.
 * @tparam T The type of the underlying data.
 */


    // Q: why a pointer?
    // Q: rename to Metadata? Has
    // consequences for Proxy<Proxy<T>>
    // that limits specialization but uses
    // T::Metadata automatically
    //
    // A: depends on how Proxy<Proxy<T>>
    // objects are used
    
    // void move(const Resource& newloc) {
    // 	LOGTRACE("Moving object from {} to {}", location, newloc);
    // 	Proxy<T> new_proxy;
	
    //     switch (location.type) {
    // 	case Resource::CPU:
    // 	    if (location.is_local()) {
    // 		new_proxy = send(location, &addr);
    // 	    } else {
    // 		Exception( NotImplementedError, "Proxy::move() non-local CPU calls" );
    // 	    }
    // 	    break;
    // 	case Resource::GPU:
    // 	    if (location.is_local()) {
    // 		Exception( NotImplementedError, "Proxy::move() local GPU calls" );
    // 	    } else {
    // 		Exception( NotImplementedError, "Proxy::move() non-local GPU calls" );
    // 	    }
    // 	    break;
    // 	case Resource::MPI:
    // 	    Exception( NotImplementedError, "MPI move (deprecate?)" );
    // 	    break;
    // 	default:
    // 	    Exception( ValueError, "Proxy::move(): Unknown resource type" );
    //     }
	
    // 	// auto Proxy<T>{ location , newaddr }
    // 	// auto tmpmeta = send(newloc, metadata);

    // 	location = newloc;
    // 	addr = new_proxy.addr;
    // 	metadata = new_proxy.metadata;
    // }

// C++17 way: template<typename T, typename Metadata = std::void_t<typename T::Metadata>>
// C++14 way: template<typename T, typename Metadata = typename std::conditional<has_metadata<T>::value, typename T::Metadata, void>::type>
// Neither needed!
template<typename T, typename Enable = void>
struct Proxy {
    /**
     * @brief Default constructor initializes the location to a default CPU resource and the address to nullptr.
     */
    Proxy() : location(Resource{Resource::CPU,0}), addr(nullptr) {};
    Proxy(const Resource& r, T* obj) : location(r), addr(obj) {};

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
	case Resource::CPU:
	    if (location.is_local()) {
		return (addr->*memberFunc)(std::forward<Args2>(args)...);
	    } else {
		Exception( NotImplementedError, "Proxy::callSync() non-local CPU calls" );
	    }
	    break;
	case Resource::GPU:
	    if (location.is_local()) {
		Exception( NotImplementedError, "Proxy::callSync() local GPU calls" );
	    } else {
		Exception( NotImplementedError, "Proxy::callSync() non-local GPU calls" );
	    }
	    break;
	case Resource::MPI:
	    Exception( NotImplementedError, "MPI sync calls (deprecate?)" );
	    break;
	default:
	    Exception( ValueError, "Proxy::callSync(): Unknown resource type" );
        }
	return RetType{};
    }

    // TODO generalize to handle void RetType 
    template <typename RetType, typename... Args1, typename... Args2>
    std::future<RetType> callAsync(RetType (T::*memberFunc)(Args1...), Args2&&... args) {
        switch (location.type) {
	case Resource::CPU:
	    if (location.is_local()) {
		return (addr->*memberFunc)(std::forward<Args2>(args)...);
	    } else {
		Exception( NotImplementedError, "Proxy::callAsync() non-local CPU calls" );
	    }
	    break;
	case Resource::GPU:
	    if (location.is_local()) {
		Exception( NotImplementedError, "Proxy::callAsync() local GPU calls" );
	    } else {
		Exception( NotImplementedError, "Proxy::callAsync() non-local GPU calls" );
	    }
	    break;
	case Resource::MPI:
	    Exception( NotImplementedError, "MPI async calls (deprecate?)" );
	    break;
	default:
	    Exception( ValueError, "Proxy::callAsync(): Unknown resource type" );
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
    Proxy() : location{Resource{Resource::CPU,0}}, addr{nullptr} {};
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
//     Proxy<int>() : location(Resource{Resource::CPU,0}), addr(nullptr) {};
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
    switch (location.type) {
    case Resource::GPU:
	if (location.is_local()) {
	    if (dest == nullptr) { // allocate if needed
		LOGTRACE("   cudaMalloc for array");
		gpuErrchk(cudaMalloc(&dest, sizeof(T)));
	    }
	    gpuErrchk(cudaMemcpy(dest, &obj, sizeof(T), cudaMemcpyHostToDevice));
	} else {
 	    Exception( NotImplementedError, "`_send_ignoring_children(...)` on non-local GPU" );
	}
	break;
    case Resource::CPU:
	if (location.is_local()) {
	    if (dest == nullptr) { // allocate if needed
		LOGTRACE("   Allocate CPU memory for {}", decltype(T));
		dest = new T;
	    }
	    memcpy(dest, &obj, sizeof(T));
	} else {
	    Exception( NotImplementedError, "`_send_ignoring_children(...)` on non-local CPU" );
	}
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
    LOGTRACE("...Sending object {} @{} to device at {}", type_name<T>().c_str(), fmt::ptr(&obj), fmt::ptr(dest));
    // Simple objects can simply be copied without worrying about contained objects and arrays
    auto ret = _send_ignoring_children(location, obj, dest);
    LOGTRACE("...done sending");
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
    LOGTRACE("Sending complex object {} @{} to device at {}", type_name<T>().c_str(), fmt::ptr(&obj), fmt::ptr(dest));
    auto dummy = obj.send_children(location); // function is expected to return an object of type obj with all pointers appropriately assigned to valid pointers on location
    Proxy<T> ret = _send_ignoring_children<T>(location, dummy, dest);
    LOGTRACE("... clearing dummy complex object");
    dummy.clear();
    LOGTRACE("... done sending");
    return ret;
}


