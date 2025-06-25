#pragma once

#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Resource.h"
#include <cstring>
#include <future>
#include <typeinfo>
#include <span>


namespace ARBD {

/**
 * @brief Concept to check if a type has a 'send_children' method that accepts a
 * Resource parameter.
 */
template <typename T>
concept has_send_children = requires(T t, Resource r) { t.send_children(r); };

/**
 * @brief Concept to check if a type has a 'no_send' member.
 */
template <typename T>
concept has_no_send = requires(T t) { t.no_send; };

// Helper alias for C++14 compatibility (still needed for Metadata_t)
template <typename...> using void_t = void;

// Used by Proxy class
template <typename T, typename = void> struct Metadata_t {
  Metadata_t(const T &obj){};
  Metadata_t(const Metadata_t<T> &other){};
};

template <typename T>
struct Metadata_t<T, void_t<typename T::Metadata>> : T::Metadata {
  Metadata_t(const T &obj) : T::Metadata(obj){};
  Metadata_t(const Metadata_t<T> &other) : T::Metadata(other){};
};

// Forward declarations and fallback implementations
namespace ProxyImpl {
// CUDA-specific implementations (defined in Proxy.cu when USE_CUDA is enabled)
#ifdef USE_CUDA
void *cuda_call_sync(void *addr, void *func_ptr, void *args, size_t args_size,
                     const Resource &location, size_t result_size);
std::future<void *> cuda_call_async(void *addr, void *func_ptr, void *args,
                                    size_t args_size, const Resource &location,
                                    size_t result_size);
#else
inline void *cuda_call_sync(void *addr, void *func_ptr, void *args, size_t args_size,
                            const Resource &location, size_t result_size) {
    ARBD::throw_not_implemented("CUDA support not enabled");
}
inline std::future<void *> cuda_call_async(void *addr, void *func_ptr, void *args,
                                           size_t args_size, const Resource &location,
                                           size_t result_size) {
    ARBD::throw_not_implemented("CUDA support not enabled");
}
#endif

// SYCL-specific implementations (defined in Proxy_sycl.cpp when USE_SYCL is enabled)
#ifdef USE_SYCL
void *sycl_call_sync(void *addr, void *func_ptr, void *args, size_t args_size,
                     const Resource &location, size_t result_size);
std::future<void *> sycl_call_async(void *addr, void *func_ptr, void *args,
                                    size_t args_size, const Resource &location,
                                    size_t result_size);
#else
inline void *sycl_call_sync(void *addr, void *func_ptr, void *args, size_t args_size,
                            const Resource &location, size_t result_size) {
    ARBD::throw_not_implemented("SYCL support not enabled");
}
inline std::future<void *> sycl_call_async(void *addr, void *func_ptr, void *args,
                                           size_t args_size, const Resource &location,
                                           size_t result_size) {
    ARBD::throw_not_implemented("SYCL support not enabled");
}
#endif

// METAL-specific implementations (defined in Proxy.mm when USE_METAL is enabled)
#if defined(USE_METAL) && defined(__OBJC__)
void *metal_call_sync(void *addr, void *func_ptr, void *args, size_t args_size,
                      const Resource &location, size_t result_size);
std::future<void *> metal_call_async(void *addr, void *func_ptr, void *args,
                                     size_t args_size, const Resource &location,
                                     size_t result_size);
#else
inline void *metal_call_sync(void *addr, void *func_ptr, void *args, size_t args_size,
                             const Resource &location, size_t result_size) {
    ARBD::throw_not_implemented("METAL support not enabled");
}
inline std::future<void *> metal_call_async(void *addr, void *func_ptr, void *args,
                                            size_t args_size, const Resource &location,
                                            size_t result_size) {
    ARBD::throw_not_implemented("METAL support not enabled");
}
#endif

// General send implementations
template <typename T>
void *send_ignoring_children(const Resource &location, T &obj, T *dest);

template <typename T>
void *construct_remote(const Resource &location, void *args, size_t args_size);
} // namespace ProxyImpl

template <typename T, typename Enable = void> struct Proxy {
  static_assert(!std::is_same<T, Proxy>::value,
                "Cannot make a Proxy of a Proxy object");

  // Constructors
  Proxy()
      : location(Resource{Resource::SYCL, 0}), addr(nullptr),
        metadata(nullptr) {
    LOGINFO("Constructing Proxy<{}> @{}", typeid(T).name(),
            static_cast<void *>(this));
  };

  Proxy(const Resource &r) : location(r), addr(nullptr), metadata(nullptr) {
    LOGINFO("Constructing Proxy<{}> @{}", typeid(T).name(),
            static_cast<void *>(this));
  };

  Proxy(const Resource &r, T &obj, T *dest = nullptr)
      : location(r), addr(dest == nullptr ? &obj : dest) {
    if (dest == nullptr)
      metadata = nullptr;
    else
      metadata = new Metadata_t<T>(obj);
    LOGINFO("Constructing Proxy<{}> @{} wrapping @{} with metadata @{}",
            typeid(T).name(), static_cast<void *>(this),
            static_cast<void *>(&obj), static_cast<void *>(metadata));
  };

  // Copy constructor
  Proxy(const Proxy<T> &other)
      : location(other.location), addr(other.addr), metadata(nullptr) {
    LOGINFO("Copy Constructing Proxy<{}> @{}", typeid(T).name(),
            static_cast<void *>(this));
    if (other.metadata != nullptr) {
      const Metadata_t<T> &tmp = *(other.metadata);
      metadata = new Metadata_t<T>(tmp);
    }
  };

  // Copy assignment
  Proxy<T> &operator=(const Proxy<T> &other) {
    if (this != &other) {
      if (metadata != nullptr)
        delete metadata;
      location = other.location;
      addr = other.addr;
      if (other.metadata != nullptr) {
        const Metadata_t<T> &tmp = *(other.metadata);
        metadata = new Metadata_t<T>(tmp);
      } else {
        metadata = nullptr;
      }
    }
    return *this;
  };

  // Move constructor
  Proxy(Proxy<T> &&other) : addr(nullptr), metadata(nullptr) {
    LOGINFO("Move Constructing Proxy<{}> @{}", typeid(T).name(),
            static_cast<void *>(this));
    location = other.location;
    addr = other.addr;
    metadata = other.metadata;
    other.metadata = nullptr;
  };

  // Destructor
  ~Proxy() {
    LOGINFO("Deconstructing Proxy<{}> @{} with metadata @{}", typeid(T).name(),
            static_cast<void *>(this), static_cast<void *>(metadata));
    if (metadata != nullptr)
      delete metadata;
  };

  /**
   * @brief Overloaded operator-> returns the address of the underlying object.
   */
  auto operator->() { return addr; };
  auto operator->() const { return addr; };

  /**
   * @brief Synchronous method call implementation
   */
  template <typename RetType, typename... Args1, typename... Args2>
  RetType callSync(RetType (T::*memberFunc)(Args1...), Args2 &&...args) {
    // Pack arguments for implementation functions
    struct ArgPack {
      std::tuple<Args2...> arguments;
      RetType (T::*func)(Args1...);
    };

    ArgPack pack{std::make_tuple(std::forward<Args2>(args)...), memberFunc};

    void *result_ptr = nullptr;

    switch (location.type) {
    case Resource::CUDA:
      result_ptr = ProxyImpl::cuda_call_sync(
          addr, reinterpret_cast<void *>(&memberFunc), &pack, sizeof(pack),
          location, sizeof(RetType));
      break;
    case Resource::SYCL:
      result_ptr = ProxyImpl::sycl_call_sync(
          addr, reinterpret_cast<void *>(&memberFunc), &pack, sizeof(pack),
          location, sizeof(RetType));
      break;
    case Resource::METAL:
      result_ptr = ProxyImpl::metal_call_sync(
          addr, reinterpret_cast<void *>(&memberFunc), &pack, sizeof(pack),
          location, sizeof(RetType));
      break;
    default:
      ARBD::throw_value_error("Proxy::callSync(): Unknown resource type");
    }

    if constexpr (sizeof(RetType) > 0) {
      RetType result = *static_cast<RetType *>(result_ptr);
      delete static_cast<RetType *>(result_ptr);
      return result;
    } else {
      return RetType{};
    }
  }

  /**
   * @brief Asynchronous method call implementation
   */
  template <typename RetType, typename... Args1, typename... Args2>
  std::future<RetType> callAsync(RetType (T::*memberFunc)(Args1...),
                                 Args2 &&...args) {
    // Pack arguments for implementation functions
    struct ArgPack {
      std::tuple<Args2...> arguments;
      RetType (T::*func)(Args1...);
    };

    ArgPack pack{std::make_tuple(std::forward<Args2>(args)...), memberFunc};

    switch (location.type) {
    case Resource::CUDA:
      return std::async(std::launch::async, [this, pack, memberFunc]() {
        void *result_ptr = ProxyImpl::cuda_call_sync(
            addr, reinterpret_cast<void *>(&memberFunc),
            const_cast<ArgPack *>(&pack), sizeof(pack), location,
            sizeof(RetType));

        if constexpr (sizeof(RetType) > 0) {
          RetType result = *static_cast<RetType *>(result_ptr);
          delete static_cast<RetType *>(result_ptr);
          return result;
        } else {
          return RetType{};
        }
      });

    case Resource::SYCL:
      return std::async(std::launch::async, [this, pack, memberFunc]() {
        void *result_ptr = ProxyImpl::sycl_call_sync(
            addr, reinterpret_cast<void *>(&memberFunc),
            const_cast<ArgPack *>(&pack), sizeof(pack), location,
            sizeof(RetType));

        if constexpr (sizeof(RetType) > 0) {
          RetType result = *static_cast<RetType *>(result_ptr);
          delete static_cast<RetType *>(result_ptr);
          return result;
        } else {
          return RetType{};
        }
      });

    case Resource::METAL:
      return std::async(std::launch::async, [this, pack, memberFunc]() {
        void *result_ptr = ProxyImpl::metal_call_sync(
            addr, reinterpret_cast<void *>(&memberFunc),
            const_cast<ArgPack *>(&pack), sizeof(pack), location,
            sizeof(RetType));

        if constexpr (sizeof(RetType) > 0) {
          RetType result = *static_cast<RetType *>(result_ptr);
          delete static_cast<RetType *>(result_ptr);
          return result;
        } else {
          return RetType{};
        }
      });

    default:
      ARBD::throw_value_error("Proxy::callAsync(): Unknown resource type");
    }

    return std::async(std::launch::async, [] { return RetType{}; });
  }

  Resource location;
  T *addr;
  Metadata_t<T> *metadata;
};

// Specialization for arithmetic types
template <typename T>
struct Proxy<T, typename std::enable_if_t<std::is_arithmetic<T>::value>> {
  Proxy() : location{Resource{Resource::SYCL, 0}}, addr{nullptr} {};
  Proxy(const Resource &r, T *obj) : location{r}, addr{obj} {};

  auto operator->() { return addr; }
  auto operator->() const { return addr; }

  Resource location;
  T *addr;
};

// Template function implementations

template <typename T>
HOST inline Proxy<T> _send_ignoring_children(const Resource &location, T &obj,
                                             T *dest = nullptr) {
  LOGTRACE("   _send_ignoring_children...");

  void *result_ptr = ProxyImpl::send_ignoring_children(location, obj, dest);
  T *typed_dest = static_cast<T *>(result_ptr);

  LOGINFO("   creating Proxy...");
  return Proxy<T>(location, obj, typed_dest);
}

template <typename T>
  requires(!has_send_children<T>)
HOST inline Proxy<T> send(const Resource &location, T &obj, T *dest = nullptr) {
  LOGINFO("...Sending object {} @{} to device at {}", typeid(T).name(),
          static_cast<void *>(&obj), static_cast<void *>(dest));

  Proxy<T> ret = _send_ignoring_children(location, obj, dest);
  LOGTRACE("...done sending");
  return ret;
}

template <typename T>
  requires has_send_children<T>
HOST inline Proxy<T> send(const Resource &location, T &obj, T *dest = nullptr) {
  LOGINFO("Sending complex object {} @{} to device at {}", typeid(T).name(),
          static_cast<void *>(&obj), static_cast<void *>(dest));

  auto dummy = obj.send_children(location);
  Proxy<T> ret = _send_ignoring_children<T>(location, dummy, dest);
  LOGTRACE("... clearing dummy complex object");
  dummy.clear();
  LOGTRACE("... done sending");
  return ret;
}

template <typename T, typename... Args>
Proxy<T> construct_remote(Resource location, Args &&...args) {
  struct ArgPack {
    std::tuple<Args...> arguments;
  };

  ArgPack pack{std::make_tuple(std::forward<Args>(args)...)};

  void *result_ptr =
      ProxyImpl::construct_remote<T>(location, &pack, sizeof(pack));
  T *typed_ptr = static_cast<T *>(result_ptr);

  // Create a temporary object for metadata (this is a limitation of the current
  // design)
  T temp_obj{std::forward<Args>(args)...};
  return Proxy<T>(location, temp_obj, typed_ptr);
}

} // namespace ARBD
