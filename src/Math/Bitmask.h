#pragma once
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <string>
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Proxy.h"
#include "Backend/Resource.h"

namespace ARBD {
class BitmaskBase {
public:
  BitmaskBase(const size_t len) : len(len) {}
  virtual ~BitmaskBase() = default;

  HOST DEVICE virtual void set_mask(size_t i, bool value) = 0;

  HOST DEVICE virtual bool get_mask(size_t i) const = 0;

  size_t get_len() const { return len; }

  virtual void print() const {
    for (size_t i = 0; i < len; ++i) {
      LOGINFO("%d", static_cast<int>(get_mask(i)));
    }
    LOGINFO("\n");
  }

protected:
  size_t len;
};

// Backend-agnostic atomic operations
namespace detail {
  template<typename T>
  HOST DEVICE inline void atomic_or(T* addr, T val) {
    // Default host implementation
    *addr |= val;
  }
  
  template<typename T>
  HOST DEVICE inline void atomic_and(T* addr, T val) {
    // Default host implementation  
    *addr &= val;
  }
}

// Don't use base because virtual member functions require device malloc
class Bitmask {
  typedef size_t idx_t;
  typedef unsigned int data_t;

public:
  Bitmask(const idx_t len) : len(len) {
    idx_t tmp = get_array_size();
    assert(tmp * data_stride >= len);
    mask = (tmp > 0) ? new data_t[tmp] : nullptr;
    for (int i = 0; i < tmp; ++i)
      mask[i] = data_t(0);
  }
  ~Bitmask() {
    if (mask != nullptr)
      delete[] mask;
  }

  HOST DEVICE idx_t get_len() const { return len; }

  HOST DEVICE void set_mask(idx_t i, bool value) {
    assert(i < len);
    idx_t ci = i / data_stride;
    data_t change_bit = (data_t(1) << (i - ci * data_stride));
    
    if (value) {
      detail::atomic_or(&mask[ci], change_bit);
    } else {
      detail::atomic_and(&mask[ci], ~change_bit);
    }
  }

  HOST DEVICE bool get_mask(const idx_t i) const {
    assert(i < len);
    const idx_t ci = i / data_stride;
    return mask[ci] & (data_t(1) << (i - ci * data_stride));
  }

  HOST DEVICE inline bool operator==(Bitmask &b) const {
    if (len != b.len)
      return false;
    for (idx_t i = 0; i < len; ++i) {
      if (get_mask(i) != b.get_mask(i))
        return false;
    }
    return true;
  }

  // Backend-agnostic memory operations (implementations in backend files)
  template<typename BackendResource>
  Proxy<Bitmask> send_to_backend(const BackendResource& resource) const;
  
  template<typename BackendResource>
  static Bitmask receive_from_backend(Proxy<Bitmask>& proxy);

  HOST auto to_string() const {
    std::string s;
    s.reserve(len);
    for (size_t i = 0; i < len; ++i) {
      s += get_mask(i) ? '1' : '0';
    }
    return s;
  }

  HOST DEVICE void print() {
    for (int i = 0; i < len; ++i) {
      printf("%d", (int)get_mask(i));
    }
    printf("\n");
  }

private:
  idx_t get_array_size() const {
    return (len == 0) ? 1 : (len - 1) / data_stride + 1;
  }
  idx_t len;
  const static idx_t data_stride = CHAR_BIT * sizeof(data_t) / sizeof(char);
  data_t *__restrict__ mask;
};
} // namespace ARBD
