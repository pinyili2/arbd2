/*********************************************************************
 * @file  Vector3.h
 *
 * @brief Declaration of templated Vector3_t class.
 *********************************************************************/
#pragma once
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Header.h"
#include <limits>
#include <type_traits>
#include <sstream>

#ifdef __CUDA_ARCH__
#include <cuda/std/limits>
#endif

namespace ARBD {

// C++20 Concepts for better template constraints
template<typename T>
concept Arithmetic = ::std::is_arithmetic_v<T>;

template<typename T>
concept FloatingPoint = ::std::is_floating_point_v<T>;

/**
 * 3D vector utility class with common operations implemented on CPU and GPU.
 *
 * Implemented with 4D data storage for better GPU alignment; extra
 * data can be stored in fourth variable this->w
 *
 * @tparam T the type of data stored in the four fields, x,y,z,w; T
 * Should usually be float or double.
 */
template<typename T>
	requires Arithmetic<T>
class alignas(4 * sizeof(T)) Vector3_t {
  public:
	HOST DEVICE constexpr Vector3_t() noexcept : x(T(0)), y(T(0)), z(T(0)), w(T(0)) {}
	HOST DEVICE constexpr Vector3_t(T s) noexcept : x(s), y(s), z(s), w(s) {}
	HOST DEVICE constexpr Vector3_t(const Vector3_t<T>& v) noexcept
		: x(v.x), y(v.y), z(v.z), w(v.w) {}
	HOST DEVICE constexpr Vector3_t(T x, T y, T z) noexcept : x(x), y(y), z(z), w(0) {}
	HOST DEVICE constexpr Vector3_t(T x, T y, T z, T w) noexcept : x(x), y(y), z(z), w(w) {}
	template<typename BackendVec>
		requires requires(BackendVec v) {
			v.x;
			v.y;
			v.z;
		}
	HOST DEVICE constexpr Vector3_t(const BackendVec& v) noexcept
		: x(static_cast<T>(v.x)), y(static_cast<T>(v.y)), z(static_cast<T>(v.z)), w(0) {}

	// C++20 concepts-based cross product
	template<typename U>
		requires Arithmetic<U>
	HOST DEVICE constexpr auto cross(const Vector3_t<U>& w) const {
		using TU = ::std::common_type_t<T, U>;
		Vector3_t<TU> v;
		v.x = y * w.z - z * w.y;
		v.y = z * w.x - x * w.z;
		v.z = x * w.y - y * w.x;
		return v;
	}

	// Assignment operators
	HOST DEVICE constexpr Vector3_t<T>& operator=(const Vector3_t<T>& v) noexcept {
		x = v.x;
		y = v.y;
		z = v.z;
		return *this;
	}

	HOST DEVICE constexpr Vector3_t<T>& operator=(Vector3_t<T>&& v) noexcept {
		x = v.x;
		y = v.y;
		z = v.z;
		return *this;
	}

	HOST DEVICE constexpr Vector3_t<T>& operator+=(const Vector3_t<T>& v) noexcept {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	HOST DEVICE constexpr Vector3_t<T>& operator-=(const Vector3_t<T>& v) noexcept {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	template<typename S>
		requires Arithmetic<S>
	HOST DEVICE constexpr Vector3_t<T>& operator*=(S s) noexcept {
		x *= s;
		y *= s;
		z *= s;
		return *this;
	}

	template<typename S>
		requires Arithmetic<S>
	HOST DEVICE constexpr Vector3_t<T>& operator/=(S s) noexcept {
		const auto inv = S(1) / s;
		x *= inv;
		y *= inv;
		z *= inv;
		return *this;
	}

	HOST DEVICE constexpr Vector3_t<T> operator-() const noexcept {
		return Vector3_t<T>(-x, -y, -z);
	}

	// Binary operators with concepts
	template<typename U>
		requires Arithmetic<U>
	HOST DEVICE constexpr auto operator+(const Vector3_t<U>& w) const noexcept {
		using TU = ::std::common_type_t<T, U>;
		return Vector3_t<TU>(x + w.x, y + w.y, z + w.z);
	}

	template<typename U>
		requires Arithmetic<U>
	HOST DEVICE constexpr auto operator-(const Vector3_t<U>& w) const noexcept {
		using TU = ::std::common_type_t<T, U>;
		return Vector3_t<TU>(x - w.x, y - w.y, z - w.z);
	}

	template<typename U>
		requires Arithmetic<U>
	HOST DEVICE constexpr auto operator*(U s) const noexcept {
		using TU = ::std::common_type_t<T, U>;
		return Vector3_t<TU>(s * x, s * y, s * z);
	}

	template<typename U>
		requires Arithmetic<U>
	HOST DEVICE constexpr auto operator/(U s) const noexcept {
		const auto inv = U(1) / s;
		return (*this) * inv;
	}

	template<typename U>
		requires Arithmetic<U>
	HOST DEVICE constexpr auto dot(const Vector3_t<U>& w) const noexcept {
		return x * w.x + y * w.y + z * w.z;
	}

	// Length operations
	HOST DEVICE constexpr auto length2() const noexcept {
		return x * x + y * y + z * z;
	}

	HOST DEVICE auto length() const noexcept
		requires FloatingPoint<T>
	{
		return sqrtf(length2());
	}

	HOST DEVICE auto rLength() const noexcept
		requires FloatingPoint<T>
	{
		auto l = length();
		return (l != T(0)) ? T(1) / l : T(0);
	}

	HOST DEVICE constexpr auto rLength2() const noexcept {
		auto l2 = length2();
		return (l2 != T(0)) ? T(1) / l2 : T(0);
	}

	// Element-wise operations
	HOST DEVICE constexpr Vector3_t<T> element_floor() const
		requires FloatingPoint<T>
	{
		return Vector3_t<T>(floor(x), floor(y), floor(z));
	}

	template<typename U>
		requires Arithmetic<U>
	HOST DEVICE constexpr auto element_mult(const U w[]) const noexcept {
		using TU = ::std::common_type_t<T, U>;
		return Vector3_t<TU>(x * w[0], y * w[1], z * w[2]);
	}

	template<typename U>
		requires Arithmetic<U>
	HOST DEVICE constexpr auto element_mult(const Vector3_t<U>& w) const noexcept {
		using TU = ::std::common_type_t<T, U>;
		return Vector3_t<TU>(x * w.x, y * w.y, z * w.z);
	}

	// Static element-wise operations
	template<typename U>
		requires Arithmetic<U>
	HOST DEVICE static constexpr auto element_mult(const Vector3_t<T>& v,
												   const Vector3_t<U>& w) noexcept {
		using TU = ::std::common_type_t<T, U>;
		return Vector3_t<TU>(v.x * w.x, v.y * w.y, v.z * w.z);
	}

	template<typename U>
		requires Arithmetic<U>
	HOST DEVICE static constexpr auto element_mult(const Vector3_t<T>& v, const U w[]) noexcept {
		using TU = ::std::common_type_t<T, U>;
		return Vector3_t<TU>(v.x * w[0], v.y * w[1], v.z * w[2]);
	}

	HOST DEVICE static auto element_sqrt(const Vector3_t<T>& w) noexcept
		requires FloatingPoint<T>
	{
		return Vector3_t<T>(sqrt(w.x), sqrt(w.y), sqrt(w.z));
	}

	// Numeric limits
	HOST DEVICE static constexpr T highest() noexcept {
#ifdef __CUDA_ARCH__
		return ::cuda::std::numeric_limits<T>::max();
#else
		return ::std::numeric_limits<T>::max();
#endif
	}

	HOST DEVICE static constexpr T lowest() noexcept {
#ifdef __CUDA_ARCH__
		return ::cuda::std::numeric_limits<T>::lowest();
#else
		return ::std::numeric_limits<T>::lowest();
#endif
	}

	// String and printing
	HOST DEVICE void print() const noexcept {
		DEVICEINFO("%0.3f %0.3f %0.3f",
				   static_cast<double>(x),
				   static_cast<double>(y),
				   static_cast<double>(z));
	}

	auto to_string() const {
		std::ostringstream oss;
		oss << x << " " << y << " " << z << " (" << w << ")";
		return oss.str();
	}

	// Comparison operators
	template<typename U>
		requires Arithmetic<U>
	HOST DEVICE constexpr bool operator==(const Vector3_t<U>& b) const noexcept {
		return x == b.x && y == b.y && z == b.z && w == b.w;
	}

	template<typename U>
		requires Arithmetic<U>
	HOST DEVICE constexpr bool operator!=(const Vector3_t<U>& b) const noexcept {
		return !(*this == b);
	}

	T x, y, z, w;
};

// Free function operators
template<typename T, typename U>
	requires Arithmetic<T> && Arithmetic<U>
HOST DEVICE constexpr auto operator/(U s, const Vector3_t<T>& v) noexcept {
	using TU = ::std::common_type_t<T, U>;
	return Vector3_t<TU>(s / v.x, s / v.y, s / v.z);
}

template<typename T>
	requires Arithmetic<T>
HOST DEVICE constexpr auto operator*(const float& s, const Vector3_t<T>& v) noexcept {
	return v * s;
}

// Helpful routines, not sure if needed
HOST DEVICE inline Vector3_t<size_t> index_to_ijk(size_t idx, size_t nx, size_t ny, size_t nz) {
	Vector3_t<size_t> res;
	res.z = idx % nz;
	res.y = (idx / nz) % ny;
	res.x = (idx / (ny * nz)) % nx;
	return res;
}
HOST DEVICE inline Vector3_t<size_t> index_to_ijk(size_t idx, const size_t n[]) {
	return index_to_ijk(idx, n[0], n[1], n[2]);
}
HOST DEVICE inline Vector3_t<size_t> index_to_ijk(size_t idx, const Vector3_t<size_t> n) {
	return index_to_ijk(idx, n.x, n.y, n.z);
}

} // namespace ARBD

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<typename T>
struct sycl::is_device_copyable<ARBD::Vector3_t<T>> : std::true_type {};
#endif

// Provide common type for vectors
namespace std {
template<typename T, typename U>
struct common_type<ARBD::Vector3_t<T>, ARBD::Vector3_t<U>> {
	using type = ARBD::Vector3_t<common_type_t<T, U>>;
};
} // namespace std

// void example_vector_kernel() {
// Resource gpu_resource{Resource::CUDA, 0};

// Create buffers for your Vector3_t type
// Vector3Buffer<float> positions(1000, gpu_resource);
// Vector3Buffer<float> velocities(1000, gpu_resource);
// Vector3Buffer<float> forces(1000, gpu_resource);

// Shamrock-style kernel launch with ARBD2 types
// kernel_call(gpu_resource,
//    MultiRef{positions, velocities},  // inputs
//    MultiRef{forces},                 // outputs
//    1000,                            // thread count
//    [](size_t i, const Vector3_t<float>* pos, const Vector3_t<float>* vel,
//       Vector3_t<float>* force) {
//        // Your kernel logic using ARBD2 Vector3_t operations
//        force[i] = pos[i].cross(vel[i]) * 0.5f;
//    });
//}
