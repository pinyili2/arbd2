/*********************************************************************
 * @file  Matrix3.h
 *
 * @brief Declaration of templated Matrix3_t class, using a
 * column-major layout for compatibility with graphics APIs.
 *********************************************************************/
#pragma once
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Backend/Resource.h"
#include "Vector3.h"
#include <cassert>
#include <memory>
#include <type_traits>

namespace ARBD {

template<typename... Args>
std::string string_format(const char* format, Args... args);

/**
 * @brief A 3x3 matrix class optimized for performance and cross-backend use.
 *
 * This matrix is stored in column-major order to align with conventions
 * in OpenGL, Vulkan, and for easier interoperability with Metal.
 *
 * Memory layout (Column-Major):
 * [ex.x, ex.y, ex.z, 0] -> [xx, yx, zx, _]
 * [ey.x, ey.y, ey.z, 0] -> [xy, yy, zy, _]
 * [ez.x, ez.y, ez.z, 0] -> [xz, yz, zz, _]
 *
 * The struct members are named to match row-major access (e.g., m.xy is
 * row 0, col 1) but are stored physically in column-major order.
 */
template<typename T, bool is_diag = false, bool check_diag = false>
	requires Arithmetic<T>
struct alignas(4 * sizeof(float)) Matrix3_t {
	using Matrix3 = Matrix3_t<T, is_diag, check_diag>;
	using Vector3 = Vector3_t<T>;

  private:
	// The matrix is physically stored as three column vectors.
	// This ensures a column-major memory layout.
	Vector3 cols[3];

  public:
	// Constructors
	HOST DEVICE constexpr Matrix3_t() noexcept {
		cols[0] = Vector3(T(1), T(0), T(0));
		cols[1] = Vector3(T(0), T(1), T(0));
		cols[2] = Vector3(T(0), T(0), T(1));
	}

	HOST DEVICE constexpr Matrix3_t(T s) noexcept {
		cols[0] = Vector3(s, T(0), T(0));
		cols[1] = Vector3(T(0), s, T(0));
		cols[2] = Vector3(T(0), T(0), s);
	}

	// Constructor for a diagonal matrix
	HOST DEVICE constexpr Matrix3_t(T x, T y, T z) noexcept {
		cols[0] = Vector3(x, T(0), T(0));
		cols[1] = Vector3(T(0), y, T(0));
		cols[2] = Vector3(T(0), T(0), z);
	}

	// Constructor from three column vectors
	HOST DEVICE constexpr Matrix3_t(const Vector3& c0,
									const Vector3& c1,
									const Vector3& c2) noexcept {
		cols[0] = c0;
		cols[1] = c1;
		cols[2] = c2;
	}

	// Element-wise multiplication
	template<typename U>
		requires Arithmetic<U>
	HOST DEVICE constexpr auto operator*(U s) const noexcept {
		using TU = ::std::common_type_t<T, U>;
		return Matrix3_t<TU, is_diag, check_diag>(cols[0] * s, cols[1] * s, cols[2] * s);
	}

	// Vector transformation (Matrix * Vector)
	template<typename U>
		requires Arithmetic<U>
	HOST DEVICE constexpr auto transform(const Vector3_t<U>& v) const noexcept {
		using TU = ::std::common_type_t<T, U>;
		if constexpr (is_diag) {
			return Vector3_t<TU>(cols[0].x * v.x, cols[1].y * v.y, cols[2].z * v.z);
		} else {
			return cols[0] * v.x + cols[1] * v.y + cols[2] * v.z;
		}
	}

	template<typename U>
		requires Arithmetic<U>
	HOST DEVICE constexpr auto operator*(const Vector3_t<U>& v) const noexcept {
		return this->transform(v);
	}

	// Matrix multiplication (this * m)
	template<typename U, bool is_diag2, bool check_diag2>
		requires Arithmetic<U>
	HOST DEVICE constexpr auto
	operator*(const Matrix3_t<U, is_diag2, check_diag2>& m) const noexcept {
		using TU = ::std::common_type_t<T, U>;
		constexpr bool result_is_diag = is_diag && is_diag2;
		constexpr bool result_check_diag = check_diag || check_diag2;

		Vector3_t<TU> new_c0 = this->transform(m.cols[0]);
		Vector3_t<TU> new_c1 = this->transform(m.cols[1]);
		Vector3_t<TU> new_c2 = this->transform(m.cols[2]);

		return Matrix3_t<TU, result_is_diag, result_check_diag>(new_c0, new_c1, new_c2);
	}

	// Matrix addition
	template<typename U, bool is_diag2, bool check_diag2>
		requires Arithmetic<U>
	HOST DEVICE constexpr auto
	operator+(const Matrix3_t<U, is_diag2, check_diag2>& m) const noexcept {
		using TU = ::std::common_type_t<T, U>;
		return Matrix3_t<TU>(cols[0] + m.cols[0], cols[1] + m.cols[1], cols[2] + m.cols[2]);
	}

	// Matrix transpose
	HOST DEVICE constexpr Matrix3 transpose() const noexcept {
		Vector3 r0(cols[0].x, cols[1].x, cols[2].x);
		Vector3 r1(cols[0].y, cols[1].y, cols[2].y);
		Vector3 r2(cols[0].z, cols[1].z, cols[2].z);
		return Matrix3(r0, r1, r2);
	}

	// Matrix inverse
	HOST DEVICE constexpr Matrix3 inverse() const noexcept {
		if constexpr (is_diag) {
			return Matrix3(T(1) / cols[0].x, T(1) / cols[1].y, T(1) / cols[2].z);
		} else {
			Vector3 c0 = cols[0], c1 = cols[1], c2 = cols[2];
			Vector3 r0 = c1.cross(c2);
			Vector3 r1 = c2.cross(c0);
			Vector3 r2 = c0.cross(c1);
			T inv_det = T(1) / c0.dot(r0);

			return Matrix3(Vector3(r0.x, r1.x, r2.x) * inv_det,
						   Vector3(r0.y, r1.y, r2.y) * inv_det,
						   Vector3(r0.z, r1.z, r2.z) * inv_det);
		}
	}

	// Determinant
	HOST DEVICE constexpr T det() const noexcept {
		if constexpr (is_diag) {
			return cols[0].x * cols[1].y * cols[2].z;
		} else {
			return cols[0].dot(cols[1].cross(cols[2]));
		}
	}

	// Extract basis vectors (columns)
	HOST DEVICE constexpr const Vector3& ex() const noexcept {
		return cols[0];
	}
	HOST DEVICE constexpr const Vector3& ey() const noexcept {
		return cols[1];
	}
	HOST DEVICE constexpr const Vector3& ez() const noexcept {
		return cols[2];
	}

	// String representation
	auto to_string() const {
		Matrix3 transposed = this->transpose(); // Print in row-major for readability
		return string_format("%2.8f %2.8f %2.8f\n%2.8f %2.8f %2.8f\n%2.8f %2.8f %2.8f",
							 transposed.cols[0].x,
							 transposed.cols[1].x,
							 transposed.cols[2].x,
							 transposed.cols[0].y,
							 transposed.cols[1].y,
							 transposed.cols[2].y,
							 transposed.cols[0].z,
							 transposed.cols[1].z,
							 transposed.cols[2].z);
	}
};

// Free function for scalar-matrix multiplication
template<typename S, typename T, bool is_diag, bool check_diag>
	requires Arithmetic<S> && Arithmetic<T>
HOST DEVICE constexpr auto operator*(S s, const Matrix3_t<T, is_diag, check_diag>& m) noexcept {
	return m * s;
}

} // namespace ARBD

// SYCL specialization to mark the type as safe for device copy
#ifdef USE_SYCL
#include <sycl/sycl.hpp>
template<typename T, bool is_diag, bool check_diag>
struct sycl::is_device_copyable<ARBD::Matrix3_t<T, is_diag, check_diag>> : std::true_type {};
#endif
