/*********************************************************************
 * @file  Vector3.h
 * 
 * @brief Declaration of templated Vector3_t class.
 *********************************************************************/
#pragma once
#include <type_traits> // for std::common_type<T,U>

/**
 * 3D vector utility class with common operations implemented on CPU and GPU.
 * 
 * Implemented with 4D data storage for better GPU alignment; extra
 * data can be stored in fourth varaible this->w
 *
 * @tparam T the type of data stored in the four fields, x,y,z,w; T
 * Should usually be float or double.
g */
template<typename T>
class MY_ALIGN(4*sizeof(T)) Vector3_t {
public:
	HOST DEVICE inline Vector3_t<T>() : x(0), y(0), z(0), w(0) {}
	HOST DEVICE inline Vector3_t<T>(T s):x(s), y(s), z(s), w(s) {}
	HOST DEVICE inline Vector3_t<T>(const Vector3_t<T>& v):x(v.x), y(v.y), z(v.z), w(v.w)  {}
	HOST DEVICE inline Vector3_t<T>(T x0, T y0, T z0) : x(x0), y(y0), z(z0), w(0) {}
	HOST DEVICE inline Vector3_t<T>(const T* d) : x(d[0]), y(d[1]), z(d[2]), w(0) {}
        HOST DEVICE inline Vector3_t<T>(const float4 a) : x(a.x), y(a.y), z(a.z), w(a.w) {}

#if __cplusplus >= 201703L
	// TODO: test if c++17 constexpr performs better than std::common_type, otherwise get rid of this #if block
	template <typename U>
	    HOST DEVICE inline Vector3_t<typename std::common_type<T,U>::type> cross(const Vector3_t<U>& w) const {
	    if constexpr(sizeof(U) < sizeof(T)) {
		Vector3_t<T> v;
		v.x = y*w.z - z*w.y;
		v.y = z*w.x - x*w.z;
		v.z = x*w.y - y*w.x;
		return v;
	    } else {
		Vector3_t<U> v;
		v.x = y*w.z - z*w.y;
		v.y = z*w.x - x*w.z;
		v.z = x*w.y - y*w.x;
		return v;
	    }
	}
#else
	template <typename U>
	    HOST DEVICE inline Vector3_t<typename std::common_type<T,U>::type> cross(const Vector3_t<U>& w) const {
	    using TU = typename std::common_type<T,U>::type;
	    Vector3_t<TU> v;
	    v.x = y*w.z - z*w.y;
	    v.y = z*w.x - x*w.z;
	    v.z = x*w.y - y*w.x;
	    return v;
	}
#endif

	HOST DEVICE inline Vector3_t<T>& operator=(const Vector3_t<T>&& v) {
		x = v.x;
		y = v.y;
		z = v.z;
		return *this;
	}

	HOST DEVICE inline Vector3_t<T>& operator+=(const Vector3_t<T>&& v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	HOST DEVICE inline Vector3_t<T>& operator-=(const Vector3_t<T>&& v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	HOST DEVICE inline Vector3_t<T>& operator*=(float s) {
		x *= s;
		y *= s;
		z *= s;
		return *this;
	}

	HOST DEVICE inline Vector3_t<T>& operator/=(float s) {
		const float sinv = 1.0f/s;
		x *= sinv;
		y *= sinv;
		z *= sinv;
		return *this;
	}

	HOST DEVICE inline Vector3_t<T> operator-() const {
		Vector3_t<T> v;
		v.x = -x;
		v.y = -y;
		v.z = -z;
		return v;
	}

	template<typename U>
	    HOST DEVICE inline Vector3_t<T> operator+(const Vector3_t<U>& w ) const {
	    using TU = typename std::common_type<T,U>::type;
	    Vector3_t<TU> v;
	    v.x = x + w.x;
	    v.y = y + w.y;
	    v.z = z + w.z;
	    return v;
	}

	HOST DEVICE inline Vector3_t<T> operator-(const Vector3_t<T>& w ) const {
		Vector3_t<T> v;
		v.x = x - w.x;
		v.y = y - w.y;
		v.z = z - w.z;
		return v;
	}

	template<typename U>
	HOST DEVICE inline Vector3_t<T> operator*(U&& s) const {
	    // TODO, throw exception if int
	    using TU = typename std::common_type<T,U>::type;
	    Vector3_t<TU> v;
	    v.x = s*x;
	    v.y = s*y;
	    v.z = s*z;
	    return v;
	}

	template<typename U>
	HOST DEVICE inline Vector3_t<T> operator/(U&& s) const {
	    const U inv = static_cast<U>(1.0)/s;
	    return (*this)*inv;
	}

	template<typename U>
	HOST DEVICE inline float dot(const Vector3_t<U>& w) const { return x*w.x + y*w.y + z*w.z; }

	HOST DEVICE inline float length() const { return sqrtf(x*x + y*y + z*z); }
	HOST DEVICE inline float length2() const { return x*x + y*y + z*z; }
	HOST DEVICE inline float rLength() const {
		float l = length();
		if (l != 0.0f)
			return 1.0f / l;
		return 0.0f;
	}

	HOST DEVICE inline float rLength2() const {
		float l2 = length2();
		if (l2 != 0.0f) return 1.0f / l2;
		return 0.0f;
	}

	template<typename U>
	HOST DEVICE static inline auto element_mult(const Vector3_t<T>&& v, const Vector3_t<U>&& w) {
	    using TU = typename std::common_type<T,U>::type;
		Vector3_t<TU> ret(
			v.x*w.x,
			v.y*w.y,
			v.z*w.z);
		return ret;
	}

	HOST DEVICE static inline Vector3_t<T> element_sqrt(const Vector3_t<T>&& w) {
		Vector3_t<T> ret(
			sqrt(w.x),
			sqrt(w.y),
			sqrt(w.z));
		return ret;
	}

	HOST DEVICE inline void print() const {
		printf("%0.3f %0.3f %0.3f\n", x,y,z);
	}

	T x, y, z, w; //append a member w	

	char* to_char() const {
	    char* s = new char[128];
	    sprintf(s, "%.10g %.10g %.10g", x, y, z);
	    s[127] = 0;
	    return s;
	}
};



/* template<typename T> */
/* HOST DEVICE inline Vector3_t<T> operator*(float s, Vector3_t<T> v) { */
/* 	v.x *= s; */
/* 	v.y *= s; */
/* 	v.z *= s; */
/* 	return v; */
/* } */

// template<typename T>
// HOST DEVICE inline Vector3_t<T> operator/(Vector3_t<T> v, float s) {
// 	const float sinv = 1.0f/s;
// 	return v*sinv;
// }

template<typename T, typename U>
HOST DEVICE inline auto operator/(U&& s, Vector3_t<T>&& v) {
    // TODO, throw exception if int
    using TU = typename std::common_type<T,U>::type;
    Vector3_t<TU> ret;
    ret.x = s / v.x;
    ret.y = s / v.y;
    ret.z = s / v.z;
    return ret;
}

namespace std {
    template<typename T,typename U>
    struct common_type<Vector3_t<T>, Vector3_t<U>> {
	using type = Vector3_t<common_type_t<T,U>>;
    };
} 
