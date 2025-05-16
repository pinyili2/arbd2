/*********************************************************************
 * @file  Matrix3.h
 * 
 * @brief Declaration of templated Vector3_t class.
 *********************************************************************/
#pragma once
#include <cassert>
#include <memory>
#include <type_traits> // for std::common_type<T,U>

/**
// IDEA: use struct for bool options
// PROBLEM: such a struct will have a size of 1 byte because it is not a "base class subobject"
//          - Might not be optimized away... have to check this empirically
//          - i.e. this approach is good for "large" flow objects, but may be bad for gpu objects
// SOLUTION?: make a base class that defines the opts subobject and?

template<bool _is_diag=false, bool _check_diag=false>
struct Matrix3_opts {
    const bool inline is_diag() const { return _is_diag; }
    const bool inline check_diag() const { return _check_diag; }
};

// Other strategies could include 

// policy packs: https://stackoverflow.com/questions/21939217/combining-policy-classes-template-template-parameters-variadic-templates && https://www.modernescpp.com/index.php/policy-and-traits
// inheritance is an interesting option here...

**/

template<typename T, bool is_diag=false, bool check_diag=false>
struct MY_ALIGN(16*sizeof(T)) Matrix3_t  {
    using Matrix3 = Matrix3_t<T,is_diag,check_diag>;
    using Vector3 = Vector3_t<T>;

    HOST DEVICE inline Matrix3_t() { (*this) = Matrix3(1); }
    HOST DEVICE Matrix3_t(T s) { (*this) = Matrix3(s,s,s); }
    HOST DEVICE Matrix3_t(T x, T y, T z) { (*this) = Matrix3( x, 0, 0, 0, y, 0, 0, 0, z); }
    HOST DEVICE Matrix3_t(const T* d) { (*this) = Matrix3(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8]); }
    HOST DEVICE Matrix3_t(const Vector3& ex, const Vector3& ey, const Vector3& ez) { (*this) = Matrix3(ex.x, ex.y, ex.z, ey.x, ey.y, ey.z, ez.x, ez.y, ez.z); }
    HOST DEVICE Matrix3_t(T xx, T xy, T xz, T yx, T yy, T yz, T zx, T zy, T zz) : xx(xx), xy(xy), xz(xz), yx(yx), yy(yy), yz(yz), zx(zx), zy(zy), zz(zz) { diag_check(); }

    HOST DEVICE inline void diag_check() const {
	// if (opts.check_diag() && opts.is_diag()) {}
	if (check_diag && is_diag) {
	    assert(xy == xz == yx == yz == zx == zy == 0);
	    // assert(xy == 0 && xz == 0 &&
	    // 	   yx == 0 && yz == 0 &&
	    // 	   zx == 0 && zy == 0);
	}
    }
	
    // Operators
    template<typename U>
	HOST DEVICE inline auto operator*(U s) const {
	Matrix3_t<std::common_type_t<T,U>, is_diag, check_diag> m;
	m.xx = s*xx; m.xy = s*xy; m.xz = s*xz;
	m.yx = s*yx; m.yy = s*yy; m.yz = s*yz;
	m.zx = s*zx; m.zy = s*zy; m.zz = s*zz;
	return m;
    }

    template<typename U>
	HOST DEVICE inline auto operator*(const Vector3_t<U>& v) const { return this->transform(v); }

    template<typename U, bool is_diag2, bool check_diag2>
	HOST DEVICE inline auto operator*(const Matrix3_t<U,is_diag2,check_diag2>& m) const { return this->transform(m); }

    template<typename U, bool is_diag2, bool check_diag2>
	HOST DEVICE inline auto operator+(const Matrix3_t<U,is_diag2,check_diag2>& m) const {
	Matrix3_t<std::common_type_t<T,U>, is_diag && is_diag2, check_diag||check_diag2 > ret;
	ret.xx = xx+m.xx;
	ret.yy = yy+m.yy;
	ret.zz = zz+m.zz;

	if ( (!is_diag) || (!is_diag2) ) {
	    ret.xy = xy+m.xy; ret.xz = xz+m.xz;
	    ret.yx = yx+m.yx; ret.yz = yz+m.yz;
	    ret.zx = zx+m.zx; ret.zy = zy+m.zy;
	}	    
	return ret;
    }

    template<typename U, bool is_diag2, bool check_diag2>
	HOST DEVICE inline auto operator-(const Matrix3_t<U,is_diag2,check_diag2>& m) const { return (*this)+(-m); }

	HOST DEVICE inline Matrix3 operator-() const { return Matrix3(-xx,-xy,-xz,-yx,-yy,-yz,-zx,-zy,-zz); }
    HOST DEVICE inline Matrix3 transpose() const { return Matrix3(xx,yx,zx,xy,yy,zy,xz,yz,zz); }

    template<typename U>
	HOST DEVICE inline auto transform(const Vector3_t<U>& v) const {
	Vector3_t<std::common_type_t<T,U>> w;
	if (is_diag) {
	    w.x = xx*v.x;
	    w.y = yy*v.y;
	    w.z = zz*v.z;
	} else {
	    w.x = xx*v.x + xy*v.y + xz*v.z;
	    w.y = yx*v.x + yy*v.y + yz*v.z;
	    w.z = zx*v.x + zy*v.y + zz*v.z;
	}
	return w;
    }

    template<typename U, bool is_diag2, bool check_diag2>
	HOST DEVICE inline auto transform(const Matrix3_t<U,is_diag2,check_diag2>& m) const {
	Matrix3_t<std::common_type_t<T,U>, is_diag && is_diag2, check_diag||check_diag2 > ret;
	ret.xx = xx*m.xx + xy*m.yx + xz*m.zx;
	ret.yy = yx*m.xy + yy*m.yy + yz*m.zy;
	ret.zz = zx*m.xz + zy*m.yz + zz*m.zz;

	if ( (!is_diag) || (!is_diag2) ) {
	    ret.yx = yx*m.xx + yy*m.yx + yz*m.zx;
	    ret.zx = zx*m.xx + zy*m.yx + zz*m.zx;

	    ret.xy = xx*m.xy + xy*m.yy + xz*m.zy;
	    ret.zy = zx*m.xy + zy*m.yy + zz*m.zy;

	    ret.xz = xx*m.xz + xy*m.yz + xz*m.zz;
	    ret.yz = yx*m.xz + yy*m.yz + yz*m.zz;
	}	    
	return ret;
    }

    HOST DEVICE
	Matrix3 inverse() const {
	Matrix3 m;
	if (is_diag) {
	    return Matrix3(T(1.0)/xx,T(1.0)/yy,T(1.0)/zz);
	} else {
	    T det = this->det();
	    return Matrix3( (yy*zz-yz*zy)/det,-(xy*zz-xz*zy)/det, (xy*yz-xz*yy)/det,
			   -(yx*zz-yz*zx)/det, (xx*zz-xz*zx)/det,-(xx*yz-xz*yx)/det,
			    (yx*zy-yy*zx)/det,-(xx*zy-xy*zx)/det, (xx*yy-xy*yx)/det );
	}
    }
	
    HOST DEVICE
        T det() const { if (is_diag) {
	    return xx*yy*zz;
	} else {
	    return xx*(yy*zz-yz*zy) - xy*(yx*zz-yz*zx) + xz*(yx*zy-yy*zx);
	}
    }

    //Han-Yi Chou
    HOST DEVICE inline Matrix3 normalized() const {                
	Vector3 x = this->ex();
	Vector3 y = this->ey();
	/*
	  x = x / x.length();
	  float error = x.dot(y);
	  y = y-(error*x);
	  y = y / y.length();
	  Vector3 z = x.cross(y);
	  z = z / z.length();*/
	//x = (0.5*(3-x.dot(x)))*x; /* approximate normalization */
	//y = (0.5*(3-y.dot(y)))*y; 
	//z = (0.5*(3-z.dot(z)))*z; 
	//return Matrix3(x,y,z);		
	Vector3 z = x.cross(y);
	T l;
	l = z.length();
	z = l > 0 ? z/l : Vector3(T(0));
	l = x.length();
	x = l > 0 ? x/l : Vector3(T(0));
	y = z.cross(x);
	l = y.length();
	y = l > 0 ? y/l : Vector3(T(0));
	return Matrix3(x,y,z);
    }
	
    HOST DEVICE inline Vector3 ex() const { return Vector3(xx,yx,zx); }
    HOST DEVICE inline Vector3 ey() const { return Vector3(xy,yy,zy); }
    HOST DEVICE inline Vector3 ez() const { return Vector3(xz,yz,zz); }

    auto to_string() const {
	return string_format("%2.8f %2.8f %2.8f\n%2.8f %2.8f %2.8f\n%2.8f %2.8f %2.8f", xx,xy,xz,yx,yy,yz,zx,zy,zz);
	// return std::format("\n{} {} {}\n{} {} {}\n{} {} {}", xx,xy,xz,yx,yy,yz,zx,zy,zz);
	// return std::format("\n{} {} {}\n{} {} {}\n{} {} {}", xx,xy,xz,yx,yy,yz,zx,zy,zz);
	// char s[256];
	// sprintf(s, "%2.8f %2.8f %2.8f\n%2.8f %2.8f %2.8f\n%2.8f %2.8f %2.8f",
	// 	xx, xy, xz, yx, yy, yz, zx, zy, zz);
	// s[255] = 0;
	// return std::string(s);
    }

    HOST DEVICE inline bool is_diagonal() const { return is_diag; }

    template<typename U>
	HOST DEVICE inline bool operator==(U b) const {
	return xx == b.xx && xy == b.xy && xz == b.xz &&
	    yx == b.yx && yy == b.yy && yz == b.yz &&
	    zx == b.zx && zy == b.zy && zz == b.zz;
    }

    // Helper function for testing
    template<typename U> void test_equal(const U&& b) const {
	CHECK( xx == b.xx );
	CHECK( xy == b.xy );
	CHECK( xz == b.xz );
	CHECK( yx == b.yx );
	CHECK( yy == b.yy );
	CHECK( yz == b.yz );
	CHECK( zx == b.zx );
	CHECK( zy == b.zy );
	CHECK( zz == b.zz );
    }
    
    T xx, xy, xz;
    T yx, yy, yz;
    T zx, zy, zz;
};

// template<typename ...Args>
// HOST std::ostream& operator << ( std::ostream& os, Matrix3_t<Args...> const& value ) {
//     printf("MATRIX OP\n");
//     os << value.to_string();
//     return os;
// }


/* template<typename U> */
/* HOST DEVICE friend inline Matrix3 operator*(float s, Matrix3 m) { return m*s; } */
/* HOST DEVICE friend inline Matrix3 operator/(Matrix3 m, float s) { */
/* 	const float sinv = 1.0f/s; */
/* 	return m*sinv; */
/* } */

    // HOST DEVICE void setIsDiag() {
    // 	isDiag = (xy == 0 && xz == 0 &&
    // 		  yx == 0 && yz == 0 &&
    // 		  zx == 0 && zy == 0) ? true : false;
    // }
