///////////////////////////////////////////////////////////////////////
// Useful classes.
// Author: Jeff Comer <jcomer2@illinois.edu>
#ifndef USEFUL_H
#define USEFUL_H

#ifdef __CUDACC__
    #define HOST __host__
    #define DEVICE __device__
#else
    #define HOST
    #define DEVICE
#endif

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>

using namespace std;

bool isReal(char c);

bool isInt(char c);

int firstSpace(const char* s, int max);

/*class int2 {
public:
	int2(int x, int y) : x(x), y(y) {}
	int x, y;
}*/

// A classic growable string class.
class String {
public:
	HOST DEVICE inline String() {
		cap = 16;
		c = new char[cap];
		c[0] = '\0';
		len = 1;
	}

	HOST DEVICE inline String(const char* s) {
		len = strlen(s) + 1;
		cap = len;
		c = new char[len];
		for (int i = 0; i < len; i++)
			c[i] = s[i];
	}

	HOST DEVICE inline String(const String& s) {
		len = s.len;
		cap = s.len;
		c = new char[cap];
		for (int i = 0; i < s.len; i++)
			c[i] = s.c[i];
	}

	HOST DEVICE inline static size_t strlen(const char * s) {
		size_t i;
		for (i = 0; s[i] != '\0'; i++);
		return i;
	}

	HOST DEVICE inline ~String() {
		delete[] c;
	}

	String& operator=(const String& s);

	void add(char c0);
	void add(const char* s);
	void add(String& s);
	void print();
	int length() const;

	// Negative indices go from the end.
	String range(int first, int last) const;

	String trim();

	static bool isWhite(char c);

	int tokenCount() const;
	int tokenCount(char delim) const;

	int tokenize(String* tokenList) const;
	int tokenize(String* tokenList, char delim) const;

	void lower();
	void upper();

	char operator[](int i) const;
	bool operator==(const String& s) const;
	bool operator==(const char* s) const;
	bool operator!=(const String& s) const;
	operator const char*() const {
		return c;
	}
	const char* val() const;

	String getNumbers() const;

private:
	char* c;
	int cap, len;

	void grow(int n);
};

String operator+(String s, int i);
String operator+(String s, const char* c);
String operator+(String s1, String s2);

// class Vector3
// Operations on 3D float vectors
//
class Vector3 {
public:
	HOST DEVICE inline Vector3() : x(0), y(0), z(0) {}
	HOST DEVICE inline Vector3(float s):x(s), y(s), z(s) {}
	HOST DEVICE inline Vector3(const Vector3& v):x(v.x), y(v.y), z(v.z) {}
	HOST DEVICE inline Vector3(float x0, float y0, float z0) : x(x0), y(y0), z(z0) {}
	HOST DEVICE inline Vector3(const float* d) : x(d[0]), y(d[1]), z(d[2]) {}

	static Vector3 random(float s);

	HOST DEVICE inline Vector3 cross(const Vector3& w) const {
		Vector3 v;
		v.x = y*w.z - z*w.y;
		v.y = z*w.x - x*w.z;
		v.z = x*w.y - y*w.x;
		return v;
	}

	HOST DEVICE inline Vector3& operator=(const Vector3& v) {
		x = v.x;
		y = v.y;
		z = v.z;
		return *this;
	}

	HOST DEVICE inline Vector3& operator+=(const Vector3& v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	HOST DEVICE inline Vector3& operator-=(const Vector3& v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	HOST DEVICE inline Vector3& operator*=(float s) {
		x *= s;
		y *= s;
		z *= s;
		return *this;
	}

	HOST DEVICE inline Vector3& operator/=(float s) {
		x /= s;
		y /= s;
		z /= s;
		return *this;
	}

	HOST DEVICE inline Vector3 operator-() const {
		Vector3 v;
		v.x = -x;
		v.y = -y;
		v.z = -z;
		return v;
	}

	HOST DEVICE inline Vector3 operator+(const Vector3& w ) const {
		Vector3 v;
		v.x = x + w.x;
		v.y = y + w.y;
		v.z = z + w.z;
		return v;
	}

	HOST DEVICE inline Vector3 operator-(const Vector3& w ) const {
		Vector3 v;
		v.x = x - w.x;
		v.y = y - w.y;
		v.z = z - w.z;
		return v;
	}


	HOST DEVICE inline Vector3 operator*(float s) const {
		Vector3 v;
		v.x = s*x;
		v.y = s*y;
		v.z = s*z;
		return v;
	}

	HOST DEVICE inline float dot(const Vector3& w) const { return x*w.x + y*w.y + z*w.z; }

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

	String toString() const;
	float x, y, z;
};

HOST DEVICE inline Vector3 operator*(float s, Vector3 v) {
	v.x *= s;
	v.y *= s;
	v.z *= s;
	return v;
}

HOST DEVICE inline Vector3 operator/(Vector3 v, float s) {
	v.x /= s;
	v.y /= s;
	v.z /= s;
	return v;
}

// class Matrix3
// Operations on 3D float matrices
class Matrix3 {
public:
	HOST DEVICE inline Matrix3() {}
	Matrix3(float s);
	Matrix3(float xx, float xy, float xz, float yx, float yy, float yz, float zx, float zy, float zz);
	Matrix3(float x, float y, float z);
	Matrix3(const Vector3& ex, const Vector3& ey, const Vector3& ez);
	Matrix3(const float* d);

	const Matrix3 operator*(float s) const;

	const Matrix3 operator*(const Matrix3& m) const;

	const Matrix3 operator-() const;

	HOST DEVICE inline Matrix3 transpose() const {
		Matrix3 m;
		m.exx = exx;
		m.exy = eyx;
		m.exz = ezx;
		m.eyx = exy;
		m.eyy = eyy;
		m.eyz = ezy;
		m.ezx = exz;
		m.ezy = eyz;
		m.ezz = ezz;
		return m;
	}

	Matrix3 inverse() const;

	float det() const;

	HOST DEVICE inline Vector3 transform(const Vector3& v) const {
		Vector3 w;
		w.x = exx*v.x + exy*v.y + exz*v.z;
		w.y = eyx*v.x + eyy*v.y + eyz*v.z;
		w.z = ezx*v.x + ezy*v.y + ezz*v.z;
		return w;
	}


	HOST DEVICE inline Matrix3 transform(const Matrix3& m) const {
		Matrix3 ret;
		ret.exx = exx*m.exx + exy*m.eyx + exz*m.ezx;
		ret.eyx = eyx*m.exx + eyy*m.eyx + eyz*m.ezx;
		ret.ezx = ezx*m.exx + ezy*m.eyx + ezz*m.ezx;

		ret.exy = exx*m.exy + exy*m.eyy + exz*m.ezy;
		ret.eyy = eyx*m.exy + eyy*m.eyy + eyz*m.ezy;
		ret.ezy = ezx*m.exy + ezy*m.eyy + ezz*m.ezy;

		ret.exz = exx*m.exz + exy*m.eyz + exz*m.ezz;
		ret.eyz = eyx*m.exz + eyy*m.eyz + eyz*m.ezz;
		ret.ezz = ezx*m.exz + ezy*m.eyz + ezz*m.ezz;
		return ret;
	}

	Vector3 ex() const;
	Vector3 ey() const;
	Vector3 ez() const;

	String toString() const;

	String toString1() const;

	float exx, exy, exz;
	float eyx, eyy, eyz;
	float ezx, ezy, ezz;
};

Matrix3 operator*(float s, Matrix3 m);
Matrix3 operator/(Matrix3 m, float s);

// class IndexList
// A growable list of integers.
class IndexList {
public:
	IndexList();
	IndexList(const IndexList& l);
	IndexList(const int* a, int n);

	~IndexList();

	IndexList& operator=(const IndexList& l);

	void add(const int val);
	void add(const IndexList& l);

	HOST DEVICE inline int length() const {
		return num;
	}

	HOST DEVICE inline int get(const int i) const {
#ifdef DEBUG
		if (i < 0 || i >= num) {
			printf("Warning! IndexList::get out of bounds.\n");
			return 0;
		}
#endif
		return lis[i];
	}

	int* getList();
	void setList(int* list) { lis = list; }

	void clear();
	String toString() const;

	int find(int key);

	IndexList range(int i0, int i1);

public:
	int num, maxnum;
	int* lis;
};
#endif
