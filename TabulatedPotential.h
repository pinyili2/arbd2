///////////////////////////////////////////////////////////////////////
// Author: Jeff Comer <jcomer2@illinois.edu>
#ifndef TABULATEDPOTENTIAL_H
#define TABULATEDPOTENTIAL_H

#ifdef __CUDACC__
    #define HOST __host__
    #define DEVICE __device__
#else
    #define HOST 
    #define DEVICE 
#endif

#include "useful.h"
#include <cuda.h>


class EnergyForce {
public:
  HOST DEVICE
	inline EnergyForce(float energy=0.0f, const Vector3& force=Vector3(0.0f)) :
			e(energy), f(force) { }
  HOST DEVICE
	inline EnergyForce operator+=(const EnergyForce& ef) {
		e += ef.e;
		f += ef.f;
		return *this;
	}
  float e;
  Vector3 f;
};

class TabulatedPotential {
public:
  TabulatedPotential();
  TabulatedPotential(const char* fileName);
  TabulatedPotential(const TabulatedPotential& tab);
  TabulatedPotential(const float* dist, const float* pot, int n0);

  ~TabulatedPotential();

  static int countValueLines(const char* fileName);

  void truncate(float cutoff);
  bool truncate(float switchDist, float cutoff, float value);

  Vector3 computeForce(Vector3 r);

  HOST DEVICE inline EnergyForce computeOLD(Vector3 r) {
		float d = r.length();
		Vector3 rUnit = -r/d;
		int home = int(floorf((d - r0)/dr));
		if (home < 0) return EnergyForce(v0[0], Vector3(0.0f));
		if (home >= n) return EnergyForce(e0, Vector3(0.0f));
		float homeR = home*dr + r0;
		float w = (d - homeR)/dr;
		
		// Interpolate.
		float energy = v3[home]*w*w*w + v2[home]*w*w + v1[home]*w + v0[home];
		Vector3 force = -(3.0f*v3[home] * w * w
										+ 2.0f*v2[home] * w
										+ v1[home]) * rUnit/dr;
		return EnergyForce(energy,force);
	}

  HOST DEVICE inline EnergyForce compute(Vector3 r) {
		float d = r.length();
		float w = (d - r0)/dr;
		int home = int( floorf(w) );
		w = w - home;
		// if (home < 0) return EnergyForce(v0[0], Vector3(0.0f));
		home = home < 0 ? 0 : home;
		if (home >= n) return EnergyForce(e0, Vector3(0.0f));
		
		float u0 = v0[home];
		float du = home+1 < n ? v0[home+1]-u0 : 0;
				
		// Interpolate.
		float energy = du*w+u0;
		Vector3 force = (-du/(d*dr))*r;
		return EnergyForce(energy,force);
	}

// private:
public:
  float* pot;
  float* v0;
  float* v1;
  float* v2;
  float* v3;
  int n;
  int numLines;
  float dr;
  float r0, r1;
  float e0;
  String fileName;
	
  void init(const float* dist, const float* pot, int n0);
  void interpolate();
 
};
#endif
