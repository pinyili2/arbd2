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

#ifndef gpuErrchk
#define delgpuErrchk
#define gpuErrchk(code) { if ((code) != cudaSuccess) {			                            \
	    fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), __FILE__, __LINE__); \
	}}
#endif

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

class TabulatedPotential;

class FullTabulatedPotential {
public:
  FullTabulatedPotential();
  FullTabulatedPotential(const char* fileName);
  FullTabulatedPotential(const FullTabulatedPotential& tab);
  ~FullTabulatedPotential();

  static int countValueLines(const char* fileName);

  /* HOST DEVICE inline EnergyForce computeOLD(Vector3 r) { */
  /* 		float d = r.length(); */
  /* 		Vector3 rUnit = -r/d; */
  /* 		int home = int(floorf((d - r0)/dr)); */
  /* 		if (home < 0) return EnergyForce(v0[0], Vector3(0.0f)); */
  /* 		if (home >= n) return EnergyForce(e0, Vector3(0.0f)); */
  /* 		float homeR = home*dr + r0; */
  /* 		float w = (d - homeR)/dr; */
		
  /* 		// Interpolate. */
  /* 		float energy = v3[home]*w*w*w + v2[home]*w*w + v1[home]*w + v0[home]; */
  /* 		Vector3 force = -(3.0f*v3[home] * w * w */
  /* 										+ 2.0f*v2[home] * w */
  /* 										+ v1[home]) * rUnit/dr; */
  /* 		return EnergyForce(energy,force); */
  /* 	} */

  TabulatedPotential* pot;

private:
  int numLines;
  String fileName;
};

class TabulatedPotential {
public:
  TabulatedPotential();
  TabulatedPotential(const TabulatedPotential& tab);
  TabulatedPotential(const float* dist, const float* pot, int n0);
    TabulatedPotential(const FullTabulatedPotential& tab) : TabulatedPotential(*tab.pot) {}
    TabulatedPotential(const char* filename) : TabulatedPotential(FullTabulatedPotential(filename)) {}
  ~TabulatedPotential();

  void truncate(float cutoff);
  bool truncate(float switchDist, float cutoff, float value);

  Vector3 computeForce(Vector3 r);

  int size() const { return n; }

    TabulatedPotential* copy_to_cuda() const {
	// Allocate data for array 
	TabulatedPotential* dev_ptr;
	TabulatedPotential tmp(*this); // TODO consider avoiding allocating v0

	float *v;
	{
	    size_t sz = sizeof(float) * n;
	    gpuErrchk(cudaMalloc(&v, sz));
	    gpuErrchk(cudaMemcpy(v, v0, sz, cudaMemcpyHostToDevice));
	}
	delete [] tmp.v0;
	tmp.v0 = v;

	size_t sz = sizeof(TabulatedPotential);
	gpuErrchk(cudaMalloc(&dev_ptr, sz));
	gpuErrchk(cudaMemcpy(dev_ptr, &tmp, sz, cudaMemcpyHostToDevice));
	tmp.v0 = NULL;
	return dev_ptr;
    }
    void free_from_cuda(TabulatedPotential* dev_ptr) const {
	TabulatedPotential tmp = TabulatedPotential();
	delete [] tmp.v0;
	gpuErrchk(cudaMemcpy(&tmp, dev_ptr, sizeof(TabulatedPotential), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(dev_ptr));
	gpuErrchk(cudaFree(tmp.v0));
	tmp.v0 = NULL;
    }


  HOST DEVICE inline EnergyForce compute(Vector3 r) {
		float d = r.length();
		float w = (d - r0) * drInv;
		int home = int( floorf(w) );
		w = w - home;
		// if (home < 0) return EnergyForce(v0[0], Vector3(0.0f));
		home = home < 0 ? 0 : home;
		if (home >= n) return EnergyForce(v0[n-1], Vector3(0.0f));
		
		float u0 = v0[home];
		float du = home+1 < n ? v0[home+1]-u0 : 0;
				
		// Interpolate.
		float energy = du*w+u0;
		Vector3 force = (du*drInv/d)*r;
		return EnergyForce(energy,force);
	}

  HOST DEVICE inline EnergyForce compute(const Vector3 r, float d) const {
		d = sqrt(d);
		// float d = r.length();
		float w = (d - r0) * drInv;
		int home = int( floorf(w) );
		w = w - home;
		// if (home < 0) return EnergyForce(v0[0], Vector3(0.0f));
		home = home < 0 ? 0 : home;
		if (home >= n) return EnergyForce(v0[n-1], Vector3(0.0f));
		
		float u0 = v0[home];
		float du = home+1 < n ? v0[home+1]-u0 : 0;
				
		// Interpolate.
		float energy = du*w+u0;
		Vector3 force = (du*drInv/d)*r;
		return EnergyForce(energy,force);
	}
  HOST DEVICE inline Vector3 computef(const Vector3 r, float d) const {
		d = sqrt(d);
		// float d = r.length();
		// RBTODO: precompute so that initial blocks are zero; reduce computation here
		float w = (d - r0)*drInv;
		int home = int( floorf(w) );
		w = w - home;
		// if (home < 0) return EnergyForce(v0[0], Vector3(0.0f));
		home = home < 0 ? 0 : home;
		if (home >= n) return Vector3(0.0f);
		
		if (home+1 < n) 
		    return ((v0[home+1]-v0[home])*drInv/d)*r;
		else
		    return Vector3(0.0f);
	}

// private:
private:
  float* v0;
  int n;
  float drInv; //TODO replace with drInv
  float r0;
};
#ifndef delgpuErrchk
#undef  delgpuErrchk
#undef  gpuErrchk(code)
#endif
#endif
