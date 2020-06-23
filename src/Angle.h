// Angle.h
// Copyright Justin Dufresne and Terrance Howard, 2013

#ifndef ANGLE_H
#define ANGLE_H

#ifdef __CUDACC__
    #define HOST __host__
    #define DEVICE __device__
#else
    #define HOST 
    #define DEVICE 
#endif

#include "useful.h"
#include "BaseGrid.h"
#include <cuda.h>

class Angle
{
public:
	Angle() {}
	Angle(int ind1, int ind2, int ind3, String fileName) :
	ind1(ind1), ind2(ind2), ind3(ind3), fileName(fileName), tabFileIndex(-1) { }
	
	int ind1, ind2, ind3;
	String fileName;
	// tabFileIndex will be assigned after ComputeForce loads the
	// TabulatedAnglePotentials. The tabefileIndex is used by ComputeForce to
	// discern which TabulatedAnglePotential this Angle uses.
	int tabFileIndex;

	inline Angle(const Angle& a) : ind1(a.ind1), ind2(a.ind2), ind3(a.ind3),
		fileName(a.fileName),
		tabFileIndex(a.tabFileIndex) { }

	HOST DEVICE inline float calcAngle(Vector3* pos, BaseGrid* sys) {
		const Vector3& posa = pos[ind1];
		const Vector3& posb = pos[ind2];
		const Vector3& posc = pos[ind3];
		const float distab = sys->wrapDiff(posa - posb).length();
		const float distbc = sys->wrapDiff(posb - posc).length();
		const float distac = sys->wrapDiff(posc - posa).length();	
		float cos = (distbc * distbc + distab * distab - distac * distac)
							  / (2.0f * distbc * distab);
		if (cos < -1.0f) cos = -1.0f;
		else if (cos > 1.0f) cos = 1.0f;
		float angle = acos(cos);
		return angle;
	}	

	HOST DEVICE inline int getIndex(int index) {
		if (index == ind1) return 1;
		if (index == ind2) return 2;
		if (index == ind3) return 3;
		return -1;
	}

	String toString();
	void print();
};

// TODO consolidate with Angle using inheritence
class BondAngle
{
public:
	BondAngle() {}
        BondAngle(int ind1, int ind2, int ind3, String angleFileName, String bondFileName1, String bondFileName2) :
	ind1(ind1), ind2(ind2), ind3(ind3), angleFileName(angleFileName), bondFileName1(bondFileName1), bondFileName2(bondFileName2), tabFileIndex1(-1), tabFileIndex2(-1), tabFileIndex3(-1) { }

	int ind1, ind2, ind3;

	String angleFileName;
	String bondFileName1;
	String bondFileName2;
	// tabFileIndex will be assigned after ComputeForce loads the
	// TabulatedAnglePotentials. The tabefileIndex is used by ComputeForce to
	// discern which TabulatedAnglePotential this Angle uses.
	int tabFileIndex1;
	int tabFileIndex2;
	int tabFileIndex3;

	inline BondAngle(const BondAngle& a) : ind1(a.ind1), ind2(a.ind2), ind3(a.ind3),
	    angleFileName(a.angleFileName),
	    bondFileName1(a.bondFileName1),
	    bondFileName2(a.bondFileName2),
	    tabFileIndex1(a.tabFileIndex1),
	    tabFileIndex2(a.tabFileIndex2),
	    tabFileIndex3(a.tabFileIndex3) { }

	HOST DEVICE inline float calcAngle(Vector3* pos, BaseGrid* sys) {
		const Vector3& posa = pos[ind1];
		const Vector3& posb = pos[ind2];
		const Vector3& posc = pos[ind3];
		const float distab = sys->wrapDiff(posa - posb).length();
		const float distbc = sys->wrapDiff(posb - posc).length();
		const float distac = sys->wrapDiff(posc - posa).length();
		float cos = (distbc * distbc + distab * distab - distac * distac)
							  / (2.0f * distbc * distab);
		if (cos < -1.0f) cos = -1.0f;
		else if (cos > 1.0f) cos = 1.0f;
		float angle = acos(cos);
		return angle;
	}

	HOST DEVICE inline int getIndex(int index) {
		if (index == ind1) return 1;
		if (index == ind2) return 2;
		if (index == ind3) return 3;
		return -1;
	}

	String toString();
	void print();
};

#endif
