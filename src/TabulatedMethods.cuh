#pragma once

#define BD_PI 3.1415927f

__device__ inline void computeAngle(const TabulatedAnglePotential* __restrict__ a, const BaseGrid* __restrict__ sys, Vector3* force, const Vector3* __restrict__ pos,
				const int& i, const int& j, const int& k) {
	    
	    
	// Particle's type and position
	Vector3 posa = pos[i];
	Vector3 posb = pos[j];
	Vector3 posc = pos[k];
		
	// The vectors between each pair of particles
	const Vector3 ab = sys->wrapDiff(posa - posb);
	const Vector3 bc = sys->wrapDiff(posb - posc);
	const Vector3 ac = sys->wrapDiff(posc - posa);
  
	// Find the distance between each pair of particles
	float distab = ab.length2();
	float distbc = bc.length2();
	const float distac2 = ac.length2();
  
	// Find the cosine of the angle we want - <ABC	
	float cos = (distab + distbc - distac2);
	distab = 1.0f/sqrt(distab); //TODO: test other functiosn
	distbc = 1.0f/sqrt(distbc);
	cos *= 0.5f * distbc * distab;
  
	// If the cosine is illegitimate, set it to 1 or -1 so that acos won't fail
	if (cos < -1.0f) cos = -1.0f;
	if (cos > 1.0f) cos = 1.0f;

	// Find the sine while we're at it.

	// Now we can use the cosine to find the actual angle (in radians)		
	float angle = acos(cos);

	// transform angle to units of tabulated array index
	angle *= a->angle_step_inv;

	// tableAngle[0] stores the potential at angle_step
	// tableAngle[1] stores the potential at angle_step * 2, etc.
	// 'home' is the index after which 'convertedAngle' would appear if it were stored in the table	
	int home = int(floor(angle));
	myAssert(home >= 0);
	myAssert(home < a->size);

	// // Make angle the distance from [0,1) from the first index in the potential array index
	// angle -= home;
		
	// Linearly interpolate the potential	
	float U0 = a->pot[home];
	float dUdx = (a->pot[(home+1)] - U0) * a->angle_step_inv;
	// float energy = (dUdx * angle) + U0;
	float sin = sqrtf(1.0f - cos*cos);
	dUdx /= abs(sin) > 1e-2 ? sin : 1e-2; // avoid singularity 

	// Calculate the forces
	Vector3 force1 = -(dUdx*distab) * (ab * (cos*distab) + bc * distbc); // force on particle 1
	Vector3 force3 = (dUdx*distbc) * (bc * (cos*distbc) + ab * distab); // force on particle 3

	// assert( force1.length() < 10000.0f );
	// assert( force3.length() < 10000.0f );
	
	atomicAdd( &force[i], force1 );
	atomicAdd( &force[j], -(force1 + force3) );
	atomicAdd( &force[k], force3 );
}


__device__ inline void computeDihedral(const TabulatedDihedralPotential* __restrict__ d,
				const BaseGrid* __restrict__ sys, Vector3* forces, const Vector3* __restrict__ pos,
				const int& i, const int& j, const int& k, const int& l) {
	const Vector3 posa = pos[i];
	const Vector3 posb = pos[j];
	const Vector3 posc = pos[k];
	const Vector3 posd = pos[l];
		
	const Vector3 ab = sys->wrapDiff(posa - posb);
	const Vector3 bc = sys->wrapDiff(posb - posc);
	const Vector3 cd = sys->wrapDiff(posc - posd);
		
	//const float distab = ab.length();
	const float distbc = bc.length();
	//const float distcd = cd.length();
	
	Vector3 crossABC = ab.cross(bc);
	Vector3 crossBCD = bc.cross(cd);
	Vector3 crossX = bc.cross(crossABC);
	// assert( crossABC.rLength2() <= 1.0f );
	// assert( crossBCD.rLength2() <= 1.0f );

	
	const float cos_phi = crossABC.dot(crossBCD) / (crossABC.length() * crossBCD.length());
	const float sin_phi = crossX.dot(crossBCD) / (crossX.length() * crossBCD.length());
		
	const float angle = -atan2(sin_phi, cos_phi);

	// float energy = 0.0f;
	float force = 0.0f;
	
	Vector3 f1, f2, f3; // forces
	f1 = -distbc * crossABC.rLength2() * crossABC;
	f3 = -distbc * crossBCD.rLength2() * crossBCD;
	f2 = -(ab.dot(bc) * bc.rLength2()) * f1 - (bc.dot(cd) * bc.rLength2()) * f3;
	
	// Shift "angle" by "PI" since    -PI < dihedral < PI
	// And our tabulated potential data: 0 < angle < 2 PI
	float t = (angle + BD_PI) * d->angle_step_inv;
	int home = (int) floorf(t);
	t = t - home;

	myAssert(home >= 0);
	myAssert(home < d->size);
	// home = home % size;
	int home1 = (home + 1) >= d->size ? (home+1-d->size) : home+1;

	//================================================
	// Linear interpolation
	float U0 = d->pot[home];       // Potential
	float dU = d->pot[home1] - U0; // Change in potential
		
	// energy = dU * t + U0;
	force = -dU * d->angle_step_inv;

	// avoid singularity when one angle is straight 
	force = (crossABC.rLength() > 1.0f || crossBCD.rLength() > 1.0f) ? 0.0f : force;
	myAssert( force < 10000.0f );
	f1 *= force;
	f2 *= force;
	f3 *= force;

	// assert( f1.length() < 10000.0f );
	// assert( f2.length() < 10000.0f );
	// assert( f3.length() < 10000.0f );

	atomicAdd( &forces[i], f1 );
	atomicAdd( &forces[j], f2-f1 );
	atomicAdd( &forces[k], f3-f2 );
	atomicAdd( &forces[l], -f3 );
}
