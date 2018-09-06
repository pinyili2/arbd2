///////////////////////////////////////////////////////////////////////
// Author: Jeff Comer <jcomer2@illinois.edu>

#include "useful.h"
#include "TabulatedPotential.h"
#include <cuda.h>

TabulatedPotential::TabulatedPotential() {
	n = 2;
	r0 = 0.0f;
	dr = 1.0f;
	r1 = r0 + n*dr;
	
	v0 = new float[n];
	v0[0] = 0.0f;
	v0[1] = 0.0f;
	interpolate();
}

TabulatedPotential::TabulatedPotential(const char* fileName) : fileName(fileName) {
	// printf("File: %s\n", fileName);
	FILE* inp = fopen(fileName, "r");
	if (inp == NULL) {
		printf("TabulatedPotential:TabulatedPotential Could not open file '%s'\n", fileName);
		exit(-1);
	}
	
	char line[256];
	
	numLines = countValueLines(fileName);
	float* r = new float[numLines];
	float* v = new float[numLines];
	
	int count = 0;
	while (fgets(line, 256, inp) != NULL) {
		// Ignore comments.
		int len = strlen(line);
		if (line[0] == '#') continue;
		if (len < 2) continue;
		
		String s(line);
		int numTokens = s.tokenCount();
		if (numTokens != 2) {
			printf("TabulatedPotential:TabulatedPotential Invalid tabulated potential file line: %s\n", line);
			exit(-1);
		}
		
		String* tokenList = new String[numTokens];
		s.tokenize(tokenList);
		if (tokenList == NULL) {
			printf("TabulatedPotential:TabulatedPotential Invalid tabulated potential file line: %s\n", line);
			exit(-1);
		}
		r[count] = (float) strtod(tokenList[0], NULL);
		v[count] = (float) strtod(tokenList[1], NULL);
		count++;
		
		delete[] tokenList;
	}
	fclose(inp);
	init(r, v, count);
	interpolate();
	delete[] r;
	delete[] v;
}

TabulatedPotential::TabulatedPotential(const TabulatedPotential& tab) {
	n = tab.n;
	dr = tab.dr;
	r0 = tab.r0;
	r1 = tab.r1;

	v0 = new float[n];
	v1 = new float[n];
	v2 = new float[n];
	v3 = new float[n];
	for (int i = 0; i < n; i++) {
		v0[i] = tab.v0[i];
		v1[i] = tab.v1[i];
		v2[i] = tab.v2[i];
		v3[i] = tab.v3[i];
	}
	e0 = tab.e0;
}

TabulatedPotential::TabulatedPotential(const float* dist, const float* pot, int n0) {
	init(dist, pot, n0);
	interpolate();
}

TabulatedPotential::~TabulatedPotential() {
	delete[] v0;
	delete[] v1;
	delete[] v2;
	delete[] v3;
}

int TabulatedPotential::countValueLines(const char* fileName) {
	FILE* inp = fopen(fileName, "r");
	char line[256];
	int count = 0;

	while (fgets(line, 256, inp) != NULL) {
		// Ignore comments.
		int len = strlen(line);
		if (line[0] == '#') continue;
		if (len < 2) continue;
      
		count++;
	}
	fclose(inp);

	return count;
}

void TabulatedPotential::truncate(float cutoff) {
	int home = int(floor((cutoff - r0)/dr));
	if (home > n) return;

	float v = v0[home];
	for (int i = home; i < n; i++) v0[i] = v;
	interpolate();
}

bool TabulatedPotential::truncate(float switchDist, float cutoff, float value) {
	int indOff = int(floor((cutoff - r0)/dr));
	int indSwitch = int(floor((switchDist - r0)/dr));

	if (indSwitch > n) return false;

	// Set everything after the cutoff to "value".
	for (int i = indOff; i < n; i++) v0[i] = value;
    
	// Apply a linear switch.
	float v = v0[indSwitch];
	float m = (value - v)/(indOff - indSwitch);
	for (int i = indSwitch; i < indOff; i++) v0[i] = m*(i - indSwitch) + v;

	interpolate();
	return true;
}

Vector3 TabulatedPotential::computeForce(Vector3 r) {
	float d = r.length();
	Vector3 rUnit = -r/d;
	int home = int(floor((d - r0)/dr));

	if (home < 0) return Vector3(0.0f);
	if (home >= n) return Vector3(0.0f);
        
	float homeR = home*dr + r0;
	float w = (d - homeR)/dr;
   
	// Interpolate.
	Vector3 force = -(3.0f*v3[home]*w*w + 2.0f*v2[home]*w + v1[home])*rUnit/dr;
	return force;
}
 
void TabulatedPotential::init(const float* dist, const float* pot, int n0) {
	n = abs(n0);
	dr = dist[1]-dist[0];
	r0 = dist[0];
	r1 = r0 + n*dr;

	v0 = new float[n];
	for (int i = 0; i < n; i++) v0[i] = pot[i];
}
  
void TabulatedPotential::interpolate() {
	v1 = new float[n];
	v2 = new float[n];
	v3 = new float[n];

	for (int i = 0; i < n; i++) {
		int i0 = i - 1;
		int i1 = i;
		int i2 = i + 1;
		int i3 = i + 2;

		if (i0 < 0) i0 = 0;
		if (i2 >= n) i2 = n-1;
		if (i3 >= n) i3 = n-1;

		v3[i] = 0.5f*(-v0[i0] + 3.0f*v0[i1] - 3.0f*v0[i2] + v0[i3]);
		v2[i] = 0.5f*(2.0f*v0[i0] - 5.0f*v0[i1] + 4.0f*v0[i2] - v0[i3]);
		v1[i] = 0.5f*(-v0[i0] + v0[i2]);
	}
	e0 = v3[n-1] + v2[n-1] + v1[n-1] + v0[n-1];
}
 
