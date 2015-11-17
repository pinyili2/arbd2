// Configuration.h (2013)
// Loads .brown config file that can be shared between simulations
// To be used by GrandBrownTown to initialize its members
//
// Authors: Terrance Howward <howard33@illinois.edu>
//          Justin Dufresne <jdufres1@friars.providence.edu>
//

#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <algorithm> // sort
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "useful.h" // String, Vector3
#include "BrownianParticleType.h"
#include "BaseGrid.h"
#include "OverlordGrid.h"
#include "ComputeForce.h"
#include "Angle.h"
#include "Reader.h"
#include "TrajectoryWriter.h"
#include "TabulatedPotential.h"
#include "TabulatedAngle.h"
#include "GPUManager.h"
#include "Dihedral.h"
#include "RigidBodyType.h"
#include "RigidBody.h"

#include <cuda.h> 
#include <cuda_runtime.h>

class Configuration {
	struct compare {
		bool operator()(const String& lhs, const String& rhs);
		bool operator()(const Bond& lhs, const Bond& rhs);
		bool operator()(const Exclude& lhs, const Exclude& rhs);
		bool operator()(const Angle& lhs, const Angle& rhs);
		bool operator()(const Dihedral& lhs, const Dihedral& rhs);
	};

	void setDefaults();
	Vector3 stringToVector3(String s);


	int readParameters(const char* config_file);
	void readAngles();
	void readAtoms();
	void readBonds();
	void readExcludes();
	void readDihedrals();

	bool readTableFile(const String& value, int currTab);
	bool readBondFile(const String& value, int currBond);
	bool readAngleFile(const String& value, int currAngle);
	bool readDihedralFile(const String& value, int currDihedral);

	// Given the numbers of each particle, populate the type list.
	void populate();

	void loadRestart(const char* file_name);
	bool loadCoordinates(const char* file_name);
	int countRestart(const char* file_name);

	void getDebugForce();

public:
	Configuration(const char * config_file, int simNum = 0, bool debug=false);
	~Configuration();

	void copyToCUDA();

	// Output variables
	Vector3 sysDim;
	BaseGrid* sys;

	bool loadedCoordinates;

	// Device Variables
	int *type_d;
	BrownianParticleType **part_d;
	RigidBodyType **rbType_d;
	BaseGrid *sys_d, *kTGrid_d;
	Bond* bonds_d;
	int2* bondMap_d;
	Exclude* excludes_d;
	int2* excludeMap_d;
	Angle* angles_d;
	Dihedral* dihedrals_d;

	// number of simulations
	int simNum;

	// Particle variables
	String* partsFromFile;
	int* indices;
	int numPartsFromFile;
	Bond* bonds;
	int numCap; // max number of particles
	int num; // current number of particles
	Vector3* pos; //  position of each particle
	int* type; // type of each particle
	int* serial; // serial number of each particle
	int currSerial; // the serial number of the next new particle
	String* name; // name of each particle
	Vector3* posLast; // used for current computation
	float timeLast; // used with posLast
	float minimumSep; // minimum separation allowed with placing new particles

	// RigidBody variables
	/* int numRB; */
	/* std::vector< std::vector<RigidBody> > rbs; */
	
	// System parameters
	String outputName;
	float timestep;
	long int steps;
	long int seed;
	// String kTGridFile;
	String temperatureGrid;
	String inputCoordinates;
	String restartCoordinates;
	int numberFluct;
	int interparticleForce;
	int tabulatedPotential;
	int fullLongRange;
	float kT;
	float coulombConst;
	float electricField;
	float cutoff;
	float switchLen;
	int outputPeriod;
	int outputEnergyPeriod;
	int outputFormat;
	float currentSegmentZ;
	int numberFluctPeriod;
	int decompPeriod;
	int numCapFactor;
	BaseGrid* kTGrid;
	BaseGrid* tGrid;
	BaseGrid* sigmaT;
	unsigned long randoSeed;

	// Other parameters.
	float switchStart;
	float maxInitialPot;
	float initialZ;

	// Particle parameters.
	BrownianParticleType* part;
	int numParts;
	int numBonds;
	int numExcludes;
	int numAngles;
	int numDihedrals;
	String partFile;
	String bondFile;
	String excludeFile;
	String angleFile;
	String dihedralFile;
	bool readPartsFromFile;
	bool readBondsFromFile;
	bool readExcludesFromFile;
	bool readAnglesFromFile;
	bool readDihedralsFromFile;
	String* partGridFile;
	String* partDiffusionGridFile;
	String* partForceXGridFile;
	String* partForceYGridFile;
	String* partForceZGridFile;
	String* partTableFile;
	String* partReservoirFile;
	int* partTableIndex0;
	int* partTableIndex1;

	String* bondTableFile;
	int numTabBondFiles;
	int2* bondMap;

	Exclude* excludes;
	int2* excludeMap;
	String excludeRule;
	int excludeCapacity;

	Angle* angles;
	String* angleTableFile;
	int numTabAngleFiles;

	Dihedral* dihedrals;
	String* dihedralTableFile;
	int numTabDihedralFiles;

	// RigidBody parameters.
	RigidBodyType* rigidBody;
	int numRigidTypes;

};

#endif
