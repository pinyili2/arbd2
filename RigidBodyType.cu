#include "RigidBodyType.h"

void RigidBodyType::clear() {
	num = 0;											// TODO: not 100% sure about this
	if (reservoir != NULL) delete reservoir;
	reservoir = NULL;
	// pmf = NULL;
	mass = 1.0;

	// TODO: make sure that this actually removes grid data
	potentialGrids.clear();
	densityGrids.clear();

	if (numPotGrids > 0) delete[] rawPotentialGrids;
	if (numDenGrids > 0) delete[] rawDensityGrids;
	rawPotentialGrids = NULL;
	rawDensityGrids = NULL;
}


// void RigidBodyType::copy(const RigidBodyType& src) {
// 	this = new RigidBodyType(src.name);
// 	num = src.num;
// 	// if (src.pmf != NULL) pmf = new BaseGrid(*src.pmf);
// 	if (src.reservoir != NULL) reservoir = new Reservoir(*src.reservoir);

// 	numPotentialGrid = src.numPotentialGrid;
// 																// TODO: fix this: BaseGrid*[]
// 	for (int i=0; i < numPotentialGrid; i++) {
		
// 	}	
	
// 	numDensityGrid = src.numDensityGrid;
// }

// RigidBodyType& RigidBodyType::operator=(const RigidBodyType& src) {
// 	clear();
// 	copy(src);
// 	return *this;
// }


// KeyGrid RigidBodyType::createKeyGrid(String s) {
// 	// tokenize and return
// 	int numTokens = s.tokenCount();
// 	if (numTokens != 2) {
// 		printf("ERROR: could not add Grid.\n"); // TODO improve this message
// 		exit(1);
// 	}
// 	String* token = new String[numTokens];
// 	s.tokenize(token);
// 	KeyGrid g;
// 	g.key = token[0];
// 	g.grid = * new BaseGrid(token[1]);
// 	return g;
// }
void RigidBodyType::addPotentialGrid(String s) {
	// tokenize and return
	int numTokens = s.tokenCount();
	if (numTokens != 2) {
		printf("ERROR: could not add Grid.\n"); // TODO improve this message
		exit(1);
	}
	String* token = new String[numTokens];
	s.tokenize(token);
	String key = token[0];
	BaseGrid g(token[1]);
	
	potentialGrids.push_back( g );
	// 	[numPotGrids] = g;
	// numPotGrids++;
}
void RigidBodyType::addDensityGrid(String s) {
	// tokenize and return
	int numTokens = s.tokenCount();
	if (numTokens != 2) {
		printf("ERROR: could not add Grid.\n"); // TODO improve this message
		exit(1);
	}
	String* token = new String[numTokens];
	s.tokenize(token);
	String key = token[0];
	BaseGrid g(token[1]);
	
	densityGrids.push_back( g );
}

void RigidBodyType::updateRaw() {
	if (numPotGrids > 0) delete[] rawPotentialGrids;
	if (numDenGrids > 0) delete[] rawDensityGrids;
	numPotGrids = potentialGrids.size();
	numDenGrids = densityGrids.size();
	if (numPotGrids > 0)
		rawPotentialGrids = new BaseGrid[numPotGrids];
	if (numDenGrids > 0)
		rawDensityGrids = new BaseGrid[numDenGrids];

	for (int i=0; i < numPotGrids; i++)
		rawPotentialGrids[i] = potentialGrids[i];
	for (int i=0; i < numDenGrids; i++)
		rawDensityGrids[i] = densityGrids[i];
	
}

