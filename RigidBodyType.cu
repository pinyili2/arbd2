#include "RigidBodyType.h"

void RigidBodyType::clear() {
	num = 0;											// RBTODO: not 100% sure about this
	if (reservoir != NULL) delete reservoir;
	reservoir = NULL;
	// pmf = NULL;
	mass = 1.0;

	// TODO: make sure that this actually removes grid data
	potentialGrids.clear();
	densityGrids.clear();
	pmfs.clear();
	
	potentialGridKeys.clear();
	densityGridKeys.clear();
	pmfKeys.clear();
	
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


// void RigidBodyType::setDampingCoeffs(float timestep, float tmp_mass, Vector3 tmp_inertia, float tmp_transDamping, float tmp_rotDamping) {
void RigidBodyType::setDampingCoeffs(float timestep) { /* MUST ONLY BE CALLED ONCE!!! */
	/*–––––––––––––––––––––––––––––––––––––––––––––––––––––––.
	| DiffCoeff = kT / dampingCoeff mass                     |
	|                                                        |
	| type->DampingCoeff has units of (1/ps)                 |
	|                                                        |
	| f[kcal/mol AA] = - dampingCoeff * momentum[amu AA/fs]  |
	|                                                        |
	| units "(1/ps) * (amu AA/fs)" "kcal_mol/AA" * 2.3900574 |
	`–––––––––––––––––––––––––––––––––––––––––––––––––––––––*/

	/*––––––––––––––––––––––––––––––––––––––––––––––––––––.
	| < f(t) f(t') > = 2 kT dampingCoeff mass delta(t-t') |
	|                                                     |
	|  units "sqrt( k K (1/ps) amu / fs )" "kcal_mol/AA"  |
	|    * 0.068916889                                    |
	`––––––––––––––––––––––––––––––––––––––––––––––––––––*/
	float Temp = 295; /* RBTODO: temperature should be read from grid? Or set in uniformly in config file */
	transForceCoeff = 0.068916889 * Vector3::element_sqrt( 2*Temp*mass*transDamping/timestep );

	// setup for langevin
	// langevin = rbParams->langevin;
	// if (langevin) {
	// T = - dampingCoeff * angularMomentum

	// < f(t) f(t') > = 2 kT dampingCoeff inertia delta(t-t')
	rotTorqueCoeff = 0.068916889 *
		Vector3::element_sqrt( 2*Temp* Vector3::element_mult(inertia,rotDamping) / timestep );


	transDamping = 2.3900574 * transDamping;
	rotDamping = 2.3900574 * rotDamping;

	// Also apply scale factors
	applyScaleFactors();
}
void RigidBodyType::applyScaleFactors() { /* currently this is called before raw is updated */
	applyScaleFactor(potentialGridScaleKeys, potentialGridScale, potentialGridKeys, potentialGrids);
	applyScaleFactor(densityGridScaleKeys, densityGridScale, densityGridKeys, densityGrids);
	applyScaleFactor(pmfScaleKeys, pmfScale, pmfKeys, pmfs);
}
void RigidBodyType::applyScaleFactor(
	const std::vector<String> &scaleKeys, const std::vector<float> &scaleFactors,
	const std::vector<String> &gridKeys, std::vector<BaseGrid> &grids) {
	for (int i = 0; i < scaleKeys.size(); i++) {
		const String &k1 = scaleKeys[i];
		for (int j = 0; j < gridKeys.size(); j++) {
			const String &k2 = gridKeys[j];
			if ( k1 == k2 )
				grids[j].scale( scaleFactors[i] );
		}
	}
}
	
void RigidBodyType::addGrid(String s, std::vector<String> &keys, std::vector<BaseGrid> &grids) {
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

	keys.push_back( key );
	grids.push_back( g );
}
void RigidBodyType::addPotentialGrid(String s) {
	addGrid(s, potentialGridKeys, potentialGrids);
}
void RigidBodyType::addDensityGrid(String s) {
	addGrid(s, densityGridKeys, densityGrids);
}
void RigidBodyType::addPMF(String s) {
	addGrid(s, pmfKeys, pmfs);
}

void RigidBodyType::addScaleFactor(String s, std::vector<String> &keys, std::vector<float> &vals) {
	// tokenize and return
	int numTokens = s.tokenCount();
	if (numTokens != 2) {
		printf("ERROR: could not add Grid.\n"); // TODO improve this message
		exit(1);
	}
	String* token = new String[numTokens];
	s.tokenize(token);
	String key = token[0];
	float v = (float) strtod(token[1], NULL);
	keys.push_back( key );
	vals.push_back( v );
}
void RigidBodyType::scalePotentialGrid(String s) {
	addScaleFactor(s, potentialGridScaleKeys, potentialGridScale);
}
void RigidBodyType::scaleDensityGrid(String s) {
	addScaleFactor(s, densityGridScaleKeys, densityGridScale);
}
void RigidBodyType::scalePMF(String s) {
	addScaleFactor(s, pmfScaleKeys, pmfScale);
}

void RigidBodyType::updateRaw() {
	if (numPotGrids > 0) delete[] rawPotentialGrids;
	if (numDenGrids > 0) delete[] rawDensityGrids;
	if (numDenGrids > 0) delete[] rawPmfs;
	numPotGrids = potentialGrids.size();
	numDenGrids = densityGrids.size();
	numPmfs = pmfs.size();
	
	if (numPotGrids > 0) {
		rawPotentialGrids		= new RigidBodyGrid[numPotGrids];
		rawPotentialBases		= new Matrix3[numPotGrids];
		rawPotentialOrigins = new Vector3[numPotGrids];
	}
	if (numDenGrids > 0) {
		rawDensityGrids			= new RigidBodyGrid[numDenGrids];
		rawDensityBases			= new Matrix3[numDenGrids];
		rawDensityOrigins		= new Vector3[numDenGrids];
	}
	if (numPmfs > 0)
		rawPmfs = new BaseGrid[numPmfs];

	for (int i=0; i < numPotGrids; i++) {
		rawPotentialGrids[i]	 = potentialGrids[i];
		rawPotentialBases[i]	 = potentialGrids[i].getBasis();
		rawPotentialOrigins[i] = potentialGrids[i].getOrigin();
	}
	for (int i=0; i < numDenGrids; i++) {
		rawDensityGrids[i]		 = densityGrids[i];
		rawDensityBases[i]		 = densityGrids[i].getBasis();
		rawDensityOrigins[i]	 = densityGrids[i].getOrigin();
	}
	for (int i=0; i < numPmfs; i++)
		rawPmfs[i] = pmfs[i];	
}

