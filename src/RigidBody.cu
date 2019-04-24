#include <iostream>
#include <typeinfo>
#include "RigidBody.h"
#include "RigidBodyType.h"
#include "RigidBodyController.h"
#include "Configuration.h"
#include "ComputeGridGrid.cuh"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>

#include "Debug.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), __FILE__, line);
      if (abort) exit(code);
   }
}


RigidBody::RigidBody(String name, const Configuration& cref, const RigidBodyType& tref, RigidBodyController* RBCref) 
    : name(name), c(&cref), t(&tref), RBC(RBCref), impulse_to_momentum(4.1867999435271e4) /*impulse_to_momentum(4.184e8f)*/ { init(); }
RigidBody::RigidBody(const RigidBody& rb)
    : name(rb.name), c(rb.c), t(rb.t), RBC(rb.RBC), impulse_to_momentum(4.1867999435271e4)/*impulse_to_momentum(4.184e8f)*/ { init(); }
void RigidBody::init() {
	// units "(kcal_mol/AA) * ns" "dalton AA/ns" * 4.184e+08	
	timestep = c->timestep;
	Temp = c->temperature * 0.0019872065f;
	// RBTODO: use temperature grids
	// tempgrid = c->temperatureGrid;
	position = t->initPos;

	// Orientation matrix that brings vector from the RB frame to the lab frame
	orientation = t->initRot;

        momentum = t->initMomentum;
        angularMomentum = t->initAngularMomentum;

	// Memory allocation for forces between particles and grids 
	const int& numGrids = t->numPotGrids;
	if (numGrids > 0) {
	    numParticles = new int[numGrids];
	    particles_d = new int*[numGrids];
	    particleForceStreams = new const cudaStream_t*[numGrids];

	    for (int i = 0; i < numGrids; ++i) {
		numParticles[i] = -1;
		const int& n = t->numParticles[i];
		if (n > 0) {
		    // gpuErrchk(cudaMalloc( &particles_d[i], 0.5*sizeof(int)*n )); // not sure why 0.5 was here; prolly bug
		    gpuErrchk(cudaMalloc( &particles_d[i], sizeof(int)*n )); // TODO: dynamically allocate memory as needed
		}
	    }
	} else {
	    numParticles = NULL;
	}
}

GPUManager RigidBody::gpuman = GPUManager();

//Boltzmann distribution
void RigidBody::Boltzmann(unsigned long int seed)
{

    gsl_rng *gslcpp_rng = gsl_rng_alloc(gsl_rng_default);
    //std::srand(time(NULL));
    gsl_rng_set (gslcpp_rng, seed);

    double sigma[4] = { sqrt(t->mass*Temp) * 2.046167135,sqrt(t->inertia.x*Temp) * 2.046167135, sqrt(t->inertia.y*Temp) * 2.046167135, sqrt(t->inertia.z*Temp) * 2.046167135 };

    //Vector3 rando = getRandomGaussVector();
    momentum = Vector3(gsl_ran_gaussian(gslcpp_rng,sigma[0]),gsl_ran_gaussian(gslcpp_rng,sigma[0]), gsl_ran_gaussian(gslcpp_rng,sigma[0]));

    angularMomentum.x = gsl_ran_gaussian(gslcpp_rng,sigma[1]);
    angularMomentum.y = gsl_ran_gaussian(gslcpp_rng,sigma[2]);
    angularMomentum.z = gsl_ran_gaussian(gslcpp_rng,sigma[3]);
    printf("%f\n", Temp);
    printf("%f\n", Temperature());
    gsl_rng_free(gslcpp_rng);
}

RigidBody::~RigidBody() {
	const int& numGrids = t->numPotGrids;
	for (int i = 0; i < numGrids; ++i) {
		const int& n = t->numParticles[i];
		if (n > 0) {
			gpuErrchk(cudaFree( particles_d[i] ));
		}
	}
	if (numParticles != NULL) {
		delete[] numParticles;
		delete[] particles_d;
		delete[] particleForceStreams;
	}
}

int RigidBody::appendNumParticleBlocks( std::vector<int>* blocks ) {
    int ret = 0;
    const int& numGrids = t->numPotGrids;
    for (int i = 0; i < numGrids; ++i) {
	numParticles[i] = -1;
	const int& n = t->numParticles[i];
	const int nb = (n/NUMTHREADS)+1; // max number of blocks
	if (n > 0) {
	    blocks->push_back(nb);
	    ret += nb;
	}
    }
    return ret;
}

void RigidBody::addForce(Force f) { 
	force += f; 
} 
void RigidBody::addTorque(Force torq) {
	torque += torq; 
}
void RigidBody::addEnergy(float e)
{
    energy += e;
}
void RigidBody::updateParticleList(Vector3* pos_d, BaseGrid* sys_d) {
	for (int i = 0; i < t->numPotGrids; ++i) {
		numParticles[i] = 0;
		int& tnp = t->numParticles[i];
		if (tnp > 0) {
		    int idx = t->potential_grid_idx[i];
			Vector3 gridCenter = t->RBC->grids[idx].getCenter();
			float cutoff = gridCenter.length();
			cutoff += t->RBC->grids[idx].getRadius();
			cutoff += c->pairlistDistance; 
		   
			int* tmp_d;
			gpuErrchk(cudaMalloc( &tmp_d, sizeof(int) ));
			gpuErrchk(cudaMemcpy( tmp_d, &numParticles[i], sizeof(int), cudaMemcpyHostToDevice ));

			int nb = floor(tnp/NUMTHREADS) + 1;
#if __CUDA_ARCH__ >= 300
			createPartlist<<<nb,NUMTHREADS>>>(pos_d, tnp, t->particles_d[i],
							tmp_d, particles_d[i],
							gridCenter + position, cutoff*cutoff, sys_d);
#else
			createPartlist<<<nb,NUMTHREADS,NUMTHREADS/WARPSIZE>>>(pos_d, tnp, t->particles_d[i],
							tmp_d, particles_d[i],
							gridCenter + position, cutoff*cutoff, sys_d);
#endif			
			gpuErrchk(cudaMemcpy(&numParticles[i], tmp_d, sizeof(int), cudaMemcpyDeviceToHost ));
			gpuErrchk(cudaFree( tmp_d ));
		}
	}
}

void RigidBody::callGridParticleForceKernel(Vector3* pos_d, Vector3* force_d, int s, float* energy, bool get_energy, int scheme, BaseGrid* sys, BaseGrid* sys_d, ForceEnergy* forcestorques_d, const std::vector<int>& forcestorques_offset, int& fto_idx) {
	// Apply the force and torque on the rigid body, and forces on particles
	
	// RBTODO: performance: consolidate CUDA stream management
	// loop over potential grids 
	for (int i = 0; i < t->numPotGrids; ++i) {
		if (numParticles[i] <= 0) continue;
		// const int nb = 500;
		/*
		  r: postion of particle in real space
		  B: grid Basis
		  o: grid origin
		  R: rigid body orientation
		  c: rigid body center

		  B': R.B 
		  c': R.o + c
		*/

		const cudaStream_t& stream = gpuman.get_next_stream();
		particleForceStreams[i] = &stream;

		int idx = t->potential_grid_idx[i];

		Vector3 c =  getOrientation()*t->RBC->grids[idx].getOrigin() + getPosition();
		Matrix3 B = (getOrientation()*t->RBC->grids[idx].getBasis()).inverse();
		
		const int nb = (numParticles[i]/NUMTHREADS)+1;		
		computePartGridForce<<< nb, NUMTHREADS, NUMTHREADS*2*sizeof(ForceEnergy), stream >>>(
			pos_d, force_d, numParticles[i], particles_d[i],
			&t->RBC->grids_d[idx],
			B, c, forcestorques_d+forcestorques_offset[fto_idx++], energy, get_energy, scheme, sys_d);
	}
}

void RigidBody::applyGridParticleForces(BaseGrid* sys, ForceEnergy* forcestorques, const std::vector<int>& forcestorques_offset, int& fto_idx) {
	// loop over potential grids 
	for (int i = 0; i < t->numPotGrids; ++i) {
		if (numParticles[i] <= 0) continue;
		const int nb = (numParticles[i]/NUMTHREADS)+1;
		int idx = t->potential_grid_idx[i];
		Vector3 c =  getOrientation()*t->RBC->grids[idx].getOrigin() + getPosition();

		// Sum and apply forces and torques
		//Vector3 f = Vector3(0.0f);
		ForceEnergy f = ForceEnergy(0.f,0.f);
		Vector3 torq = Vector3(0.0f);
		for (int k = 0; k < nb; ++k) {
		    int j = forcestorques_offset[fto_idx]+2*k;
		    f = f + forcestorques[j];
		    torq = torq + forcestorques[j+1].f;
		}
		++fto_idx;
	        //why the force points are at the origin of the potential?	
		torq = -torq + (sys->wrapDiff(getPosition()-c)).cross( f.f ); 
		addForce( -f.f );
		addTorque( torq );
                addEnergy( f.e );
	}
}

	/*===========================================================================\
	| Following "Algorithm for rigid-body Brownian dynamics" Dan Gordon, Matthew |
	|   Hoyles, and Shin-Ho Chung                                                |
	|   http://langevin.anu.edu.au/publications/PhysRevE_80_066703.pdf           |
	|                                                                            |
	|                                                                            |
	| BUT: assume diagonal friction tensor and no Wiener process / stochastic    |
	|   calculus then this is just the same as for translation                   |
	|                                                                            |
	|   < T_i(t) T_i(t) > = 2 kT friction inertia                                |
	|                                                                            |
	|   friction / kt = Diff                                                     |
	\===========================================================================*/
void RigidBody::addLangevin(Vector3 w1, Vector3 w2) 
{
    Vector3 transForceCoeff = Vector3::element_sqrt( 2. * Temp * t->mass*t->transDamping / timestep );
    Vector3  rotTorqueCoeff = Vector3::element_sqrt( 2. * Temp * Vector3::element_mult( t->inertia,t->rotDamping) / timestep );

    Force f = Vector3::element_mult(transForceCoeff,w1) -
              Vector3::element_mult(t->transDamping, orientation.transpose()*momentum) * 10000;
    
    Force torq = Vector3::element_mult(rotTorqueCoeff,w2) -
                 Vector3::element_mult(t->rotDamping, angularMomentum) * 10000;

    f = orientation * f;
    torq = orientation * torq;

    addForce(f);
    addTorque(torq);
}

  /*==========================================================================\
	| from: Dullweber, Leimkuhler, Maclachlan. Symplectic splitting methods for |
	| rigid body molecular dynamics. JCP 107. (1997)                            |
	| http://jcp.aip.org/resource/1/jcpsa6/v107/i15/p5840_s1                    |
	\==========================================================================*/
void RigidBody::integrateDLM(int startFinishAll) 
{
    Vector3 trans; // = *p_trans;
    //Matrix3 rot = Matrix3(1); // = *p_rot;

    if ( isnan(force.x) || isnan(torque.x) ) 
    {   
        // NaN check
        printf("Rigid Body force or torque was NaN!\n");
        exit(-1);
    }

    if (startFinishAll == 0 || startFinishAll == 2) 
    {
        // propogate momenta by half step
        momentum += 0.5f * timestep * force * impulse_to_momentum;
        angularMomentum += 0.5f * timestep * (orientation.transpose()*torque) * impulse_to_momentum;
    } 
    else if (startFinishAll == 1)
    {
        position += timestep * momentum / t->mass * 1e4; // update CoM a full timestep
        // update orientations a full timestep
        Matrix3 R; // represents a rotation about a principle axis
        R = Rx(0.5*timestep * angularMomentum.x / t->inertia.x * 1e4); // R1
        applyRotation(R);

        R = Ry(0.5*timestep * angularMomentum.y / t->inertia.y * 1e4); // R2
        applyRotation(R);
                        
        R = Rz(    timestep * angularMomentum.z / t->inertia.z * 1e4); // R3
        applyRotation(R);
                        
        R = Ry(0.5*timestep * angularMomentum.y / t->inertia.y * 1e4); // R4
        applyRotation(R);

        R = Rx(0.5*timestep * angularMomentum.x / t->inertia.x * 1e4); // R5
        applyRotation(R);               
        // TODO make this periodic
        // printf("det: %.12f\n", orientation.det());
        orientation = orientation.normalized();
        // orientation = orientation/orientation.det();
        // printf("det2: %.12f\n", orientation.det());
        // orientation = orientation/orientation.det(); // TODO: see if this can be somehow eliminated (wasn't in original DLM algorithm...)
    }
}
/* Following:
Brownian Dynamics Simulation of Rigid Particles of Arbitrary Shape in External Fields
Miguel X. Fernandes, José García de la Torre
*/

//Chris original implementation for Brownian motion
void RigidBody::integrate(int startFinishAll)
{
    // UNITS
    // Temp: kcal_mol
    // t->transDamping: (kcal_mol/AA) / (amu AA/ns)
    // t->mass: amu
    // diffusion: AA**2/ns

    //if (startFinishAll == 1) return;

    //Matrix3 rot = Matrix3(1); // = *p_rot;

    if ( isnan(force.x) || isnan(torque.x) ) 
    {
        printf("Rigid Body force or torque was NaN!\n");
        exit(-1);
    }
    //float Temp = 1;
    Vector3 diffusion    = Temp / (t->transDamping*t->mass); // TODO: assign diffusion in config file, or elsewhere
    //Vector3 diffusion    = Temp / (t->transDamping*t->mass);
    Vector3 rotDiffusion = Temp / (Vector3::element_mult(t->rotDamping,t->inertia));

    Vector3 rando  = getRandomGaussVector();
    Vector3 offset = Vector3::element_mult( (diffusion / Temp), orientation.transpose() * force ) * timestep +
                     Vector3::element_mult( Vector3::element_sqrt( 2.0f * diffusion * timestep), rando) ;

    position += orientation*offset;

    rando = getRandomGaussVector();
    Vector3 rotationOffset = Vector3::element_mult( (rotDiffusion / Temp) , orientation.transpose() * torque * timestep) +
                             Vector3::element_mult( Vector3::element_sqrt( 2.0f * rotDiffusion * timestep), rando );

    // Consider whether a DLM-like decomposition of rotations is needed for time-reversibility
    orientation = orientation * (Rz(rotationOffset.z * 0.5) * Ry(rotationOffset.y * 0.5) * Rx(rotationOffset.x)
                              *  Ry(rotationOffset.y * 0.5) * Rz(rotationOffset.z * 0.5));
    //orientation = orientation * Rz(rotationOffset.z) * Ry(rotationOffset.y) * Rx(rotationOffset.x);
    orientation = orientation.normalized();
}
 
float RigidBody::Temperature()
{
    return (momentum.length2() / t->mass + 
            angularMomentum.x * angularMomentum.x / t->inertia.x + 
            angularMomentum.y * angularMomentum.y / t->inertia.y + 
            angularMomentum.z * angularMomentum.z / t->inertia.z) * 0.50 / Temp * (2.388458509e-1);
}

void RigidBody::applyRotation(const Matrix3& R) {
	angularMomentum = R * angularMomentum;
	// According to DLM, but rotations work the wrong way; I think DLM update is wrong
	// orientation = orientation * R.transpose(); 

	// This makes sense: apply a rotation in the body frame followed by a transformation from body to lab frame
	// Also works in statistical test
	// Consistent with www.archer.ac.uk/documentation/white-papers/lammps-elba/lammps-ecse.pdf
	orientation = orientation * R; 
        orientation.normalized();	
}

// Rotations about axes
// for very small angles 10^-8, cos^2+sin^2 != 1 
// concerned about the accumulation of errors in non-unitary transformations!
Matrix3 RigidBody::Rx(BigReal t) {
	BigReal qt = 0.25*t*t;  // for approximate calculations of sin(t) and cos(t)
	BigReal cos = (1-qt)/(1+qt);
	BigReal sin = t/(1+qt);

	return Matrix3(
		1.0f, 0.0f, 0.0f,
		0.0f,  cos, -sin,
		0.0f,  sin,  cos);
}
Matrix3 RigidBody::Ry(BigReal t) {
	BigReal qt = 0.25*t*t;  // for approximate calculations of sin(t) and cos(t)
	BigReal cos = (1-qt)/(1+qt);
	BigReal sin = t/(1+qt);

	return Matrix3(
		cos,  0.0f,  sin,
		0.0f, 1.0f, 0.0f,
		-sin, 0.0f,  cos);
}
Matrix3 RigidBody::Rz(BigReal t) {
	BigReal qt = 0.25*t*t;  // for approximate calculations of sin(t) and cos(t)
	BigReal cos = (1-qt)/(1+qt);
	BigReal sin = t/(1+qt);

	return Matrix3(
		cos,  -sin, 0.0f,
		sin,   cos, 0.0f,
		0.0f, 0.0f, 1.0f);
}
Matrix3 RigidBody::eulerToMatrix(const Vector3 e) {
	// convert euler angle input to rotation matrix
	// http://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Conversion_formulae_between_formalisms
	return Rz(e.z) * Ry(e.y) * Rx(e.x);
}
