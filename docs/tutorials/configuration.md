# ARBD Configuration Reference

This tutorial explains the configuration parameters used in ARBD `.bd` files.

## Basic Parameters

- `seed <number>`: Random number generator seed
- `timestep <value>`: Simulation timestep
- `steps <number>`: Total number of simulation steps
- `temperature <value>`: Simulation temperature (K)
- `ParticleDynamicType <type>`: Integration method (Brownian, Langevin, etc.)

## Output Settings

- `outputPeriod <number>`: How frequently to save trajectory frames
- `outputEnergyPeriod <number>`: How frequently to save energy data
- `outputFormat <format>`: Output file format (default: dcd)

## System Settings

- `systemSize <x> <y> <z>`: Simulation box dimensions
- `cutoff <value>`: Cutoff distance for non-bonded interactions
- `pairlistDistance <value>`: Extra distance for neighborlist construction
- `decompPeriod <number>`: How frequently to update domain decomposition

## Particle Definitions

Each particle type is defined with:

```
particle <name>
num <count>
diffusion <value>  # For Brownian dynamics
mass <value>       # For Langevin dynamics
transDamping <x> <y> <z>  # For Langevin dynamics
gridFile <filename>  # External potential grid
```

## Interactions

- `tabulatedPotential <0/1>`: Enable tabulated potentials
- `tabulatedFile <i>@<j>@<filename>`: Define interaction between particle types i and j

### Bonded Interactions

- `inputBonds <filename>`: File with bond definitions
- `inputAngles <filename>`: File with angle definitions
- `inputDihedrals <filename>`: File with dihedral definitions
- `inputExcludes <filename>`: File with exclusion definitions

## Advanced Settings

- `fullLongRange <0/1>`: Enable full long-range interactions
- `interparticleForce <0/1>`: Enable inter-particle forces
- `numberFluct <0/1>`: Enable number fluctuations

For more details, refer to the specific examples in the Examples section.
