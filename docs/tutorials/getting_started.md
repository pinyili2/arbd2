# Getting Started with ARBD

This tutorial will guide you through the basics of using the ARBD library for molecular dynamics simulations.

## Prerequisites

Before you begin, make sure you have:

1. Compiled the ARBD library successfully
2. CUDA toolkit installed (recommended version: 10.0 or newer)
3. Basic understanding of molecular dynamics concepts

## Basic Simulation Structure

All ARBD simulations follow this general structure:

1. Define a simulation configuration (`.bd` file)
2. Prepare input coordinates and parameters
3. Run the simulation using the `arbd` executable
4. Analyze the results with post-processing scripts

## Example: Simple Argon Simulation

Let's walk through a simple Argon simulation as an example.

### 1. Configuration File

The simulation is defined in a `.bd` file. Here's an example from the argon-small test:

```
seed 20101992
timestep 0.01
steps 10000
interparticleForce 1
fullLongRange 0
temperature 120.0
ParticleDynamicType Brownian

outputPeriod 1000
outputEnergyPeriod 1000

decompPeriod 40
cutoff 2.5
pairlistDistance 0.5

systemSize 5 5 5

particle argon
num 100
diffusion 1
gridFile null.dx
```

This configuration:
- Sets a random seed
- Defines timestep and number of steps
- Specifies Brownian dynamics
- Sets output frequency
- Defines system size and parameters
- Creates 100 argon particles

### 2. Running the Simulation

Use the `arbd` executable to run the simulation:

```bash
./arbd BrownDyn.bd output/arbd
```

This will run the simulation and output files to the `output/arbd` directory.

### 3. Analyzing Results

The output includes trajectory files that can be analyzed with VMD or custom scripts.

## Next Steps

Check out the other examples in the Examples section to learn about:

- Bonded interactions
- Langevin dynamics
- Non-bonded pair interactions

Each example includes a complete configuration file and analysis scripts.
