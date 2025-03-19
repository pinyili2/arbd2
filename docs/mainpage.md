# ARBD C++ Library Documentation

Welcome to the documentation for the ARBD (Acceleration of Rigid Body Dynamics) C++ library!

## Overview

The ARBD library provides high-performance molecular dynamics simulations using CUDA acceleration. It is designed to efficiently handle both particle-based and rigid-body simulations for a variety of molecular systems.

## Key Features

- CUDA-accelerated molecular dynamics simulations
- Support for particle-based and rigid body simulations
- Tabulated potentials for flexible interaction modeling
- Grid-based acceleration for large systems
- Various integration methods including Brownian dynamics and Langevin dynamics

## Getting Started

To get started with ARBD, check out the tutorials section which walks through example simulations.

## Module Organization

The ARBD codebase is organized into the following modules:

- **Core**: Main simulation driver classes
- **Particles**: Brownian particle implementations
- **RigidBodies**: Rigid body simulation components
- **Potentials**: Tabulated and other potential implementations
- **Interactions**: Various interaction types (bonds, angles, dihedrals)
- **Grids**: Grid-based acceleration structures
- **Forces**: Force computation routines
- **Utils**: Utility classes and functions
- **IO**: Input/output functionality
- **GPU**: GPU management and cell decomposition

## Examples

The `tests` directory contains several example simulations that demonstrate how to use the library:

- Simple Argon simulation
- Systems with bonded interactions
- Langevin dynamics
- Non-bonded pair interactions

These examples are documented in the Tutorials section.
