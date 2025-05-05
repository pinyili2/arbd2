#!/bin/bash

# Base directories
ORIGINAL_DIR="src"
NEW_DIR="src_refactored"

# Create new directory structure
echo "Creating new directory structure..."
mkdir -p "${NEW_DIR}"/{Core/{Types,Memory,Random},Objects/Particles,Simulation/{Engine,Compute,Integrator,Interactions},Patch,Backend/{CPU,GPU,MPI},Utils/{System,Misc}}

# Core directory
echo "Organizing Core directory..."
mv "${ORIGINAL_DIR}/core/Types"/* "${NEW_DIR}/Core/Types/"
mv "${ORIGINAL_DIR}/core/Types.h" "${NEW_DIR}/Core/Types/"
mv "${ORIGINAL_DIR}/core/Proxy.h" "${NEW_DIR}/Core/Memory/"
mv "${ORIGINAL_DIR}/core/MemoryManager.h" "${NEW_DIR}/Core/Memory/"
mv "${ORIGINAL_DIR}/core/Resource.h" "${NEW_DIR}/Backend/"
mv "${ORIGINAL_DIR}/core/Resource.cu" "${NEW_DIR}/Backend/"
mv "${ORIGINAL_DIR}/core/Resource.bak.h" "${NEW_DIR}/Backend/"
mv "${ORIGINAL_DIR}/core/Resource.h.bak" "${NEW_DIR}/Backend/"
mv "${ORIGINAL_DIR}/core/Resource.cu.bak" "${NEW_DIR}/Backend/"
mv "${ORIGINAL_DIR}/core/GPUManager.h" "${NEW_DIR}/Backend/GPU/"
mv "${ORIGINAL_DIR}/core/GPUManager.cpp" "${NEW_DIR}/Backend/GPU/"
mv "${ORIGINAL_DIR}/core/SignalManager.h" "${NEW_DIR}/Utils/System/"
mv "${ORIGINAL_DIR}/core/SignalManager.cpp" "${NEW_DIR}/Utils/System/"

# Move Random files
mv "${ORIGINAL_DIR}/random/Random.h" "${NEW_DIR}/Core/Random/"
mv "${ORIGINAL_DIR}/random/Random.cu" "${NEW_DIR}/Core/Random/"
mv "${ORIGINAL_DIR}/random/Random_old.h" "${NEW_DIR}/Core/Random/"

# Move common files
mv "${ORIGINAL_DIR}/common.h" "${NEW_DIR}/Core/"
mv "${ORIGINAL_DIR}/type_name.h" "${NEW_DIR}/Utils/Misc/"
mv "${ORIGINAL_DIR}/useful.h" "${NEW_DIR}/Core/"
mv "${ORIGINAL_DIR}/useful.cu" "${NEW_DIR}/Core/"

# Simulation directory
echo "Organizing Simulation directory..."
# Move Integrator files
mv "${ORIGINAL_DIR}/Integrator/CPU.h" "${NEW_DIR}/Simulation/Integrator/CPUIntegrator.h"
mv "${ORIGINAL_DIR}/Integrator/CPU.cpp" "${NEW_DIR}/Simulation/Integrator/CPUIntegrator.cpp"
mv "${ORIGINAL_DIR}/Integrator/CUDA.h" "${NEW_DIR}/Simulation/Integrator/GPUIntegrator.h"
mv "${ORIGINAL_DIR}/Integrator/CUDA.cu" "${NEW_DIR}/Simulation/Integrator/GPUIntegrator.cu"
mv "${ORIGINAL_DIR}/Integrator/kernels.h" "${NEW_DIR}/Simulation/Integrator/"
mv "${ORIGINAL_DIR}/Integrator.h" "${NEW_DIR}/Simulation/Integrator/"
mv "${ORIGINAL_DIR}/Integrator.cpp" "${NEW_DIR}/Simulation/Integrator/"

# Move Interaction files
mv "${ORIGINAL_DIR}/Interaction/CPU.h" "${NEW_DIR}/Simulation/Interactions/CPUInteraction.h"
mv "${ORIGINAL_DIR}/Interaction/CPU.cpp" "${NEW_DIR}/Simulation/Interactions/CPUInteraction.cpp"
mv "${ORIGINAL_DIR}/Interaction/CUDA.h" "${NEW_DIR}/Simulation/Interactions/GPUInteraction.h"
mv "${ORIGINAL_DIR}/Interaction/CUDA.cu" "${NEW_DIR}/Simulation/Interactions/GPUInteraction.cu"
mv "${ORIGINAL_DIR}/Interaction/kernels.h" "${NEW_DIR}/Simulation/Interactions/"
mv "${ORIGINAL_DIR}/Interaction.h" "${NEW_DIR}/Simulation/Interactions/"
mv "${ORIGINAL_DIR}/Interaction.cpp" "${NEW_DIR}/Simulation/Interactions/"

# Move SimManager and SimSystem files
mv "${ORIGINAL_DIR}/SimManager.h" "${NEW_DIR}/Simulation/Engine/"
mv "${ORIGINAL_DIR}/SimManager.cu" "${NEW_DIR}/Simulation/Engine/"
mv "${ORIGINAL_DIR}/SimManager.cu.bak" "${NEW_DIR}/Simulation/Engine/"
mv "${ORIGINAL_DIR}/SimSystem.h" "${NEW_DIR}/Simulation/Engine/"
mv "${ORIGINAL_DIR}/SimSystem.cpp" "${NEW_DIR}/Simulation/Engine/"

# Patch directory
echo "Organizing Patch directory..."
mv "${ORIGINAL_DIR}/Patch/ParticlePatch"/* "${NEW_DIR}/Patch/"
mv "${ORIGINAL_DIR}/Patch/ParticlePatch.h" "${NEW_DIR}/Patch/"
mv "${ORIGINAL_DIR}/Patch/ParticlePatch.cpp" "${NEW_DIR}/Patch/"
mv "${ORIGINAL_DIR}/Patch/PatchOp.h" "${NEW_DIR}/Patch/"
mv "${ORIGINAL_DIR}/Patch/PatchOp.cu" "${NEW_DIR}/Patch/"
mv "${ORIGINAL_DIR}/Patch/PatchOp.cuh" "${NEW_DIR}/Patch/"
mv "${ORIGINAL_DIR}/Patch.h" "${NEW_DIR}/Patch/"
mv "${ORIGINAL_DIR}/Patch.cpp" "${NEW_DIR}/Patch/"

# Python bindings
echo "Organizing Python bindings..."
mkdir -p "${NEW_DIR}/Python"
mv "${ORIGINAL_DIR}/pybindings"/* "${NEW_DIR}/Python/"

# Tests
echo "Organizing Tests..."
mv "${ORIGINAL_DIR}/Tests" "${NEW_DIR}/"

# Move main file and CMake
mv "${ORIGINAL_DIR}/arbd.cpp" "${NEW_DIR}/"
mv "${ORIGINAL_DIR}/CMakeLists.txt" "${NEW_DIR}/"

# Create backup of original directory
echo "Creating backup of original directory..."
mv "${ORIGINAL_DIR}" "${ORIGINAL_DIR}_backup"

# Rename new directory to original name
mv "${NEW_DIR}" "${ORIGINAL_DIR}"

# Create new placeholder files
echo "Creating placeholder files..."
touch "${NEW_DIR}/Objects/Particles/Particle.h"
touch "${NEW_DIR}/Objects/Particles/ParticleType.h"
touch "${NEW_DIR}/Simulation/Compute/ComputeForce.h"
touch "${NEW_DIR}/Patch/CellDecomposition.h"
touch "${NEW_DIR}/Backend/CPU/CPUBackend.h"
touch "${NEW_DIR}/Backend/GPU/GPUBackend.h"
touch "${NEW_DIR}/Backend/MPI/MPIBackend.h"

echo "Refactoring complete. Original files backed up in ${ORIGINAL_DIR}_backup"
echo "Note: You may need to update include paths in source files to match new directory structure"
