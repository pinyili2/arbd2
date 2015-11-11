#!/bin/bash
# Usage: source nvcc.sh

# module load cuda-toolkit
# /software/cuda-toolkit-4.1-x86_64/cuda/bin/nvcc

# Add CUDA 'bin' folder to PATH
# export PATH="/Developer/NVIDIA/CUDA-5.5/bin:$PATH"

# Remove duplicate entries from PATH
# PATH=$(echo "$PATH" | awk -v RS=':' -v ORS=":" '!a[$1]++{if (NR > 1) printf ORS; printf $a[$1]}')
