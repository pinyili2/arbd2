#!/bin/bash
# Usage: source nvcc.sh

# Add CUDA 'bin' folder to PATH
export PATH="/Developer/NVIDIA/CUDA-5.5/bin:$PATH"

# Remove duplicate entries from PATH
PATH=$(echo "$PATH" | awk -v RS=':' -v ORS=":" '!a[$1]++{if (NR > 1) printf ORS; printf $a[$1]}')
