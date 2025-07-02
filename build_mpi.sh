mkdir build 
cd build
cmake -S .. -B . -DCMAKE_CXX_COMPILER=/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/openmpi-5.0.2-o2g3ojh/bin/mpicxx -DUSE_MPI=ON -DUSE_CUDA=OFF

cmake --build . --target arbd_test_mpi --parallel 2


#command used for run: srun --mem=16g --nodes=1 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=1 --partition=gpuA40x4-interactive,gpuA100x4-interactive,gpuMI100x8-interactive --gpus-per-node=1 --account=becs-delta-gpu --time=00:10:00 mpirun -np 8 ./arbd_test_mpi


