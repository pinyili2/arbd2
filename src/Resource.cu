#include "Resource.h"

HOST DEVICE size_t current_thread_idx() {
#ifdef USE_MPI
    Exception( NotImplementedError, "current_thread_idx() not implemented on GPU" );
    int world_rank;
    return MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
#else
#ifdef __CUDA_ACC__
    Exception( NotImplementedError, "current_thread_idx() not implemented on GPU" );
#else
    return 0;
#endif
#endif
}
