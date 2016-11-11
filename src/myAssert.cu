#include "myAssert.h"

#ifdef __APPLE__
__host__ __device__ void myAssert(bool statement) {  }
#else
#include <assert.h>
__host__ __device__ void myAssert(bool statement) { assert(statement); }
#endif