#pragma once

#define WARPSIZE 32

/* __device__ int atomicAggInc(int *ctr, int warpLane) { */
/* 	int mask = __ballot(1); */
/* 	int leadThread = __ffs(mask)-1; */

/* 	int res; */
/* 	if ( warpLane == leadThread ) */
/* 		res = atomicAdd(ctr, __popc(mask)); */

/* 	res = warp_ */
										
	
class localIntBuffer {
public:
	__device__ void localIntBuffer(const int s, const int nt, const int c) :
		size(s), numThreads(nt), subChunksPerBlock(c),
		chunkSize(c*nt), numChunks(s/(c*nt)) {
		// numChunks = size / chunkSize; /* assume 0 remainder */
		
		buffer = (int*) malloc( size * sizeof(int) );

		freeChunk = (bool*) malloc( numChunks * sizeof(bool) );
		for (int i = 0; i < numChunks; i++)	freeChunk = true;
		chunk = getNextChunk();
		chunkOffset = 0;
	}
	__device__ void ~localIntBuffer() {
		free buffer;
	}
	__device__ void append( const int tid, const int val ) {
		// RBTODO
		buffer[chunk*chunkSize + chunkOffset + tid] = val;

		// update offset
	}

	__device__ void sendChunks( const int tid, int* globalDest, int& g_offset ) {
		int chunksSent = 0;
		for (int i = 0; i < numChunks; i++) {
			if (!freeChunk[i] && i != chunk) { // found a chunk to send
				for (int c = 0; c < subChunksPerBlock; c++) {
					globalDest[g_offset + tid] = buffer[i*chunkSize + c*nt + tid];
					g_offset += nt;
				}
				// free chunk
				freeChunk[i] = true;
				// NOT DONE
			}
		}
	}

private:
	__device__ int getNextChunk() {
		for (int i = 0; i < numChunks; i++) {
			if (freeChunk[i]) {
				freeChunk[i] = false;
				return i;
			}
		}
		return -1;									/* out of chunks */
	}

	const int size;
  const int numThreads;
	const int subChunksPerBlock;
	const int chunkSize;
  const int numChunks;
	int* buffer
  bool* freeChunk;
  int chunk;
	int chunkOffset;
};

class sharedIntBuffer {
public:
	__device__ void sharedIntBuffer(const int s, const int nt, const int c) :
		size(s), numThreads(nt), subChunksPerBlock(c),
		chunkSize(c*nt), numChunks(s/(c*nt)) {
		// numChunks = size / chunkSize; /* assume 0 remainder */
		
		buffer = (int*) malloc( size * sizeof(int) );

		freeChunk = (bool*) malloc( numChunks * sizeof(bool) );
		for (int i = 0; i < numChunks; i++)	freeChunk = true;
		chunk = getNextChunk();
		chunkOffset = 0;
	}
	__device__ void ~sharedIntBuffer() {
		free buffer;
	}
	__device__ void append( const int tid, const int val ) {
		// RBTODO
		buffer[chunk*chunkSize + chunkOffset + tid] = val;

		// update offset
	}

	__device__ void sendChunks( const int tid, int* globalDest, int& g_offset ) {
		int chunksSent = 0;
		for (int i = 0; i < numChunks; i++) {
			if (!freeChunk[i] && i != chunk) { // found a chunk to send
				for (int c = 0; c < subChunksPerBlock; c++) {
					globalDest[g_offset + tid] = buffer[i*chunkSize + c*nt + tid];
					g_offset += nt;
				}
			}
		}
	}

private:
	__device__ int getNextChunk() {
		for (int i = 0; i < numChunks; i++) {
			if (freeChunk[i]) {
				freeChunk[i] = false;
				return i;
			}
		}
		return -1;									/* out of chunks */
	}

	const int size;
  const int numThreads;
	const int subChunksPerBlock;
	const int chunkSize;
  const int numChunks;
	int* buffer
  bool* freeChunk;
  int chunk;
	int chunkOffset;
};
	
__device__ inline void exclIntCumSum(int* in, const int n) {
	// 1) int* in must point to shared memory
	// 2) int n must be power of 2
	const int tid = threadIdx.x;
	// RBTODO: worry about possible bank conflicts http://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf
	
	// build tree of sums
	int stride = 1;
	for (int d = n>>1; d > 0; d >>= 1) {
		__syncthreads();
		if (tid < d) {
			int id = 2*stride*(tid+1)-1;
			in[id] += in[id-stride];
		}
		stride *= 2;
	}
	if (tid == 0) in[n-1] = 0;		/* exclusive cumsum (starts at 0) */

	// traverse down tree and build 'scan'
	for (int d = 1; d < n; d*= 2) {
		stride >>= 1;
		__syncthreads();

		if (tid < d) { // RBTODO: this could be incorrect ==> test
			int id = 2*stride*(tid+1)-1;
			int t = in[id];
			in[id] += in[id-stride];
			in[id-stride] = t;
		}
	}
	__syncthreads();
}


__device__ inline void inclIntCumSum(int* in, const int n) {
	// 1) int* in must point to shared memory
	// 2) int n must be power of 2
	const int tid = threadIdx.x;
	
	// RBTODO: worry about possible bank conflicts http://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf
	
	// build tree of sums
	int stride = 1;
	for (int d = n>>1; d > 0; d >>= 1) {
		__syncthreads();
		if (tid < d) {
			int id = 2*stride*(tid+1)-1;
			in[id] += in[id-stride];
		}
		stride *= 2;
	}
	// if (tid == 0) in[n-1] = 0;		/* exclusive cumsum (starts at 0) */

	// traverse down tree and build 'scan'
	for (int d = 1; d < n; d*= 2) {
		stride >>= 1;
		__syncthreads();

		if (tid < d) { // RBTODO: this could be incorrect ==> test
			int id = 2*stride*(tid+1)-1;
			in[id+stride] += in[id];
			/* int t = in[id]; */
			/* in[id] += in[id-stride]; */
			/* in[id-stride] = t; */
		}
	}
	__syncthreads();
}

__device__ void atomicAdd( Vector3* address, Vector3 val) {
	atomicAdd( &(address->x), val.x);
	atomicAdd( &(address->y), val.y);
	atomicAdd( &(address->z), val.z);
}
