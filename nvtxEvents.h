#pragma once

#include <nvToolsExt.h>

// http://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvtx
nvtxRangeId_t nvtxStart(const char* name, int colorId) {
	return nvtxRangeStartA(name);
}


