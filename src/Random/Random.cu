#include "Math/Types.h"
#include "Math/Vector3.h"

// Vector3 kernel wrappers
template __global__ void cuda_kernel_wrapper<void (*)(size_t, Vector3<float>*, Vector3<float>*)>(
	size_t,
	void (*)(size_t, Vector3<float>*, Vector3<float>*));
