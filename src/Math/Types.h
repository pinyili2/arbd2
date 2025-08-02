#pragma once
#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Array.h"
#include "Bitmask.h"
#include "Matrix3.h"
#include "TypeName.h"
#include "Vector3.h"
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory> // For std::unique_ptr
#include <ranges>
#include <sstream>
#include <stdarg.h> // For va_start, etc.
#include <string>
#include <string_view>
#include <vector> // Added for std::vector
#include "Backend/Header.h"

namespace ARBD {

// Simplified string formatting function for CUDA compatibility
template<typename... Args>
inline std::string string_format(const char* format, Args... args) {
	// Calculate required size
	int size = snprintf(nullptr, 0, format, args...);
	if (size <= 0)
		return std::string();

	// Allocate buffer and format
	size++; // for null terminator
	std::string result(size, '\0');
	snprintf(&result[0], size, format, args...);
	result.resize(size - 1); // remove null terminator
	return result;
}

// Includes of various types (allows those to be used simply by including
// Types.h)

using Vector3 = Vector3_t<float>;
using Matrix3 = Matrix3_t<float, false>;
using VecArray = std::vector<Vector3>;

using idx_t = size_t;
} // namespace ARBD
