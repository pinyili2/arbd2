#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <stdexcept>
#include <array>
#include <vector>
#include <string>
#include <span>
#include <optional>
#include <chrono>


namespace ARBD {
namespace METAL {

inline void check_metal_error(void* device, std::string_view file, int line) {
    if (device == nullptr) {
        throw std::runtime_error("METAL error: Device is null");
    }
}

#define METAL_CHECK(call) check_metal_error(call, __FILE__, __LINE__)

}
}