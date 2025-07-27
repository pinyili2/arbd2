#pragma once

#include "ARBDException.h"
#include "ARBDLogger.h"
#include "Resource.h"
#include <memory>
#include <tuple>
#include <vector>
#ifdef USE_CUDA
#include "CUDA/CUDAManager.h"
#include <cuda_runtime.h>
#endif

#ifdef USE_SYCL
#include "SYCL/SYCLManager.h"
#include <sycl/sycl.hpp>
#endif

#ifdef USE_METAL
#include "METAL/METALManager.h"
#endif

namespace ARBD {
namespace BACKEND {
class Event {
  private:
	Resource resource_;
	std::shared_ptr<void> event_impl_; // Type-erased backend event

  public:
	Event() = default;

	// Null constructor for cases where no event is needed
	Event(std::nullptr_t, const Resource& res) : resource_(res), event_impl_(nullptr) {}

	// Backend-specific constructors (only available if backend is enabled)
#ifdef USE_CUDA
	Event(cudaEvent_t cuda_event, const Resource& res)
		: resource_(res), event_impl_(std::make_shared<cudaEvent_t>(cuda_event)) {}
#endif

#ifdef USE_SYCL
	Event(sycl::event sycl_event, const Resource& res)
		: resource_(res), event_impl_(std::make_shared<sycl::event>(std::move(sycl_event))) {}
#endif

#ifdef USE_METAL
	Event(const ARBD::METAL::Event& metal_event, const Resource& res)
		: resource_(res), event_impl_(std::make_shared<ARBD::METAL::Event>(
							  std::move(const_cast<ARBD::METAL::Event&>(metal_event)))) {}
#endif

	void wait() const {
		if (!event_impl_)
			return;

		// Use resource type to determine which backend implementation to use
		if (resource_.is_device()) {
#ifdef USE_CUDA
			// Try CUDA implementation first if available
			if (auto* cuda_event = static_cast<cudaEvent_t*>(event_impl_.get())) {
				cudaEventSynchronize(*cuda_event);
				return;
			}
#endif
#ifdef USE_SYCL
			// Try SYCL implementation if available
			if (auto* sycl_event = static_cast<sycl::event*>(event_impl_.get())) {
				sycl_event->wait();
				return;
			}
#endif
#ifdef USE_METAL
			// Try Metal implementation if available
			if (auto* metal_event = static_cast<ARBD::METAL::Event*>(event_impl_.get())) {
				if (!metal_event->is_complete()) {
					metal_event->wait();
				}
				return;
			}
#endif
		}
		// Host events don't need waiting
	}

	bool is_complete() const {
		if (!event_impl_)
			return true;

		if (resource_.is_device()) {
#ifdef USE_CUDA
			// Try CUDA implementation first if available
			if (auto* cuda_event = static_cast<cudaEvent_t*>(event_impl_.get())) {
				cudaError_t status = cudaEventQuery(*cuda_event);
				return status == cudaSuccess;
			}
#endif
#ifdef USE_SYCL
			// Try SYCL implementation if available
			if (auto* sycl_event = static_cast<sycl::event*>(event_impl_.get())) {
				return sycl_event->get_info<sycl::info::event::command_execution_status>() ==
					   sycl::info::event_command_status::complete;
			}
#endif
#ifdef USE_METAL
			// Try Metal implementation if available
			if (auto* metal_event = static_cast<ARBD::METAL::Event*>(event_impl_.get())) {
				return metal_event->is_complete();
			}
#endif
		}

		// Host events are always complete
		return true;
	}

	const Resource& get_resource() const {
		return resource_;
	}
	bool is_valid() const {
		return event_impl_ != nullptr;
	}
	void* get_event_impl() const {
		return event_impl_.get();
	}
};

class EventList {
  private:
	std::vector<Event> events_;

  public:
	EventList() = default;
	EventList(std::initializer_list<Event> events) : events_(events) {}

	void add(const Event& event) {
		if (event.is_valid()) {
			events_.push_back(event);
		}
	}

	void wait_all() const {
		for (const auto& event : events_) {
			event.wait();
		}
	}

	bool all_complete() const {
		for (const auto& event : events_) {
			if (!event.is_complete())
				return false;
		}
		return true;
	}

	const std::vector<Event>& get_events() const {
		return events_;
	}
	bool empty() const {
		return events_.empty();
	}
	void clear() {
		events_.clear();
	}
	#ifdef USE_CUDA
    /**
     * @brief Extracts the raw cudaEvent_t handles from the list.
     * * @return A vector of cudaEvent_t, which can be used with cudaStreamWaitEvent.
     */
    std::vector<cudaEvent_t> get_cuda_events() const {
        std::vector<cudaEvent_t> cuda_events;
        cuda_events.reserve(events_.size());
        for (const auto& event : events_) {
            if (event.is_valid() && event.get_resource().is_device()) {
                if (auto* impl = static_cast<cudaEvent_t*>(event.get_event_impl())) {
                    cuda_events.push_back(*impl);
                }
            }
        }
        return cuda_events;
    }
#endif

#ifdef USE_SYCL
	std::vector<sycl::event> get_sycl_events() const {
		std::vector<sycl::event> sycl_events;
		for (const auto& event : events_) {
			if (event.get_resource().is_device() && event.is_valid()) {
				// Extract SYCL event from type-erased pointer
				auto impl = static_cast<sycl::event*>(event.get_event_impl());
				if (impl) {
					sycl_events.push_back(*impl);
				}
			}
		}
		return sycl_events;
	}
#endif

#ifdef USE_METAL
	std::vector<ARBD::METAL::Event*> get_metal_events() const {
		std::vector<ARBD::METAL::Event*> metal_events;
		for (const auto& event : events_) {
			if (event.get_resource().is_device() && event.is_valid()) {
				// Extract Metal event from type-erased pointer
				auto impl = static_cast<ARBD::METAL::Event*>(event.get_event_impl());
				if (impl) {
					metal_events.push_back(impl);
				}
			}
		}
		return metal_events;
	}
#endif
};
} // namespace BACKEND

// Make Event and EventList directly accessible from ARBD namespace
using Event = BACKEND::Event;
using EventList = BACKEND::EventList;

} // namespace ARBD