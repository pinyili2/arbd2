#pragma once

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

#include <functional>
#include <memory> // std::make_unique
#include <vector> // std::vector

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#endif

#ifdef USE_METAL
#include <Metal/Metal.h>
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
#endif

#include "Math/Types.h"

#include "../Patch/Patch.h"

template<typename Pos, typename Force>
class Patch {
	int num_replicas;

  public:
	Patch(SimParameters* sim);

	void compute();

  private:
	SimParameters* sim;

	Pos minimum;
	Pos maximum;

	ParticleTypes* types;
	ParticleTypes* types_d;

	// Particle data
	size_t num_particles;

	idx_t* global_idx; // global index of particle
	size_t* type_ids;
	Pos* pos;
	Force* force;

	size_t* type_ids_d;
	Pos* pos_d;
	Force* force_d;
};
// Storage class that should be
class Patch : public BasePatch {
  public:
	Patch() : BasePatch(), metadata() {
		LOGINFO("Creating Patch");
		initialize();
		LOGINFO("Done Creating Patch");
	}
	Patch(size_t capacity) : BasePatch(capacity) {
		initialize();
	}

	// Particle data arrays pointing to either CPU or GPU memory
	struct Data {
		Data(const size_t capacity = 0) {
			if (capacity == 0) {
				pos_force = momentum = nullptr;
				particle_types = particle_order = nullptr;
			} else {
				pos_force = new VecArray(capacity);
				momentum = new VecArray(capacity);
				particle_types = new Array<size_t>(capacity);
				particle_order = new Array<size_t>(capacity);
			}
		}
		VecArray* pos_force;
		VecArray* momentum;
		Array<size_t>* particle_types;
		Array<size_t>* particle_order;

		HOST DEVICE inline Vector3& get_pos(size_t i) {
			return (*pos_force)[i * 2];
		};
		HOST DEVICE inline Vector3& get_force(size_t i) {
			return (*pos_force)[i * 2 + 1];
		};
		HOST DEVICE inline Vector3& get_momentum(size_t i) {
			return (*momentum)[i];
		};
		HOST DEVICE inline size_t& get_type(size_t i) {
			return (*particle_types)[i];
		};
		HOST DEVICE inline size_t& get_order(size_t i) {
			return (*particle_order)[i];
		};

		// Replace with auto? Return number of particles sent?
		// template<typename T>
		// size_t send_particles_filtered( Proxy<Data>& destination, T filter ); //
		// = [](size_t idx, Patch::Data d)->bool { return true; } );
		size_t send_particles_filtered(
			Proxy<Data>& destination,
			std::function<bool(size_t, Data)> filter); // = [](size_t idx, Patch::Data
													   // d)->bool { return true; } );
	};

	// Metadata stored on host even if Data is on device
	struct Metadata {
		Metadata() : num(0), capacity(0), min(0), max(0), data(){};
		Metadata(const Patch& p)
			: num(p.metadata.num), capacity(p.metadata.capacity), min(p.metadata.min),
			  max(p.metadata.max), data(p.metadata.data){};
		Metadata(const Metadata& other)
			: num(other.num), capacity(other.capacity), min(other.min), max(other.max),
			  data(other.data){};
		size_t num;
		size_t capacity;
		Vector3 min;
		Vector3 max;
		Proxy<Data> data; // actual data may be found elsewhere
	};

	// void deleteParticles(IndexList& p);
	// void addParticles(size_t n, size_t typ);
	// template<class T>
	// void add_compute(std::unique_ptr<T>&& p) {
	// 	std::unique_ptr<BasePatchOp> base_p =
	// static_cast<std::unique_ptr<BasePatchOp>>(p);
	// 	local_computes.emplace_back(p);
	// };

	// void add_compute(std::unique_ptr<BasePatchOp>&& p) {
	// 	local_computes.emplace_back(std::move(p));
	// };

	void add_point_particles(size_t num_added);
	void add_point_particles(size_t num_added, Vector3* positions, Vector3* momenta = nullptr);

	Patch send_children(Resource location) const;

	// TODO deprecate copy_to_cuda
	Patch* copy_to_cuda(Patch* dev_ptr = nullptr) const;
	static Patch copy_from_cuda(Patch* dev_ptr, Patch* dest = nullptr);
	static void remove_from_cuda(Patch* dev_ptr);

	// TODO? emplace_point_particles
	void compute();

	// Communication
	size_t send_particles(Proxy<Patch>* destination); // Same as send_children?
	// void send_particles_filtered( Proxy<Patch> destination,
	// std::function<bool(size_t, Patch::Data)> = [](size_t idx, Patch::Data
	// d)->bool { return true; } );

	// Replace with auto? Return number of particles sent?
	// template<typename T>
	// size_t send_particles_filtered( Proxy<Patch>& destination, T filter );
	// // [](size_t idx, Patch::Data d)->bool { return true; } );
	size_t send_particles_filtered(Proxy<Patch>& destination,
								   std::function<bool(size_t, Data)> filter);
	// [](size_t idx, Patch::Data d)->bool { return true; } );

	void clear() {
		LOGWARN("Patch::clear() was called but is not implemented");
	}

	size_t test() {
		LOGWARN("Patch::test() was called but is not implemented");
		return 1;
	}

  private:
	void initialize();

	// void randomize_positions(size_t start = 0, size_t num = -1);

	// TODO: move computes to another object; this should simply be a dumb data
	// store std::vector<PatchProxy> neighbors;
	std::vector<std::unique_ptr<PatchOp>> local_computes;	 // Operations that will be performed on
															 // this patch each timestep
	std::vector<std::unique_ptr<PatchOp>> nonlocal_computes; // Operations that will be performed on
															 // this patch each timestep

	Metadata metadata; // Usually associated with proxy, but can use it here too

	// size_t num_rb;
	// Vector3* rb_pos;
	// Matrix3* rb_orient;
	// Vector3* rb_mom;
	// Vector3* rb_ang_mom;

	// size_t* type;	     // particle types: 0, 1, ... -> num * numReplicas
	// size_t* rb_type;	     // rigid body types: 0, 1, ... -> num * numReplicas

	// size_t num_rb_attached_particles;

	// TODO: add a rb_attached bitmask

	size_t num_group_sites;
};
} // namespace ARBD
