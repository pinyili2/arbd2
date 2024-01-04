#include "ParticlePatch.h"

// Patch::Patch() : BasePatch() {
//     // pos_force = VectorArr();
//     // momentum = VectorArr();
//     // num_rb = num_rb_attached_particles = 0;
//     // rb_pos = rb_orient = rb_mom = rb_ang_mom = type = rb_type = nullptr;
// };

void Patch::initialize() {
    data->pos_force = data->momentum = nullptr;
    data->particle_types = data->particle_order = nullptr;
};

Patch* Patch::copy_to_cuda(Patch* dev_ptr) const {
    
    if (dev_ptr == nullptr) { // allocate if needed
	gpuErrchk(cudaMalloc(&dev_ptr, sizeof( typeid(this) )));
    }

    Patch* tmp;
    tmp = new Patch(0);
    tmp->data->pos_force = data->pos_force->copy_to_cuda();
    tmp->data->momentum = data->momentum->copy_to_cuda();
    tmp->data->particle_types = data->particle_types->copy_to_cuda();
    tmp->data->particle_order = data->particle_order->copy_to_cuda();

    return tmp;
    // tmp.pos_force;
    // // Allocate member data
    // pos_force.
    // 	if (num > 0) { 
    // 	    size_t sz = sizeof(T) * num;
    // 	    // printf("   cudaMalloc for %d items\n", num);
    // 	    gpuErrchk(cudaMalloc(&values_d, sz));

    // 	    // Copy values
    // 	    for (size_t i = 0; i < num; ++i) {
    // 		values[i].copy_to_cuda(values_d + i);
    // 	    }
    // 	}

};

Patch Patch::send_children(Resource location) const {
    Patch tmp(0);
    switch (location.type) {
    case Resource::GPU:
	tmp.data->pos_force = data->pos_force->copy_to_cuda();
	tmp.data->momentum = data->momentum->copy_to_cuda();
	tmp.data->particle_types = data->particle_types->copy_to_cuda();
	tmp.data->particle_order = data->particle_order->copy_to_cuda();
	break;
    case Resource::CPU:
	Exception( NotImplementedError, "`send_children(...)` not implemented on CPU" );
	break;
    default:
	Exception( ValueError, "`send_children(...)` passed unknown resource type" );
    }
    tmp.data.location = location;
    return tmp;
};


Patch Patch::copy_from_cuda(Patch* dev_ptr, Patch* dest) {
    // TODO
    Patch tmp = Patch();
    return tmp;
};
void Patch::remove_from_cuda(Patch* dev_ptr) {
    // TODO
};
    

void Patch::compute() {
    for (auto& c_p: local_computes) {
	c_p->compute(this);
    }
};
