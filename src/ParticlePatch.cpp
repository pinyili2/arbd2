#include "ParticlePatch.h"

// Patch::Patch() : BasePatch() {
//     // pos_force = VectorArr();
//     // momentum = VectorArr();
//     // num_rb = num_rb_attached_particles = 0;
//     // rb_pos = rb_orient = rb_mom = rb_ang_mom = type = rb_type = nullptr;
// };

void Patch::initialize() {
    Data* data = new Data{nullptr,nullptr,nullptr,nullptr};	// TODO free
    metadata = Metadata{0,0,0};
    // data->pos_force = data->momentum = nullptr;
    // data->particle_types = data->particle_order = nullptr;
    metadata.data = send( Resource{Resource::CPU,0}, *data );
};

Patch* Patch::copy_to_cuda(Patch* dev_ptr) const {
    Exception(NotImplementedError, "Deprecated");
    
    if (dev_ptr == nullptr) { // allocate if needed
	gpuErrchk(cudaMalloc(&dev_ptr, sizeof( typeid(this) )));
    }

    // Patch* tmp;
    // tmp = new Patch(0);
    // tmp->metadata->data->pos_force = data->pos_force->copy_to_cuda();
    // tmp->data->momentum = data->momentum->copy_to_cuda();
    // tmp->data->particle_types = data->particle_types->copy_to_cuda();
    // tmp->data->particle_order = data->particle_order->copy_to_cuda();

    // return tmp;
    return nullptr;

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
    Proxy<Data> newdata;
    switch (location.type) {
    case Resource::GPU:
	// metadata.data is already a proxy because the data may reside on the GPU;
	// Q: In what cases should we move metadata.data.addr?
	//     For example, we could always move, or we could never move, or we could move if location metadata.data.location is a CPU resource

	// if (location != metadata.data,location) 
	// newdata = metadata.data.callSync( send, addr
	// data->pos_force = metadata.data.callSync( send,  ->pos_force->copy_to_cuda();
	// data->momentum = metadata.data->momentum->copy_to_cuda();
	// data->particle_types = metadata.data->particle_types->copy_to_cuda();
	// data->particle_order = metadata.data->particle_order->copy_to_cuda();

	// Nothing to do then?
	
	break;
    case Resource::CPU:
	Exception( NotImplementedError, "`send_children(...)` not implemented on CPU" );
	break;
    default:
	Exception( ValueError, "`send_children(...)` passed unknown resource type" );
    }
    newdata.location = location;
    tmp.metadata.data = newdata;
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


// template<typename T>
// size_t Patch::send_particles_filtered( Proxy<Patch>& destination, T filter ) {
size_t Patch::send_particles_filtered( Proxy<Patch>* destination, std::function<bool(size_t,Patch::Data)> filter ) {
// TODO determine who allocates and cleans up temporary data
    // TODO determine if there is a good way to avoid having temporary data (this can be done later if delegated to a reasonable object)

    // TODO think about the design and whether it's possible to have a single array with start/end
    // TODO think about the design for parallel sending; should a receiver do some work?

    // TODO currently everything is sychronous, but that will change; how do we avoid race conditions?
    // E.g. have a single array allocated with known start and end for each PE/Patch?
    
    Data buffer;		// not sure which object should allocate
    { 
    using _filter_t = decltype(filter);
    // using A = decltype(destination->metadata.data);
    using A = Proxy<Patch::Data>&;
    // size_t num_sent = metadata.data.callSync<size_t,A>( &Patch::Data::send_particles_filtered, destination->metadata.data, filter );
    // size_t num_sent = metadata.data.callSync<size_t,A>( static_cast<size_t (Patch::Data::*)(Proxy<Patch::Data>&, _filter_t)>(&Patch::Data::send_particles_filtered), destination->metadata.data, filter );

    // size_t num_sent = metadata.data.callSync<size_t,A>( &Patch::Data::send_particles_filtered, destination->metadata.data, filter );
    size_t num_sent = metadata.data.callSync( &Patch::Data::send_particles_filtered, &((*destination)->metadata.data), filter );

    }
    // size_t num_sent = metadata.data.callSync<size_t,Proxy<Patch::Data>&,_filter_t>( &Patch::Data::send_particles_filtered<_filter_t>, destination->metadata.data, filter );
    return 0;
};

// template<typename T>
// size_t Patch::Data::send_particles_filtered( Proxy<Patch::Data>& destination, T filter ) { }
size_t Patch::Data::send_particles_filtered( Proxy<Patch::Data>* destination, std::function<bool(size_t,Patch::Data)> filter ) { 
    Data buffer;		// not sure which object should allocate
    // metadata.data.callSync( &Patch::Data::send_particles_filtered, destination, filter );
    // x
    return 0;
};

