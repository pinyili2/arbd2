#include "ParticlePatch.h"

Patch::Patch() : BasePatch() {
    pos_force = momentum = nullptr;
    // num_rb = num_rb_attached_particles = 0;
    // rb_pos = rb_orient = rb_mom = rb_ang_mom = type = rb_type = nullptr;
};


void Patch::compute() {
    for (auto& c_p: local_computes) {
	c_p->compute(this);
    }
};
