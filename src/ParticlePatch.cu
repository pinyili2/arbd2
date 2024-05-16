#include "ParticlePatch.h"

// BasePatch::BasePatch(size_t num, short thread_id, short gpu_id) { ;};
// BasePatch::BasePatch() {};
// BasePatch::~BasePatch() {};

Patch::Patch(size_t num, short thread_id, short gpu_id) {};

void Patch::compute() {
    for (auto& c_p: local_computes) {
	c_p->compute(this);
    }
};

