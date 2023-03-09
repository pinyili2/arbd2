#include "ParticlePatch.h"

void Patch::compute() {
    for (auto& c_p: local_computes) {
	c_p->compute(this);
    }
};
