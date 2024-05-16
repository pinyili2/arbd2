#pragma once

#include "useful.h"

class Interactions {
    // Object to store all kinds of info about the simulation system, but no particle data

public:
    size_t num_interactions;
    
};

class BondInteractions : public Interactions {

private:
    static const char* type = "Bond";
    size_t bondlist;
};
