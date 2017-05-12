///////////////////////////////////////////////////////////////////////
// Author: Jeff Comer <jcomer2@illinois.edu>
#include <cstdio>
#include <cuda.h>
#include <sstream>

#include "useful.h"
#include "GrandBrownTown.h"
#include "Configuration.h"
#include "GPUManager.h"

// using namespace std;
using std::max;

const unsigned int kIMDPort = 71992;

int main(int argc, char* argv[]) {
	if (argc == 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
		// --help
		printf("Usage: %s [OPTIONS [ARGS]] CONFIGFILE OUTPUT [SEED]\n", argv[0]);
		printf("\n");
		printf("  -r, --replicas=    Number of replicas to run\n");
		printf("  -g, --gpu=         Index of gpu to use (defaults to 0)\n");
		printf("  -i, --imd=         IMD port (defaults to %d)\n", kIMDPort);
		printf("  -d, --debug        Debug mode: allows user to choose which forces are computed\n");
		printf("  --safe             Do not use GPUs that may timeout\n");
		printf("  --unsafe           Use GPUs that may timeout (default)\n");
		printf("  -h, --help         Display this help and exit\n");
		printf("  --info             Output CPU and GPU information and exit\n");
		printf("  --version          Output version information and exit\n");
		return 0;
	} else if (argc == 2 && (strcmp(argv[1], "--version") == 0)) {
		// --version
		// printf("%s Nov 2016 (alpha)\n", argv[0]);
#ifdef VERSION
	    printf("%s %s\n", argv[0], VERSION);
#else
	    printf("%s Nov 2016 (alpha)\n", argv[0]);
#endif
		return 0;
	} else if (argc == 2 && (strcmp(argv[1], "--info") == 0)) {
		// --info
		GPUManager::load_info();
		// size_t n_gpus = max(GPUManager::gpus.size(), 1lu);
		return 0;
	} else if (argc < 3) {
		printf("%s: missing arguments\n", argv[0]);
    printf("Try '%s --help' for more information.\n", argv[0]);
    return 1;
  }
	// printf("Everything's great when you're...BrownTown\n");
	printf("  –––––––––––––––––––––––––––––––––––––––––––––\n");
	printf("  |    Atomic Resolution Brownian Dynamics    |\n");
	printf("  –––––––––––––––––––––––––––––––––––––––––––––\n");
	GPUManager::init();

	size_t n_gpus = max(GPUManager::gpus.size(), 1lu);
	int gpuID = -1;
	
	bool debug = false, safe = false;
	int replicas = 1;
	unsigned int imd_port = 0;
	bool imd_on = false;
	int num_flags = 0;
	for (int pos = 1; pos < argc; pos++) {
		const char *arg = argv[pos];
		if (strcmp(arg, "--safe") == 0) {
			safe = true;
			num_flags++;
		} else if (strcmp(arg, "--unsafe") == 0) {
			safe = false;
			num_flags++;
		} else if (strcmp(arg, "-d") == 0 || strcmp(arg, "--debug") == 0) {
			debug = true;
			num_flags++;

		} else if (strcmp(arg, "-g") == 0 || strcmp(arg, "--gpu") == 0) {
			unsigned int arg_val = atoi(argv[pos + 1]);
			safe = false;
			gpuID = arg_val;
			num_flags += 2;
			if (arg_val < 0 || arg_val > n_gpus) {
				printf("ERROR: Invalid argument given to %s\n", arg);
				return 1;
			}
			
		} else if (strcmp(arg, "-r") == 0 || strcmp(arg, "--replicas") == 0) {
			int arg_val = atoi(argv[pos + 1]);
			if (arg_val <= 0) {
				printf("ERROR: Invalid argument given to %s\n", arg);
				return 1;
			}
			replicas = arg_val;
			num_flags += 2;
		} else if (strcmp(arg, "-i") == 0 || strcmp(arg, "--imd") == 0) {
			int arg_val = atoi(argv[pos + 1]);
			if (arg_val <= 0) {
				imd_port = kIMDPort;
			} else {
				imd_port = arg_val;
				num_flags++;
			}
			imd_on = true;
			num_flags++;
		}
		
		if (argc - num_flags < 3) {
			printf("%s: missing arguments\n", argv[0]);
			printf("Try '%s --help' for more information.\n", argv[0]);
			return 1;
		}
	}

	long int randomSeed = 1992;
	char* configFile = NULL;
	char* outArg = NULL;
	if (argc - num_flags >= 4) {
		configFile = argv[argc - 3];
		outArg = argv[argc - 2];
		randomSeed = atol(argv[argc - 1]);
	} else if (argc - num_flags >= 3) {
		configFile = argv[argc - 2];
		outArg = argv[argc - 1];
	}

	GPUManager::safe(safe);
	if (gpuID == -1)
	    gpuID = GPUManager::getInitialGPU();

	Configuration config(configFile, replicas, debug);
	// GPUManager::set(0);
	printf("Setting gpuID to %d\n",gpuID);
	GPUManager::set(gpuID);
	//MLog: this copyToCUDA function (along with the one in GrandBrownTown.cpp) was split into pieces to allocate memory into the ComputeForce, due to the location of this call we may get some memory error as a ComputeForce class isn't allocated until later on.
	config.copyToCUDA();
	

	GrandBrownTown brown(config, outArg, randomSeed,
			debug, imd_on, imd_port, replicas);
	printf("Running on GPU %d...\n", GPUManager::current());
	cudaDeviceProp prop = GPUManager::properties[GPUManager::current()];
	if (prop.kernelExecTimeoutEnabled != 0)
		printf("WARNING: GPU may timeout\n");
	brown.run();
  return 0;

}
