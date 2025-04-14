#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_CUDA
#include <cuda.h>
#endif

#include <cstdio>
#include <sstream>

// #include "useful.h"
#include "GPUManager.h"
#include "SignalManager.h"

#include "SimManager.h"

// using namespace std;
using std::max;

const unsigned int kIMDPort = 71992;

int main(int argc, char* argv[]) {

#ifdef USE_MPI
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print off a hello world message
    printf("Hello world from processor %s, rank %d out of %d processors\n",
           processor_name, world_rank, world_size);

    // Finalize the MPI environment.
    MPI_Finalize();
#endif

    SignalManager::manage_segfault();

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
	    printf("%s unknown version\n", argv[0]);
#endif
		return 0;
	} else if (argc == 2 && (strcmp(argv[1], "--info") == 0)) {
#ifdef USE_CUDA
	    GPUManager::load_info();
#else
	    printf("ARBD not compiled with CUDA support\n");
#endif
	    printf("Returning\n");
	    return 0;
	} else if (argc < 3) {
		printf("%s: missing arguments\n", argv[0]);
    printf("Try '%s --help' for more information.\n", argv[0]);
    return 1;
  }
	printf("--- Atomic Resolution Brownian Dynamics ---\n");

	size_t n_gpus = 0;
#ifdef USE_CUDA
	GPUManager::init();
	n_gpus = GPUManager::allGpuSize();
	std::vector<unsigned int> gpuIDs;
#endif
	
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
#ifdef USE_CUDA
		    /*
		    String argval(argv[pos+1]);
		    int nTokens = argval.tokenCount(',');
		    String* tokens = new String[nTokens];
		    argval.tokenize(tokens,',');
		    for (int i = 0; i < nTokens; ++i) {
			unsigned int arg_val = atoi(tokens[i].val());
			if (arg_val < 0 || arg_val > n_gpus) {
			    printf("ERROR: Invalid argument given to %s: %s\n", arg, tokens[i].val());
				return 1;
			}
			// std::vector<unsigned int>::iterator it;
			// it = std::find(gpuIDs.begin(), gpuIDs.end(), arg_val);
			// if (it != gpuIDs.end()) {
			//     printf("WARNING: ignoring repeated GPU ID %d\n", arg_val);
			// } else {
			gpuIDs.push_back(arg_val);
			// }
		    }
		    delete[] tokens;
		    safe = false;
		    num_flags += 2;
		    */
#else
		    printf("ARBD compiled without CUDA support. Exiting.\n");
		    return 1;
#endif			
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

	char* configFile = NULL;
	char* outArg = NULL;
	if (argc - num_flags == 3) {
		configFile = argv[argc - 2];
		outArg = argv[argc - 1];
	} else {
	    printf("%s: too many arguments\n", argv[0]);
	    printf("Try '%s --help' for more information.\n", argv[0]);
	    return 1;
	}

#ifdef USE_CUDA
	GPUManager::safe(safe);
	if (gpuIDs.size() == 0)
	    gpuIDs.push_back( GPUManager::getInitialGPU() );

	#ifndef USE_NCCL
	if (gpuIDs.size() > 1) {
	    Exception(ValueError, "Using more than one GPU requires compilation with USE_NCCL flag");
	    return 1;
	}
	#endif

	GPUManager::select_gpus(gpuIDs);
#endif

	SimManager sim;
	sim.run();
	
	// Configuration config(configFile, replicas, debug);
	// config.copyToCUDA();
	// // GPUManager::set(0);
	// GrandBrownTown brown(config, outArg,
	// 		debug, imd_on, imd_port, replicas);
	// brown.run();
  return 0;

}
