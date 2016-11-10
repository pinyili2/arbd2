### Paths, libraries, includes, options
include ./findcudalib.mk
CUDA_PATH ?= /usr/local/cuda-8.0

$(info CUDA_PATH=$(CUDA_PATH))
INCLUDE = $(CUDA_PATH)/include

ifeq ($(dbg),1)
	DEBUG = -g -O0
	NV_FLAGS = -v -g -G
	EX_FLAGS = -O0 -m$(OS_SIZE)
else
	NV_FLAGS = -v
	EX_FLAGS = -O3 -m$(OS_SIZE)
endif

CC_FLAGS = -Wall -Wno-write-strings -I$(INCLUDE) $(DEBUG) -std=c++0x -pedantic
NV_FLAGS += -lineinfo

ifneq ($(MAVERICKS),)
    CC = $(CLANG)
    CC_FLAGS += -stdlib=libstdc++
    NV_FLAGS += -Xcompiler -arch -Xcompiler x86_64
else
    CC = $(GCC)
endif

ifneq ($(DARWIN),)
    LIBRARY = $(CUDA_PATH)/lib
else
    LIBRARY = $(CUDA_PATH)/lib64
endif

# NV_FLAGS += -ftz=true			# TODO: test if this preserves accurate simulation

SMS ?= 20 30 35 37 50 52 60
## Generate SASS code
$(foreach SM,$(SMS), $(eval NV_FLAGS += -gencode arch=compute_$(SM),code=sm_$(SM)) )

## Generate PTX code for highest SM architecture for future GPUs
SM=$(lastword $(sort $(SMS)))
NV_FLAGS += -gencode arch=compute_$(SM),code=sm_$(SM)

NVLD_FLAGS := $(NV_FLAGS) --device-link 

LD_FLAGS = -L$(LIBRARY) -lcurand -lcudart -lcudadevrt -Wl,-rpath,$(LIBRARY)
LD_FLAGS += -lcuda 


### Sources
CC_SRC := $(wildcard *.cpp)
CC_SRC := $(filter-out arbd.cpp, $(CC_SRC))
CU_SRC := $(wildcard *.cu)

CC_OBJ := $(patsubst %.cpp, %.o, $(CC_SRC))
CU_OBJ := $(patsubst %.cu, %.o, $(CU_SRC))


### Targets
TARGET = arbd

all: $(TARGET)
	@echo "Done ->" $(TARGET)

$(TARGET): $(CU_OBJ) $(CC_OBJ) arbd.cpp vmdsock.c imd.c imd.h
	$(NVCC) $(NVLD_FLAGS) $(CU_OBJ) $(CC_OBJ) -o $(TARGET)_link.o
	$(NVCC) $(NVLD_FLAGS) $(CU_OBJ) -o $(TARGET)_link.o
	$(CC) $(CC_FLAGS) $(EX_FLAGS) arbd.cpp vmdsock.c imd.c $(TARGET)_link.o $(CU_OBJ) $(CC_OBJ) $(LD_FLAGS)  -o $(TARGET)

.SECONDEXPANSION:
$(CU_OBJ): %.o: %.cu $$(wildcard %.h) $$(wildcard %.cuh)
	$(NVCC) $(NV_FLAGS) $(EX_FLAGS) -dc $< -o $@

$(CC_OBJ): %.o: %.cpp %.h 
	$(CC) $(CC_FLAGS) $(EX_FLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(CU_OBJ) $(CC_OBJ)
