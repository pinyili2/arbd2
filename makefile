### Paths, libraries, includes, options

# CUDA_PATH ?= /Developer/NVIDIA/CUDA-6.5
# CUDA_PATH ?= /software/cuda-toolkit-6.5-x86_64
CUDA_PATH ?= /usr/local/cuda-7.0

#CUDA_PATH ?= /usr/local/encap/cuda-5.5

include ./findcudalib.mk

INCLUDE = $(CUDA_PATH)/include

DEBUG = -g
CC_FLAGS = -Wall -Wno-write-strings -I$(INCLUDE) $(DEBUG) -std=c++0x -pedantic
NV_FLAGS = $(DEBUG)
EX_FLAGS = -O3 -m$(OS_SIZE)

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


#CODE_10 := -gencode arch=compute_10,code=sm_10
#CODE_12 := -gencode arch=compute_12,code=sm_12
#CODE_20 := -gencode arch=compute_20,code=sm_20
CODE_20 := -arch=sm_20
#CODE_30 := -gencode arch=compute_30,code=sm_30
#CODE_35 := -gencode arch=compute_35,code=\"sm_35,compute_35\"

NV_FLAGS += $(CODE_10) $(CODE_12) $(CODE_20) $(CODE_30) $(CODE_35)

NVLD_FLAGS := $(NV_FLAGS) --device-link
NV_FLAGS += -rdc=true

LD_FLAGS = -L$(LIBRARY) -lcurand -lcudart -Wl,-rpath,$(LIBRARY)


### Sources
CC_SRC := $(wildcard *.cpp)
CC_SRC := $(filter-out runBrownTown.cpp, $(CC_SRC))
CU_SRC := $(wildcard *.cu)

CC_OBJ := $(patsubst %.cpp, %.o, $(CC_SRC))
CU_OBJ := $(patsubst %.cu, %.o, $(CU_SRC))


### Targets
EXEC = #@echo

TARGET = runBrownCUDA

all: $(TARGET)
	@echo "Done ->" $(TARGET)

$(TARGET): $(CU_OBJ) $(CC_OBJ) runBrownTown.cpp vmdsock.c imd.c imd.h
	$(EXEC) $(NVCC) $(NVLD_FLAGS) $(CU_OBJ) $(CC_OBJ) -o $(TARGET)_link.o
	$(EXEC) $(CC) $(CC_FLAGS) $(EX_FLAGS) runBrownTown.cpp vmdsock.c imd.c $(TARGET)_link.o $(CU_OBJ) $(CC_OBJ) $(LD_FLAGS)  -o $(TARGET)

# $(EXEC) $(NVCC) $(NVLD_FLAGS) $(CU_OBJ) -o $(TARGET)_link.o
# $(EXEC) $(CC) $(CC_FLAGS) $(LD_FLAGS) $(EX_FLAGS) runBrownTown.cpp vmdsock.c imd.c $(CU_OBJ) $(CC_OBJ) -o $(TARGET)

.SECONDEXPANSION:
$(CU_OBJ): %.o: %.cu $$(wildcard %.h) $$(wildcard %.cuh)
	$(EXEC) $(NVCC) $(NV_FLAGS) $(EX_FLAGS) -c $< -o $@

$(CC_OBJ): %.o: %.cpp %.h 
	$(EXEC) $(CC) $(CC_FLAGS) $(EX_FLAGS) -c $< -o $@

clean:
	$(EXEC) rm -f $(TARGET) $(CU_OBJ) $(CC_OBJ)
