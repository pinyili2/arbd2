### Paths, libraries, includes, options
include ./findcudalib.mk
INCLUDE = $(CUDA_PATH)/include


ifeq ($(dbg),1)
	NV_FLAGS += -g -G
	EX_FLAGS = -g -O0 -m$(OS_SIZE)
else
	EX_FLAGS = -g -O3 -m$(OS_SIZE)
endif

## Determine the version; TODO add tags for versions
ifndef (VERSION)
ifeq ($(shell git status >& /dev/null && echo true),true)
	VERSION:=git rev. $(shell git rev-parse HEAD | sed 's/\(.\{8\}\).*/\1/')
ifneq ($(shell git status --porcelain),)
	VERSION:=$(VERSION) (modified)
endif
else 
	VERSION:=unknown (not compiled in a git repository)
endif
endif
CC_FLAGS += -DVERSION="\"$(VERSION)\""


CC_FLAGS += -I$(CUDA_PATH)/include
CC_FLAGS += -Wall -Wno-write-strings -std=c++0x -pedantic # TODO: test on Mac OSX and other architectures
ifeq ($(dbg),1)
NV_FLAGS += -Xcompiler -rdynamic -lineinfo
else
NV_FLAGS += -lineinfo
endif

ifneq ($(DARWIN),)
    LIBRARY = $(CUDA_PATH)/lib
else
    LIBRARY = $(CUDA_PATH)/lib64
endif
# NV_FLAGS += -ftz=true			# TODO: test if this preserves accurate simulation

$(info $(NVCC))

## Find valid compute capabilities for this machine
SMS ?= 30 35 37 50 52 60 61
$(info Testing CUDA toolkit with compute capabilities SMS='$(SMS)')
SMS := $(shell for sm in $(SMS); do $(NVCC) cuda-test.c -arch=sm_$$sm -o /dev/null &> /dev/null && echo $$sm; done)

ifeq (,$(SMS))
    $(error nvcc ($(NVCC)) failed with all tested compute capabilities.)
endif

SMPTXS ?= $(lastword $(sort $(SMS)))
$(info Building SASS code for SMS='$(SMS)' and PTX code for '$(SMPTXS)')

## Generate SASS and PTX code
$(foreach SM,$(SMS), $(eval NV_FLAGS += -gencode arch=compute_$(SM),code=sm_$(SM)) )
$(foreach SM,$(SMPTXS), $(eval NV_FLAGS += -gencode arch=compute_$(SM),code=compute_$(SM)) )

NVLD_FLAGS := $(NV_FLAGS) --device-link 
LD_FLAGS = -L$(LIBRARY) -lcurand -lcudart -lcudadevrt -Wl,-rpath,$(LIBRARY)

### Sources
CC_SRC := $(wildcard *.cpp)
CC_SRC := $(filter-out arbd.cpp, $(CC_SRC))
CU_SRC := $(wildcard *.cu)

CC_OBJ := $(patsubst %.cpp, %.o, $(CC_SRC))
CU_OBJ := $(patsubst %.cu, %.o, $(CU_SRC))

### Targets
ifeq ($(dbg),1)
TARGET = arbd_dbg
else
# TARGET = arbd_lin_cubic
TARGET = arbd
endif
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
