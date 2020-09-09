# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda-10.2
TARGET_ARCH ?= x86_64


HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m64 -DSHARED

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      BUILD_TYPE := debug
else
      BUILD_TYPE := release
endif

# Gencode arguments
SMS ?= 30 35 37 50 52 60 61 70 75

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif



# Common includes and paths for CUDA
INCLUDES  := -I$(CUDA_PATH)/include -I./include -I./include/opencv/2.4.9

LIBRARIES := -L$(CUDA_PATH)/lib64 -L./lib/OpenCV
LDFLAGS :=  -lcudart -lopencv_core -lopencv_highgui -lopencv_imgproc 


SOURCE=$(wildcard ./src/*.cu)
OBJECT=$(foreach n,${SOURCE},$(addsuffix .o , $(basename ${n})))

# Target rules
all: build

build: main

%.o:%.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS)  $(GENCODE_FLAGS) -c $< -o $@ 


main: ${OBJECT}
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $(GENCODE_FLAGS) -o main ${OBJECT} ${LIBRARIES}
	cp $@ ./bin


clean:
	rm -f main ${OBJECT}
	rm -rf ./bin/main



###  /usr/local/cuda-10.2/bin/nvcc vecAdd.cu -o vecAdd   为啥不用include .h 与链接 .so