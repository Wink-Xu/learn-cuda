# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda-10.2



HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m64

# Common includes and paths for CUDA
INCLUDES  := -I$(CUDA_PATH)/include
LIBRARIES := -L$(CUDA_PATH)/lib64

LDFLAGS := -lcudart

SOURCE=$(wildcard ./src/*.cu)
OBJECT=$(foreach n,${SOURCE},$(addsuffix .o , $(basename ${n})))


# Target rules
all: build

build: vectorAdd

${OBJECT}:$(SOURCE)
	$(NVCC) $(INCLUDES) $(NVCCFLAGS)  -o $@ -c $<

vectorAdd: ${OBJECT}
	$(NVCC) $(NVCCFLAGS)  -o vectorAdd ${OBJECT}
	cp $@ ./bin


clean:
	rm -f vectorAdd vectorAdd.o
	rm -rf ./bin/vectorAdd

