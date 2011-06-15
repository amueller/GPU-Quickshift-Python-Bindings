Darwin_i386_ARCH := maci
Linux_i386_ARCH := glx
Linux_i686_ARCH := glx
Linux_x86_64_ARCH := a64

UNAME := $(shell uname -sm)
ARCH ?= $($(shell echo "$(UNAME)" | tr \  _)_ARCH)

CFLAGS   = -fno-strict-aliasing -O3 -DUNIX 

# Mac OS X Intel 32 / 64
ifeq ($(ARCH),maci)
GPUROOT := /usr/local/cuda
GPULIB  := $(GPUROOT)/lib
GDKROOT := /Developer/GPU\ Computing
SDKROOT := /Developer/SDKs/MacOSX10.5.sdk
MACOPTS := -m32 
MACLD   := $(MACOPTS) -Xlinker -rpath $(GPUROOT)/lib
LDFLAGS += -lm -mmacosx-version-min=10.5
CPU := i386
PLATFORM := darwin
endif


# Linux 32
ifeq ($(ARCH),glx)
GPUROOT := /usr/local/cuda
GPULIB  := $(GPUROOT)/lib
GDKROOT := /home/brian/NVIDIA_GPU_Computing_SDK
MACOPTS := 
MACLD   := 
CPU := i386
PLATFORM := linux
endif

# Linux 64
ifeq ($(ARCH),a64)
GPUROOT := /usr/local/cuda
GPULIB  := $(GPUROOT)/lib64
GDKROOT := /usr/local/src/NVIDIA_GPU_Computing_SDK
MACOPTS := 
MACLD   := 
CPU := x86_64
PLATFORM := linux
endif

CFLAGS += -I. -I$(GPUROOT)/include -I$(GDKROOT)/C/common/inc -I$(GDKROOT)/shared/inc

NVCC     = $(GPUROOT)/bin/nvcc
GENCODE  = -gencode=arch=compute_10,code=\"sm_10,compute_10\"  -gencode=arch=compute_20,code=\"sm_20,compute_20\"
LDFLAGS += -L$(GPULIB) -L$(GDKROOT)/C/lib -L$(GDKROOT)/C/common/lib/$(PLATFORM) -L$(GDKROOT)/shared/lib
LDFLAGS += -lcudart -lcutil_$(CPU) -lshrutil_$(CPU)

all: quickshift

%.cpp.o: %.cpp
	g++ $(MACOPTS) -Wall $(CFLAGS) -o $@ -c $<

%.cu.o: %.cu
	$(NVCC) $(GENCODE) $(MACOPTS) --compiler-options $(CFLAGS) -o $@ -c $<

quickshift: Image.cpp.o quickshift_cpu.cpp.o main.cpp.o quickshift_gpu.cu.o
	g++ -fPIC $(MACLD) -o quickshift $^ $(LDFLAGS)

clean:
	rm -f quickshift *.o
