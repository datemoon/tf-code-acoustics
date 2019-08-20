CXX = g++
#-std=gnu++0x
AR = ar
AS = as
RANLIB = ranlib

# Base configuration
#
DOUBLE_PRECISION = 0
KALDIROOT = /search/speech/hubo/git/kaldi
KALDISRC = $(KALDIROOT)/src
KALDILIBDIR := /search/speech/hubo/git/kaldi/src/lib

OPENFSTINC = $(KALDIROOT)/tools/openfst/include
OPENFSTLIBS = $(KALDIROOT)/tools/openfst/lib/libfst.so
OPENFSTLDFLAGS = -Wl,-rpath=$(KALDIROOT)/tools/openfst/lib

CUBROOT = $(KALDIROOT)/tools/cub
ATLASINC = $(KALDIROOT)/tools/ATLAS/include
ATLASLIBS = /usr/lib64/atlas/libsatlas.so /usr/lib64/atlas/libtatlas.so.3 -Wl,-rpath=/usr/lib64/atlas

LIBFLAG = dynamic

CXXFLAGS = -std=c++11 -I. -isystem $(OPENFSTINC) -I $(KALDISRC) \
		   -Wall -Wno-sign-compare -Wno-unused-local-typedefs \
		   -Wno-deprecated-declarations -Winit-self \
		   -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION) \
		   -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_ATLAS -I$(ATLASINC) \
		   -msse -msse2 -pthread \
		   -D_GLIBCXX_USE_CXX11_ABI=0 \
		   -O2

CUDA = true
CUDATKDIR = /usr/local/cuda-8.0
CUDA_ARCH = -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62

ifndef DOUBLE_PRECISION
	$(error DOUBLE_PRECISION not defined.)
endif
ifndef CUDATKDIR
	$(error CUDATKDIR not defined.)
endif

CUDA_INCLUDE= -I$(CUDATKDIR)/include
CUDA_FLAGS = -g -Xcompiler -fPIC --verbose --machine 64 -DHAVE_CUDA \
			 -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION)

CXXFLAGS += -DHAVE_CUDA -I$(CUDATKDIR)/include
CUDA_LDFLAGS += -L$(CUDATKDIR)/lib64 -Wl,-rpath,$(CUDATKDIR)/lib64
CUDA_LDLIBS += -lcublas -lcudart -lcurand #LDLIBS : The libs are loaded later than static libs in implicit rule

# Environment configuration
CXXFLAGS += -DKALDI_NO_EXPF

EXTRA_CXXFLAGS += -Wno-sign-compare

LDFLAGS = -rdynamic
LDLIBS = -lm -lpthread -ldl

LIBNAME = chain-loss

OBJFILES = fst-convert-openfst.o  tf-2-kaldi-api.o

ifeq ($(LIBFLAG), dynamic)
	CXXFLAGS += -fPIC
	LIBFILE = lib${LIBNAME}.so
else
	LIBFILE = $(LIBNAME).a
endif

all: $(LIBFILE)

$(LIBFILE): $(OBJFILES)
	$(AR) -cru $(LIBNAME).a $(OBJFILES)
	$(RANLIB) $(LIBNAME).a
ifeq ($(LIBFLAG), dynamic)
	$(CXX) -shared -o $@ -Wl,--no-undefined -Wl,--as-needed  -Wl,-soname=$@,--whole-archive $(LIBNAME).a -Wl,--no-whole-archive $(LDFLAGS) $(LDLIBS)
endif

clean:
	rm -f *.o *.a *.so


