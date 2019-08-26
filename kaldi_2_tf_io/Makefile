CXX = g++
#-std=gnu++0x
AR = ar
AS = as
RANLIB = ranlib

# Base configuration
#
DOUBLE_PRECISION = 0
KALDIROOT = /search/speech/hubo/git/tf-code-acoustics/kaldi_2_tf_io/kaldi
KALDISRC = $(KALDIROOT)/src
KALDILIBDIR := $(KALDIROOT)/src/lib

OPENFSTINC = $(KALDIROOT)/tools/openfst/include
OPENFSTLIBS = $(KALDIROOT)/tools/openfst/lib/libfst.so
OPENFSTLDFLAGS = -Wl,-rpath=$(KALDIROOT)/tools/openfst/lib

CUBROOT = $(KALDIROOT)/tools/cub
ATLASINC = $(KALDIROOT)/tools/ATLAS_headers/include
ATLASLIBS = /usr/lib64/atlas/libsatlas.so.3 /usr/lib64/atlas/libtatlas.so.3 -Wl,-rpath=/usr/lib64/atlas

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
CUDATKDIR = /usr/local/cuda
CUDA_ARCH = -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70

ifndef DOUBLE_PRECISION
	$(error DOUBLE_PRECISION not defined.)
endif
ifndef CUDATKDIR
	$(error CUDATKDIR not defined.)
endif

CXXFLAGS += -DHAVE_CUDA -I$(CUDATKDIR)/include -fPIC -pthread -isystem $(OPENFSTINC)

CUDA_INCLUDE= -I$(CUDATKDIR)/include -I$(CUBROOT)
CUDA_FLAGS = --machine 64 -DHAVE_CUDA \
			 -ccbin $(CXX) -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION) \
			 -std=c++11 -DCUDA_API_PER_THREAD_DEFAULT_STREAM  -lineinfo \
			 --verbose -Xcompiler "$(CXXFLAGS)"

CUDA_LDFLAGS += -L$(CUDATKDIR)/lib64 -Wl,-rpath,$(CUDATKDIR)/lib64
CUDA_LDLIBS += -lcublas -lcusparse -lcudart -lcurand -lcufft -lnvToolsExt #LDLIBS : The .so libs are loaded later than static libs in implicit rule
CUDA_LDLIBS += -lcusolver
# Environment configuration
CXXFLAGS += -DKALDI_NO_EXPF

EXTRA_CXXFLAGS += -Wno-sign-compare

LDFLAGS = -rdynamic
LDLIBS = -lm -lpthread -ldl

LIBNAME = chain-loss

ADDLIBS = $(KALDISRC)/nnet3/kaldi-nnet3.a $(KALDISRC)/chain/kaldi-chain.a \
		  $(KALDISRC)/cudamatrix/kaldi-cudamatrix.a $(KALDISRC)/lat/kaldi-lat.a \
		  $(KALDISRC)/fstext/kaldi-fstext.a $(KALDISRC)/hmm/kaldi-hmm.a \
		  $(KALDISRC)/transform/kaldi-transform.a $(KALDISRC)/tree/kaldi-tree.a \
		  $(KALDISRC)/util/kaldi-util.a \
		  $(KALDISRC)/matrix/kaldi-matrix.a $(KALDISRC)/base/kaldi-base.a 

OBJFILES = tf-2-kaldi-api.o fst-convert-openfst.o

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
	$(CXX) -shared -o $@ -Wl,--no-undefined -Wl,--as-needed  -Wl,-soname=$@,--whole-archive $(LIBNAME).a -Wl,--no-whole-archive $(OPENFSTLDFLAGS) -rdynamic -L$(CUDATKDIR)/lib64 -Wl,-rpath,$(CUDATKDIR)/lib64 -Wl,-rpath=$(KALDILIBDIR) $(ADDLIBS) $(OPENFSTLIBS)  $(ATLASLIBS) $(LDLIBS) $(CUDA_LDLIBS)
endif

clean:
	rm -f *.o *.a *.so


