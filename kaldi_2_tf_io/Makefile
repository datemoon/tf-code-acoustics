
all:
EXTRA_CXXFLAGS += -Wno-sign-compare
include hubo.mk
include $(KALDIDIR)/kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

CXXFLAGS += -I$(KALDIDIR)

#-DDEBUG_SPEED
#-DDEBUG_PRINT

TESTFILES = chain-loss-test

OBJFILES = fst-convert-openfst.o tf-2-kaldi-api.o chain-loss-test.o

LIBNAME = chain-loss


ADDLIBS = $(KALDIDIR)/nnet3/kaldi-nnet3.a $(KALDIDIR)/chain/kaldi-chain.a $(KALDIDIR)/cudamatrix/kaldi-cudamatrix.a \
		  $(KALDIDIR)/decoder/kaldi-decoder.a $(KALDIDIR)/lat/kaldi-lat.a \
		  $(KALDIDIR)/fstext/kaldi-fstext.a $(KALDIDIR)/hmm/kaldi-hmm.a \
		  $(KALDIDIR)/transform/kaldi-transform.a $(KALDIDIR)/gmm/kaldi-gmm.a \
		  $(KALDIDIR)/tree/kaldi-tree.a $(KALDIDIR)/util/kaldi-util.a $(KALDIDIR)/matrix/kaldi-matrix.a \
		  $(KALDIDIR)/base/kaldi-base.a 



include $(KALDIDIR)/makefiles/default_rules.mk

LDFLAGS_STATIC=$(OPENFSTLDFLAGS) -rdynamic $(CUDA_LDFLAGS)
MKLLIB_STATIC=-Wl,--start-group $(MKLLIB)/libmkl_intel_lp64.a $(MKLLIB)/libmkl_core.a $(MKLLIB)/libmkl_sequential.a -Wl,--end-group -ldl -lpthread -lm
LDLIBS_STATIC=$(OPENFSTLIBS) -lm -lpthread -ldl $(CUDA_LDLIBS)

STATIC_LIB=lib$(LIBNAME)-static.so
.PHONY:
static:fst-convert-openfst.o tf-2-kaldi-api.o
	g++ -shared -o $(STATIC_LIB) $(LDFLAGS_STATIC) $^ $(ADDLIBS) $(MKLLIB_STATIC) $(LDLIBS_STATIC)



