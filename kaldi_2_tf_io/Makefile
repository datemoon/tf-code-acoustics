
all:

EXTRA_CXXFLAGS += -Wno-sign-compare
include hubo.mk
include $(KALDIDIR)/kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

CXXFLAGS += -I$(KALDIDIR)

#-DDEBUG_SPEED
#-DDEBUG_PRINT

OBJFILES = fst-convert-openfst.o tf-2-kaldi-api.o

LIBNAME = chain-loss

TESTFILES = chain-loss-test

ADDLIBS = $(KALDIDIR)/nnet3/kaldi-nnet3.a $(KALDIDIR)/chain/kaldi-chain.a $(KALDIDIR)/cudamatrix/kaldi-cudamatrix.a \
		  $(KALDIDIR)/decoder/kaldi-decoder.a $(KALDIDIR)/lat/kaldi-lat.a \
		  $(KALDIDIR)/fstext/kaldi-fstext.a $(KALDIDIR)/hmm/kaldi-hmm.a \
		  $(KALDIDIR)/transform/kaldi-transform.a $(KALDIDIR)/gmm/kaldi-gmm.a \
		  $(KALDIDIR)/tree/kaldi-tree.a $(KALDIDIR)/util/kaldi-util.a $(KALDIDIR)/matrix/kaldi-matrix.a \
		  $(KALDIDIR)/base/kaldi-base.a 

include $(KALDIDIR)/makefiles/default_rules.mk

