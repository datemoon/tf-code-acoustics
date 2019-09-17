
#TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
#TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

all:

EXTRA_CXXFLAGS += -Wno-sign-compare
include ../hubo.mk
include $(KALDIDIR)/kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

LDLIBS += -L/usr/lib/python3.4/site-packages/tensorflow -ltensorflow_framework
CXXFLAGS += -I/usr/lib/python3.4/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0 -I$(KALDIDIR)



#-DDEBUG_SPEED
#-DDEBUG_PRINT

OBJFILES = tf_chain_loss.o

LIBNAME = tf_chain_api

TESTFILES = chain-loss-test

ADDLIBS = ../chain-loss.a \
		  $(KALDIDIR)/nnet3/kaldi-nnet3.a $(KALDIDIR)/chain/kaldi-chain.a $(KALDIDIR)/cudamatrix/kaldi-cudamatrix.a \
		  $(KALDIDIR)/decoder/kaldi-decoder.a $(KALDIDIR)/lat/kaldi-lat.a \
		  $(KALDIDIR)/fstext/kaldi-fstext.a $(KALDIDIR)/hmm/kaldi-hmm.a \
		  $(KALDIDIR)/transform/kaldi-transform.a $(KALDIDIR)/gmm/kaldi-gmm.a \
		  $(KALDIDIR)/tree/kaldi-tree.a $(KALDIDIR)/util/kaldi-util.a $(KALDIDIR)/matrix/kaldi-matrix.a \
		  $(KALDIDIR)/base/kaldi-base.a 

include $(KALDIDIR)/makefiles/default_rules.mk

LDFLAGS_STATIC=$(OPENFSTLDFLAGS) -rdynamic $(CUDA_LDFLAGS)
MKLLIB_STATIC=-Wl,--start-group $(MKLLIB)/libmkl_intel_lp64.a $(MKLLIB)/libmkl_core.a $(MKLLIB)/libmkl_sequential.a -Wl,--end-group -ldl -lpthread -lm
LDLIBS_STATIC=$(OPENFSTLIBS) -lm -lpthread -ldl $(CUDA_LDLIBS) -L/usr/lib/python3.4/site-packages/tensorflow -ltensorflow_framework

STATIC_LIB=lib$(LIBNAME)-static.so
.PHONY:
static:$(OBJFILES)
	g++ -shared -o $(STATIC_LIB) $(LDFLAGS_STATIC) $^ $(ADDLIBS) $(MKLLIB_STATIC) $(LDLIBS_STATIC)

#LDFLAGS_STATIC=$(OPENFSTLDFLAGS) -rdynamic $(CUDA_LDFLAGS)
#LDLIBS_STATIC=$(OPENFSTLIBS) -lm -lpthread -ldl $(CUDA_LDLIBS) -L/usr/lib/python3.4/site-packages/tensorflow -ltensorflow_framework

#STATIC_LIB=lib$(LIBNAME)-static.so
#.PHONY:
#static:$(OBJFILES)
#	g++ -shared -o $(STATIC_LIB) $(LDFLAGS_STATIC) $^ $(ADDLIBS) $(LDLIBS_STATIC)

#STATIC_LIB=lib$(LIBNAME)-static.so
#LDLIBS_STATIC = -L/usr/lib/python3.4/site-packages/tensorflow -ltensorflow_framework 
#.PHONY:
#static:$(OBJFILES) ../libchain-loss-static.so
#	g++ -shared -o $(STATIC_LIB) -rdynamic -Wl,-rpath=/search/speech/hubo/git/tf-code-acoustics/kaldi_2_tf_io/tf_chain_api $^ $(LDLIBS_STATIC)

