
all:

include ../kaldi.mk

EXTRA_CXXFLAGS += -Wno-sign-compare

OBJFILES = sparse-lattice-function.o loss.o


LIBNAME = mmi-loss

TESTFILES = convert-lattice-test nnet-train-mmi-sequential

ADDLIBS = ../lat/kaldi-lat.a ../lm/kaldi-lm.a ../fstext/kaldi-fstext.a \
          ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a

include ../makefiles/default_rules.mk
