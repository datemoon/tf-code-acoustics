
CXX = g++ -std=gnu++0x
AR = ar
AS = as
RANLIB = ranlib


LIBFLAG = dynamic

CXXFLAGS = -std=c++11 -I. \
		   -Wall -Wno-sign-compare -Wno-unused-local-typedefs \
		   -Wno-deprecated-declarations -Winit-self \
		   -msse -msse2 -pthread \
		   -O2
EXTRA_CXXFLAGS += -Wno-sign-compare 
#-DDEBUG_SPEED
#-DDEBUG_PRINT

LDFLAGS = -rdynamic
LDLIBS = -lm -lpthread -ldl

LIBNAME = mmi-loss

OBJFILES = sparse-lattice-function.o loss.o

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

