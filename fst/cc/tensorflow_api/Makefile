

#TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_CFLAGS = -I/usr/local/lib/python3.5/dist-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0


#TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
TF_LFLAGS = -L/usr/local/lib/python3.5/dist-packages/tensorflow -ltensorflow_framework


ADDLIBS=../mmi-loss.a

CXXFLAGS += -std=c++11 -fPIC  -I ../ 
#-DDEBUG_SPEED

all:tf_mmi_api.so


tf_mmi_api.so: mmi_loss_op.o mpe_loss_op.o
	g++ -std=c++11 -shared $^ -o $@ -fPIC ${TF_CFLAGS} ${CXXFLAGS} ${TF_LFLAGS} ${ADDLIBS} -O2

%.o:%.cc
	$(CXX) $(CXXFLAGS) ${TF_CFLAGS} -c  -o $@ $^

%.o:%.cpp
	$(CXX) $(CXXFLAGS) ${TF_CFLAGS} -c  -o $@ $^

.PHONY:

clean:
	rm -f tf_mmi_api.so *.o
