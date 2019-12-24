
g++ -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/usr/local/python35/lib/python3.5/site-packages/tensorflow_core/include -I/usr/local/python35/lib/python3.5/site-packages/tensorflow_core -I/usr/local/python35/lib/python3.5/site-packages/tensorflow_core/include/../../external/nsync/public -I/usr/local/python35/include/python3.5m -c test.cc -o test.o -std=c++11 -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -Wno-return-type -DDEBUG_SPEED -Wno-sign-compare -Wall -Wno-sign-compare -Wno-unused-local-typedefs -Wno-deprecated-declarations -Winit-self

g++ -pthread -shared test.o -L/usr/local/python35/lib -lpython3.5m -o test.so -L/usr/local/python35/lib/python3.5/site-packages/tensorflow_core -ltensorflow_framework
