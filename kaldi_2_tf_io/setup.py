from __future__ import print_function

import os
import sys

import warnings
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as orig_build_ext

try:
    import tensorflow as tf
except ImportError:
    raise RuntimeError("Tensorflow must be installed to build the tensorflow wrapper.")

if "TENSORFLOW_SRC_PATH" not in os.environ:
    print("Please define the TENSORFLOW_SRC_PATH environment variable.\n"
            "This should be a path to the Tensorflow source directory.",
            file=sys.stderr)
    sys.exit(1)

lib_srcs = ['tf_chain_api/tf_chain_loss.cc']

root_path = os.path.realpath(os.path.dirname(__file__))

tf_include = tf.sysconfig.get_include()
tf_src_dir = os.environ["TENSORFLOW_SRC_PATH"]
tf_includes = [tf_include, tf_src_dir]
chain_includes = [os.path.join(root_path, './')]
include_dirs = tf_includes + chain_includes

if tf.__version__ >= '1.4':
    include_dirs += [tf_include + '/../../external/nsync/public']

if os.getenv("TF_CXX11_ABI") is not None:
    TF_CXX11_ABI = os.getenv("TF_CXX11_ABI")
else:
    warnings.warn("Assuming tensorflow was compiled without C++11 ABI. "
            "It is generally true if you are using binary pip package. "
            "If you compiled tensorflow from source with gcc >= 5 and didn't set "
            "-D_GLIBCXX_USE_CXX11_ABI=0 during compilation, you need to set "
            "environment variable TF_CXX11_ABI=1 when compiling this bindings. "
            "Also be sure to touch some files in src to trigger recompilation. "
            "Also, you need to set (or unsed) this environment variable if getting "
            "undefined symbol: _ZN10tensorflow... errors")
    TF_CXX11_ABI = "0"


chain_path = './'

extra_compile_args = ['-std=c++11', '-fPIC', '-D_GLIBCXX_USE_CXX11_ABI=' + TF_CXX11_ABI]
extra_compile_args += ['-Wno-return-type']

extra_link_args = []
if tf.__version__ >= '1.4':
    if os.path.exists(os.path.join(tf_src_dir, 'libtensorflow_framework.so')):
        extra_link_args = ['-L' + tf.sysconfig.get_lib(), '-ltensorflow_framework']

if tf.__version__ >= '1.4':
    include_dirs += [tf_include + '/../../external/nsync/public']

# add kaldi config

# kaldi fst mkl
kaldi_src_dir = '/root/kaldi/src/'
mklroot = '/opt/intel/mkl/'
fstroot = kaldi_src_dir + '../tools/openfst-1.6.7/'
cudalib='/usr/local/cuda/lib64/'
kaldi_include = [ kaldi_src_dir]
fst_include = [ fstroot + '/include' ]
cuda_include = [ cudalib + '/../include']

if os.path.exists(kaldi_src_dir) is False:
    print("no kaldi dir.")
    sys.exit(1)

extra_compile_args += ['-Wno-sign-compare', '-Wall', '-Wno-sign-compare',
        '-Wno-unused-local-typedefs', '-Wno-deprecated-declarations',
        '-Winit-self', '-DKALDI_DOUBLEPRECISION=0', 
        '-DHAVE_EXECINFO_H=1', '-DHAVE_CXXABI_H', '-DHAVE_MKL', '-isystem ' + fstroot + 'include' ] 

# add mkl link extra args
extra_link_args += [ '-Wl,-rpath=' + mklroot + '/lib/intel64']
        

# add fst and cuda link
extra_link_args += ['-Wl,--no-undefined', '-Wl,--as-needed',
#        '-Wl,-soname=' + 'libtf_chain_api.so' + ',--whole-archive ' + 'tf_chain_api.a',
        '-Wl,--no-whole-archive',
        '-Wl,-rpath=' + fstroot + '/lib', '-rdynamic']

extra_link_args += ['-Wl,-rpath,' + cudalib, '-Wl,-rpath=' + kaldi_src_dir + '/lib']

include_dirs += kaldi_include + cuda_include + fst_include

shared_libraries = ['chain-loss', 
        'kaldi-nnet3', 'kaldi-chain',
        'kaldi-cudamatrix', 'kaldi-decoder',
        'kaldi-lat', 'kaldi-lat', 
        'kaldi-fstext', 'kaldi-hmm', 
        'kaldi-transform', 'kaldi-gmm',
        'kaldi-tree', 'kaldi-util',
        'kaldi-matrix', 'kaldi-base',
        'fst',
        'cublas', 'cusparse', 'cudart', 'curand', 'cufft', 'nvToolsExt', 
        'mkl_intel_lp64', 'mkl_core', 'mkl_sequential']

library_dirs = [chain_path] + [kaldi_src_dir+'/lib'] + [fstroot + '/lib'] + [mklroot + '/lib/intel64'] + [cudalib]

class build_tf_ext(orig_build_ext):
    def build_extensions(self):
#        self.compiler.compiler_so.remove('-Wstrict-prototypes')
        orig_build_ext.build_extensions(self)

ext = Extension('tf_chain_py_api.chainloss',
        sources = lib_srcs,
        language = 'c++',
        include_dirs = include_dirs,
        library_dirs = library_dirs,
        runtime_library_dirs = [os.path.realpath(chain_path)], #kaldi_src_dir + '/lib', fstroot + '/lib'],
        libraries = shared_libraries,
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args)

setup(name="tf_chain_py_api",
        version = "1.0",
        author = "hubo",
        description = "tf wrapper for chian loss",
        packages = ["tf_chain_py_api"],
        ext_modules = [ext],
        cmdclass = {'build_ext': build_tf_ext})

