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

lib_srcs = ['tensorflow_api/mmi_loss_op.cc', 'tensorflow_api/mpe_loss_op.cc']

root_path = os.path.realpath(os.path.dirname(__file__))

tf_include = tf.sysconfig.get_include()
tf_src_dir = os.environ["TENSORFLOW_SRC_PATH"]
tf_includes = [tf_include, tf_src_dir]
mmi_includes = [os.path.join(root_path, './')]
include_dirs = tf_includes + mmi_includes

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


mmi_path = './'

extra_compile_args = ['-std=c++11', '-fPIC', '-D_GLIBCXX_USE_CXX11_ABI=' + TF_CXX11_ABI]
extra_compile_args += ['-Wno-return-type']

extra_link_args = []
if tf.__version__ >= '1.4':
    if os.path.exists(os.path.join(tf_src_dir, 'libtensorflow_framework.so')):
        extra_link_args = ['-L' + tf.sysconfig.get_lib(), '-ltensorflow_framework']

if tf.__version__ >= '1.4':
    include_dirs += [tf_include + '/../../external/nsync/public']

class build_tf_ext(orig_build_ext):
    def build_extensions(self):
        self.compiler.compiler_so.remove('-Wstrict-prototypes')
        orig_build_ext.build_extensions(self)

ext = Extension('tensorflow_py_api.mmi',
        sources = lib_srcs,
        language = 'c++',
        include_dirs = include_dirs,
        library_dirs = [mmi_path],
        runtime_library_dirs = [os.path.realpath(mmi_path)],
        libraries = ['mmi-loss', 'tensorflow_framework'],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args)

setup(name="tensorflow_py_api",
        version = "1.0",
        author = "hubo",
        description = "tf wrapper for mmi loss",
        packages = ["tensorflow_py_api"],
        ext_modules = [ext],
        cmdclass = {'build_ext': build_tf_ext})

