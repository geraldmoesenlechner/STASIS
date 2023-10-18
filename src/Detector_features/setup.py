from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [Extension(name="detector_features_py", sources=["detector_features_py.pyx"], extra_compile_args=['-fPIC', '-g', '-lm', '-lfftw3', '-lxml2', '-fopenmp', '-O3', '-lfftw3_omp', '-Wall'],
                   extra_link_args=['-fPIC' ,'-g', '-pthread', '-lfftw3', '-lxml2', '-fopenmp', '-O3', '-lfftw3_omp', '-Wall'],include_dirs=[np.get_include()], libraries=["detector_features"])]

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
