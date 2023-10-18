from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [Extension(name="datasim_utility_py", sources=["datasim_utility_py.pyx"],include_dirs=[np.get_include()], libraries = ["datasim_utility"])]

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
