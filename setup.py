from os import path
from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize
import Cython
import numpy as np
import subprocess as  sp

extensions = [
    Extension("STASIS.Utility", ["./STASIS/Utilities/_STASIS_utility.pyx"],
        include_dirs=[np.get_include()],
        libraries = ["STASIS_utility"],
        build_dir = ["."]
        ),
    Extension("STASIS.Detector", ["./STASIS/Detector_features/_STASIS_detector.pyx"],
        extra_compile_args=['-fPIC', '-g', '-lm', '-fopenmp', '-O3', '-Wall'],
        extra_link_args=['-fPIC' ,'-g', '-pthread', '-fopenmp', '-O3', '-Wall'],
        include_dirs=[np.get_include()],
        libraries=["STASIS_detector"],
        build_dir=["."]
        )
]
setup(
    name="STASIS",
    ext_modules=cythonize(extensions),
    cmdclass={'build_ext': Cython.Build.build_ext},
    packages=find_packages(),
    package_data={
        "STASIS":["*pxd"],
        "STASIS/Detector_features":["*.pxd"],
        "STASIS/Utilities":["*.pxd"]
    },
)