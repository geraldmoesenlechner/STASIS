from os import path
from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize
import numpy as np
import subprocess as  sp

sp.run("./src/build.sh", shell=True)

extensions = [
    Extension("Utility", ["./src/Utilities/_STASIS_utility.pyx"],
        include_dirs=[np.get_include()],
        libraries = ["STASIS_utility"]
        ),
    Extension("Detector", ["./src/Detector_features/_STASIS_detector.pyx"],
        extra_compile_args=['-fPIC', '-g', '-lm', '-lfftw3', '-lxml2', '-fopenmp', '-O3', '-lfftw3_omp', '-Wall'],
        extra_link_args=['-fPIC' ,'-g', '-pthread', '-lfftw3', '-lxml2', '-fopenmp', '-O3', '-lfftw3_omp', '-Wall'],
        include_dirs=[np.get_include()],
        libraries=["STASIS_detector"]
        ),
    Extension("Kinematics", ["./src/Sc_kinematics/_sc_kinematics_py.pyx"],
        extra_compile_args=['-fPIC', '-g', '-lm', '-Wall'],
        extra_link_args=['-fPIC' ,'-g', '-Wall'],
        include_dirs=[np.get_include()],
        libraries=["sc_kinematics"])
]
setup(
    name="STASIS",
    ext_modules=cythonize(extensions),
    packages=find_packages(),
)