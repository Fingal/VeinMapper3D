from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    #ext_modules = cythonize("cone_calculation.pyx"),
    ext_modules = cythonize("convert_to_tiff_smooth.pyx"),
    include_dirs=[numpy.get_include()]
)
#python setup.py build_ext --inplace