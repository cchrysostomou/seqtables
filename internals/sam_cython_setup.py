from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("sam_to_arr", ["sam_to_arr.pyx"])]

setup(
    name='SAM to np array app',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
