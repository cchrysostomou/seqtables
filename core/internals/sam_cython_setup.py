from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    Extension(
        "sam_to_arr",    # location of the resulting .so
        sources=[
            "cython/sam_to_arr.pyx"
        ],
        include_dirs=[numpy.get_include()],
        # language="c++"
    )
]


setup(
    name='package',
    packages=find_packages(),
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
)
