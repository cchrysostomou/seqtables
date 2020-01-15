#!/usr/bin/env python

from setuptools import setup, find_packages, Extension

import numpy


have_cython = False
try:
    from Cython.Distutils import build_ext as _build_ext
    have_cython = True
except ImportError:
    from distutils.command.build_ext import build_ext as _build_ext

print('CYTHON WAS FOUND: ', have_cython)

INSTALL_REQUIRES = ['future', 'numpy >= 1.14', 'pandas >= 0.18.0', 'xarray', 'orderedset']  # scipy

if have_cython:    
    ext_modules  = [
        Extension(
            "seqtables.core.internals.sam_to_arr",    # location of the resulting .so
            sources=[
                "seqtables/core/internals/cython/sam_to_arr.pyx"
            ],
            include_dirs=[numpy.get_include()],
            # language="c++"
        )
    ]
else:
    ext_modules  = [
        Extension(
            "seqtables.core.internals.sam_to_arr", 
            sources=[
                "seqtables/core/internals/cython/sam_to_arr.c"
            ],
            # include_dirs=["seqtables/core/internals/cython/"]
            include_dirs=[numpy.get_include()]
        )
    ]

setup(
     name='seqtables',
     version='0.1',
     license='MIT',
     description='Package for efficient analysis of next generation sequencing amplicon data using numpy and dataframes', 
     author='Constantine Chrysostomou', 
     author_email='cchrysos1@gmail.com',
     url='https://github.com/cchrysostomou/seqtables',
     download_url='https://github.com/user/seqtables/archive/v_01.tar.gz',
     keywords=['NGS', 'Next generation sequencing', 'dataframe', 'protein engineering', 'amplicon', 'numpy', 'variant analysis', 'sequence logo', 'enrichment'],
     packages=find_packages(),
     cmdclass={'build_ext': _build_ext},
     install_requires=INSTALL_REQUIRES,
     ext_modules=ext_modules,
     classifiers=[
        'Development Status :: 4 - Beta',     
        'Intended Audience :: Developers and biological data scientists',     
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License', 
        'Programming Language :: Python :: 3',    
    
     ]
)
