#!/usr/bin/env python

import numpy as np
from distutils.core import setup, Extension

setup(name='patchextraction',
      version='1.0',
      description='Patch extraction tools for coronary angiograms',
      author='Fernando Cervantes',
      author_email='iie.fercer@gmail.com',
      
      ext_modules=[Extension('patchextraction', ['include/patchextraction.c', 'include/random_numbers_generator.c'],
                             define_macros=[('NDEBUG',)],
                             include_dirs=[np.get_include()],
                             )],
      )
