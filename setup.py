import setuptools
from distutils.core import setup, Extension
import numpy.distutils.misc_util

setup(name='elasticdeform',
      version='0.1.0',
      description='Elastic deformations for N-D images.',
      author='Gijs van Tulder',
      author_email='gvtulder@gmail.com',
      url='https://github.com/gvtulder/elasticdeform',
      license='BSD',
      packages=['elasticdeform'],
      ext_modules=[Extension('elasticdeform._deform_grid',
                             ['elasticdeform/_deform_grid.c',
                              'elasticdeform/deform.c',
                              'elasticdeform/from_nd_image.c'],
                             include_dirs=['elasticdeform'])],
      include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: BSD License",
      ],
)
