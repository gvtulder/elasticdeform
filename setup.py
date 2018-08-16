from distutils.core import setup, Extension
import numpy.distutils.misc_util

setup(name='elasticdeform',
      version='0.1',
      description='Elastic deformations for N-D images.',
      author='Gijs van Tulder',
      packages=['elasticdeform'],
      ext_modules=[Extension("elasticdeform._deform_grid",
                             ["elasticdeform/_deform_grid.c",
                              "elasticdeform/deform.c",
                              "elasticdeform/from_nd_image.c"])],
      include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)
