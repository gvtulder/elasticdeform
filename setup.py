from distutils.core import setup, Extension
import numpy.distutils.misc_util

setup(
    ext_modules=[Extension("_deform_grid", ["_deform_grid.c", "deform.c", "from_nd_image.c"])],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)
