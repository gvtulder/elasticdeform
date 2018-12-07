from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

with open("README.md") as f:
    readme_txt = f.read()

setup(name='elasticdeform',
      version='0.4.0',
      description='Elastic deformations for N-D images.',
      long_description_content_type='text/markdown',
      long_description=readme_txt,
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
      cmdclass={'build_ext': build_ext},
      setup_requires=['numpy'],
      install_requires=['numpy', 'scipy'],
      classifiers=[
          "Programming Language :: Python :: 2",
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: BSD License",
      ],
)
