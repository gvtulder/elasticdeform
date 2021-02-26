Elastic deformations for N-dimensional images (Python, SciPy, NumPy, TensorFlow, PyTorch)
=========================================================================================

[![Documentation Status](https://readthedocs.org/projects/elasticdeform/badge/?version=latest)](https://elasticdeform.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/gvtulder/elasticdeform.svg?branch=master)](https://travis-ci.com/gvtulder/elasticdeform)
[![Build status](https://ci.appveyor.com/api/projects/status/air4dambkpcummeh/branch/master?svg=true)](https://ci.appveyor.com/project/gvtulder/elasticdeform/branch/master)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4563172.svg)](https://doi.org/10.5281/zenodo.4563172)

This library implements elastic grid-based deformations for N-dimensional images.

The elastic deformation approach is described in
*   Ronneberger, Fischer, and Brox, "U-Net: Convolutional Networks for Biomedical
    Image Segmentation" (<https://arxiv.org/abs/1505.04597>)
*   Çiçek et al., "3D U-Net: Learning Dense Volumetric
    Segmentation from Sparse Annotation" (<https://arxiv.org/abs/1606.06650>)

The procedure generates a coarse displacement grid with a random displacement
for each grid point. This grid is then interpolated to compute a displacement for
each pixel in the input image. The input image is then deformed using the
displacement vectors and a spline interpolation.

In addition to the normal, forward deformation, this package also provides a
function that can backpropagate the gradient through the deformation. This makes
it possible to use the deformation as a layer in a convolutional neural network.
For convenience, TensorFlow and PyTorch wrappers are provided in `elasticdeform.tf`
and `elasticdeform.torch`.


Installation
------------

```
pip install elasticdeform
or
pip install git+https://github.com/gvtulder/elasticdeform
```

This library requires Python 3 and NumPy development headers.

On Windows, try to install the precompiled binaries directly using `pip install elasticdeform`.
If that does not work, [these precompiled packages](https://www.lfd.uci.edu/~gohlke/pythonlibs/#elasticdeform) might be an alternative option.


Examples
--------

This basic example deforms an image with a random 3 x 3 deformation grid:
```python
import numpy, imageio, elasticdeform
X = numpy.zeros((200, 300))
X[::10, ::10] = 1

# apply deformation with a random 3 x 3 grid
X_deformed = elasticdeform.deform_random_grid(X, sigma=25, points=3)

imageio.imsave('test_X.png', X)
imageio.imsave('test_X_deformed.png', X_deformed)
```

### Multiple inputs

If you have multiple images, e.g., an image and a segmentation image, you can
deform both simultaneously by providing a list of inputs. You can specify
a different spline order for each input.
```python
# apply deformation to inputs X and Y
[X_deformed, Y_deformed] = elasticdeform.deform_random_grid([X, Y])

# apply deformation to inputs X and Y,
# with a different interpolation for each input
[X_deformed, Y_deformed] = elasticdeform.deform_random_grid([X, Y], order=[3, 0])
```

### Multi-channel images

By default, a deformation will be applied to every dimension of the input. If you
have multi-channel images, you can use the `axis` parameter to specify which axes
should be deformed. The same deformation will be applied for each channel.

For example, to deform an RGB image across the first two dimensions, run:
```python
X_deformed = elasticdeform.deform_random_grid(X, axis=(0, 1))
```

When deforming multiple inputs, you can provide a tuple of axes for each input:
```python
X = numpy.random.rand(3, 200, 300)
Y = numpy.random.rand(200, 300)
[X_deformed, Y_deformed] = elasticdeform.deform_random_grid([X, Y], axis=[(1, 2), (0, 1)])
```

### Cropping

If you intend to crop a small subpatch from the deformed image, you can provide
the crop dimensions to the deform function. It will then compute only the cropped
output pixels, while still computing the deformation grid based on the full image
dimensions. This saves computation time.
```python
X = numpy.random.rand(200, 300)

# define a crop region
crop = (slice(50, 150), slice(0, 100))

# generate a deformation grid
displacement = numpy.random.randn(2, 3, 3) * 25

# deform full image
X_deformed = elasticdeform.deform_grid(X, displacement)
# compute only the cropped region
X_deformed_crop = elasticdeform.deform_grid(X, displacement, crop=crop)

# the deformation is the same
numpy.testing.assert_equal(X_deformed[crop], X_deformed_crop)
```


### Rotate and zoom

The deformation functions accept `rotate` and `zoom` parameters, which allows you
to combine the elastic deformation with rotation and scaling. This can be useful
as data augmentation step. The rotation and zoom are applied to the output
coordinates, using the center pixel of the output patch as the origin.
```python
# apply deformation with a random 3 x 3 grid,
# rotate by 30 degrees and rescale with a factor 1.5
X_deformed = elasticdeform.deform_random_grid(X, sigma=25, points=3,
                                              rotate=30, zoom=1.5)
```
Note that the output shape remains the same. The mapping of the input to the
output is rotated within the given output frame.

Rotate and zoom can be combined with the `crop` argument. In that case, the
scaling and rotation is performed relative to the center of the cropped output.

For more advanced transformations, it is also possible to provide an affine
transformation matrix directly.


### Gradient

The `deform_grid_gradient` function can be used to backpropagate the gradient of
the output with respect to the input. Call `deform_grid_gradient` with the
parameters that were used for the forward step.
```python
X = numpy.random.rand(200, 300)

# generate a deformation grid
displacement = numpy.random.randn(2, 3, 3) * 25

# perform forward deformation
X_deformed = elasticdeform.deform_grid(X, displacement)

# obtain the gradient w.r.t. X_deformed (e.g., with backpropagation)
dX_deformed = numpy.random.randn(*X_deformed.shape)

# compute the gradient w.r.t. X
dX = elasticdeform.deform_grid_gradient(dX_deformed, displacement)
```

Note: The gradient function will assume that the input has the same size as the
output. If you used the `crop` parameter in the forward phase, it is necessary to
provide the gradient function with the original, uncropped input shape in the
`X_shape` parameter.


### TensorFlow wrapper

The `elasticdeform.tf` module provides a wrapper for `deform_grid` in TensorFlow.
The function uses TensorFlow Tensors as input and output, but otherwise uses
the same parameters.
```python
import numpy
import elasticdeform.tf as etf

displacement_val = numpy.random.randn(2, 3, 3) * 5
X_val = numpy.random.rand(200, 300)
dY_val = numpy.random.rand(200, 300)

# construct TensorFlow input and top gradient
displacement = tf.Variable(displacement_val)
X = tf.Variable(X_val)
dY = tf.Variable(dY_val)

# the deform_grid function is similar to the plain Python equivalent,
# but it accepts and returns TensorFlow Tensors
X_deformed = etf.deform_grid(X, displacement, order=3)

# the gradient w.r.t. X can be computed in the normal TensorFlow manner
[dX] = tf.gradients(X_deformed, X, dY)
```


### PyTorch wrapper

The `elasticdeform.torch` module provides a wrapper for `deform_grid` in PyTorch.
The function uses PyTorch Tensors as input and output, but otherwise uses
the same parameters.
```python
import numpy
import elasticdeform.torch as etorch

displacement_val = numpy.random.randn(2, 3, 3) * 5
X_val = numpy.random.rand(200, 300)
dY_val = numpy.random.rand(200, 300)

# construct PyTorch input and top gradient
displacement = torch.tensor(displacement_val)
X = torch.tensor(X_val, requires_grad=True)
dY = torch.tensor(dY_val)

# the deform_grid function is similar to the plain Python equivalent,
# but it accepts and returns PyTorch Tensors
X_deformed = etorch.deform_grid(X, displacement, order=3)

# the gradient w.r.t. X can be computed in the normal PyTorch manner
X_deformed.backward(dY)
print(X.grad)
```


License information
-------------------

This library was written by [Gijs van Tulder](https://vantulder.net/) at the
[Biomedical Imaging Group Rotterdam](https://www.bigr.nl/),
Erasmus MC, Rotterdam, the Netherlands

It is inspired by a similar, Python-based implementation by
[Florian Calvet](https://github.com/fcalvet/image_tools).
This C-based implementation gives the same results, but is faster and has
a gradient implementation.

This C implementation includes a modified version of the `NI_GeometricTransform`
from [SciPy's ndimage library](https://github.com/scipy/scipy/blob/28636fbc3f16d562eab7b823546276111f6da98a/scipy/ndimage/src/ni_interpolation.c#L242).

This code is made available under the BSD license. See ``LICENSE.txt`` for details.

If you want to cite this library, please use [DOI:10.5281/zenodo.4563172](https://doi.org/10.5281/zenodo.4563172).

* <https://github.com/gvtulder/elasticdeform>
* <https://elasticdeform.readthedocs.io/>
