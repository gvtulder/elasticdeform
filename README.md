Elastic deformations for N-dimensional images (Python, SciPy, NumPy)
====================================================================

This library implements elastic grid-based deformations for N-dimensional images.

The elastic deformation approach is described in
*   Ronneberger, Fischer, and Brox, "U-Net: Convolutional Networks for Biomedical
    Image Segmentation" (https://arxiv.org/abs/1505.04597)
*   Çiçek et al., "3D U-Net: Learning Dense Volumetric
    Segmentation from Sparse Annotation" (https://arxiv.org/abs/1606.06650)

The procedure generates a coarse displacement grid with a random displacement
for each grid point. This grid is then interpolated to compute a displacement for
each pixel in the input image. The input image is then deformed using the
displacement vectors and a spline interpolation.


Installation
------------

```
pip install elasticdeform
or
pip install git+https://github.com/gvtulder/elasticdeform
```

This library requires Python 3 and NumPy development headers.


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

License information
-------------------

This library was written by Gijs van Tulder (https://vantulder.net/) at the
Biomedical Imaging Group Rotterdam, Erasmus MC, Rotterdam, the Netherlands
(https://www.bigr.nl/).

It is inspired by a similar, Python-based implementation by Florian Calvet
(https://github.com/fcalvet/image_tools). This C-based implementation gives
the same results but is faster.

This C implementation includes a modified version of the `NI_GeometricTransform`
from SciPy's ndimage library. (https://github.com/scipy/scipy/blob/28636fbc3f16d562eab7b823546276111f6da98a/scipy/ndimage/src/ni_interpolation.c#L242)

This code is made available under the BSD license. See ``LICENSE.txt`` for details.

https://github.com/gvtulder/elasticdeform
