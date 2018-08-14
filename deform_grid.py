import numpy
import scipy.ndimage

import _deform_grid

def deform_random_grid(X, sigma=25, points=3, **kwargs):
    """
    Elastic deformation with a random deformation grid

    Parameters
    ----------
    sigma: standard deviation of the normal distribution
    points: number of points of the each side of the square grid

    See deform_grid for more details.
    """
    if isinstance(X, numpy.ndarray):
        X_shape = X.shape
    else:
        assert isinstance(X, (list, tuple))
        X_shape = X[0].shape

    if not isinstance(points, (list, tuple)):
        points = [points] * len(X_shape)

    displacement = numpy.random.randn(len(X_shape), *points) * sigma
    return deform_grid(X, displacement, **kwargs)


def deform_grid(X, displacement, order=3, mode='constant', cval=0.0, crop=None):
    """
    Elastic deformation with a deformation grid

    Parameters
    ----------
    X: image, or list of images of the same size
    displacement: displacement vectors for each control point
    order: interpolation order
    mode: border mode (nearest, wrap, reflect, mirror, constant)
    cval: constant value to be used if mode == 'constant'
    crop: None, or a list of slice() objects to crop the output

    displacement is a NumPy array with displacement vectors for each
    control points. For example, to deform a 2D image with 3 x 5 control
    points, provide a displacement matrix of shape 2 x 3 x 5.

    If X is a list of images, the values for order, mode and cval can be lists
    to specify a different value for every image in X.

    crop can be a list of slice() objects to crop the output with.
    Only very simple slicing is supported: the slice start and stop values must
    be positive and should not be larger than the output.

    Returns
    -------
    Returns the deformed image, or a list of deformed images if a list
    of inputs is given.

    Notes
    -----
    See the SciPy documentation for scipy.ndimage.interpolation.map_coordinates
    for more details on some of the parameters.

    Based on Florian Calvet's Python implementation.

    Elastic deformation approach found in
        Ronneberger, Fischer, and Brox, "U-Net: Convolutional Networks for Biomedical
        Image Segmentation" also used in Çiçek et al., "3D U-Net: Learning Dense Volumetric
        Segmentation from Sparse Annotation"
    based on a coarsed displacement grid interpolated to generate displacement for every pixel
    deemed to represent more realistic, biologically explainable deformation of the image
    for each dimension, a value for the displacement is generated on each point of the grid
    then interpolated to give an array of displacement values, which is then added to the corresponding array of coordinates
    the resulting (list of) array of coordinates is mapped to the original image to give the final image
    """
    if isinstance(X, numpy.ndarray):
        Xs = [X]
    elif isinstance(X, list):
        Xs = X
    else:
        raise Exception('X should be a numpy.ndarray or a list of numpy.ndarrays.')

    # check X inputs
    assert len(Xs) > 0, 'You must provide at least one image.'
    assert all(isinstance(x, numpy.ndarray) for x in Xs), 'All elements of X should be numpy.ndarrays.'
    n_inputs = len(Xs)
    X_ndim = Xs[0].ndim
    X_shape = Xs[0].shape
    X_strides = Xs[0].strides
    assert all(X_shape == x.shape for x in Xs), 'All elements of X should have the same shape.'
    assert all(x.flags['C_CONTIGUOUS'] for x in Xs), 'All elements of X must be contiguous.'

    # prepare output cropping
    if crop is not None:
        assert isinstance(crop, (tuple, list)), "crop must be a tuple or a list."
        assert len(crop) == X_ndim
        output_shape = [X_shape[d] for d in range(X_ndim)]
        output_offset = [0 for d in range(X_ndim)]
        for d in range(X_ndim):
            if isinstance(crop[d], slice):
                assert crop[d].step is None
                start = (crop[d].start or 0)
                stop = (crop[d].stop or X_shape[d])
                assert start >= 0
                assert start < stop and stop <= X_shape[d]
                output_shape[d] = stop - start
                if start > 0:
                    output_offset[d] = start
            else:
                raise Exception('Crop must be a slice.')
        if any(o > 0 for o in output_offset):
            output_offset = numpy.array(output_offset).astype('int64')
        else:
            output_offset = None
    else:
        output_shape = X_shape
        output_offset = None

    # check displacement
    assert isinstance(displacement, numpy.ndarray), 'Displacement matrix should be a numpy.ndarray.'
    assert displacement.ndim == X_ndim + 1, 'Number of dimensions of displacement does not match X.'
    assert displacement.shape[0] == X_ndim, 'First dimension of displacement should match dimensions in X.'
    # prepare order
    if not isinstance(order, (tuple, list)):
        order = [order] * n_inputs
    assert n_inputs == len(order), 'Number of order parameters should be equal to number of inputs.'
    assert all(0 <= o and o <= 5 for o in order), 'order should be 0, 1, 2, 3, 4 or 5.'
    order = numpy.array(order).astype('int64')

    # prepare mode
    if not isinstance(mode, (tuple, list)):
        mode = [mode] * n_inputs
    mode = [_extend_mode_to_code(o) for o in mode]
    assert n_inputs == len(mode), 'Number of mode parameters should be equal to number of inputs.'
    mode = numpy.array(mode).astype('int64')

    # prepare cval
    if not isinstance(cval, (tuple, list)):
        cval = [cval] * n_inputs
    assert n_inputs == len(cval), 'Number of cval parameters should be equal to number of inputs.'
    cval = numpy.array(cval).astype('float64')

    # prefilter inputs
    Xs_f = []
    for i in range(n_inputs):
        if order[i] > 1:
            Xs_f.append(scipy.ndimage.spline_filter(Xs[i], order=order[i]))
        else:
            Xs_f.append(Xs[i])
    # prefilter displacement
    displacement_f = numpy.zeros_like(displacement)
    for d in range(1, displacement.ndim):
        scipy.ndimage.spline_filter1d(displacement, axis=d, order=3, output=displacement_f)
        displacement = displacement_f

    # prepare output arrays
    outputs = [numpy.zeros(output_shape, dtype=x.dtype) for x in Xs]
    _deform_grid.deform_grid(Xs_f, displacement_f, output_offset, outputs, order, mode, cval)

    if isinstance(X, list):
        return outputs
    else:
        return outputs[0]




def _extend_mode_to_code(mode):
    """Convert an extension mode to the corresponding integer code.
    """
    if mode == 'nearest':
        return 0
    elif mode == 'wrap':
        return 1
    elif mode == 'reflect':
        return 2
    elif mode == 'mirror':
        return 3
    elif mode == 'constant':
        return 4
    else:
        raise RuntimeError('boundary mode not supported')

