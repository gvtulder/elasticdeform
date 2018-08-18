import numpy as np
import scipy.ndimage
import unittest
import itertools

import elasticdeform

# Python implementation
def deform_grid_py(X, displacement, order=3, mode='constant', cval=0.0, crop=None, prefilter=True, axis=None):
    if axis is None:
        axis = tuple(range(X.ndim))
    elif isinstance(axis, int):
        axis = (axis,)

    # compute number of control points in each dimension
    points = [displacement[0].shape[d] for d in range(len(axis))]

    # creates the grid of coordinates of the points of the image (an ndim array per dimension)
    coordinates = np.meshgrid(*[np.arange(X.shape[d]) for d in axis], indexing='ij')
    # creates the grid of coordinates of the points of the image in the "deformation grid" frame of reference
    xi = np.meshgrid(*[np.linspace(0, p - 1, X.shape[d]) for d, p in zip(axis, points)], indexing='ij')

    if crop is not None:
        coordinates = [c[crop] for c in coordinates]
        xi = [x[crop] for x in xi]
        # crop is given only for the axes in axis, convert to all dimensions for the output
        crop = tuple(crop[axis.index(i)] if i in axis else slice(None) for i in range(X.ndim))
    else:
        crop = (slice(None),) * X.ndim

    # add the displacement to the coordinates
    for i in range(len(axis)):
        yd = scipy.ndimage.map_coordinates(displacement[i], xi, order=3)
        # adding the displacement
        coordinates[i] = np.add(coordinates[i], yd)

    out = np.zeros(X[crop].shape, dtype=X.dtype)
    # iterate over the non-deformed axes
    iter_axes = [range(X.shape[d]) if d not in axis else [slice(None)]
                 for d in range(X.ndim)]
    for a in itertools.product(*iter_axes):
        scipy.ndimage.map_coordinates(X[a], coordinates, output=out[a],
                                      order=order, cval=cval, mode=mode, prefilter=prefilter)
    return out

# C implementation wrapper
def deform_grid_c(X_in, displacement, order=3, mode='constant', cval=0.0, crop=None, prefilter=True, axis=None):
    return elasticdeform.deform_grid(X_in, displacement, order, mode, cval, crop, prefilter, axis)


class TestDeformGrid(unittest.TestCase):
    def test_random(self):
        for points in (3, (3, 5)):
            for shape in ((100, 100), (100, 75)):
                for order in (0, 1, 2, 3, 4):
                    X = np.random.rand(*shape)
                    elasticdeform.deform_random_grid(X, points=points)

    def test_basic_2d(self):
        for points in ((3, 3), (3, 5), (1, 5)):
            for shape in ((100, 100), (100, 75)):
                for order in (0, 1, 2, 3, 4):
                    for mode in ('nearest', 'wrap', 'reflect', 'mirror', 'constant'):
                        self.run_comparison(shape, points, order=order, mode=mode)

    def test_basic_3d(self):
        for points in ((3, 3, 3), (3, 5, 7), (1, 3, 5)):
            for shape in ((50, 50, 50), (100, 50, 25)):
                for order in (0, 1, 2, 3, 4):
                    self.run_comparison(shape, points, order=order)

    def test_crop_2d(self):
        points = (3, 3)
        shape = (100, 100)
        for crop in ((slice(0, 50), slice(0, 50)),
                     (slice(20, 60), slice(20, 60)),
                     (slice(50, 100), slice(50, 100))):
            for order in (0, 1, 2, 3, 4):
                self.run_comparison(shape, points, crop=crop, order=order)

    def test_crop_3d(self):
        points = (3, 3, 5)
        shape = (25, 25, 25)
        order = 3
        for crop in ((slice(15, 25), slice(None), slice(None)),):
            self.run_comparison(shape, points, crop=crop, order=order)

    def test_multi_2d(self):
        points = (3, 3)
        shape = (100, 75)
        sigma = 25
        for order in (0, 1, 2, 3, 4, [0, 3]):
            for crop in (None, (slice(15, 25), slice(15, 50))):
                for cval in (0.0, 1.0, [0.0, 1.0]):
                    for mode in ('constant', ['constant', 'reflect']):
                        # generate random displacement vector
                        displacement = np.random.randn(len(shape), *points) * sigma
                        # generate random data
                        X = np.random.rand(*shape).astype('float64')
                        # generate more random data, force a different data type
                        Y = np.random.rand(*shape).astype('float32')

                        # test and compare
                        order_list = order if isinstance(order, list) else [order] * 2
                        mode_list = mode if isinstance(mode, list) else [mode] * 2
                        cval_list = cval if isinstance(cval, list) else [cval] * 2
                        res_X_ref = deform_grid_py(X, displacement, order=order_list[0], crop=crop, cval=cval_list[0], mode=mode_list[0])
                        res_Y_ref = deform_grid_py(Y, displacement, order=order_list[1], crop=crop, cval=cval_list[1], mode=mode_list[1])
                        [res_X_test, res_Y_test] = deform_grid_c([X, Y], displacement, order=order, crop=crop, cval=cval, mode=mode)

                        np.testing.assert_array_almost_equal(res_X_ref, res_X_test)
                        np.testing.assert_array_almost_equal(res_Y_ref, res_Y_test)

    def test_multi_3d(self):
        points = (3, 3, 3)
        shape = (25, 25, 30)
        sigma = 25
        for order in (0, 1, 2, 3, 4):
            for crop in (None, (slice(15, 20), slice(15, 25), slice(2, 10))):
                # generate random displacement vector
                displacement = np.random.randn(len(shape), *points) * sigma
                # generate random data
                X = np.random.rand(*shape)
                # generate more random data
                Y = np.random.rand(*shape)

                # test and compare
                res_X_ref = deform_grid_py(X, displacement, order=order, crop=crop)
                res_Y_ref = deform_grid_py(Y, displacement, order=order, crop=crop)
                [res_X_test, res_Y_test] = deform_grid_c([X, Y], displacement, order=order, crop=crop)

                np.testing.assert_array_almost_equal(res_X_ref, res_X_test)
                np.testing.assert_array_almost_equal(res_Y_ref, res_Y_test)

    def test_different_strides(self):
        # test for multiple inputs with unequal strides
        shape = (200, 150)

        X = np.random.rand(*shape)
        Y = np.array(X, order='F')
        # the inputs have the same values, but different strides
        self.assertNotEqual(X.strides, Y.strides)

        displacement = np.random.randn(2, 3, 3) * 25
        # test and compare
        # disable prefiltering, since that would create new input arrays with equal strides
        res_X_ref = deform_grid_py(X, displacement, prefilter=False)
        res_Y_ref = deform_grid_py(Y, displacement, prefilter=False)
        [res_X_test, res_Y_test] = deform_grid_c([X, Y], displacement, prefilter=False)

    def test_axis(self):
        self.run_comparison(shape=(30, 20, 3), points=(3, 3), axis=(0, 1))
        self.run_comparison(shape=(20, 3, 30), points=(3, 3), axis=(0, 2))
        self.run_comparison(shape=(100, 200, 3), points=(3, 3), axis=(0, 1))
        self.run_comparison(shape=(200, 3, 100), points=(3, 3), axis=(0, 2))
        self.run_comparison(shape=(200, 3, 100, 4), points=(3, 3), axis=(0, 2))

        # test multiple inputs, same axes
        X = np.random.rand(3, 90, 80, 7)
        Y = np.random.rand(7, 90, 80)
        displacement = np.random.randn(2, 5, 3) * 25
        res_X_ref = deform_grid_py(X, displacement, axis=(1, 2))
        res_Y_ref = deform_grid_py(Y, displacement, axis=(1, 2))
        res_X_test, res_Y_test = deform_grid_c([X, Y], displacement, axis=(1, 2))
        np.testing.assert_array_almost_equal(res_X_ref, res_X_test)
        np.testing.assert_array_almost_equal(res_Y_ref, res_Y_test)

        # test multiple inputs, different axes
        X = np.random.rand(3, 20, 30)
        Y = np.random.rand(20, 30)
        displacement = np.random.randn(2, 5, 3) * 25
        res_X_ref = deform_grid_py(X, displacement, axis=(1, 2))
        res_Y_ref = deform_grid_py(Y, displacement, axis=(0, 1))
        res_X_test, res_Y_test = deform_grid_c([X, Y], displacement, axis=[(1, 2), (0, 1)])
        np.testing.assert_array_almost_equal(res_X_ref, res_X_test)
        np.testing.assert_array_almost_equal(res_Y_ref, res_Y_test)

        # test multiple inputs, with cropping
        X = np.random.rand(3, 90, 80, 7)
        Y = np.random.rand(7, 90, 80)
        displacement = np.random.randn(2, 5, 3) * 25
        for crop in [(slice(30, 50), slice(20, 40)), (slice(0, 30), slice(0, 80))]:
            res_X_ref = deform_grid_py(X, displacement, axis=(1, 2), crop=crop)
            res_Y_ref = deform_grid_py(Y, displacement, axis=(1, 2), crop=crop)
            res_X_test, res_Y_test = deform_grid_c([X, Y], displacement, axis=(1, 2), crop=crop)
            np.testing.assert_array_almost_equal(res_X_ref, res_X_test)
            np.testing.assert_array_almost_equal(res_Y_ref, res_Y_test)

    def run_comparison(self, shape, points, order=3, sigma=25, crop=None, mode='constant', axis=None):
        # generate random displacement vector
        displacement = np.random.randn(len(shape) if axis is None else len(axis), *points) * sigma
        # generate random data
        X = np.random.rand(*shape)

        # test and compare
        res_ref = deform_grid_py(X, displacement, order=order, crop=crop, mode=mode, axis=axis)
        res_test = deform_grid_c(X, displacement, order=order, crop=crop, mode=mode, axis=axis)

        np.testing.assert_array_almost_equal(res_ref, res_test)


if __name__ == '__main__':
    unittest.main()

