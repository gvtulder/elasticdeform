import numpy as np
import scipy.ndimage
import unittest

import deform as deform_grid

# Python implementation
def deform_grid_py(X, displacement, order=3, mode='constant', cval=0.0, crop=None):
    # compute number of control points in each dimension
    points = [displacement[0].shape[d] for d in range(X.ndim)]

    # creates the grid of coordinates of the points of the image (an ndim array per dimension)
    coordinates = np.meshgrid(*[np.arange(X.shape[d]) for d in range(X.ndim)], indexing='ij')
    # creates the grid of coordinates of the points of the image in the "deformation grid" frame of reference
    xi = np.meshgrid(*[np.linspace(0, points[d] - 1, X.shape[d]) for d in range(X.ndim)], indexing='ij')

    if crop is not None:
        coordinates = [c[crop] for c in coordinates]
        xi = [x[crop] for x in xi]

    # add the displacement to the coordinates
    for i in range(X.ndim):
        yd = scipy.ndimage.map_coordinates(displacement[i], xi, order=3)
        # adding the displacement
        coordinates[i] = np.add(coordinates[i], yd)

    return scipy.ndimage.map_coordinates(X, coordinates, order=order, cval=cval, mode=mode)

# C implementation wrapper
def deform_grid_c(X_in, displacement, order=3, mode='constant', cval=0.0, crop=None):
    return deform_grid.deform_grid(X_in, displacement, order, mode, cval, crop)


class TestDeformGrid(unittest.TestCase):
    def test_random(self):
        for points in (3, (3, 5)):
            for shape in ((100, 100), (100, 75)):
                for order in (0, 1, 2, 3, 4):
                    X = np.random.rand(*shape)
                    deform_grid.deform_random_grid(X, points=points)

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

    def run_comparison(self, shape, points, order=3, sigma=25, crop=None, mode='constant'):
        # generate random displacement vector
        displacement = np.random.randn(len(shape), *points) * sigma
        # generate random data
        X = np.random.rand(*shape)

        # test and compare
        res_ref = deform_grid_py(X, displacement, order=order, crop=crop, mode=mode)
        res_test = deform_grid_c(X, displacement, order=order, crop=crop, mode=mode)

        np.testing.assert_array_almost_equal(res_ref, res_test)


if __name__ == '__main__':
    unittest.main()

