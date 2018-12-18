import os
import numpy as np
import scipy.ndimage
import unittest
import itertools

try:
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
except Exception as e:
    print(e)
    tf = None

import elasticdeform
if tf is not None:
    import elasticdeform.tf as etf

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
def deform_grid_gradient_c(X_in, displacement, order=3, mode='constant', cval=0.0, crop=None, prefilter=True, axis=None, X_shape=None):
    return elasticdeform.deform_grid_gradient(X_in, displacement, order, mode, cval, crop, prefilter, axis, X_shape)


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

    def test_grad_2d(self):
        points = (3, 5)
        shape = (30, 25)
        for order in (0, 1, 2, 3, 4):
            for mode in ('nearest', 'wrap', 'reflect', 'mirror', 'constant'):
                X = np.random.rand(*shape)
                displacement = np.random.randn(2, *points) * 3
                def fn(X):
                    return deform_grid_c(X, displacement, order=order, mode=mode)
                def grad_fn(gY, X):
                    return deform_grid_gradient_c(gY, displacement, order=order, mode=mode)
                self.verify_grad(X, fn, grad_fn, n_tests=5)

    def test_grad_crop(self):
        points = (3, 3)
        shape = (20, 20)
        for crop in ((slice(0, 10), slice(0, 10)),
                     (slice(4, 12), slice(4, 12)),
                     (slice(10, 20), slice(10, 20))):
            X = np.random.rand(*shape)
            displacement = np.random.randn(2, *points) * 3
            def fn(X):
                return deform_grid_c(X, displacement, crop=crop)
            def grad_fn(gY, X):
                return deform_grid_gradient_c(gY, displacement, crop=crop, X_shape=shape)
            self.verify_grad(X, fn, grad_fn)

    def test_grad_with_list(self):
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
                        # compute forward
                        Xdeformed, Ydeformed = deform_grid_c([X, Y], displacement, order=order, crop=crop, cval=cval, mode=mode)
                        # generate random gradients
                        dXdeformed = np.random.rand(*Xdeformed.shape).astype('float64')
                        dYdeformed = np.random.rand(*Ydeformed.shape).astype('float32')

                        # test and compare
                        order_list = order if isinstance(order, list) else [order] * 2
                        mode_list = mode if isinstance(mode, list) else [mode] * 2
                        cval_list = cval if isinstance(cval, list) else [cval] * 2
                        res_dX_ref = deform_grid_gradient_c(dXdeformed, displacement, order=order_list[0], crop=crop, cval=cval_list[0], mode=mode_list[0], X_shape=X.shape)
                        res_dY_ref = deform_grid_gradient_c(dYdeformed, displacement, order=order_list[1], crop=crop, cval=cval_list[1], mode=mode_list[1], X_shape=Y.shape)
                        [res_dX_test, res_dY_test] = deform_grid_gradient_c([dXdeformed, dYdeformed], displacement, order=order, crop=crop, cval=cval, mode=mode, X_shape=[X.shape, Y.shape])

                        np.testing.assert_array_almost_equal(res_dX_ref, res_dX_test)
                        np.testing.assert_array_almost_equal(res_dY_ref, res_dY_test)

    def verify_grad(self, X, fn, grad_fn, eps=1e-4, n_tests=10):
        # test the gradient computed by grad_fn by comparing it with the numeric gradient of fn
        output_shape = fn(X).shape

        # test for multiple random projections
        for t in range(n_tests):
            random_projection = np.random.rand(*output_shape) + 0.5

            # define a gradient cost function
            def cost_fn(x):
                return np.sum(fn(x) * random_projection)

            # compute baseline result at X
            f_x = cost_fn(X)

            # initialize input that we can disturb later
            X_copy = X.copy()

            # iterate over all elements of X and compute the gradient
            gx_ref = np.zeros_like(X)
            for i in range(X.size):
                X_copy[:] = X
                X_copy.flat[i] += eps
                f_eps = cost_fn(X_copy)
                gx_ref.flat[i] = ((f_eps - f_x) / eps)

            # now compute the gradient directly using grad_fn
            gx_test = grad_fn(random_projection, X)
            np.testing.assert_array_almost_equal(gx_ref, gx_test)

    def run_comparison(self, shape, points, order=3, sigma=25, crop=None, mode='constant', axis=None):
        # generate random displacement vector
        displacement = np.random.randn(len(shape) if axis is None else len(axis), *points) * sigma
        # generate random data
        X = np.random.rand(*shape)

        # test and compare
        res_ref = deform_grid_py(X, displacement, order=order, crop=crop, mode=mode, axis=axis)
        res_test = deform_grid_c(X, displacement, order=order, crop=crop, mode=mode, axis=axis)

        np.testing.assert_array_almost_equal(res_ref, res_test)

    def test_basic_2d_tensorflow(self):
        points = (3, 3)
        shape = (100, 100)
        for order in (0, 1, 2):
            for crop in (None, (slice(20, 80), slice(30, 70))):
                for mode in ('nearest', 'wrap', 'reflect', 'mirror', 'constant'):
                    self.run_comparison_tensorflow(shape, points, order=order, mode=mode, crop=crop)

    def run_comparison_tensorflow(self, shape, points, order=3, sigma=25, crop=None, mode='constant', axis=None):
        if tf is None:
            raise unittest.SkipTest("TensorFlow was not loaded.")

        # generate random displacement vector
        displacement = np.random.randn(len(shape) if axis is None else len(axis), *points) * sigma
        # generate random data
        X_val = np.random.rand(*shape)

        # compute forward reference value
        X_deformed_ref = elasticdeform.deform_grid(X_val, displacement, order=order, crop=crop, mode=mode, axis=axis)

        # generate gradient
        dY_val = np.random.rand(*X_deformed_ref.shape)

        # compute backward reference value
        dX_ref = elasticdeform.deform_grid_gradient(dY_val, displacement, order=order, crop=crop, mode=mode, axis=axis, X_shape=shape)

        # build tensorflow graph
        X = tf.Variable(X_val)
        dY = tf.Variable(dY_val)
        X_deformed = etf.deform_grid(X, displacement, order=order, crop=crop, mode=mode, axis=axis)
        [dX] = tf.gradients(X_deformed, X, dY)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            X_deformed_val, dX_val = sess.run([X_deformed, dX])

            np.testing.assert_almost_equal(X_deformed_ref, X_deformed_val)
            np.testing.assert_almost_equal(dX_ref, dX_val)


if __name__ == '__main__':
    unittest.main()

