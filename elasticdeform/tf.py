import tensorflow
import elasticdeform

def deform_grid(X, *args, **kwargs):
    """
    Elastic deformation with a deformation grid, wrapped in a TensorFlow Op.

    This function wraps the ``elasticdeform.deform_grid`` function in a TensorFlow Op
    with a custom gradient.

    Parameters
    ----------
    X : Tensor or list of Tensors
        input image or list of input images
    displacement : numpy array
        displacement vectors for each control point

    Returns
    -------
    Tensor
       the deformed image, or a list of deformed images

    See Also
    --------
    elasticdeform.deform_grid : for the other parameters
    """
    @tensorflow.custom_gradient
    def f(*xs):
        def fwd(*xs):
            return elasticdeform.deform_grid(list(xs), *args, **kwargs)

        def bwd(*dys):
            def grad(*dys_xs):
                dys = dys_xs[:len(xs)]
                X_shape = [x.shape for x in dys_xs[len(xs):]]
                return elasticdeform.deform_grid_gradient(list(dys), *args, X_shape=X_shape, **kwargs)
            return tensorflow.py_func(grad, dys + xs, [dy.dtype for dy in dys],
                                      stateful=False, name='DeformGridGrad')

        y = tensorflow.py_func(fwd, xs, [x.dtype for x in xs], stateful=False, name='DeformGrid')
        return y, bwd

    if isinstance(X, (list, tuple)):
        return f(*X)
    else:
        return f(X)[0]
