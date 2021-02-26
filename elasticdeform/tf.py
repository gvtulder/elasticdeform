import numpy
import tensorflow
import elasticdeform

def deform_grid(X, displacement, *args, **kwargs):
    """
    Elastic deformation with a deformation grid, wrapped in a TensorFlow Op.

    This function wraps the ``elasticdeform.deform_grid`` function in a TensorFlow Op
    with a custom gradient.

    Parameters
    ----------
    X : Tensor or list of Tensors
        input image or list of input images
    displacement : Tensor or numpy array
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
    def f(displacement, *xs):
        def fwd(displacement, *xs):
            return elasticdeform.deform_grid(list(xs), displacement, *args, **kwargs)

        def bwd(*dys):
            def grad(*dys_disp_xs):
                dys = dys_disp_xs[:len(xs)]
                displacement = dys_disp_xs[len(xs)]
                X_shape = [x.shape for x in dys_disp_xs[len(xs) + 1:]]
                dXs = elasticdeform.deform_grid_gradient(list(dys), displacement,
                                                         *args, X_shape=X_shape, **kwargs)
                return [numpy.nan * displacement] + dXs
            grad_inputs = dys + (displacement,) + xs
            grad_output_dtypes = [displacement.dtype] + [x.dtype for x in xs]
            if hasattr(tensorflow, 'py_func'):
                # TensorFlow 1
                return tensorflow.py_func(grad, grad_inputs, grad_output_dtypes,
                                          stateful=False, name='DeformGridGrad')
            else:
                # TensorFlow 2
                return tensorflow.numpy_function(grad, grad_inputs, grad_output_dtypes,
                                                 name='DeformGridGrad')

        inputs = (displacement,) + xs
        output_dtypes = [x.dtype for x in xs]
        if hasattr(tensorflow, 'py_func'):
            # TensorFlow 1
            y = tensorflow.py_func(fwd, inputs, output_dtypes, stateful=False, name='DeformGrid')
        else:
            # TensorFlow 2
            y = tensorflow.numpy_function(fwd, inputs, output_dtypes, name='DeformGrid')
        return y, bwd

    if isinstance(X, (list, tuple)):
        return f(displacement, *X)
    else:
        if hasattr(tensorflow, 'py_func'):
            # TensorFlow 1
            return f(displacement, X)[0]
        else:
            # TensorFlow 2
            return f(displacement, X)
