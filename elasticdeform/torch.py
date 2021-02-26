import numpy
import torch
import elasticdeform

class ElasticDeform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, displacement, deform_args, deform_kwargs, *xs):
        ctx.save_for_backward(displacement)
        ctx.deform_args = deform_args
        ctx.deform_kwargs = deform_kwargs
        ctx.x_shapes = [x.shape for x in xs]

        xs_numpy = [x.detach().cpu().numpy() for x in xs]
        displacement = displacement.detach().cpu().numpy()
        ys = elasticdeform.deform_grid(xs_numpy, displacement, *deform_args, **deform_kwargs)
        return tuple(torch.tensor(y, device=x.device) for x, y in zip(xs, ys))

    @staticmethod
    def backward(ctx, *dys):
        displacement, = ctx.saved_tensors
        deform_args = ctx.deform_args
        deform_kwargs = ctx.deform_kwargs
        x_shapes = ctx.x_shapes

        dys_numpy = [dy.detach().cpu().numpy() for dy in dys]
        displacement = displacement.detach().cpu().numpy()
        dxs = elasticdeform.deform_grid_gradient(dys_numpy, displacement,
                                                 *deform_args, X_shape=x_shapes, **deform_kwargs)
        return (None, None, None) + tuple(torch.tensor(dx, device=dy.device) for dx, dy in zip(dxs, dys))



def deform_grid(X, displacement, *args, **kwargs):
    """
    Elastic deformation with a deformation grid, wrapped for PyTorch.

    This function wraps the ``elasticdeform.deform_grid`` function in a PyTorch function
    with a custom gradient.

    Parameters
    ----------
    X : torch.Tensor or list of torch.Tensors
        input image or list of input images
    displacement : torch.Tensor
        displacement vectors for each control point

    Returns
    -------
    torch.Tensor
       the deformed image, or a list of deformed images

    See Also
    --------
    elasticdeform.deform_grid : for the other parameters
    """
    if not isinstance(X, (list, tuple)):
        X_list = [X]
    else:
        X_list = X
    displacement = torch.as_tensor(displacement)
    y = ElasticDeform.apply(displacement, args, kwargs, *X_list)

    if isinstance(X, (list, tuple)):
        return y
    else:
        return y[0]
