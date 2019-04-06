#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
import numpy.random as random
import sys
from dipy.align.fused_types cimport floating
from dipy.align import vector_fields as vf

from dipy.align.vector_fields cimport(_apply_affine_3d_x0,
                                      _apply_affine_3d_x1,
                                      _apply_affine_3d_x2,
                                      _apply_affine_2d_x0,
                                      _apply_affine_2d_x1)

from dipy.align.transforms cimport (Transform)

cdef extern from "dpy_math.h" nogil:
    double cos(double)
    double sin(double)
    double log(double)

class SSDMetricComputation(object):
    def __init__(self):
        r""" Compute ssd metric and derivatives of transformation matrix.

        Parameters
        ----------
        None

        Notes
        --------
        We need this class in cython to allow __gradient_dense_2d and
        _gradient_dense_3d to use a nogil Jacobian function (obtained
        from an instance of the Transform class), which allows us to evaluate
        Jacobians at all the sampling points (maybe the full grid) inside a
        nogil loop.

        The reason we need a class is to encapsulate all the parameters related to transformation matrix

        """
        self.setup_called = False

    def setup(self, static, moving):
        r""" Initializes static and moving images

        Parameters
        ----------
        static : array
            static image
        moving : array
            moving image

        """
        self.ssd_grad = None
        self.delta = np.zeros_like(moving)
        self.setup_called = True


    def update_delta_field(self, static, moving):
        r"""
        Computes delta for static and moving points

        Parameters:
        -----------
        sval: array, shape (n, ) or (S, R, C) - intensities of static image either sampled or full image

        mval: array, shape (n,) or (S, R, C) - intensities of moving image  either sampled
              of full image
        """
        if not self.setup_called:
            self.setup(static,moving)

        self.delta = (static - moving)

    def update_gradient_dense(self, theta, transform, static, moving,
                              grid2world, mgradient):
        r"""
        Compute the gradient with respect to transformation parameters.

        The gradient is stored in - self.ssd_gradient.

        Parameters
        ----------
        theta : array, shape (n,)
            parameters to compute the gradient at
        transform : instance of Transform
            the transformation with respect to whose parameters the gradient
            must be computed
        static : array, shape (S, R, C)
            static image
        moving : array, shape (S, R, C)
            moving image
        grid2world : array, shape (4, 4)
            we assume that both images have already been sampled at a common
            grid. This transform must map voxel coordinates of this common grid
            to physical coordinates of its corresponding voxel in the moving
            image. For example, if the moving image was sampled on the static
            image's grid (this is the typical setting) using an aligning
            matrix A, then

            (1) grid2world = A.dot(static_affine)

            where static_affine is the transformation mapping static image's
            grid coordinates to physical space.

        mgradient : array, shape (S, R, C, 3)
            the gradient of the moving image
        """
        if static.shape != moving.shape:
            raise ValueError("Images must have the same shape")
        dim = len(static.shape)
        if not dim in [2, 3]:
            msg = 'Only dimensions 2 and 3 are supported. ' +\
                str(dim) + ' received'
            raise ValueError(msg)

        if mgradient.shape != moving.shape + (dim,):
            raise ValueError('Invalid gradient field dimensions.')

        if not self.setup_called:
            self.setup(static, moving)

        n = theta.shape[0] ## number of transformation parameters

        if self.ssd_grad is None:
            self.ssd_grad = np.zeros((n,)) ## number of parameters

        if dim == 2:
            if mgradient.dtype == np.float64:
                gradient_dense_2d[cython.double](theta, transform,
                    static, moving, grid2world, mgradient,self.ssd_grad)
            elif mgradient.dtype == np.float32:
                gradient_dense_2d[cython.float](theta, transform,
                    static, moving, grid2world, mgradient,self.ssd_grad)
            else:
                raise ValueError('Grad. field dtype must be floating point')

        elif dim == 3:
            if mgradient.dtype == np.float64:
                gradient_dense_3d[cython.double](theta, transform,
                    static, moving, grid2world, mgradient,self.ssd_grad)
            elif mgradient.dtype == np.float32:
                gradient_dense_3d[cython.float](theta, transform,
                    static, moving, grid2world, mgradient, self.ssd_grad)
            else:
                raise ValueError('Grad. field dtype must be floating point')


    def update_gradient_sparse(self, theta, transform, sval, mval,
                               sample_points, mgradient):
        r""" Computes the Gradient of the  w.r.t. transform parameters


        The gradient is stored in self.ssd_gradient.

        Parameters
        ----------
        theta : array, shape (n,)
            parameters to compute the gradient at
        transform : instance of Transform
            the transformation with respect to whose parameters the gradient
            must be computed
        sval : array, shape (m,)
            sampled intensities from the static image at sampled_points
        mval : array, shape (m,)
            sampled intensities from the moving image at sampled_points
        sample_points : array, shape (m, 3)
            coordinates (in physical space) of the points the images were
            sampled at
        mgradient : array, shape (m, 3)
            the gradient of the moving image at the sample points
        """

        dim = sample_points.shape[1]
        if mgradient.shape[1] != dim:
            raise ValueError('Dimensions of gradients and points are different')

        nsamples = sval.shape[0]
        if ((mgradient.shape[0] != nsamples) or (mval.shape[0] != nsamples)
            or sample_points.shape[0] != nsamples):
            raise ValueError('Number of points and gradients are different.')

        if not mgradient.dtype in [np.float32, np.float64]:
            raise ValueError('Gradients dtype must be floating point')

        n = theta.shape[0]


        if self.ssd_grad is None:
            self.ssd_grad = np.zeros(shape=(n,))

        if dim == 2:
            if mgradient.dtype == np.float64:
                gradient_sparse_2d[cython.double](theta, transform,
                    sval, mval, sample_points, mgradient, self.ssd_grad)
            elif mgradient.dtype == np.float32:
                gradient_sparse_2d[cython.float](theta, transform,
                    sval, mval, sample_points, mgradient, self.ssd_grad)
            else:
                raise ValueError('Gradients dtype must be floating point')

        elif dim == 3:

            if mgradient.dtype == np.float64:
                gradient_sparse_3d[cython.double](theta, transform,
                    sval, mval, sample_points, mgradient, self.ssd_grad)
            elif mgradient.dtype == np.float32:
                gradient_sparse_3d[cython.float](theta, transform,
                    sval, mval, sample_points, mgradient,self.ssd_grad)
            else:
                raise ValueError('Gradients dtype must be floating point')

        else:
            msg = 'Only dimensions 2 and 3 are supported. ' + str(dim) +\
                ' received'
            raise ValueError(msg)



cdef gradient_dense_2d(double[:] theta, Transform transform,
                                  double[:, :] static,
                                  double[:, :] moving,
                                  double[:, :] grid2world,
                                  floating[:, :,:] mgradient,
                                  double[:] grad):
    r""" Gradient of the metric w.r.t. transform parameters theta

    Computes the vector of partial derivatives 
    each transformation parameter. The transformation itself is not necessary
    to compute the gradient, but only its Jacobian.

    Parameters
    ----------
    theta : array, shape (n,)
        parameters of the transformation to compute the gradient from
    transform : instance of Transform
        the transformation with respect to whose parameters the gradient
        must be computed
    static : array, shape (R, C)
        static image
    moving : array, shape (R, C)
        moving image
    grid2world : array, shape (3, 3)
        the grid-to-space transform associated with images static and moving
        (we assume that both images have already been sampled at a common grid)
    mgradient : array, shape (R, C, 2)
        the gradient of the moving image
    
    grad : array, shape (len(theta))
        the array to write the gradient to
    """
    cdef:
        cnp.npy_intp nrows = static.shape[0]
        cnp.npy_intp ncols = static.shape[1]
        cnp.npy_intp n = theta.shape[0]
        cnp.npy_intp offset, valid_points
        int constant_jacobian = 0
        cnp.npy_intp k, i, j, r, c
        double rn, cn
        double val, spline_arg, norm_factor
        double[:, :] J = np.empty(shape=(2, n), dtype=np.float64)
        double[:] prod = np.empty(shape=(n,), dtype=np.float64)
        double[:] x = np.empty(shape=(2,), dtype=np.float64)


    with nogil:
        valid_points = 0
        for i in range(nrows):
            for j in range(ncols):
                x[0] = _apply_affine_2d_x0(i, j, 1, grid2world)
                x[1] = _apply_affine_2d_x1(i, j, 1, grid2world)

                if constant_jacobian == 0:
                    constant_jacobian = transform._jacobian(theta, x, J)

                for k in range(n):
                    grad[k] += (J[0, k] * mgradient[i, j, 0] +
                               J[1, k] * mgradient[i, j, 1])


cdef gradient_dense_3d(double[:] theta, Transform transform,
                                  double[:, :, :] static,
                                  double[:, :, :] moving,
                                  double[:, :] grid2world,
                                  floating[:, :,:,:] mgradient,
                                  double[:] grad):
    r""" Gradient of the metric  w.r.t. transform parameters theta

    Computes the vector of partial derivatives of the  w.r.t.
    each transformation parameter. The transformation itself is not necessary
    to compute the gradient, but only its Jacobian.

    Parameters
    ----------
    theta : array, shape (n,)
        parameters of the transformation to compute the gradient from
    transform : instance of Transform
        the transformation with respect to whose parameters the gradient
        must be computed
    static : array, shape (S, R, C)
        static image
    moving : array, shape (S, R, C)
        moving image
    grid2world : array, shape (4, 4)
        the grid-to-space transform associated with images static and moving
        (we assume that both images have already been sampled at a common grid)
    mgradient : array, shape (S, R, C, 3)
        the gradient of the moving image
    grad: array, shape (len(theta),)
        the array to write the gradient to
    """
    cdef:
        cnp.npy_intp nslices = static.shape[0]
        cnp.npy_intp nrows = static.shape[1]
        cnp.npy_intp ncols = static.shape[2]
        cnp.npy_intp n = theta.shape[0]
        cnp.npy_intp offset, valid_points
        int constant_jacobian = 0
        cnp.npy_intp l, k, i, j, r, c
        double rn, cn
        double val, spline_arg, norm_factor
        double[:, :] J = np.empty(shape=(3, n), dtype=np.float64)
        double[:] prod = np.empty(shape=(n,), dtype=np.float64)
        double[:] x = np.empty(shape=(3,), dtype=np.float64)


    with nogil:
        valid_points = 0
        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):

                    valid_points += 1
                    x[0] = _apply_affine_3d_x0(k, i, j, 1, grid2world)
                    x[1] = _apply_affine_3d_x1(k, i, j, 1, grid2world)
                    x[2] = _apply_affine_3d_x2(k, i, j, 1, grid2world)

                    if constant_jacobian == 0:
                        constant_jacobian = transform._jacobian(theta, x, J)

                    for l in range(n):
                        grad[l] += (J[0, l] * mgradient[k, i, j, 0] +
                                   J[1, l] * mgradient[k, i, j, 1] +
                                   J[2, l] * mgradient[k, i, j, 2])



cdef gradient_sparse_2d(double[:] theta, Transform transform,
                                   double[:] sval, double[:] mval,
                                   double[:,:] sample_points,
                                   floating[:, :] mgradient,
                                   double[:] grad):
    r""" Gradient of the gradient w.r.t. transform parameters theta

    Computes the vector of partial derivatives w.r.t.
    each transformation parameter. The transformation itself is not necessary
    to compute the gradient, but only its Jacobian.

    Parameters
    ----------
    theta : array, shape (n,)
        parameters to compute the gradient at
    transform : instance of Transform
        the transformation with respect to whose parameters the gradient
        must be computed
    sval : array, shape (m,)
        sampled intensities from the static image at sampled_points
    mval : array, shape (m,)
        sampled intensities from the moving image at sampled_points
    sample_points : array, shape (m, 2)
        positions (in physical space) of the points the images were sampled at
    mgradient : array, shape (m, 2)
        the gradient of the moving image at the sample points
    grad : array, shape (len(theta))
        the array to write the gradient to
    """
    cdef:
        cnp.npy_intp n = theta.shape[0]
        cnp.npy_intp m = sval.shape[0]
        cnp.npy_intp offset
        int constant_jacobian = 0
        cnp.npy_intp i, j, r, c, valid_points
        double rn, cn
        double val, spline_arg, norm_factor
        double[:, :] J = np.empty(shape=(2, n), dtype=np.float64)
        double[:] prod = np.empty(shape=(n,), dtype=np.float64)

    grad[:] = 0
    with nogil:
        valid_points = 0
        for i in range(m):
            valid_points += 1
            if constant_jacobian == 0:
                constant_jacobian = transform._jacobian(theta,
                                                        sample_points[i], J)


            for j in range(n):
                grad[j] += (J[0, j] * mgradient[i, 0] +
                           J[1, j] * mgradient[i, 1])


cdef gradient_sparse_3d(double[:] theta, Transform transform,
                                   double[:] sval, double[:] mval,
                                   double[:, :] sample_points,
                                   floating[:, :] mgradient,
                                   double[:] grad):
    r""" Gradient of the  w.r.t. transform parameters theta

    Computes the vector of partial derivatives of the joint histogram w.r.t.
    each transformation parameter. The transformation itself is not necessary
    to compute the gradient, but only its Jacobian.

    Parameters
    ----------
    theta : array, shape (n,)
        parameters to compute the gradient at
    transform : instance of Transform
        the transformation with respect to whose parameters the gradient
        must be computed
    sval : array, shape (m,)
        sampled intensities from the static image at sampled_points
    mval : array, shape (m,)
        sampled intensities from the moving image at sampled_points
    sample_points : array, shape (m, 3)
        positions (in physical space) of the points the images were sampled at
    mgradient : array, shape (m, 3)
        the gradient of the moving image at the sample points
    
    grad : array, shape ( len(theta))
        the array to write the gradient to
    """
    cdef:
        cnp.npy_intp n = theta.shape[0]
        cnp.npy_intp m = sval.shape[0]
        cnp.npy_intp offset, valid_points
        int constant_jacobian = 0
        cnp.npy_intp i, j, r, c
        double rn, cn
        double val, spline_arg, norm_factor
        double[:, :] J = np.empty(shape=(3, n), dtype=np.float64)
        double[:] prod = np.empty(shape=(n,), dtype=np.float64)


    with nogil:
        valid_points = 0
        for i in range(m):
            valid_points += 1

            if constant_jacobian == 0:
                constant_jacobian = transform._jacobian(theta,
                                                        sample_points[i], J)

            for j in range(n):
                grad[j] += (J[0, j] * mgradient[i, 0] +
                           J[1, j] * mgradient[i, 1] +
                           J[2, j] * mgradient[i, 2])



def compute_ssd_mi_2d(double[:]ssd_grad, double[:,:] delta, double[:] mi_gradient):
    r""" Computes the ssd and its gradient (if requested)

    Parameters
    ----------
    ssd_grad : array, shape (n, )
        gradient with respect to transformation parameters
    delta : array, shape (R,C)
        gradient between moving and static image
    mi_gradient : array, shape (n,)
        the buffer in which to write the gradient of the mutual information.
        If None, the gradient is not computed
    """
    cdef:
        double epsilon = 2.2204460492503131e-016
        double metric_value
        double delta_2
        cnp.npy_intp nrows = delta.shape[0]
        cnp.npy_intp ncols = delta.shape[1]
        cnp.npy_intp n = ssd_grad.shape[0]

    with nogil:
        mi_gradient[:] = 0
        metric_value = 0
        delta_2 = 0
        for i in range(nrows):
            for j in range(ncols):
                if delta[i,j] < epsilon:
                    continue

            if mi_gradient is not None:
                for k in range(n):
                    mi_gradient[k] += delta[i,j] * ssd_grad[k] * 2

            delta_2 = delta[i, j]**2
            metric_value += delta_2

    return metric_value


def compute_ssd_mi_3d(double[:]ssd_grad, double[:, :, :] delta, double[:] mi_gradient):
    r"""Computes the ssd and its gradient (if requested)

        Parameters
        ----------
        ssd_grad : array, shape (n, )
        gradient with respect to transformation parameters
        delta : array, shape (R, C, K)
        mi_gradient : array, shape (n,)
        the buffer in which to write the gradient of the mutual information.
        If None, the gradient is not computed
    """
    cdef:
        double epsilon = 2.2204460492503131e-016
        double metric_value
        double delta_2

        cnp.npy_intp nrows = delta.shape[1]
        cnp.npy_intp ncols = delta.shape[2]
        cnp.npy_intp nslices = delta.shape[0]
        cnp.npy_intp n = ssd_grad.shape[0]

    with nogil:
        mi_gradient[:] = 0
        metric_value = 0
        delta_2 = 0
        for s in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    if delta[s, i,j] < epsilon:
                        continue

                if mi_gradient is not None:
                    for k in range(n):
                        mi_gradient[k] += delta[s, i, j] * ssd_grad[k] * 2

                delta_2 = delta[s, i, j]**2
                metric_value += delta_2

    return metric_value

