#ifndef DEFORM_H
#define DEFORM_H

/* the iterator structure: */
typedef struct {
    int rank_m1;
    npy_intp dimensions[NPY_MAXDIMS];
    npy_intp coordinates[NPY_MAXDIMS];
    npy_intp strides[NPY_MAXDIMS];
    npy_intp backstrides[NPY_MAXDIMS];
    npy_intp steps;
    npy_intp step_stride;
} NI_SteppingIterator;

int DeformGrid(int, int, PyArrayObject**, PyArrayObject*, PyArrayObject*, PyArrayObject**,
               int, int*, int*, int*, double*, double*);

int NI_SplineFilter1DGrad(PyArrayObject*, int, int, PyArrayObject*);

#endif
