#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_13_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL Deform_ARRAY_API
#include <numpy/arrayobject.h>
#include "from_scipy.h"

/* initialize iterations over single array elements: */
int NI_InitPointIterator(PyArrayObject *array, NI_Iterator *iterator)
{
    int ii;

    iterator->rank_m1 = PyArray_NDIM(array) - 1;
    for(ii = 0; ii < PyArray_NDIM(array); ii++) {
        /* adapt dimensions for use in the macros: */
        iterator->dimensions[ii] = PyArray_DIM(array, ii) - 1;
        /* initialize coordinates: */
        iterator->coordinates[ii] = 0;
        /* initialize strides: */
        iterator->strides[ii] = PyArray_STRIDE(array, ii);
        /* calculate the strides to move back at the end of an axis: */
        iterator->backstrides[ii] =
                PyArray_STRIDE(array, ii) * iterator->dimensions[ii];
    }
    return 1;
}
